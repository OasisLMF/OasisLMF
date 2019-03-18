# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open as io_open
from builtins import str

from future import standard_library
standard_library.install_aliases()

__all__ = [
    'get_il_input_items',
    'get_layer_calcrule_id',
    'get_sub_layer_calcrule_id',
    'get_sub_layer_calcrule_ids',
    'unified_fm_profile_by_level',
    'unified_fm_profile_by_level_and_term_group',
    'unified_fm_terms_by_level_and_term_group',
    'unified_id_terms',
    'write_il_input_files',
    'write_fmsummaryxref_file',
    'write_fm_policytc_file',
    'write_fm_profile_file',
    'write_fm_programme_file',
    'write_fm_xref_file'
]

import copy
import io
import json
import multiprocessing
import os
import sys

from collections import OrderedDict
from itertools import (
    chain,
    groupby,
    product,
)

from future.utils import (
    viewitems,
    viewkeys,
    viewvalues,
)

import pandas as pd
pd.options.mode.chained_assignment = None

from ..utils.concurrency import (
    multiprocess,
    multithread,
    Task,
)
from ..utils.data import (
    get_dataframe,
    merge_dataframes,
)
from ..utils.defaults import (
    get_default_accounts_profile,
    get_default_exposure_profile,
    get_default_fm_aggregation_profile,
    OASIS_FILES_PREFIXES,
)
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.path import as_path
from ..utils.metadata import (
    COVERAGE_TYPES,
    FM_LEVELS,
)


def get_policytc_ids(il_inputs_df):

    policytc_terms = ['limit', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'share', 'calcrule_id']

    policytc_df = il_inputs_df[policytc_terms]

    for col in policytc_df.columns:
        policytc_df[col] = policytc_df[col].astype(float) if col != 'calcrule_id' else policytc_df[col].astype(int)

    policytc_ids = {
        k: i + 1 for i, (k, _) in enumerate(policytc_df.groupby(policytc_terms, sort=False))
    }

    return policytc_ids, policytc_terms


def get_layer_calcrule_id(att=0.0, lim=9999999999, shr=1.0):

    if att > 0 or lim > 0 or shr > 0:
        return 2


def get_sub_layer_calcrule_id(ded, ded_min, ded_max, lim, ded_code=0, lim_code=0):

    if (ded > 0 and ded_code == 0) and (ded_min == ded_max == 0) and (lim > 0 and lim_code == 0):
        return 1
    elif (ded > 0 and ded_code == 2) and (ded_min == ded_max == 0) and (lim > 0 and lim_code == 0):
        return 4
    elif (ded > 0 and ded_code == 1) and (ded_min == ded_max == 0) and (lim > 0 and lim_code == 1):
        return 5
    elif (ded > 0 and ded_code == 2) and (ded_min == ded_max == 0) and (lim == lim_code == 0):
        return 6
    elif (ded == ded_code == 0) and (ded_min == 0 and ded_max > 0) and (lim > 0 and lim_code == 0):
        return 7
    elif (ded == ded_code == 0) and (ded_min > 0 and ded_max == 0) and (lim > 0 and lim_code == 0):
        return 8
    elif (ded == ded_code == 0) and (ded_min == 0 and ded_max > 0) and (lim == lim_code == 0):
        return 10
    elif (ded == ded_code == 0) and (ded_min > 0 and ded_max == 0) and (lim == lim_code == 0):
        return 11
    elif (ded >= 0 and ded_code == 0) and (ded_min == ded_max == 0) and (lim == lim_code == 0):
        return 12
    elif (ded == ded_code == 0) and (ded_min > 0 and ded_max > 0) and (lim == lim_code == 0):
        return 13
    elif (ded == ded_code == 0) and (ded_min == ded_max == 0) and (lim > 0 and lim_code == 0):
        return 14
    elif (ded == ded_code == 0) and (ded_min == ded_max == 0) and (lim > 0 and lim_code == 1):
        return 15
    elif (ded > 0 and ded_code == 1) and (ded_min == ded_max == 0) and (lim == lim_code == 0):
        return 16
    elif (ded > 0 and ded_code == 1) and (ded_min > 0 and ded_max > 0) and (lim == lim_code == 0):
        return 19
    elif (ded > 0 and ded_code in [0, 2]) and (ded_min > 0 and ded_max > 0) and (lim == lim_code == 0):
        return 21


def get_sub_layer_calcrule_ids(gul_or_il_inputs_df):
    calcrule_id_terms = ['deductible', 'deductible_min', 'deductible_max', 'limit']
    drop_cols = set(gul_or_il_inputs_df.columns).difference(calcrule_id_terms)

    calcrule_ids_df = gul_or_il_inputs_df.drop(drop_cols, axis=1)[calcrule_id_terms].drop_duplicates()

    calcrule_ids = {
        k: get_sub_layer_calcrule_id(*k) for k, _ in calcrule_ids_df.groupby(calcrule_id_terms, sort=False)
    }

    return calcrule_ids


def unified_fm_profile_by_level(profiles=[], profile_paths=[]):

    if not (profiles or profile_paths):
        raise OasisException('A list of source profiles (loc. or acc.) or a list of source profile paths must be provided')

    if not profiles:
        for pp in profile_paths:
            with io_open(pp, 'r', encoding='utf-8') as f:
                profiles.append(json.load(f))

    comb_prof = {k: v for p in profiles for k, v in ((k, v) for k, v in viewitems(p) if 'FMLevel' in v)}

    return OrderedDict({
        int(k): {v['ProfileElementName']: v for v in g} for k, g in groupby(sorted(viewvalues(comb_prof), key=lambda v: v['FMLevel']), key=lambda v: v['FMLevel'])
    })


def unified_fm_profile_by_level_and_term_group(profiles=[], profile_paths=[], unified_profile_by_level=None):

    ufp = copy.deepcopy(unified_profile_by_level or {})

    if not (profiles or profile_paths or ufp):
        raise OasisException(
            'A list of source profiles (loc. or acc.), or source profile paths '
            ', or a unified FM profile grouped by level, must be provided'
        )

    ufp = ufp or unified_fm_profile_by_level(profiles=profiles, profile_paths=profile_paths)

    from_profile_fm_term_types = {'deductible': 'deductible', 'deductiblemin': 'deductible_min', 'deductiblemax': 'deductible_max', 'limit': 'limit', 'share': 'share'}

    return OrderedDict({
        k: {
            _k: {(from_profile_fm_term_types.get(v['FMTermType'].lower()) or v['FMTermType'].lower()): v for v in g} for _k, g in groupby(sorted(viewvalues(ufp[k]), key=lambda v: v['FMTermGroupID']), key=lambda v: v['FMTermGroupID'])
        } for k in sorted(ufp)
    })


def unified_fm_terms_by_level_and_term_group(profiles=[], profile_paths=[], unified_profile_by_level=None, unified_profile_by_level_and_term_group=None):

    ufpl = copy.deepcopy(unified_profile_by_level or {})
    ufp = copy.deepcopy(unified_profile_by_level_and_term_group or {})

    if not (profiles or profile_paths or ufpl or ufp):
        raise OasisException(
            'A list of source profiles (loc. or acc.), or source profile paths '
            ', or a unified FM profile grouped by level or by level and term '
            'group, must be provided'
        )

    ufpl = ufpl or (unified_fm_profile_by_level(profiles=profiles, profile_paths=profile_paths) if profiles or profile_paths else {})
    ufp = ufp or unified_fm_profile_by_level_and_term_group(unified_profile_by_level=ufpl)

    return OrderedDict({
        level: {
            tiv_tgid: {
                term_type: (
                    ufp[level][tiv_tgid][term_type]['ProfileElementName'].lower() if ufp[level][tiv_tgid].get(term_type) else None
                ) for term_type in ('deductible', 'deductible_min', 'deductible_max', 'limit', 'share',)
            } for tiv_tgid in ufp[level]
        } for level in sorted(ufp)[1:]
    })


def unified_id_terms(profiles=[], profile_paths=[], unified_profile_by_level=None, unified_profile_by_level_and_term_group=None):

    ufpl = copy.deepcopy(unified_profile_by_level or {})
    ufp = copy.deepcopy(unified_profile_by_level_and_term_group or {})

    if not (profiles or profile_paths or ufpl or ufp):
        raise OasisException(
            'A list of source profiles (loc. or acc.), or source profile paths '
            ', or a unified FM profile grouped by level or by level and term '
            'group, must be provided'
        )

    ufpl = ufpl or (unified_fm_profile_by_level(profiles=profiles, profile_paths=profile_paths) if profiles or profile_paths else {})
    ufp = ufp or unified_fm_profile_by_level_and_term_group(unified_profile_by_level=ufpl)

    id_terms = OrderedDict({
        k.lower(): v['ProfileElementName'].lower()
        for k, v in sorted(viewitems(ufp[0][1]))
    })
    id_terms.setdefault('locid', 'locnumber')
    id_terms.setdefault('accid', 'accnumber')
    id_terms.setdefault('polid', 'polnumber')
    id_terms.setdefault('portid', 'portnumber')

    return id_terms


@oasis_log
def get_il_input_items(
    exposure_df,
    gul_inputs_df,
    accounts_df=None,
    accounts_fp=None,
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile(),
    fm_aggregation_profile=get_default_fm_aggregation_profile()
):
    """
    Generates and returns a Pandas dataframe of IL input items.

    :param exposure_df: Source exposure
    :type exposure_df: pandas.DataFrame

    :param gul_inputs_df: GUL input items
    :type gul_inputs_df: pandas.DataFrame

    :param accounts_df: Source accounts dataframe (optional)
    :param accounts_df: pandas.DataFrame

    :param accounts_fp: Source accounts file path (optional)
    :param accounts_fp: str

    :param exposure_profile: Source exposure profile (optional)
    :type exposure_profile: dict

    :param accounts_profile: Source accounts profile (optional)
    :type accounts_profile: dict

    :param fm_aggregation_profile: FM aggregation profile (optional)
    :param fm_aggregation_profile: dict

    :return: IL inputs dataframe
    :rtype: pandas.DataFrame

    :return Accounts dataframe
    :rtype: pandas.DataFrame 
    """
    # Get the OED profiles describing exposure, accounts, and using these also
    # unified exposure + accounts profile and the aggregation profile
    exppf = exposure_profile
    accpf = accounts_profile
    
    ufp = unified_fm_profile_by_level_and_term_group(profiles=(exppf, accpf,))

    if not ufp:
        raise OasisException(
            'Unable to get a unified FM profile by level and term group. '
            'Canonical loc. and/or acc. profiles are possibly missing FM term information: '
            'FM term definitions for TIV, deductibles, limit, and/or share.'
        )

    fmap = fm_aggregation_profile

    if not fmap:
        raise OasisException(
            'FM aggregation profile is empty - this is required to perform aggregation'
        )

    # Get the ID terms profile and use that to define the column names for loc.
    # ID, acc. ID, policy no. and portfolio no., as used in the source exposure
    # and accounts files. This is to ensure that the method never makes hard
    # coded references to the corresponding columns in the source files, as
    # that would mean that changes to these column names in the source files
    # may break the method
    id_terms = unified_id_terms(unified_profile_by_level_and_term_group=ufp)
    loc_id = id_terms['locid']
    acc_id = id_terms['accid']
    policy_num = id_terms['polid']
    portfolio_num = id_terms['portid']

    # Get the accounts frame either directly or from a file path if provided
    accounts_df = accounts_df if accounts_df is not None else get_dataframe(
        src_fp=accounts_fp,
        col_dtypes={acc_id: 'str', policy_num: 'str', portfolio_num: 'str'},
        required_cols=(acc_id, policy_num, portfolio_num,),
        empty_data_error_msg='No accounts found in the source accounts (loc.) file'
    )

    if not (accounts_df is not None or accounts_fp):
        raise OasisException('No accounts frame or file path provided')

    # Define the FM levels from the unified profile, including the coverage
    # level (the first level) and the layer level (the last level) - the FM
    # levels thus obtained should correspond to the FM levels in the OED
    # spec., as the profiles are based on the same spec. Also get the FM
    # terms profile
    fm_levels = tuple(ufp)[1:]
    cov_level = min(fm_levels)
    layer_level = max(fm_levels)

    fm_terms = unified_fm_terms_by_level_and_term_group(unified_profile_by_level_and_term_group=ufp)

    try:
        # Merge the combined exposure and GUL inputs frame with the accounts
        # frame on acc. ID - this will be the main IL inputs frame that the
        # method will continue to work on and eventually return to the caller.
        il_inputs_df = merge_dataframes(
            merge_dataframes(
                exposure_df,
                gul_inputs_df,
                left_on=loc_id,
                right_on=loc_id,
                how='outer'
            ),
            accounts_df,
            left_on=acc_id,
            right_on=acc_id,
            how='outer'
        )

        # Start the layering process - as the previous merge will have linked
        # locations with accounts with multiple policies, producing inputs
        # with multiple layers, and we need to start by completing the coverage
        # level inputs for layer 1 items only, we layer the items by using a
        # dictionary of distinct account/policy. no. combinations
        layers = OrderedDict({
            (k, p): tuple(v[policy_num].unique()).index(p) + 1 for k, v in il_inputs_df.groupby([acc_id]) for p in v[policy_num].unique()
        })

        def get_layer_id(row):
            return layers[(row[acc_id], row[policy_num])]

        # Perform the initial layering
        il_inputs_df['layer_id'] = il_inputs_df[[acc_id, policy_num]].apply(get_layer_id, axis=1)

        # Select only the layer 1 items and resequence the index
        il_inputs_df = il_inputs_df[il_inputs_df['layer_id'] == 1].reset_index()
        n = len(il_inputs_df)

        # Set the cov. level ID, item IDs, and defaults and/or temp. values for
        # attachment, share and policy TC ID - the attachment and share columns
        # are not relevant to the coverage level, but are required for the layer
        # level
        il_inputs_df = il_inputs_df.assign(
            item_id=range(1, n + 1),
            level_id=cov_level,
            agg_id=range(1, n + 1),
            attachment=0,
            share=0,
            calcrule_id=-1,
            index=il_inputs_df.index
        )

        # At this stage the IL inputs frame should only contain coverage level
        # layer 1 inputs, and the financial terms are already present from the
        # earlier merge with the exposure and GUL inputs frame - the GUL inputs
        # frame should already contain the coverage level terms

        # A helper method to determine whether a given level of IL inputs has
        # non-zero financial terms
        def has_nonzero_financial_terms(level_df, terms):
            try:
                return level_df[terms].any().any()
            except KeyError:
                return

        # Filter out any intermediate FM levels from the original list of FM
        # levels which have no financial terms
        intermediate_fm_levels = [
            l for l in fm_levels[1:-1] if has_nonzero_financial_terms(il_inputs_df, [v for v in viewvalues(fm_terms[l][1]) if v])
        ]

        # Define a list of all supported OED coverage types in the exposure
        all_cov_types = [
            v['id'] for k, v in viewitems(COVERAGE_TYPES) if k in ['buildings','other','contents','bi']
        ]

        # A helper method to resequence the levels in the current IL inputs frame
        def reset_levels(levels_df, orig_levels):
            new_levels = levels_df['level_id'].unique().tolist()
            orig_levels.update(new_levels)
            return levels_df[['level_id']].apply(lambda row: new_levels.index(row['level_id']) + 1, axis=1)

        # A helper method to perform aggregation for a given level inputs DF
        def set_level_agg_ids(level_df, level):
            agg_key = tuple(v['field'].lower() for v in viewvalues(fmap[level]['FMAggKey']))
            agg_groups = [
                [it['item_id'] for _, it in v.iterrows()] for _, v in level_df.groupby(by=list(agg_key))
            ]
            def get_agg_id(row):
                try:
                    item_group = [g for g in agg_groups if row['item_id'] in g][0]
                except IndexError:
                    return -1
                return agg_groups.index(item_group) + 1

            return level_df.apply(get_agg_id, axis=1)

        # The basic list of financial term types for sub-layer levels - the
        # layer level has the same list of terms but has an additional
        # ``share`` term
        terms = ['deductible', 'deductible_min', 'deductible_max', 'limit']

        # This is used to store the level IDs prior to any resequencing of the
        # levels as a result of removing levels with no financial terms, as
        # determined by the value of ``reduced``
        orig_levels = set()

        # The main loop for processing the financial terms for each sub-layer
        # non-coverage level (currently, 2, 3, 6, 9), including setting the 
        # calc. rule IDs, and append each level to the current IL inputs frame
        for level in intermediate_fm_levels:
            level_df = il_inputs_df[il_inputs_df['level_id'] == cov_level].copy(deep=True)
            level_df['level_id'] = level
            level_df[terms] = 0.0
            level_df[terms] = level_df[[(v[t] or t) for v in viewvalues(fm_terms[level]) for t in terms]]
            if ufp[level][1].get('deductible'):
                level_df['deductible'] = level_df['deductible'].where(level_df['coverage_type_id'].isin(ufp[level][1]['deductible'].get('CoverageTypeID') or all_cov_types), 0)
            if ufp[level][1].get('limit'):
                level_df['limit'] = level_df['limit'].where(level_df['coverage_type_id'].isin(ufp[level][1]['limit'].get('CoverageTypeID') or all_cov_types), 0)
            level_df['deductible'] = level_df['deductible'].where(
                (level_df['deductible'] == 0) | (level_df['deductible'] >= 1),
                level_df['tiv'] * level_df['deductible'],
                axis=0
            )
            level_df['limit'] = level_df['limit'].where(
                (level_df['limit'] == 0) | (level_df['limit'] >= 1),
                level_df['tiv'] * level_df['limit'],
                axis=0
            )
            level_df['agg_id'] = set_level_agg_ids(level_df, level)
            il_inputs_df = pd.concat([il_inputs_df, level_df], sort=True, ignore_index=True)

        # Create the sub-layer calc. rule IDs dict and set calcrule IDs
        calcrule_ids = get_sub_layer_calcrule_ids(il_inputs_df)
        def _get_sub_layer_calcrule_id(row):
            return calcrule_ids[(row['deductible'], row['deductible_min'], row['deductible_max'], row['limit'])]

        il_inputs_df['calcrule_id'] = il_inputs_df[terms].apply(_get_sub_layer_calcrule_id, axis=1)

        # Resequence the index and item IDs, as the earlier repeated
        # concatenation would have produced a non-sequential index
        il_inputs_df['index'] = il_inputs_df.index
        il_inputs_df['item_id'] = range(1, len(il_inputs_df) + 1)

        # Process the layer level inputs separately - we start with merging
        # the coverage level layer 1 inputs with the accounts frame to create
        # a separate layer level frame
        layer_df = merge_dataframes(
            il_inputs_df[il_inputs_df['level_id'] == cov_level],
            accounts_df,
            left_on=acc_id,
            right_on=acc_id,
            how='outer'
        )

        # In the layer frame set the layer level ID, acc. ID and policy num.,
        # and perform the initial layering
        layer_df['level_id'] = layer_level
        layer_df['layer_id'] = layer_df[[acc_id, policy_num]].apply(get_layer_id, axis=1)

        # The layer level calc. rule ID setter
        def _get_layer_calcrule_id(row):
            return get_layer_calcrule_id(row['attachment'], row['limit'], row['share'])

        terms = ['deductible', 'limit', 'share']

        # Still in the layer frame, now process the financial terms for this
        # level, and then append that to the main IL inputs frame
        layer_df[terms] = 0.0, 9999999999, 1.0
        layer_df[terms] = layer_df[[(v[t] or t) for v in viewvalues(fm_terms[layer_level]) for t in terms]]
        layer_df['limit'] = layer_df['limit'].where(layer_df['limit'] != 0, 9999999999)
        layer_df['attachment'] = layer_df['deductible']
        layer_df['share'] = layer_df['share'].where(layer_df['share'] != 0, 1.0)
        layer_df['calcrule_id'] = layer_df[['attachment', 'limit', 'share']].apply(_get_layer_calcrule_id, axis=1)
        layer_df['agg_id'] = set_level_agg_ids(layer_df, layer_level)
        il_inputs_df = pd.concat([il_inputs_df, layer_df], sort=True, ignore_index=True)

        # Only keep the required columns and resequence the levels, index and
        # item IDs - this is necessary as the earlier filtering out of
        # intermediate FM levels with no financial terms would have produced
        # a non-sequential list of FM levels and/or index and item IDs
        il_inputs_df['level_id'] = reset_levels(il_inputs_df, orig_levels)
        il_inputs_df['index'] = il_inputs_df.index
        il_inputs_df['item_id'] = range(1, len(il_inputs_df) + 1)

    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
        raise OasisException(e)

    return il_inputs_df, accounts_df


def write_fm_policytc_file(il_inputs_df, fm_policytc_fp, chunksize=100000):
    """
    Writes an FM policy T & C file.

    :param il_inputs_df: IL inputs dataframe
    :type il_inputs_df: pandas.DataFrame

    :param fm_policytc_fp: FM policy TC file path
    :type fm_policytc_fp: str

    :return: FM policy TC file path
    :rtype: str
    """
    try:
        cols = ['layer_id', 'level_id', 'agg_id', 'calcrule_id', 'limit', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'share']
        fm_policytc_df = il_inputs_df[cols].drop_duplicates()

        fm_policytc_df['policytc_id'] = pd.factorize(
            pd._libs.lib.fast_zip([
                fm_policytc_df['calcrule_id'].values,
                fm_policytc_df['deductible'].values,
                fm_policytc_df['deductible_min'].values,
                fm_policytc_df['deductible_max'].values,
                fm_policytc_df['attachment'].values,
                fm_policytc_df['limit'].values,
                fm_policytc_df['share'].values
            ])
        )[0] + 1

        fm_policytc_df = fm_policytc_df[cols[:3] + ['policytc_id']]
        fm_policytc_df.to_csv(
            path_or_buf=fm_policytc_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(fm_policytc_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fm_policytc_fp

def write_fm_profile_file(il_inputs_df, fm_profile_fp, chunksize=100000):
    """
    Writes an FM profile file.

    :param il_inputs_df: IL inputs dataframe
    :type il_inputs_df: pandas.DataFrame

    :param fm_profile_fp: FM profile file path
    :type fm_profile_fp: str

    :return: FM profile file path
    :rtype: str
    """
    try:
        cols = ['calcrule_id', 'limit', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'share']

        fm_profile_df = il_inputs_df[cols].drop_duplicates()

        fm_profile_df['policytc_id'] = pd.factorize(
            pd._libs.lib.fast_zip([
                fm_profile_df['calcrule_id'].values,
                fm_profile_df['deductible'].values,
                fm_profile_df['deductible_min'].values,
                fm_profile_df['deductible_max'].values,
                fm_profile_df['attachment'].values,
                fm_profile_df['limit'].values,
                fm_profile_df['share'].values
            ])
        )[0] + 1

        fm_profile_df.rename(
            columns={
                'deductible': 'deductible1',
                'deductible_min': 'deductible2',
                'deductible_max': 'deductible3',
                'attachment': 'attachment1',
                'limit': 'limit1',
                'share': 'share1'
            },
            inplace=True
        )

        fm_profile_df['share2'] = fm_profile_df['share3'] = 0

        fm_profile_df.to_csv(
            columns=['policytc_id', 'calcrule_id', 'deductible1', 'deductible2', 'deductible3', 'attachment1', 'limit1', 'share1', 'share2', 'share3'],
            path_or_buf=fm_profile_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(fm_profile_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fm_profile_fp

def write_fm_programme_file(il_inputs_df, fm_programme_fp, chunksize=100000):
    """
    Writes an FM programme file.

    :param il_inputs_df: IL inputs dataframe
    :type il_inputs_df: pandas.DataFrame

    :param fm_programme_fp: FM programme file path
    :type fm_programme_fp: str

    :return: FM programme file path
    :rtype: str
    """
    try:
        cov_level = FM_LEVELS['site coverage']['id']

        cov_level_df = il_inputs_df[il_inputs_df['level_id'] == 1][['agg_id']].assign(level_id=0)
        fm_programme_df = pd.DataFrame(
            pd.concat([cov_level_df, il_inputs_df], sort=True, ignore_index=True)[['level_id', 'agg_id']],
            dtype=int
        ).reset_index(drop=True)

        def from_agg_id_to_agg_id(from_level_id, to_level_id):
            iterator = (
                (from_level_it, to_level_it)
                for (_, from_level_it), (_, to_level_it) in zip(
                    fm_programme_df[fm_programme_df['level_id'] == from_level_id].iterrows(),
                    fm_programme_df[fm_programme_df['level_id'] == to_level_id].iterrows()
                )
            )
            for from_level_it, to_level_it in iterator:
                yield from_level_it['agg_id'], to_level_id, to_level_it['agg_id']

        levels = list(set(fm_programme_df['level_id']))

        data = [
            (from_agg_id, level_id, to_agg_id)
            for from_level_id, to_level_id in zip(levels, levels[1:])
            for from_agg_id, level_id, to_agg_id in from_agg_id_to_agg_id(from_level_id, to_level_id)
        ]

        fm_programme_df = pd.DataFrame(columns=['from_agg_id', 'level_id', 'to_agg_id'], data=data, dtype=int).drop_duplicates()

        fm_programme_df.to_csv(
            path_or_buf=fm_programme_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(fm_programme_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fm_programme_fp

def write_fm_xref_file(il_inputs_df, fm_xref_fp, chunksize=100000):
    """
    Writes an FM xref file.

    :param il_inputs_df: IL inputs dataframe
    :type il_inputs_df: pandas.DataFrame

    :param fm_xref_fp: FM xref file path
    :type fm_xref_fp: str

    :return: FM xref file path
    :rtype: str
    """
    try:
        data = [
            (i + 1, agg_id, layer_id) for i, (agg_id, layer_id) in enumerate(product(set(il_inputs_df['agg_id']), set(il_inputs_df['layer_id'])))
        ]

        fm_xref_df = pd.DataFrame(columns=['output', 'agg_id', 'layer_id'], data=data, dtype=int)

        fm_xref_df.to_csv(
            path_or_buf=fm_xref_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(fm_xref_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fm_xref_fp

def write_fmsummaryxref_file(il_inputs_df, fmsummaryxref_fp, chunksize=100000):
    """
    Writes a summary xref file.

    :param il_inputs_df: IL inputs dataframe
    :type il_inputs_df: pandas.DataFrame

    :param fmsummaryxref_fp: Summary xref file path
    :type fmsummaryxref_fp: str

    :return: Summary xref file path
    :rtype: str
    """
    try:
        fmsummaryxref_df = pd.DataFrame(
            columns=['output', 'summary_id', 'summaryset_id'],
            data=[(i + 1, 1, 1) for i, _ in enumerate(product(set(il_inputs_df['agg_id']), set(il_inputs_df['layer_id'])))],
            dtype=int
        )

        fmsummaryxref_df.to_csv(
            path_or_buf=fmsummaryxref_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(fmsummaryxref_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fmsummaryxref_fp


@oasis_log
def write_il_input_files(
    exposure_df,
    gul_inputs_df,
    target_dir,
    accounts_df=None,
    accounts_fp=None,
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile(),
    fm_aggregation_profile=get_default_fm_aggregation_profile(),
    oasis_files_prefixes=copy.deepcopy(OASIS_FILES_PREFIXES['il']),
    write_inputs_table_to_file=False
):
    """
    Writes standard Oasis IL input files, namely
    ::

        fm_policytc.csv
        fm_profile.csv
        fm_programme.csv
        fm_xref.csv
        fmsummaryxref.csv

    :param exposure_df: Exposure dataframe
    :type exposure_df: pandas.DataFrame

    :param gul_inputs_df: GUL inputs dataframe
    :type gul_inputs_df: pandas.DataFrame

    :param target_dir: Target directory in which to write the files
    :type target_dir: str

    :param accounts_df: Accounts dataframe (optional)
    :type accounts_df: pandas.DataFrame

    :param accounts_fp: Accounts file path (optional)
    :type accounts_fp: str

    :param exposure_profile: Exposure profile (optional)
    :type exposure_profile: dict

    :param accounts_profile: Accounts profile (optional)
    :type accounts_profile: dict

    :param fm_aggregation_profile: FM aggregation profile (optional)
    :param fm_aggregation_profile: dict

    :param oasis_files_prefixes: Oasis IL input file name prefixes
    :param oasis_files_prefixes: dict

    :param write_inputs_table_to_file: Whether to write the IL inputs table to file
    :param write_inputs_table_to_file: bool

    :return: IL input files dict
    :rtype: dict

    :return: IL inputs dataframe
    :rtype: pandas.DataFrame

    :return: Accounts dataframe
    :rtype: pandas.DataFrame
    """
    # Clean the target directory path
    target_dir = as_path(target_dir, 'Target IL input files directory', is_dir=True, preexists=False)

    il_inputs_df, _ = get_il_input_items(
        exposure_df,
        gul_inputs_df,
        accounts_df=accounts_df,
        accounts_fp=accounts_fp,
        exposure_profile=exposure_profile,
        accounts_profile=accounts_profile,
        fm_aggregation_profile=fm_aggregation_profile
    )

    if write_inputs_table_to_file:
        il_inputs_df.to_csv(path_or_buf=os.path.join(target_dir, 'il_inputs.csv'), index=False, encoding='utf-8', chunksize=100000)

    il_input_files = {
        fn: os.path.join(target_dir, '{}.csv'.format(oasis_files_prefixes[fn])) for fn in oasis_files_prefixes
    }

    this_module = sys.modules[__name__]
    concurrent_tasks = (
        Task(getattr(this_module, 'write_{}_file'.format(fn)), args=(il_inputs_df.copy(deep=True), il_input_files[fn], 100000,), key=fn)
        for fn in il_input_files
    )
    num_ps = min(len(il_input_files), multiprocessing.cpu_count())
    for _, _ in multithread(concurrent_tasks, pool_size=num_ps):
        pass

    return il_input_files, il_inputs_df, accounts_df
