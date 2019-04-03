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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
import numpy as np

from ..utils.concurrency import (
    get_num_cpus,
    multiprocess,
    multithread,
    Task,
)
from ..utils.data import (
    factorize_ndarray,
    fast_zip_arrays,
    get_dataframe,
    merge_dataframes,
    set_dataframe_column_dtypes,
)
from ..utils.defaults import (
    get_calc_rules,
    get_default_accounts_profile,
    get_default_exposure_profile,
    get_default_fm_aggregation_profile,
    OASIS_FILES_PREFIXES,
    STATIC_DATA_FP,
)
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.path import as_path
from ..utils.metadata import (
    COVERAGE_TYPES,
    FM_LEVELS,
)


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


def unified_fm_terms_by_level_and_term_group(
    profiles=[],
    profile_paths=[],
    unified_profile_by_level=None,
    unified_profile_by_level_and_term_group=None,
    lowercase=True
):

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
                    (
                        ufp[level][tiv_tgid][term_type]['ProfileElementName'].lower() if lowercase
                        else ufp[level][tiv_tgid][term_type]['ProfileElementName']
                    ) if ufp[level][tiv_tgid].get(term_type) else None
                ) for term_type in ('deductible', 'deductible_min', 'deductible_max', 'limit', 'share',)
            } for tiv_tgid in ufp[level]
        } for level in sorted(ufp)[1:]
    })


def unified_id_terms(profiles=[], profile_paths=[], unified_profile_by_level=None, unified_profile_by_level_and_term_group=None, lowercase=True):

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
        k.lower(): (v['ProfileElementName'].lower() if lowercase else v['ProfileElementName'])
        for k, v in sorted(viewitems(ufp[0][1]))
    })
    id_terms.setdefault('locid', ('locnumber' if lowercase else 'LocNumber'))
    id_terms.setdefault('accid', 'accnumber'  if lowercase else 'AccNumber')
    id_terms.setdefault('polid', 'polnumber'  if lowercase else 'PolNumber')
    id_terms.setdefault('portid', 'portnumber'  if lowercase else 'PortNumber')

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

    accounts_il_terms = unified_fm_terms_by_level_and_term_group(profiles=(accpf,))
    accounts_il_cols = [__v for v in viewvalues(accounts_il_terms) for _v in viewvalues(v) for __v in viewvalues(_v) if __v]
    col_dtypes = {
        **{t: 'str' for t in [loc_id, acc_id, portfolio_num, policy_num]},
        **{t: 'float32' for t in accounts_il_cols},
        **{t: 'int32' for t in ['layer_id', 'layerid']}
    }

    # Get the accounts frame either directly or from a file path if provided
    accounts_df = accounts_df if accounts_df is not None else get_dataframe(
        src_fp=accounts_fp,
        col_dtypes=col_dtypes,
        required_cols=(acc_id, policy_num, portfolio_num,),
        empty_data_error_msg='No accounts found in the source accounts (loc.) file',
        memory_map=True,
    )

    if not (accounts_df is not None or accounts_fp):
        raise OasisException('No accounts frame or file path provided')

    # The layer ID function = use this to set a layer ID column in the accounts
    # frame for enumerating (acc. num., policy num.) for different accounts
    def get_layer_ids(df):
        acc_ids = df[acc_id].values
        policy_nums = df[policy_num].values
        return np.hstack((
            factorize_ndarray(np.asarray(list(accnum_group)), col_idxs=range(2))[0]
            for _, accnum_group in groupby(fast_zip_arrays(acc_ids, policy_nums), key=lambda t: t[0])
        ))

    if 'layer_id' not in accounts_df or 'layerid' not in accounts_df:
        accounts_df['layer_id'] = get_layer_ids(accounts_df)

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
                how='inner'
            ),
            accounts_df,
            left_on=acc_id,
            right_on=acc_id,
            how='inner'
        )

        # At this point the IL inputs frame will contain essentially only
        # coverage level items, but will include multiple items relating to
        # single GUL input items (the higher layer items).

        # If the merge is empty raise an exception - this will happen usually
        # if there are no common acc. numbers between the GUL input items and
        # the accounts listed in the accounts file
        if il_inputs_df.empty:
            raise OasisException(
                'Inner merge of the GUL inputs + exposure file dataframe '
                'and the accounts file dataframe ({}) on acc. number '
                'is empty - '
                'please check that the acc. number columns in the exposure '
                'and accounts files respectively have a non-empty '
                'intersection'.format(accounts_fp)
            )

        # Drop all unnecessary columns.
        usecols = (
            gul_inputs_df.columns.to_list() +
            [loc_id, acc_id, portfolio_num, policy_num] +
            ['is_bi_coverage', 'group_id', 'item_id', 'coverage_id', 'layer_id', 'agg_id', 'summary_id', 'summaryset_id'] +
            [__v for v in viewvalues(fm_terms) for _v in viewvalues(v) for __v in viewvalues(_v) if __v]
        )
        il_inputs_df.drop(
            [c for c in il_inputs_df.columns if c not in usecols],
            axis=1,
            inplace=True
        )

        # Mark the GUL inputs and exposure file dataframes for deletion
        del [gul_inputs_df, exposure_df]

        # Set the GUL input item IDs by enumerating loc. number + coverage type
        # ID combinations
        gul_input_ids = factorize_ndarray(il_inputs_df[[loc_id, 'coverage_type_id']].values, col_idxs=range(2))[0]
        il_inputs_df['gul_input_id'] = gul_input_ids

        # Now set the IL input item IDs, and some other required columns such
        # as the level ID, and initial values for some financial terms,
        # including the calcrule ID and policy TC ID
        il_inputs_df = il_inputs_df.assign(
            item_id=il_inputs_df.index + 1,
            level_id=cov_level,
            attachment=0,
            share=0,
            calcrule_id=-1,
            policytc_id=-1
        )

        # Set data types for the newer columns just added
        col_dtypes = {
            **{t: 'int32' for t in ['gul_input_id', 'item_id', 'level_id', 'calcrule_id', 'policytc_id']},
            **{t: 'float32' for t in ['attachment', 'share']}
        }
        set_dataframe_column_dtypes(il_inputs_df, col_dtypes)

        # Drop any items with layer IDs > 1, reset index ad order items by
        # GUL input ID.
        il_inputs_df = il_inputs_df[il_inputs_df['layer_id'] == 1]
        il_inputs_df.reset_index(drop=True, inplace=True)
        il_inputs_df.sort_values('gul_input_id', axis=0, inplace=True)

        # At this stage the IL inputs frame should only contain coverage level
        # layer 1 inputs, and the financial terms are already present from the
        # earlier merge with the exposure and GUL inputs frame - the GUL inputs
        # frame should already contain the coverage level terms

        # Filter out any intermediate FM levels from the original list of FM
        # levels which have no financial terms
        def level_has_fm_terms(level):
            try:
                return il_inputs_df[[v for v in viewvalues(fm_terms[level][1]) if v]].any().any()
            except KeyError:
                return False
        #import ipdb; ipdb.set_trace()

        intermediate_fm_levels = tuple(l for l in fm_levels[1:-1] if level_has_fm_terms(l))

        # Define a list of all supported OED coverage types in the exposure
        all_cov_types = [
            v['id'] for k, v in viewitems(COVERAGE_TYPES) if k in ['buildings','other','contents','bi']
        ]

        # The basic list of financial term types for sub-layer levels - the
        # layer level has the same list of terms but has an additional
        # ``share`` term
        terms = ['deductible', 'deductible_min', 'deductible_max', 'limit']

        # The main loop for processing the financial terms for each sub-layer
        # non-coverage level (currently, 2, 3, 6, 9), including setting the 
        # calc. rule IDs, and append each level to the current IL inputs frame
        for level in intermediate_fm_levels:
            term_cols = [(term_col or term) for term, term_col in viewitems(fm_terms[level][1]) if term != 'share']
            level_df = il_inputs_df[il_inputs_df['level_id'] == cov_level]
            level_df['level_id'] = level
            agg_key = [v['field'].lower() for v in viewvalues(fmap[level]['FMAggKey'])]
            level_df['agg_id'] = factorize_ndarray(level_df[agg_key].values, col_idxs=range(len(agg_key)))[0]
            level_df.loc[:, term_cols] = level_df.loc[:, term_cols].where(level_df.loc[:, term_cols].notnull(), 0.0).values
            level_df.loc[:, terms] = 0.0
            level_df.loc[:, terms] = level_df.loc[:, term_cols].values
            level_df['deductible'] = np.where(
                level_df['coverage_type_id'].isin((ufp[level][1].get('deductible') or {}).get('CoverageTypeID') or all_cov_types),
                level_df['deductible'],
                0
            )
            level_df['deductible'] = np.where(
                (level_df['deductible'] == 0) | (level_df['deductible'] >= 1),
                level_df['deductible'],
                level_df['tiv'] * level_df['deductible'],
            )
            level_df['limit'] = np.where(
                level_df['coverage_type_id'].isin((ufp[level][1].get('limit') or {}).get('CoverageTypeID') or all_cov_types),
                level_df['limit'],
                0
            )
            level_df['limit'] = np.where(
                (level_df['limit'] == 0) | (level_df['limit'] >= 1),
                level_df['limit'],
                level_df['tiv'] * level_df['limit'],
            )
            il_inputs_df = pd.concat([il_inputs_df, level_df], sort=True, ignore_index=True)

        # Resequence the item IDs, as the earlier repeated concatenation of
        # the intermediate level frames may have produced a non-sequential index
        il_inputs_df['item_id'] = range(1, len(il_inputs_df) + 1)

        # Process the layer level inputs separately - we start with merging
        # the coverage level layer 1 items with the accounts frame to create
        # a separate layer level frame, on which further processing is 
        cov_level_layer1_df = il_inputs_df[il_inputs_df['level_id'] == cov_level]
        layer_df = merge_dataframes(
            cov_level_layer1_df,
            accounts_df,
            left_on=acc_id,
            right_on=acc_id,
            how='inner'
        )

        # Set the layer level, layer IDs and agg. IDs
        layer_df['level_id'] = layer_level
        agg_key = [v['field'].lower() for v in viewvalues(fmap[layer_level]['FMAggKey'])]
        layer_df['agg_id'] = factorize_ndarray(layer_df[agg_key].values, col_idxs=range(len(agg_key)))[0]

        # The layer level FM terms
        terms = ['deductible', 'limit', 'share']

        # Process the financial terms for the layer level
        term_cols = [(v[t] or t) for v in viewvalues(fm_terms[layer_level]) for t in terms]
        layer_df.loc[:, term_cols] = layer_df.loc[:, term_cols].where(layer_df.notnull(), 0.0).values
        layer_df.loc[:, terms] = layer_df.loc[:, term_cols].values
        layer_df['limit'] = layer_df['limit'].where(layer_df['limit'] != 0, 9999999999)
        layer_df['attachment'] = layer_df['deductible']
        layer_df['share'] = layer_df['share'].where(layer_df['share'] != 0, 1.0)

        # Join the IL inputs and layer level frames, and set layer ID, level ID
        # and IL item IDs
        il_inputs_df = pd.concat([il_inputs_df, layer_df], sort=True, ignore_index=True)

        del layer_df

        # Resequence the level IDs and item IDs, but also store the "old" level
        # IDs (before the resequencing)
        il_inputs_df['orig_level_id'] = il_inputs_df['level_id']
        il_inputs_df['level_id'] = factorize_ndarray(il_inputs_df[['level_id']].values, col_idxs=[0])[0]
        il_inputs_df['item_id'] = il_inputs_df.index + 1

        # Set the calc. rule IDs
        calc_rules = get_calc_rules().drop(['desc'], axis=1)
        calc_rules['id_key'] = calc_rules['id_key'].apply(eval)

        terms = ['deductible', 'deductible_min', 'deductible_max', 'limit', 'share', 'attachment']
        terms_indicators = ['{}_gt_0'.format(t) for t in terms]
        types_and_codes = ['deductible_type', 'deductible_code', 'limit_type', 'limit_code']

        il_inputs_calc_rules_df = il_inputs_df.loc[:, ['item_id'] + terms + terms_indicators + types_and_codes + ['calcrule_id']]
        for t, ti in zip(terms, terms_indicators):
            il_inputs_calc_rules_df[ti] = np.where(il_inputs_calc_rules_df[t] > 0, 1, 0)
        for t in types_and_codes:
            il_inputs_calc_rules_df[t] = 0
        il_inputs_calc_rules_df['id_key'] = [t for t in fast_zip_arrays(*il_inputs_calc_rules_df[terms_indicators + types_and_codes].transpose().values)]
        il_inputs_calc_rules_df = merge_dataframes(il_inputs_calc_rules_df, calc_rules, how='left', on='id_key')
        il_inputs_df['calcrule_id'] = il_inputs_calc_rules_df['calcrule_id']
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
        raise OasisException(e)

    return il_inputs_df, accounts_df


@oasis_log
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
        fm_policytc_df = fm_policytc_df[
            (fm_policytc_df['layer_id'] == 1) |
            (fm_policytc_df['level_id'] == fm_policytc_df['level_id'].max())
        ]
        fm_policytc_df['policytc_id'] = factorize_ndarray(fm_policytc_df[cols[3:]].values, col_idxs=range(len(cols[3:])))[0]

        fm_policytc_df[cols[:3] + ['policytc_id']].to_csv(
            path_or_buf=fm_policytc_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(fm_policytc_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fm_policytc_fp


@oasis_log
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
        cols = ['policytc_id', 'calcrule_id', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'limit', 'share']

        fm_profile_df = il_inputs_df[cols].drop_duplicates()

        fm_profile_df['policytc_id'] = factorize_ndarray(fm_profile_df[cols].values, col_idxs=range(len(cols)))[0]

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

        fm_profile_df = fm_profile_df.assign(share2=0.0, share3=0.0)

        fm_profile_df[fm_profile_df.columns].to_csv(
            path_or_buf=fm_profile_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(fm_profile_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fm_profile_fp


@oasis_log
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
        fm_programme_df = pd.concat(
            [
                il_inputs_df[il_inputs_df['level_id'] == il_inputs_df['level_id'].min()][['agg_id']].assign(level_id=0),
                il_inputs_df[['level_id', 'agg_id']]
            ]
        ).reset_index(drop=True)

        min_level, max_level = 0, fm_programme_df['level_id'].max()

        fm_programme_df = pd.DataFrame(
            {
                'from_agg_id': fm_programme_df[fm_programme_df['level_id'] < max_level]['agg_id'],
                'level_id': fm_programme_df[fm_programme_df['level_id'] > min_level]['level_id'].reset_index(drop=True),
                'to_agg_id': fm_programme_df[fm_programme_df['level_id'] > min_level]['agg_id'].reset_index(drop=True)
            },
        ).dropna(axis=0).drop_duplicates()

        set_dataframe_column_dtypes(fm_programme_df, {t: 'int32' for t in fm_programme_df.columns})

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


@oasis_log
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
        cov_level_layers_df = il_inputs_df[il_inputs_df['level_id'] == il_inputs_df['level_id'].max()]
        pd.DataFrame(
            {
                'output': factorize_ndarray(cov_level_layers_df[['gul_input_id', 'layer_id']].values, col_idxs=range(2))[0],
                'agg_id': cov_level_layers_df['gul_input_id'],
                'layer_id': cov_level_layers_df['layer_id']
            }
        ).drop_duplicates().to_csv(
            path_or_buf=fm_xref_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(fm_xref_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fm_xref_fp


@oasis_log
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
        cov_level_layers_df = il_inputs_df[il_inputs_df['level_id'] == il_inputs_df['level_id'].max()]
        pd.DataFrame(
            {
                'output': factorize_ndarray(cov_level_layers_df[['gul_input_id', 'layer_id']].values, col_idxs=range(2))[0],
                'summary_id': 1,
                'summaryset_id': 1
            }
        ).drop_duplicates().to_csv(
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
    il_inputs_df,
    target_dir,
    oasis_files_prefixes=copy.deepcopy(OASIS_FILES_PREFIXES['il']),
    write_inputs_table_to_file=False
):
    """
    Writes standard Oasis IL input files to a target directory using a
    pre-generated dataframe of IL inputs dataframe. The files written are
    ::

        fm_policytc.csv
        fm_profile.csv
        fm_programme.csv
        fm_xref.csv
        fmsummaryxref.csv

    :param il_inputs_df: IL inputs dataframe
    :type exposure_df: pandas.DataFrame

    :param target_dir: Target directory in which to write the files
    :type target_dir: str

    :param oasis_files_prefixes: Oasis IL input file name prefixes
    :param oasis_files_prefixes: dict

    :param write_inputs_table_to_file: Whether to write the IL inputs table to file
    :param write_inputs_table_to_file: bool

    :return: IL input files dict
    :rtype: dict
    """
    # Clean the target directory path
    target_dir = as_path(target_dir, 'Target IL input files directory', is_dir=True, preexists=False)

    chunksize = min(2*10**5, len(il_inputs_df))

    if write_inputs_table_to_file:
        il_inputs_df.to_csv(path_or_buf=os.path.join(target_dir, 'il_inputs.csv'), index=False, encoding='utf-8', chunksize=chunksize)

    il_input_files = {
        fn: os.path.join(target_dir, '{}.csv'.format(oasis_files_prefixes[fn])) for fn in oasis_files_prefixes
    }

    this_module = sys.modules[__name__]
    cpu_count = get_num_cpus()

    if len(il_inputs_df) <= chunksize or cpu_count >= len(il_input_files):
        tasks = (
            Task(getattr(this_module, 'write_{}_file'.format(fn)), args=(il_inputs_df.copy(deep=True), il_input_files[fn], chunksize,), key=fn)
            for fn in il_input_files
        )
        num_ps = min(len(il_input_files), cpu_count)
        for _, _ in multithread(tasks, pool_size=num_ps):
            pass
    else:
        for fn, fp in viewitems(il_input_files):
            getattr(this_module, 'write_{}_file'.format(fn))(il_inputs_df, fp, chunksize)

    return il_input_files
