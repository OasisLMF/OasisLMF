__all__ = [
    'get_il_input_items',
    'unified_fm_profile_by_level',
    'unified_fm_profile_by_level_and_term_group',
    'unified_fm_terms_by_level_and_term_group',
    'unified_hierarchy_terms',
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
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import OrderedDict
from itertools import (
    groupby,
)

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

from ..utils.concurrency import (
    get_num_cpus,
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

    _profiles = copy.deepcopy(profiles)
    if not _profiles:
        for pp in profile_paths:
            with io.open(pp, 'r', encoding='utf-8') as f:
                _profiles.append(json.load(f))

    comb_prof = {k: v for p in _profiles for k, v in ((k, v) for k, v in p.items() if 'FMLevel' in v)}

    return OrderedDict({
        int(k): {v['ProfileElementName']: v for v in g} for k, g in groupby(sorted(comb_prof.values(), key=lambda v: v['FMLevel']), key=lambda v: v['FMLevel'])
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
            _k: {(from_profile_fm_term_types.get(v['FMTermType'].lower()) or v['FMTermType'].lower()): v for v in g} for _k, g in groupby(sorted(ufp[k].values(), key=lambda v: v['FMTermGroupID']), key=lambda v: v['FMTermGroupID'])
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


def unified_hierarchy_terms(profiles=[], profile_paths=[], unified_profile_by_level=None, unified_profile_by_level_and_term_group=None, lowercase=True):

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

    hierarchy_terms = OrderedDict({
        k.lower(): (v['ProfileElementName'].lower() if lowercase else v['ProfileElementName'])
        for k, v in sorted(ufp[0][1].items())
    })
    hierarchy_terms.setdefault('locid', ('locnumber' if lowercase else 'LocNumber'))
    hierarchy_terms.setdefault('accid', 'accnumber' if lowercase else 'AccNumber')
    hierarchy_terms.setdefault('polid', 'polnumber' if lowercase else 'PolNumber')
    hierarchy_terms.setdefault('portid', 'portnumber' if lowercase else 'PortNumber')
    hierarchy_terms.setdefault('condid', 'condnumber' if lowercase else 'CondNumber')

    return hierarchy_terms


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
    hierarchy_terms = unified_hierarchy_terms(unified_profile_by_level_and_term_group=ufp)
    loc_num = hierarchy_terms['locid']
    acc_num = hierarchy_terms['accid']
    policy_num = hierarchy_terms['polid']
    portfolio_num = hierarchy_terms['portid']
    cond_num = hierarchy_terms['condid']

    accounts_il_terms = unified_fm_terms_by_level_and_term_group(profiles=(accpf,))
    accounts_il_cols = [__v for v in accounts_il_terms.values() for _v in v.values() for __v in _v.values() if __v]

    col_defaults = {t: (0.0 if t in accounts_il_cols else 0) for t in accounts_il_cols + [portfolio_num, cond_num]}
    col_dtypes = {
        **{t: 'str' for t in [loc_num, acc_num, portfolio_num, policy_num]},
        **{t: 'float32' for t in accounts_il_cols},
        **{t: 'int32' for t in [cond_num, 'layer_id', 'layerid']}
    }

    # Get the accounts frame either directly or from a file path if provided
    accounts_df = accounts_df if accounts_df is not None else get_dataframe(
        src_fp=accounts_fp,
        col_dtypes=col_dtypes,
        col_defaults=col_defaults,
        required_cols=(acc_num, policy_num, portfolio_num,),
        empty_data_error_msg='No accounts found in the source accounts (loc.) file',
        memory_map=True,
    )

    if not (accounts_df is not None or accounts_fp):
        raise OasisException('No accounts frame or file path provided')

    # The layer ID function = use this to set a layer ID column in the accounts
    # frame for enumerating (acc. num., policy num.) for different accounts
    def get_layer_ids(df):
        portfolio_nums = df[portfolio_num].values
        acc_nums = df[acc_num].values
        policy_nums = df[policy_num].values
        return np.hstack((
            factorize_ndarray(np.asarray(list(accnum_group)), col_idxs=range(3))[0]
            for _, accnum_group in groupby(fast_zip_arrays(portfolio_nums, acc_nums, policy_nums), key=lambda t: t[0])
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
        # Create a list of all the IL columns for the site pd and site all
        # levels
        site_pd_and_site_all_terms = [
            t for level in [FM_LEVELS['site pd']['id'], FM_LEVELS['site all']['id']]
            for t in fm_terms[level][1].values() if t
        ]

        # Check if any of these columns are missing in the exposure frame, and if so
        # set the missing columns with a default value of 0.0 in the exposure frame
        missing = set(site_pd_and_site_all_terms).difference(exposure_df.columns)
        if missing:
            exposure_df = get_dataframe(src_data=exposure_df, col_defaults={t: 0.0 for t in missing})

        # First, merge the exposure and GUL inputs frame to augment the GUL inputs
        # frame with financial terms for level 2 (site PD) and level 3 (site all) -
        # the GUL inputs frame effectively only contains financial terms related to
        # FM level 1 (site coverage)
        gul_inputs_df = merge_dataframes(
            exposure_df[site_pd_and_site_all_terms + [loc_num]],
            gul_inputs_df,
            on=loc_num,
            how='inner'
        )
        gul_inputs_df.rename(columns={'item_id': 'gul_input_id'}, inplace=True)

        # Construct a basic IL inputs frame by merging the combined exposure +
        # GUL inputs frame above, with the accounts frame, on portfolio no.,
        # account no. and layer ID (by default items in the GUL inputs frame
        # are set with a layer ID of 1)
        il_inputs_df = merge_dataframes(
            gul_inputs_df,
            accounts_df,
            on=[portfolio_num, acc_num, 'layer_id'],
            how='left'
        )

        il_inputs_df['item_id'] = il_inputs_df.index + 1
        cond_numbers = merge_dataframes(
            il_inputs_df[['item_id', 'gul_input_id', cond_num]],
            gul_inputs_df[['gul_input_id', cond_num]],
            on='gul_input_id',
            how='left'
        )[cond_num]

        il_inputs_df[cond_num] = cond_numbers

        # Mark the GUL inputs and exposure file dataframes for deletion
        del [exposure_df, cond_numbers]

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
            [policy_num, 'gul_input_id'] +
            [__v for v in fm_terms.values() for _v in v.values() for __v in _v.values() if __v]
        )
        il_inputs_df.drop(
            [c for c in il_inputs_df.columns if c not in usecols],
            axis=1,
            inplace=True
        )

        # Mark the GUL inputs frame for deletion
        del gul_inputs_df

        # Now set the IL input item IDs, and some other required columns such
        # as the level ID, and initial values for some financial terms,
        # including the calcrule ID and policy TC ID
        il_inputs_df = il_inputs_df.assign(
            level_id=cov_level,
            attachment=0,
            share=0,
            calcrule_id=-1,
            policytc_id=-1
        )

        # Set data types for the newer columns just added
        col_dtypes = {
            **{t: 'int32' for t in ['level_id', 'calcrule_id', 'policytc_id']},
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
                return il_inputs_df[[v for v in fm_terms[level][1].values() if v]].any().any()
            except KeyError:
                return False

        intermediate_fm_levels = tuple(l for l in fm_levels[1:-1] if level_has_fm_terms(l))

        # Define a list of all supported OED coverage types in the exposure
        all_cov_types = [
            v['id'] for k, v in COVERAGE_TYPES.items() if k in ['buildings', 'other', 'contents', 'bi']
        ]

        # The basic list of financial terms for the sub-layer levels - the
        # layer level terms are deductible (attachment), share and limit
        terms = ['deductible', 'deductible_min', 'deductible_max', 'limit']

        # The main loop for processing the financial terms for the sub-layer
        # non-coverage levels - currently these are site pd (2), site all (3),
        # cond. all (6), policy all (9). Each level is represented by a frame 
        # copy of the main IL inputs frame, which is then processed for the
        # level's financial terms and the calc. rule ID, and then appended
        # to the main IL inputs frame
        for level in intermediate_fm_levels:
            term_cols = [(term_col or term) for term, term_col in fm_terms[level][1].items() if term != 'share']
            level_df = il_inputs_df[il_inputs_df['level_id'] == cov_level]
            level_df['level_id'] = level

            agg_key = [v['field'].lower() for v in fmap[level]['FMAggKey'].values()]
            level_df['agg_id'] = factorize_ndarray(level_df[agg_key].values, col_idxs=range(len(agg_key)))[0]

            level_df.loc[:, term_cols] = level_df.loc[:, term_cols].where(
                level_df.loc[:, term_cols].notnull(),
                0.0
            ).values
            if level == FM_LEVELS['cond all']['id']:
                level_df.loc[:, term_cols] = level_df.loc[:, term_cols].where(level_df[cond_num] != 0, 0.0)

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
            left_on=acc_num,
            right_on=acc_num,
            how='inner'
        )

        # Set the layer level, layer IDs and agg. IDs
        layer_df['level_id'] = layer_level
        agg_key = [v['field'].lower() for v in fmap[layer_level]['FMAggKey'].values()]
        layer_df['agg_id'] = factorize_ndarray(layer_df[agg_key].values, col_idxs=range(len(agg_key)))[0]

        # The layer level financial terms
        terms = ['deductible', 'limit', 'share']

        # Process the financial terms for the layer level
        term_cols = [(v[t] or t) for v in fm_terms[layer_level].values() for t in terms]
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
        il_inputs_df['calcrule_id'] = il_inputs_df['calcrule_id'].astype('int32')
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
        raise OasisException from e

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
        raise OasisException from e

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
        raise OasisException from e

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
        raise OasisException from e

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
        raise OasisException from e

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
        raise OasisException from e

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

    # Set chunk size for writing the CSV files - default is 100K
    chunksize = min(2 * 10**5, len(il_inputs_df))

    # A debugging option
    if write_inputs_table_to_file:
        il_inputs_df.to_csv(path_or_buf=os.path.join(target_dir, 'il_inputs.csv'), index=False, encoding='utf-8', chunksize=chunksize)

    # A dict of GUL input file names and file paths
    il_input_files = {
        fn: os.path.join(target_dir, '{}.csv'.format(oasis_files_prefixes[fn])) for fn in oasis_files_prefixes
    }

    # GUL input file writers have the same filename prefixes as the input files
    # and we use this property to dynamically retrieve the methods from this
    # module
    this_module = sys.modules[__name__]
    cpu_count = get_num_cpus()

    # If the IL inputs size doesn't exceed the chunk size, or there are
    # sufficient physical CPUs to cover the number of input files to be written,
    # then use multiple threads to write the files, otherwise write them
    # serially
    if len(il_inputs_df) <= chunksize or cpu_count >= len(il_input_files):
        tasks = (
            Task(getattr(this_module, 'write_{}_file'.format(fn)), args=(il_inputs_df.copy(deep=True), il_input_files[fn], chunksize,), key=fn)
            for fn in il_input_files
        )
        num_ps = min(len(il_input_files), cpu_count)
        for _, _ in multithread(tasks, pool_size=num_ps):
            pass
    else:
        for fn, fp in il_input_files.items():
            getattr(this_module, 'write_{}_file'.format(fn))(il_inputs_df, fp, chunksize)

    return il_input_files
