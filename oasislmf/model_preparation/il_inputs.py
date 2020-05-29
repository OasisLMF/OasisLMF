__all__ = [
    'get_calc_rule_ids',
    'get_grouped_fm_profile_by_level_and_term_group',
    'get_grouped_fm_terms_by_level_and_term_group',
    'get_il_input_items',
    'get_policytc_ids',
    'get_step_calc_rule_ids',
    'get_step_policytc_ids',
    'write_il_input_files',
    'write_fm_policytc_file',
    'write_fm_profile_file',
    'write_fm_programme_file',
    'write_fm_xref_file'
]

import copy
import os
import sys
import warnings

import pandas as pd
import numpy as np

from ast import literal_eval

from ..utils.calc_rules import (
    get_calc_rules,
    get_step_calc_rules
)
from ..utils.coverages import SUPPORTED_COVERAGE_TYPES
from ..utils.data import (
    factorize_array,
    factorize_ndarray,
    fast_zip_arrays,
    get_dataframe,
    get_ids,
    merge_check,
    merge_dataframes,
    set_dataframe_column_dtypes,
)
from ..utils.defaults import (
    assign_defaults_to_il_inputs,
    get_default_accounts_profile,
    get_default_exposure_profile,
    get_default_fm_aggregation_profile,
    OASIS_FILES_PREFIXES,
    SOURCE_IDX,
)
from ..utils.exceptions import OasisException
from ..utils.fm import (
    DEDUCTIBLE_AND_LIMIT_TYPES,
    SUPPORTED_FM_LEVELS,
    STEP_TRIGGER_TYPES,
    COVERAGE_AGGREGATION_METHODS,
    CALCRULE_ASSIGNMENT_METHODS
)
from ..utils.log import oasis_log
from ..utils.path import as_path
from ..utils.profiles import (
    get_fm_terms_oed_columns,
    get_grouped_fm_profile_by_level_and_term_group,
    get_grouped_fm_terms_by_level_and_term_group,
    get_oed_hierarchy,
    get_step_policies_oed_mapping,
)

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_calc_rule_ids(il_inputs_df):
    """
    Returns a Numpy array of calc. rule IDs from a table of IL input items

    :param il_inputs_df: IL input items dataframe
    :type il_inputs_df: pandas.DataFrame

    :return: Numpy array of calc. rule IDs
    :rtype: numpy.ndarray
    """
    calc_rules = get_calc_rules().drop(['desc'], axis=1)
    try:
        calc_rules['id_key'] = calc_rules['id_key'].apply(literal_eval)
    except ValueError as e:
        raise OasisException("Exception raised in 'get_calc_rule_ids'", e)

    terms = ['deductible', 'deductible_min', 'deductible_max', 'limit', 'share', 'attachment']
    terms_indicators = ['{}_gt_0'.format(t) for t in terms]
    types_and_codes = ['ded_type', 'ded_code', 'lim_type', 'lim_code']

    calc_mapping_cols = ['item_id'] + terms + terms_indicators + types_and_codes + ['calcrule_id']
    il_inputs_calc_rules_df = il_inputs_df.reindex(columns=calc_mapping_cols)
    il_inputs_calc_rules_df.loc[:, terms_indicators] = np.where(il_inputs_calc_rules_df[terms] > 0, 1, 0)
    il_inputs_calc_rules_df['id_key'] = [t for t in fast_zip_arrays(*il_inputs_calc_rules_df.loc[:, terms_indicators + types_and_codes].transpose().values)]
    il_inputs_calc_rules_df = merge_dataframes(il_inputs_calc_rules_df, calc_rules, how='left', on='id_key').fillna(0)
    il_inputs_calc_rules_df['calcrule_id'] = il_inputs_calc_rules_df['calcrule_id'].astype('uint32')

    if 0 in il_inputs_calc_rules_df.calcrule_id.unique():
        err_msg = 'Calculation Rule mapping error, non-matching keys:\n'
        no_match_keys = il_inputs_calc_rules_df.loc[
            il_inputs_calc_rules_df.calcrule_id == 0
        ].id_key.unique()

        err_msg += '   {}\n'.format(tuple(terms_indicators + types_and_codes))
        for key_id in no_match_keys:
            err_msg += '   {}\n'.format(key_id)
        raise OasisException(err_msg)

    return il_inputs_calc_rules_df['calcrule_id'].values


def get_step_calc_rule_ids(il_inputs_df, step_trigger_type_cols):
    """
    Returns a Numpy array of calc. rule IDs from a table of IL input items that
    include step policies

    :param il_inputs_df: IL input items dataframe
    :type il_inputs_df: pandas.DataFrame

    :param step_trigger_type_cols: column names used to determine values for
    terms indicators and types
    :type step_trigger_type_cols: list

    :return: Numpy array of calc. rule IDs
    :rtype: numpy.ndarray
    """
    calc_rules_step = get_step_calc_rules().drop(['desc'], axis=1)
    try:
        calc_rules_step['id_key'] = calc_rules_step['id_key'].apply(literal_eval)
    except ValueError as e:
        raise OasisException("Exception raised in 'get_step_calc_rule_ids'", e)

    terms = ['deductible1', 'payout_start', 'payout_end', 'limit1', 'limit2']
    terms_indicators = ['{}_gt_0'.format(t) for t in terms]
    types = ['trigger_type', 'payout_type']

    cols = [
        'item_id', 'level_id', 'steptriggertype', 'assign_step_calcrule',
        'coverage_type_id'
    ]
    calc_mapping_cols = cols + step_trigger_type_cols + terms + terms_indicators + types + ['calcrule_id']
    il_inputs_calc_rules_df = il_inputs_df.reindex(columns=calc_mapping_cols)

    # Fill columns used to determine values for terms indicators and types
    # Columns used depend on step trigger type or sub step trigger type if
    # applicable
    # Set terms indicators and types to 0 if calc. rule should not be assigned
    def assign_terms_indicators_and_types(term, row):
        step_trigger_type = row['steptriggertype']
        # Reassign step trigger type if sub step trigger types exist
        if STEP_TRIGGER_TYPES[step_trigger_type].get('sub_step_trigger_types'):
            sub_trigger_types = STEP_TRIGGER_TYPES[step_trigger_type]['sub_step_trigger_types']
            if row['coverage_type_id'] in sub_trigger_types.keys():
                step_trigger_type = sub_trigger_types[row['coverage_type_id']]

        if get_step_policies_oed_mapping(step_trigger_type).get(term) and row['assign_step_calcrule'] != False:
            return row[get_step_policies_oed_mapping(step_trigger_type)[term]]
        else:
            return 0

    for term in terms + types:
        il_inputs_calc_rules_df[term] = il_inputs_calc_rules_df.apply(
            lambda row: assign_terms_indicators_and_types(term, row),
            axis=1
        )

    il_inputs_calc_rules_df.loc[:, terms_indicators] = np.where(il_inputs_calc_rules_df[terms] > 0, 1, 0)
    il_inputs_calc_rules_df[types] = il_inputs_calc_rules_df[types].fillna(0).astype('uint8')
    il_inputs_calc_rules_df['id_key'] = [t for t in fast_zip_arrays(*il_inputs_calc_rules_df.loc[:, terms_indicators + types].transpose().values)]

    il_inputs_calc_rules_df = merge_dataframes(il_inputs_calc_rules_df, calc_rules_step, how='left', on='id_key').fillna(0)
    # Assign passthrough calcrule ID 100 to first level
    il_inputs_calc_rules_df.loc[il_inputs_calc_rules_df['level_id'] == il_inputs_calc_rules_df['level_id'].min(), 'calcrule_id'] = 100
    il_inputs_calc_rules_df['calcrule_id'] = il_inputs_calc_rules_df['calcrule_id'].astype('uint32')

    if 0 in il_inputs_calc_rules_df.calcrule_id.unique():
        err_msg = 'Calculation Rule mapping error, non-matching keys:\n'
        no_match_keys = il_inputs_calc_rules_df.loc[
            il_inputs_calc_rules_df.calcrule_id == 0
        ].id_key.unique()

        err_msg += '   {}\n'.format(tuple(terms_indicators + types))
        for key_id in no_match_keys:
            err_msg += '   {}\n'.format(key_id)
        raise OasisException(err_msg)

    return il_inputs_calc_rules_df['calcrule_id'].values


def get_policytc_ids(il_inputs_df):
    """
    Returns a Numpy array of policy TC IDs from a table of IL input items

    :param il_inputs_df: IL input items dataframe
    :type il_inputs_df: pandas.DataFrame

    :return: Numpy array of policy TC IDs
    :rtype: numpy.ndarray
    """
    policytc_cols = [
        'layer_id', 'level_id', 'agg_id', 'calcrule_id', 'limit',
        'deductible', 'deductible_min', 'deductible_max', 'attachment',
        'share'
    ]
    fm_policytc_df = il_inputs_df.loc[:, ['item_id'] + policytc_cols].drop_duplicates()
    fm_policytc_df = fm_policytc_df[
        (fm_policytc_df['layer_id'] == 1) | (fm_policytc_df['level_id'] == fm_policytc_df['level_id'].max())
    ]

    return factorize_ndarray(fm_policytc_df.loc[:, policytc_cols[3:]].values, col_idxs=range(len(policytc_cols[3:])))[0]


def get_step_policytc_ids(
    il_inputs_df,
    step_trigger_type_cols,
    offset=0,
    idx_cols=[]
):
    """
    Returns a Numpy array of policy TC IDs from a table of IL input items that
    include step policies

    :param il_inputs_df: IL input items dataframe
    :type il_inputs_df: pandas.DataFrame

    :param step_trigger_type_cols: column names used to determine values for
    terms indicators and types
    :type step_trigger_type_cols: list

    :return: Numpy array of policy TC IDs
    :rtype: numpy.ndarray
    """
    policytc_cols = [
        'layer_id', 'level_id', 'agg_id', 'calcrule_id', 'deductible1',
        'limit1', 'step_id', 'trigger_start', 'trigger_end', 'payout_start',
        'payout_end', 'limit2', 'scale1', 'scale2'
    ]
    fm_policytc_df = il_inputs_df.loc[:, ['item_id'] + idx_cols + ['coverage_id', 'steptriggertype', 'assign_step_calcrule'] + policytc_cols[:4] + step_trigger_type_cols].drop_duplicates()

    for col in policytc_cols[4:]:
        fm_policytc_df[col] = fm_policytc_df.apply(lambda x: x[get_step_policies_oed_mapping(x['steptriggertype'])[col]] if get_step_policies_oed_mapping(x['steptriggertype']).get(col) is not None and x['assign_step_calcrule'] == True else 0, axis=1)
        fm_policytc_df[col].fillna(0, inplace=True)

    fm_policytc_df = fm_policytc_df[
        (fm_policytc_df['layer_id'] == 1) | (fm_policytc_df['level_id'] == fm_policytc_df['level_id'].max())
    ]
    fm_policytc_df['policytc_id'] = factorize_ndarray(fm_policytc_df.loc[:, policytc_cols[3:]].values, col_idxs=range(len(policytc_cols[3:])))[0]
    fm_policytc_df['pol_id'] = factorize_ndarray(fm_policytc_df.loc[:, idx_cols + ['coverage_id']].values, col_idxs=range(len(idx_cols) + 1))[0]

    step_calcrule_policytc_agg = pd.DataFrame(
        fm_policytc_df[fm_policytc_df['assign_step_calcrule'] == True]['policytc_id'].to_list(),
        index=fm_policytc_df[fm_policytc_df['assign_step_calcrule'] == True]['pol_id']
    ).groupby('pol_id').aggregate(list).to_dict()[0]

    fm_policytc_df.loc[
        fm_policytc_df['assign_step_calcrule'] == True, 'policytc_id'
    ] = fm_policytc_df.loc[fm_policytc_df['assign_step_calcrule'] == True]['pol_id'].map(step_calcrule_policytc_agg)
    fm_policytc_df['policytc_id'] = fm_policytc_df['policytc_id'].apply(
        lambda x: tuple(x) if isinstance(x, list) else x
    )

    return factorize_array(fm_policytc_df['policytc_id'])[0] + offset


def get_programme_ids(il_inputs_df, level):
    """
    Returns a Serise of Agg ids by level for the FM programme file generation

    :param il_inputs_df: IL input items dataframe
    :type il_inputs_df: pandas.DataFrame

    :param level: fm_programme level number
    :type  level: int
    """
    return il_inputs_df[il_inputs_df['level_id'] == level][['agg_id', 'coverage_id']].drop_duplicates(
        subset=['agg_id', 'coverage_id'], keep="first"
    ).agg_id.reset_index(drop=True)


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
    # Get the grouped exposure + accounts profile - this describes the
    # financial terms found in the source exposure and accounts files,
    # which are for the following FM levels: site coverage (# 1),
    # site pd (# 2), site all (# 3), cond. all (# 6), policy all (# 9),
    # policy layer (# 10).  It also describes the OED hierarchy terms
    # present in the exposure and accounts files, namely portfolio num.,
    # acc. num., loc. num., and cond. num.
    profile = get_grouped_fm_profile_by_level_and_term_group(exposure_profile, accounts_profile)

    if not profile:
        raise OasisException(
            'Unable to get a unified FM profile by level and term group. '
            'Canonical loc. and/or acc. profiles are possibly missing FM term information: '
            'FM term definitions for TIV, deductibles, limit, and/or share.'
        )

    # Get the FM aggregation profile - this describes how the IL input
    # items are to be aggregated in the various FM levels
    fmap = fm_aggregation_profile

    if not fmap:
        raise OasisException(
            'FM aggregation profile is empty - this is required to perform aggregation'
        )

    # Get the OED hierarchy terms profile - this defines the column names for loc.
    # ID, acc. ID, policy no. and portfolio no., as used in the source exposure
    # and accounts files. This is to ensure that the method never makes hard
    # coded references to the corresponding columns in the source files, as
    # that would mean that changes to these column names in the source files
    # may break the method
    oed_hierarchy = get_oed_hierarchy(exposure_profile, accounts_profile)
    acc_num = oed_hierarchy['accnum']['ProfileElementName'].lower()
    policy_num = oed_hierarchy['polnum']['ProfileElementName'].lower()
    portfolio_num = oed_hierarchy['portnum']['ProfileElementName'].lower()
    cond_num = oed_hierarchy['condnum']['ProfileElementName'].lower()

    # Get the FM terms profile (this is a simplfied view of the main grouped
    # profile, containing only information about the financial terms)
    fm_terms = get_grouped_fm_terms_by_level_and_term_group(grouped_profile_by_level_and_term_group=profile)

    # Get the list of financial terms columns for the cond. all (# 6),
    # policy all (# 9) and policy layer (# 10) FM levels - all of these columns
    # are in the accounts file, not the exposure file, so will have to be
    # sourced from the accounts dataframe
    cond_pol_layer_levels = ['cond all', 'policy all', 'policy layer']
    terms_floats = ['deductible', 'deductible_min', 'deductible_max', 'limit', 'attachment', 'share']
    terms_ints = ['ded_code', 'ded_type', 'lim_code', 'lim_type']
    terms = terms_floats + terms_ints
    term_cols_floats = get_fm_terms_oed_columns(
        fm_terms,
        levels=cond_pol_layer_levels,
        terms=terms_floats
    )
    term_cols_ints = get_fm_terms_oed_columns(
        fm_terms,
        levels=cond_pol_layer_levels,
        terms=terms_ints
    )
    term_cols = term_cols_floats + term_cols_ints

    # Set defaults and data types for all the financial terms columns in the
    # accounts dataframe
    defaults = {
        **{t: 0.0 for t in term_cols_floats},
        **{t: 0 for t in term_cols_ints},
        **{cond_num: 0},
        **{portfolio_num: '1'}
    }
    dtypes = {
        **{t: 'str' for t in [acc_num, portfolio_num, policy_num]},
        **{t: 'float64' for t in term_cols_floats},
        **{t: 'uint8' for t in term_cols_ints},
        **{t: 'uint16' for t in [cond_num]},
        **{t: 'uint32' for t in ['layer_id']}
    }

    # Get the accounts frame either directly or from a file path if provided
    accounts_df = accounts_df if accounts_df is not None else get_dataframe(
        src_fp=accounts_fp,
        col_dtypes=dtypes,
        col_defaults=defaults,
        required_cols=(acc_num, policy_num, portfolio_num,),
        empty_data_error_msg='No accounts found in the source accounts (loc.) file',
        memory_map=True,
    )
    accounts_df[SOURCE_IDX['acc']] = accounts_df.index

    if not (accounts_df is not None or accounts_fp):
        raise OasisException('No accounts frame or file path provided')

    # Determine whether step policies are listed, are not full of nans and step
    # numbers are greater than zero
    step_policies_present = False
    if 'steptriggertype' in accounts_df and 'stepnumber' in accounts_df:
        if accounts_df['steptriggertype'].notnull().any():
            if accounts_df[accounts_df['steptriggertype'].notnull()]['stepnumber'].gt(0).any():
                step_policies_present = True

    # Look for a `layer_id` column in the accounts dataframe - this column
    # will exist if the accounts file has the column - the user has the option
    # of doing this before calling the MDK. The `layer_id` column is simply
    # an enumeration of the unique (portfolio num., acc. num., policy num.)
    # combinations in the accounts file. If the column doesn't exist then
    # a custom method is called that will generate this column and set it
    # in the accounts dataframe
    # If step policies are listed use `stepnumber` column in combination
    layers_cols = [portfolio_num, acc_num]
    if step_policies_present:
        layers_cols += ['stepnumber']
    accounts_df['layer_id'] = get_ids(
        accounts_df[layers_cols + [policy_num]].drop_duplicates(keep='first'),
        layers_cols + [policy_num], group_by=layers_cols,
    ).reindex(range(len(accounts_df))).fillna(method='ffill').astype('uint32')

    # Drop all columns from the accounts dataframe which are not either one of
    # portfolio num., acc. num., policy num., cond. numb., layer ID, or one of
    # the source columns for the financial terms present in the accounts file
    # (the file should contain all financial terms relating to the cond. all
    # (# 6), policy all (# 9) and policy layer (# 10) FM levels)
    usecols = [acc_num, portfolio_num, policy_num, cond_num, 'layer_id', SOURCE_IDX['acc']] + term_cols
    # If step policies listed, keep step trigger type and columns associated
    # with those step trigger types that are present
    if step_policies_present:
        usecols += ['steptriggertype']
        # Find unique values of step policies to determine columns that need to
        # be kept
        step_trigger_types = accounts_df['steptriggertype'].dropna().unique()
        step_trigger_type_cols = [
            col for step_trigger_type in step_trigger_types for col in get_step_policies_oed_mapping(step_trigger_type, only_cols=True)
        ]
        step_trigger_type_cols = list(set(step_trigger_type_cols))
        usecols += step_trigger_type_cols
    accounts_df.drop([c for c in accounts_df.columns if c not in usecols], axis=1, inplace=True)

    # Create a list of all the IL columns for the site pd (# 2) and site all (# 3)
    # levels - these columns are in the exposure file, not the accounts
    # file, and so must be sourced from the exposure dataframe
    site_pd_and_site_all_term_cols_floats = get_fm_terms_oed_columns(fm_terms, levels=['site pd', 'site all'], terms=terms_floats)
    site_pd_and_site_all_term_cols_ints = get_fm_terms_oed_columns(fm_terms, levels=['site pd', 'site all'], terms=terms_ints)
    site_pd_and_site_all_term_cols = site_pd_and_site_all_term_cols_floats + site_pd_and_site_all_term_cols_ints

    # Check if any of these columns are missing in the exposure frame, and if so
    # set the missing columns with a default value of 0.0 in the exposure frame
    missing_floats = set(site_pd_and_site_all_term_cols_floats).difference(exposure_df.columns)
    missing_ints = set(site_pd_and_site_all_term_cols_ints).difference(exposure_df.columns)
    defaults = {
        **{t: 0.0 for t in missing_floats},
        **{t: 0 for t in missing_ints}
    }
    if defaults:
        exposure_df = get_dataframe(src_data=exposure_df, col_defaults=defaults)

    # First, merge the exposure and GUL inputs frame to augment the GUL inputs
    # frame with financial terms for level 2 (site PD) and level 3 (site all) -
    # the GUL inputs frame effectively only contains financial terms related to
    # FM level 1 (site coverage)
    gul_inputs_df = merge_dataframes(
        exposure_df.loc[:, site_pd_and_site_all_term_cols + ['loc_id']],
        gul_inputs_df,
        join_on='loc_id',
        how='inner'
    )
    gul_inputs_df.rename(columns={'item_id': 'gul_input_id'}, inplace=True)
    dtypes = {t: 'float64' for t in site_pd_and_site_all_term_cols}
    gul_inputs_df = set_dataframe_column_dtypes(gul_inputs_df, dtypes)

    # check for empty intersection between dfs
    merge_check(
        gul_inputs_df[[portfolio_num, acc_num, 'layer_id', cond_num]],
        accounts_df[[portfolio_num, acc_num, 'layer_id', cond_num]],
        on=[portfolio_num, acc_num, 'layer_id', cond_num]
    )

    # Construct a basic IL inputs frame by merging the combined exposure +
    # GUL inputs frame above, with the accounts frame, on portfolio no.,
    # account no. and layer ID (by default items in the GUL inputs frame
    # are set with a layer ID of 1)
    il_inputs_df = merge_dataframes(
        gul_inputs_df,
        accounts_df,
        on=[portfolio_num, acc_num, 'layer_id', cond_num],
        how='left',
        drop_duplicates=True
    )

    # Mark the exposure dataframes for deletion
    del exposure_df

    # At this point the IL inputs frame will contain essentially only
    # items for the coverage FM level, but will include multiple items
    # relating to single GUL input items (the higher layer items).

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

    # Drop all columns from the IL inputs dataframe which aren't one of
    # necessary columns in the GUL inputs dataframe, or one of policy num.,
    # GUL input item ID, or one of the source columns for the
    # non-coverage FM levels (site PD (# 2), site all (# 3), cond. all (# 6),
    # policy all (# 9), policy layer (# 10))
    usecols = (
        gul_inputs_df.columns.to_list() +
        [policy_num, 'gul_input_id'] +
        ([SOURCE_IDX['loc']] if SOURCE_IDX['loc'] in il_inputs_df else []) +
        ([SOURCE_IDX['acc']] if SOURCE_IDX['acc'] in il_inputs_df else []) +
        (['steptriggertype'] if 'steptriggertype' in il_inputs_df else []) +
        site_pd_and_site_all_term_cols +
        term_cols
    )
    il_inputs_df.drop(
        [c for c in il_inputs_df.columns if c not in usecols],
        axis=1,
        inplace=True
    )

    # Mark the GUL inputs frame for deletion - no longer needed
    del gul_inputs_df

    # The coverage FM level (site coverage, # 1) ID
    cov_level_id = SUPPORTED_FM_LEVELS['site coverage']['id']

    # Now set the IL input item IDs, and some other required columns such
    # as the level ID, and initial values for some financial terms,
    # including the calcrule ID and policy TC ID
    il_inputs_df = il_inputs_df.assign(
        level_id=cov_level_id,
        attachment=0.0,
        share=0.0,
        calcrule_id=0,
        policytc_id=0
    )

    # Set data types for the newer columns just added
    dtypes = {
        **{t: 'float64' for t in ['attachment', 'share']},
        **{t: 'uint32' for t in ['level_id', 'calcrule_id', 'policytc_id']}
    }
    il_inputs_df = set_dataframe_column_dtypes(il_inputs_df, dtypes)

    # Drop any items with layer IDs > 1, reset index ad order items by
    # GUL input ID.
    il_inputs_df = il_inputs_df[il_inputs_df['layer_id'] == 1]
    il_inputs_df.reset_index(drop=True, inplace=True)
    il_inputs_df.sort_values('gul_input_id', axis=0, inplace=True)

    # At this stage the IL inputs frame should only contain coverage level
    # layer 1 inputs, and the financial terms are already present from the
    # earlier merge with the exposure and GUL inputs frame - the GUL inputs
    # frame should already contain the coverage level terms

    # The list of financial terms for the sub-layer levels, which are
    # site pd (# 2), site all (# 3), cond. all (# 6), policy all (# 9) -
    # the terms for these levels do not include "attachment" or share",
    # which do exist for the (policy) layer level (# 10); also the
    # layer level terms do not include ded. or limit codes or types
    terms_floats.remove('attachment')
    terms_floats.remove('share')
    terms = terms_floats + terms_ints

    # Steps to filter out any intermediate FM levels which have no
    # financial terms, and also drop all the OED columns for the terms
    # defined for these levels
    def level_has_fm_terms(level, terms):
        try:
            level_terms_cols = get_fm_terms_oed_columns(fm_terms, levels=[level], terms=terms)
            return il_inputs_df.loc[:, level_terms_cols].any().any()
        except KeyError:
            return False

    intermediate_fm_levels = [
        level for level in list(SUPPORTED_FM_LEVELS)[1:-1]
        if level_has_fm_terms(level, terms)
    ]
    fm_levels_with_no_terms = list(set(list(SUPPORTED_FM_LEVELS)[1:-1]).difference(intermediate_fm_levels))
    no_terms_cols = get_fm_terms_oed_columns(fm_terms, levels=fm_levels_with_no_terms, terms=terms)
    il_inputs_df.drop(no_terms_cols, axis=1, inplace=True)

    # Define a list of all supported OED coverage types in the exposure
    supp_cov_types = [v['id'] for v in SUPPORTED_COVERAGE_TYPES.values()]

    # For coverage level (level_id = 1) set the `agg_id` to `coverage id`
    il_inputs_df.agg_id = il_inputs_df.coverage_id

    # The main loop for processing the financial terms for the sub-layer
    # non-coverage levels - currently these are site pd (# 2), site all (# 3),
    # cond. all (# 6), policy all (# 9).
    #
    # Each level is initially a dataframe copy of the main IL inputs
    # dataframe, which at the start only represents coverage level input
    # items. Using the level terms profile the following steps take place
    # in the loop:
    #
    # (1) financial terms defined for the level are set
    # (2) coverage type filters for the blanket deductibles and limits, if
    # they are defined in the profiles, are applied
    # (3) any blanket deductibles or limits which are expressed as TIV
    # ratios are converted to TIV shares
    #
    # Finally, the processed level dataframe is concatenated with the
    # main IL inputs dataframe, with the financial terms OED columns for
    # level removed
    for level in intermediate_fm_levels:
        level_id = SUPPORTED_FM_LEVELS[level]['id']
        level_terms = [t for t in terms if fm_terms[level_id][1].get(t)]
        level_term_cols = get_fm_terms_oed_columns(fm_terms, level_ids=[level_id], terms=terms)
        level_df = il_inputs_df[il_inputs_df['level_id'] == cov_level_id].drop_duplicates()
        level_df['level_id'] = level_id

        agg_key = [v['field'].lower() for v in fmap[level_id]['FMAggKey'].values()]
        level_df['agg_id'] = factorize_ndarray(level_df.loc[:, agg_key].values, col_idxs=range(len(agg_key)))[0]

        if level in ['cond all', 'site all', 'site coverage']:
            level_df.loc[:, level_term_cols] = level_df.loc[:, level_term_cols].fillna(0)
        else:
            level_df.loc[:, level_term_cols] = level_df.loc[:, level_term_cols].fillna(method='ffill')
            level_df.loc[:, level_term_cols] = level_df.loc[:, level_term_cols].fillna(0)

        level_df.loc[:, level_terms] = level_df.loc[:, level_term_cols].values

        level_df['deductible'] = np.where(
            level_df['coverage_type_id'].isin((profile[level_id][1].get('deductible') or {}).get('CoverageTypeID') or supp_cov_types),
            level_df['deductible'],
            0
        )
        level_df['limit'] = np.where(
            level_df['coverage_type_id'].isin((profile[level_id][1].get('limit') or {}).get('CoverageTypeID') or supp_cov_types),
            level_df['limit'],
            0
        )

        il_inputs_df = pd.concat([il_inputs_df, level_df], sort=True, ignore_index=True)
        il_inputs_df.drop(level_term_cols, axis=1, inplace=True)

    # Resequence the item IDs, as the earlier repeated concatenation of
    # the intermediate level frames may have produced a non-sequential index
    il_inputs_df['item_id'] = il_inputs_df.index + 1

    # Process the layer FM level (policy layer, # 10) inputs separately - we
    # start with merging the coverage level layer 1 items with the accounts
    # dataframe to create a separate layer level frame, on which further
    # processing is done
    cov_level_layer1_df = il_inputs_df[il_inputs_df['level_id'] == cov_level_id]
    layer_df = merge_dataframes(
        cov_level_layer1_df,
        accounts_df,
        on=[portfolio_num, acc_num],
        how='inner'
    )

    # If step policies listed, create additional column to determine agg id
    # from coverage aggregation method
    if 'steptriggertype' in layer_df:
        def assign_cov_agg_id(row):
            try:
                cov_agg_method = STEP_TRIGGER_TYPES[row['steptriggertype']]['coverage_aggregation_method']
                return COVERAGE_AGGREGATION_METHODS[cov_agg_method][row['coverage_type_id']]
            except KeyError:
                return 0

        layer_df['cov_agg_id'] = layer_df.apply(lambda row: assign_cov_agg_id(row), axis=1)

        def assign_calcrule_flag(row):
            try:
                calcrule_assign_method = STEP_TRIGGER_TYPES[row['steptriggertype']]['calcrule_assignment_method']
                return CALCRULE_ASSIGNMENT_METHODS[calcrule_assign_method][row['cov_agg_id']]

            except KeyError:
                return False

        layer_df['assign_step_calcrule'] = layer_df.apply(lambda row: assign_calcrule_flag(row), axis=1)

    # Remove the source columns for all non-layer FM levels - this includes the
    # site pd (# 2), site all (# 3), cond. all (# 6), policy all (# 9) FM levels
    cond_all_and_pol_all_term_cols = get_fm_terms_oed_columns(fm_terms, levels=['cond all', 'policy all'])
    layer_df.drop(
        [c for c in layer_df.columns if c in site_pd_and_site_all_term_cols + cond_all_and_pol_all_term_cols],
        axis=1, inplace=True
    )

    # The layer FM level (policy layer, # 10) ID
    layer_level_id = SUPPORTED_FM_LEVELS['policy layer']['id']

    # Set the layer level, layer IDs and agg. IDs
    layer_df['level_id'] = layer_level_id
    agg_key = [v['field'].lower() for v in fmap[layer_level_id]['FMAggKey'].values()]
    # If step policies listed, use agg id from coverage aggregation method
    if 'cov_agg_id' in layer_df:
        agg_key += ['cov_agg_id']
    layer_df['agg_id'] = factorize_ndarray(layer_df.loc[:, agg_key].values, col_idxs=range(len(agg_key)))[0]

    # The layer level financial terms
    terms = ['limit', 'attachment', 'share']

    # Process the financial terms for the layer level
    term_cols = get_fm_terms_oed_columns(fm_terms, levels=['policy layer'], terms=terms)
    layer_df.loc[:, term_cols] = layer_df.loc[:, term_cols].where(layer_df.notnull(), 0.0).values
    layer_df.loc[:, terms] = layer_df.loc[:, term_cols].values

    # Join the IL inputs and layer level frames, drop the FM terms
    # source columns for the layer level, and mark the layer level dataframe
    # for deletion
    il_inputs_df = pd.concat([il_inputs_df, layer_df], sort=True, ignore_index=True)
    il_inputs_df.drop(term_cols, axis=1, inplace=True)
    del layer_df

    # Assign default values to IL inputs
    il_inputs_df = assign_defaults_to_il_inputs(il_inputs_df)

    # il_inputs are not necessarily in the same order for the topmost level when layers are present,
    # fix by sorting the il_inputs_df
    il_inputs_df = il_inputs_df.sort_values(['level_id', 'loc_id', 'coverage_id']).reset_index(drop=True)

    # Resequence the level IDs and item IDs, but also store the "original"
    # FM level IDs (before the resequencing)
    il_inputs_df['orig_level_id'] = il_inputs_df['level_id']
    il_inputs_df['level_id'] = factorize_ndarray(il_inputs_df.loc[:, ['level_id']].values, col_idxs=[0])[0]
    il_inputs_df['item_id'] = il_inputs_df.index + 1

    # Set datatypes again for the deductible code and type columns, as
    # they may have changed since the processing of the intermediate level
    # terms
    dtypes = {t: 'uint8' for t in ['ded_code', 'ded_type', 'lim_code', 'lim_type']}
    il_inputs_df = set_dataframe_column_dtypes(il_inputs_df, dtypes)

    # Group and sum TIVS for items by loc. ID and agg. ID, within each
    # level, and store in a new ``agg_tiv`` column - this step is
    # preparation for the next step which is to convert % TIV deductibles
    # to TIV fractional amounts
    agg_tivs = pd.DataFrame(
        il_inputs_df.loc[:, ['level_id', 'agg_id', 'tiv']].groupby(['level_id', 'agg_id'])['tiv'].sum()
    ).reset_index()
    agg_tivs.rename(columns={'tiv': 'agg_tiv'}, inplace=True)
    il_inputs_df['agg_tiv'] = il_inputs_df.loc[:, ['level_id', 'agg_id']].merge(
        agg_tivs,
        on=['level_id', 'agg_id'],
        how='inner'
    )['agg_tiv']

    # Apply rule to convert type 2 deductibles and limits to TIV shares
    il_inputs_df['deductible'] = np.where(
        il_inputs_df['ded_type'] == DEDUCTIBLE_AND_LIMIT_TYPES['pctiv']['id'],
        il_inputs_df['deductible'] * il_inputs_df['agg_tiv'],
        il_inputs_df['deductible']
    )
    il_inputs_df['limit'] = np.where(
        il_inputs_df['lim_type'] == DEDUCTIBLE_AND_LIMIT_TYPES['pctiv']['id'],
        il_inputs_df['limit'] * il_inputs_df['agg_tiv'],
        il_inputs_df['limit']
    )

    # Set the calc. rule IDs
    if 'cov_agg_id' in il_inputs_df:
        il_inputs_df.loc[
            ~(il_inputs_df['steptriggertype'] > 0), 'calcrule_id'
        ] = get_calc_rule_ids(
            il_inputs_df[~(il_inputs_df['steptriggertype'] > 0)]
        )
        il_inputs_df.loc[
            il_inputs_df['steptriggertype'] > 0, 'calcrule_id'
        ] = get_step_calc_rule_ids(
            il_inputs_df[il_inputs_df['steptriggertype'] > 0],
            step_trigger_type_cols
        )
    else:
        il_inputs_df['calcrule_id'] = get_calc_rule_ids(il_inputs_df)

    # Set the policy TC IDs
    if 'cov_agg_id' in il_inputs_df:
        il_inputs_df.loc[
            ~(il_inputs_df['steptriggertype'] > 0), 'policytc_id'
        ] = get_policytc_ids(
            il_inputs_df[~(il_inputs_df['steptriggertype'] > 0)]
        )
        il_inputs_df.loc[
            il_inputs_df['steptriggertype'] > 0, 'policytc_id'
        ] = get_step_policytc_ids(
            il_inputs_df[il_inputs_df['steptriggertype'] > 0],
            step_trigger_type_cols,
            offset=il_inputs_df['policytc_id'].max(),
            idx_cols=[acc_num, policy_num, portfolio_num]
        )
    else:
        il_inputs_df['policytc_id'] = get_policytc_ids(il_inputs_df)

    # Final setting of data types before returning the IL input items
    dtypes = {
        **{t: 'float64' for t in ['tiv', 'agg_tiv', 'deductible', 'deductible_min', 'deductible_max', 'limit', 'attachment', 'share']},
        **{t: 'uint32' for t in ['agg_id', 'item_id', 'layer_id', 'level_id', 'orig_level_id', 'calcrule_id', 'policytc_id']},
        **{t: 'uint16' for t in [cond_num]},
        **{t: 'uint8' for t in ['ded_code', 'ded_type', 'lim_code', 'lim_type']}
    }
    il_inputs_df = set_dataframe_column_dtypes(il_inputs_df, dtypes)

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
        fm_policytc_df = il_inputs_df.loc[:, ['layer_id', 'level_id', 'agg_id', 'policytc_id']]
        fm_policytc_df.drop_duplicates().to_csv(
            path_or_buf=fm_policytc_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(fm_policytc_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException("Exception raised in 'write_fm_policytc_file'", e)

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
        # Step policies exist
        if 'cov_agg_id' in il_inputs_df:
            fm_profile_df = il_inputs_df.loc[:, ['policytc_id', 'calcrule_id']]
            cols = [
                'deductible1', 'deductible2', 'deductible3', 'attachment1',
                'limit1', 'share1', 'share2', 'share3', 'step_id',
                'trigger_start', 'trigger_end', 'payout_start', 'payout_end',
                'limit2', 'scale1', 'scale2'
            ]
            non_step_cols_map = {
                'deductible1': 'deductible',
                'deductible2': 'deductible_min',
                'deductible3': 'deductible_max',
                'attachment1': 'attachment',
                'limit1': 'limit',
                'share1': 'share'
            }
            for col in cols:
                fm_profile_df[col] = il_inputs_df.apply(lambda x: x[get_step_policies_oed_mapping(x['steptriggertype'])[col]] if x['steptriggertype'] > 0 and get_step_policies_oed_mapping(x['steptriggertype']).get(col) is not None and x['assign_step_calcrule'] == True else 0, axis=1)
            for col in non_step_cols_map.keys():
                fm_profile_df.loc[
                    ~(il_inputs_df['steptriggertype'] > 0), col
                ] = il_inputs_df.loc[
                    ~(il_inputs_df['steptriggertype'] > 0),
                    non_step_cols_map[col]
                ]
            fm_profile_df.fillna(0, inplace=True)
            fm_profile_df = fm_profile_df.drop_duplicates()

            # Ensure step_id is of int data type and set default value to 1
            dtypes = {t: 'int64' if t == 'step_id' else 'float64' for t in cols}
            fm_profile_df = set_dataframe_column_dtypes(fm_profile_df, dtypes)
            fm_profile_df.loc[fm_profile_df['step_id'] == 0, 'step_id'] = 1

            fm_profile_df.loc[:, ['policytc_id', 'calcrule_id'] + cols].to_csv(
                path_or_buf=fm_profile_fp,
                encoding='utf-8',
                mode=('w' if os.path.exists(fm_profile_fp) else 'a'),
                chunksize=chunksize,
                index=False
            )
        # No step policies
        else:
            cols = ['policytc_id', 'calcrule_id', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'limit', 'share']
            fm_profile_df = il_inputs_df.loc[:, cols]

            fm_profile_df.loc[:, cols[2:]] = fm_profile_df.loc[:, cols[2:]].round(7).values

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
            fm_profile_df = fm_profile_df.drop_duplicates()

            fm_profile_df = fm_profile_df.assign(share2=0.0, share3=0.0)

            cols = ['policytc_id', 'calcrule_id', 'deductible1', 'deductible2', 'deductible3', 'attachment1', 'limit1', 'share1', 'share2', 'share3']
            fm_profile_df.loc[:, cols].to_csv(
                path_or_buf=fm_profile_fp,
                encoding='utf-8',
                mode=('w' if os.path.exists(fm_profile_fp) else 'a'),
                chunksize=chunksize,
                index=False
            )
    except (IOError, OSError) as e:
        raise OasisException("Exception raised in 'write_fm_profile_file'", e)

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
        max_level = il_inputs_df['level_id'].max()
        programme_levels = list()

        for level in range(max_level):
            # Select The Agg ids based on the current level in the hierarchy
            if level == 0:
                # Items level (first)
                agg_from = il_inputs_df[il_inputs_df['level_id'] == il_inputs_df['level_id'].min()].drop_duplicates(subset=['loc_id', 'coverage_type_id', 'peril_id']).gul_input_id
                agg_to = il_inputs_df[il_inputs_df['level_id'] == level + 1].drop_duplicates(subset=['loc_id', 'coverage_type_id', 'peril_id']).agg_id

            else:
                # All other levels
                agg_from = get_programme_ids(il_inputs_df, level)
                agg_to = get_programme_ids(il_inputs_df, level + 1)

            programme_levels.append(
                pd.DataFrame({
                    'from_agg_id': agg_from,
                    'level_id': level + 1,
                    'to_agg_id': agg_to
                }).drop_duplicates(subset=['from_agg_id', 'level_id'], keep="first")
            )

        fm_programme_df = pd.concat(programme_levels)

        dtypes = {t: 'uint32' for t in fm_programme_df.columns}
        fm_programme_df = set_dataframe_column_dtypes(fm_programme_df, dtypes)

        fm_programme_df.to_csv(
            path_or_buf=fm_programme_fp,
            encoding='utf-8',
            mode=('w' if os.path.exists(fm_programme_fp) else 'a'),
            chunksize=chunksize,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException("Exception raised in 'write_fm_programme_file'", e)

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
                'output': factorize_ndarray(cov_level_layers_df.loc[:, ['gul_input_id', 'layer_id']].values, col_idxs=range(2))[0],
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
        raise OasisException("Exception raised in 'write_fm_xref_file'", e)

    return fm_xref_fp


@oasis_log
def write_il_input_files(
    il_inputs_df,
    target_dir,
    oasis_files_prefixes=copy.deepcopy(OASIS_FILES_PREFIXES['il']),
    chunksize=(2 * 10 ** 5)
):
    """
    Writes standard Oasis IL input files to a target directory using a
    pre-generated dataframe of IL inputs dataframe. The files written are
    ::

        fm_policytc.csv
        fm_profile.csv
        fm_programme.csv
        fm_xref.csv

    :param il_inputs_df: IL inputs dataframe
    :type exposure_df: pandas.DataFrame

    :param target_dir: Target directory in which to write the files
    :type target_dir: str

    :param oasis_files_prefixes: Oasis IL input file name prefixes
    :param oasis_files_prefixes: dict

    :param chunksize: The chunk size to use when writing out the
                      input files
    :type chunksize: int

    :return: IL input files dict
    :rtype: dict
    """
    # Clean the target directory path
    target_dir = as_path(target_dir, 'Target IL input files directory', is_dir=True, preexists=False)

    # Set chunk size for writing the CSV files - default is the minimum of 100K
    # or the IL inputs frame size
    chunksize = chunksize or min(2 * 10**5, len(il_inputs_df))

    # A dict of IL input file names and file paths
    il_input_files = {
        fn: os.path.join(target_dir, '{}.csv'.format(oasis_files_prefixes[fn])) for fn in oasis_files_prefixes
    }

    this_module = sys.modules[__name__]
    # Write the files serially
    for fn in il_input_files:
        getattr(this_module, 'write_{}_file'.format(fn))(il_inputs_df.copy(deep=True), il_input_files[fn], chunksize)

    return il_input_files
