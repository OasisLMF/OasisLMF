__all__ = [
    'set_calc_rule_ids',
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
    get_dtypes_and_required_cols,
)
from ..utils.defaults import (
    assign_defaults_to_il_inputs,
    get_default_accounts_profile,
    get_default_exposure_profile,
    get_default_fm_aggregation_profile,
    get_acc_dtypes,
    get_oed_default_values,
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
    get_default_step_policies_profile,
)
from .summaries import get_usefull_summary_cols, get_xref_df

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define a list of all supported OED coverage types in the exposure
supp_cov_types = [v['id'] for v in SUPPORTED_COVERAGE_TYPES.values()]


step_profile_cols = [
    'policytc_id', 'calcrule_id',
    'deductible1', 'deductible2', 'deductible3', 'attachment1',
    'limit1', 'share1', 'share2', 'share3', 'step_id',
    'trigger_start', 'trigger_end', 'payout_start', 'payout_end',
    'limit2', 'scale1', 'scale2'
]

profile_cols = ['policytc_id', 'calcrule_id', 'deductible1', 'deductible2', 'deductible3', 'attachment1', 'limit1',
                'share1', 'share2', 'share3']
profile_cols_map = {
                    'deductible': 'deductible1',
                    'deductible_min': 'deductible2',
                    'deductible_max': 'deductible3',
                    'attachment': 'attachment1',
                    'limit': 'limit1',
                    'share': 'share1'
                }


def set_calc_rule_ids(
    il_inputs_calc_rules_df,
    terms,
    terms_indicators,
    types_and_codes,
    types,
    policy_layer=False
):
    """
    Lookup and assign calc. rule IDs

    :param il_inputs_df: IL inputs items dataframe
    :type il_inputs_df: pandas.DataFrame

    :param terms: list of relevant terms
    :type terms: list

    :param terms_indicators: list indicating whether relevant terms > 0
    :type terms_indicators: list

    :param types_and_codes: list of types and codes if applicable
    :type types_and_codes: list

    :param types: list of types if applicable
    :type types: list or NoneType

    :param policy_layer: flag to indicate whether policy layer present
    :type policy_layer: bool

    :return: Numpy array of calc. rule IDs
    :type: numpy.ndarray
    """
    calc_rules = get_calc_rules(policy_layer).drop(['desc'], axis=1)
    try:
        calc_rules['id_key'] = calc_rules['id_key'].apply(literal_eval)
    except ValueError as e:
        raise OasisException("Exception raised in 'set_calc_rule_ids'", e)

    # Percentage type assigned same calc. rule as flat type so set both to flat
    # type to reduce permutations
    if types:
        il_inputs_calc_rules_df.loc[:, types] = np.where(
            il_inputs_calc_rules_df[types] == DEDUCTIBLE_AND_LIMIT_TYPES['pctiv']['id'],
            DEDUCTIBLE_AND_LIMIT_TYPES['flat']['id'],
            il_inputs_calc_rules_df[types]
        )
    il_inputs_calc_rules_df.loc[:, terms_indicators] = np.where(
        il_inputs_calc_rules_df[terms] > 0, 1, 0
    )
    il_inputs_calc_rules_df['id_key'] = [
        t for t in fast_zip_arrays(*il_inputs_calc_rules_df.loc[
            :, terms_indicators + types_and_codes
        ].transpose().values)
    ]
    il_inputs_calc_rules_df = merge_dataframes(
        il_inputs_calc_rules_df, calc_rules, how='left', on='id_key', drop_duplicates=False
    ).fillna(0)
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


def get_calc_rule_ids(il_inputs_df):
    """
    Returns a Numpy array of calc. rule IDs from a table of IL input items

    :param il_inputs_df: IL input items dataframe
    :type il_inputs_df: pandas.DataFrame

    :return: Numpy array of calc. rule IDs
    :rtype: numpy.ndarray
    """
    policy_layer_id = SUPPORTED_FM_LEVELS['policy layer']['id']

    # Get calc. rule IDs for all levels except policy layer
    terms = ['deductible', 'deductible_min', 'deductible_max', 'limit']
    terms_indicators = ['{}_gt_0'.format(t) for t in terms]
    types_and_codes = ['ded_type', 'ded_code', 'lim_type', 'lim_code']
    types = ['ded_type', 'lim_type']
    calc_mapping_cols = ['item_id'] + terms + terms_indicators + types_and_codes + ['orig_level_id', 'calcrule_id']
    il_inputs_calc_rules_df = il_inputs_df.reindex(columns=calc_mapping_cols)
    il_inputs_df.loc[
        il_inputs_df['orig_level_id'] != policy_layer_id, 'calcrule_id'
    ] = set_calc_rule_ids(
        il_inputs_calc_rules_df[
            il_inputs_calc_rules_df['orig_level_id'] != policy_layer_id
        ],
        terms, terms_indicators, types_and_codes, types
    )

    # Get calc. rule IDs for policy layer level
    terms = ['limit', 'share', 'attachment']
    terms_indicators = ['{}_gt_0'.format(t) for t in terms]
    types_and_codes = []
    types = None
    calc_mapping_cols = ['item_id'] + terms + terms_indicators + ['orig_level_id', 'calcrule_id']
    il_inputs_calc_rules_df = il_inputs_df.reindex(columns=calc_mapping_cols)
    il_inputs_df.loc[
        il_inputs_df['orig_level_id'] == policy_layer_id, 'calcrule_id'
    ] = set_calc_rule_ids(
        il_inputs_calc_rules_df[
            il_inputs_calc_rules_df['orig_level_id'] == policy_layer_id
        ],
        terms, terms_indicators, types_and_codes, types, policy_layer=True
    )

    return il_inputs_df['calcrule_id'].values


def get_step_calc_rule_ids(il_inputs_df):
    """
    Returns a Numpy array of calc. rule IDs from a table of IL input items that
    include step policies

    :param il_inputs_df: IL input items dataframe
    :type il_inputs_df: pandas.DataFrame

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

    calc_mapping_cols = cols + terms + terms_indicators + types + ['calcrule_id']
    il_inputs_calc_rules_df = il_inputs_df.reindex(columns=calc_mapping_cols)

    il_inputs_calc_rules_df.loc[:, terms_indicators] = np.where(il_inputs_calc_rules_df[terms] > 0, 1, 0)
    il_inputs_calc_rules_df[types] = il_inputs_calc_rules_df[types].fillna(0).astype('uint8')
    il_inputs_calc_rules_df['id_key'] = [t for t in fast_zip_arrays(*il_inputs_calc_rules_df.loc[:, terms_indicators + types].transpose().values)]
    il_inputs_calc_rules_df = merge_dataframes(il_inputs_calc_rules_df, calc_rules_step, how='left', on='id_key', drop_duplicates=False).fillna(0)

    # Assign passthrough calcrule ID 100 to first level
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
    policytc_cols = ['calcrule_id', 'limit', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'share']
    return factorize_ndarray(il_inputs_df.loc[:, policytc_cols].values, col_idxs=range(len(policytc_cols)))[0]


def get_step_policytc_ids(
    il_inputs_df,
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
    fm_policytc_cols = [col for col in idx_cols + ['layer_id', 'level_id', 'agg_id', 'coverage_id', 'assign_step_calcrule'] + step_profile_cols
                        if col in il_inputs_df.columns]
    fm_policytc_df = il_inputs_df[fm_policytc_cols]
    fm_policytc_df['policytc_id'] = factorize_ndarray(fm_policytc_df.loc[:, ['layer_id', 'level_id', 'agg_id']].values, col_idxs=range(3))[0]
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


def __level_has_fm_terms(df, fm_column_names):
    """Check if a df has non null values for a list of column name
    :param df: dataframe
    :type df: pandas.DataFrame

    :param fm_column_names: column names used
    :type fm_column_names: list

    :return: True is any column has at least one non null value
    :rtype: bool
    """
    return df.loc[:, set(fm_column_names).intersection(df.columns)].any().any()


def __get_bi_tiv_col_name(profile):
    """get the name of the bi tiv column
    :param profile: profile containing all the terms
    :type profile: dict

    :return: the name of the bi tiv column
    :rtype: str
    """
    cov_level_id = SUPPORTED_FM_LEVELS['site coverage']['id']
    bi_cov_id = SUPPORTED_COVERAGE_TYPES['bi']['id']
    return profile[cov_level_id][bi_cov_id]['tiv']['ProfileElementName'].lower()


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


def get_account_df(accounts_fp, accounts_profile):
    """
    Get the accounts frame from a file path
    :param accounts_fp: Source accounts file path
    :type accounts_fp: str

    :param accounts_profile: Source accounts profile
    :type accounts_profile: dict

    :return: the accounts dataframe
    :rtype: pandas.DataFrame
    """
    acc_num = accounts_profile['AccNumber']['ProfileElementName'].lower()
    policy_num = accounts_profile['PolNumber']['ProfileElementName'].lower()
    portfolio_num = accounts_profile['PortNumber']['ProfileElementName'].lower()
    cond_num = accounts_profile['CondNumber']['ProfileElementName'].lower()
    layer_num = accounts_profile['LayerNumber']['ProfileElementName'].lower()

    # Get the FM terms profile (this is a simplfied view of the main grouped
    # profile, containing only information about the financial terms)
    profile = get_grouped_fm_profile_by_level_and_term_group(accounts_profile)
    fm_terms = get_grouped_fm_terms_by_level_and_term_group(grouped_profile_by_level_and_term_group=profile)

    # Get the list of financial terms columns for the cond. all (# 6),
    # policy all (# 9) and policy layer (# 10) FM levels - all of these columns
    # are in the accounts file, not the exposure file, so will have to be
    # sourced from the accounts dataframe
    cond_pol_layer_levels = ['cond all', 'policy all', 'policy layer']
    terms_floats = ['deductible', 'deductible_min', 'deductible_max', 'limit', 'attachment', 'share']
    terms_ints = ['ded_code', 'ded_type', 'lim_code', 'lim_type']

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
    defaults = get_oed_default_values(terms=term_cols)
    defaults[cond_num] = 0
    defaults[portfolio_num] = 1
    oed_acc_dtypes, _ = get_dtypes_and_required_cols(get_acc_dtypes)
    dtypes = {
        **{t: 'str' for t in [acc_num, portfolio_num, policy_num]},
        **{t: 'float64' for t in term_cols_floats},
        **{t: 'uint8' for t in term_cols_ints},
        **{t: 'uint16' for t in [cond_num]},
        **{t: 'uint32' for t in ['layer_id']},
        **oed_acc_dtypes
    }

    accounts_df = get_dataframe(
        src_fp=accounts_fp,
        col_dtypes=dtypes,
        col_defaults=defaults,
        required_cols=(acc_num, policy_num, portfolio_num,),
        empty_data_error_msg='No accounts found in the source accounts (loc.) file',
        memory_map=True,
    )
    accounts_df[SOURCE_IDX['acc']] = accounts_df.index

    # Determine whether step policies are listed, are not full of nans and step
    # numbers are greater than zero
    step_policies_present = False
    if 'steptriggertype' in accounts_df and 'stepnumber' in accounts_df:
        if accounts_df['steptriggertype'].notnull().any():
            if accounts_df[accounts_df['steptriggertype'].notnull()]['stepnumber'].gt(0).any():
                step_policies_present = True

    # Determine whether layer num. column exists in the accounts dataframe and
    # create it if needed, filling it with default value. The layer num. field
    # is used to identify unique layers in cases where layers share the same
    # policy num.
    # Create `layer_id` column, which is simply an enumeration of the unique
    # (portfolio_num., acc. num., policy num., layer num.) combinations in the
    # accounts file.
    # If step policies are listed use `stepnumber` column in combination
    if layer_num not in accounts_df:
        accounts_df[layer_num] = 1
    accounts_df[layer_num].fillna(1, inplace=True)
    layers_cols = [portfolio_num, acc_num]
    if step_policies_present:
        layers_cols += ['stepnumber']
    id_df = accounts_df[layers_cols + [policy_num, layer_num]].drop_duplicates(keep='first')
    id_df['layer_id'] = get_ids(id_df,
        layers_cols + [policy_num, layer_num], group_by=layers_cols,
    ).astype('uint32')
    accounts_df = merge_dataframes(accounts_df, id_df, join_on=layers_cols + [policy_num, layer_num])

    # Drop all columns from the accounts dataframe which are not either one of
    # portfolio num., acc. num., policy num., cond. numb., layer ID, or one of
    # the source columns for the financial terms present in the accounts file
    # (the file should contain all financial terms relating to the cond. all
    # (# 6), policy all (# 9) and policy layer (# 10) FM levels)
    usecols = [acc_num, portfolio_num, policy_num, cond_num, 'layer_id', SOURCE_IDX['acc'], 'condpriority'] + term_cols
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

    return accounts_df


def __merge_exposure_and_gul(exposure_df, gul_inputs_df, fm_terms, profile, oed_hierarchy):
    """merge the exposure df and the gul df

    :param exposure_df: exposure dataframe
    :type exposure_df: pandas.DataFrame

    :param gul_inputs_df: gul dataframe
    :type gul_inputs_df: pandas.DataFrame

    :param fm_terms: fm terms to use
    :type fm_terms: dict

    :param profile: Source profile
    :type profile: dict

    :param oed_hierarchy: oed_hierarchy
    :type oed_hierarchy: dict

    :return: merge dataframe between exposure_df and gul_inputs_df
    :rtype: pandas.DataFrame


    """
    cond_num = oed_hierarchy['condnum']['ProfileElementName'].lower()

    # get usefull term columns from exposure_df
    site_pd_and_site_all_term_cols = get_fm_terms_oed_columns(fm_terms, levels=['site pd', 'site all'])
    exposure_usefull_fm_cols = list(set(site_pd_and_site_all_term_cols).intersection(exposure_df.columns))

    # The coverage FM level (site coverage, #1) ID
    cov_level_id = SUPPORTED_FM_LEVELS['site coverage']['id']
    # Get the TIV column names and corresponding coverage types
    tiv_terms = {
        v['tiv']['CoverageTypeID']: v['tiv']['ProfileElementName'].lower()
        for v in profile[cov_level_id].values()
    }

    # Calculate sum of TIV columns
    exposure_df['tiv_sum'] = exposure_df[tiv_terms.values()].sum(axis=1)

    # Identify BI TIV column
    bi_tiv_col = __get_bi_tiv_col_name(profile)

    # merge exposure_df and gul_inputs_df #####
    return merge_dataframes(
        exposure_df.loc[:, exposure_usefull_fm_cols + [
            'loc_id', 'tiv_sum', bi_tiv_col, cond_num
        ]],
        gul_inputs_df,
        join_on='loc_id',
        how='inner'
    ).rename(columns={'item_id': 'gul_input_id'})


def __merge_gul_and_account(gul_inputs_df, accounts_df, fm_terms, oed_hierarchy):
    """prepare gul and account df and merge them based on [portfolio_num, acc_num, cond_num]"""

    acc_num = oed_hierarchy['accnum']['ProfileElementName'].lower()
    portfolio_num = oed_hierarchy['portnum']['ProfileElementName'].lower()
    cond_num = oed_hierarchy['condnum']['ProfileElementName'].lower()
    loc_num = oed_hierarchy['locnum']['ProfileElementName'].lower()

    ###### prepare accounts_df #####
    # create account line without condition
    null_cond = accounts_df[accounts_df[cond_num] != 0]
    null_cond[cond_num] = 0
    level_id = SUPPORTED_FM_LEVELS['cond all']['id']
    level_term_cols = get_fm_terms_oed_columns(fm_terms, level_ids=[level_id])
    null_cond[level_term_cols] = 0
    accounts_df = pd.concat([accounts_df, null_cond])

    ##### merge accounts_df and gul_inputs_df #####
    # check for empty intersection between dfs
    merge_check(
        gul_inputs_df[[portfolio_num, acc_num, cond_num]],
        accounts_df[[portfolio_num, acc_num, cond_num]],
        on=[portfolio_num, acc_num, cond_num]
    )
    # Construct a basic IL inputs frame by merging the combined exposure +
    # GUL inputs frame above, with the accounts frame, on portfolio no.,
    # account no. and condition no. (by default items in the GUL inputs frame
    # are set with a condition no. of 0)
    column_base_il_df =  merge_dataframes(
        gul_inputs_df,
        accounts_df,
        on=[portfolio_num, acc_num, cond_num],
        how='left',
        drop_duplicates=True
    )

    missing_account_row = column_base_il_df.loc[column_base_il_df['layer_id'].isna()]
    if not missing_account_row.empty:
        raise OasisException("locations have policies, accounts combination not present in the account file \n" +
                             missing_account_row[[loc_num, portfolio_num, acc_num, cond_num]].drop_duplicates().to_string(index=False))

    # If the merge is empty raise an exception - this will happen usually
    # if there are no common acc. numbers between the GUL input items and
    # the accounts listed in the accounts file
    if column_base_il_df.empty:
        raise OasisException(
            'Inner merge of the GUL inputs + exposure file dataframe '
            'and the accounts file dataframe  on acc. number '
            'is empty - '
            'please check that the acc. number columns in the exposure '
            'and accounts files respectively have a non-empty '
            'intersection'
        )
    return column_base_il_df


def __update_level(group, min_level, groups):
    """
    update the sub levels of all the parents of a group
    the sublevel is increased by 1 to match the depth of the childs of the parent group
    """

    while True:
        if group['level'] < min_level:
            group['level'] = min_level
            if group.get('parent') is not None:
                group = groups[group['parent']]
                min_level += 1
            else:

                return min_level
        else:
            return min_level


def __extract_level_location_to_agg_cond_dict(account_groups, base_level, max_level):
    """
    from the groups of account_groups, create a dictionary that will give the mapping between
    (ptf, account, level_id, layer_id, location) => (agg_id, condition)
    """
    level_location_to_agg_cond_dict = {}
    level_grouping = {}
    agg_dict = {level_id: 1 for level_id in range(base_level, max_level + 1)}

    def set_agg_cond(ptf, account, group_id, group, level_id, layer_id, condition):
        level_group_key = (ptf, account, group_id, level_id)
        if level_group_key not in level_grouping:
            level_grouping[level_group_key] = agg_dict[level_id]
            agg_dict[level_id] += 1
        for _, _, location in group['locations']:
            level_location_to_agg_cond_dict[(ptf, account, level_id, layer_id, location)] = (level_grouping[level_group_key], condition)

    for (ptf, account), groups in account_groups.items():
        #first we iter through groups to create agg
        no_parent_not_top_groups = set(groups)
        for group_id, group in groups.items():
            for layer_id, ((_, _, condition), _) in group['layers'].items():
                for level_id in range(base_level, group['level']):
                    set_agg_cond(ptf, account, group_id, group, level_id, layer_id, 0)

                if group['level'] == max_level:
                    no_parent_not_top_groups.discard(group_id)
                set_agg_cond(ptf, account, group_id, group, group['level'], layer_id, condition)

                for child_group_id in group.get('childs', []):
                    no_parent_not_top_groups.discard(child_group_id)
                    set_agg_cond(ptf, account, group_id, groups[child_group_id], group['level'], layer_id, condition)

        for group_id in no_parent_not_top_groups:
            for level_id in range(group['level'] + 1, max_level + 1):
                group = groups[group_id]
                for layer_id in group['layers']:
                    set_agg_cond(ptf, account, group_id, group, level_id, layer_id, 0)

    return level_location_to_agg_cond_dict


def __get_cond_grouping_hierarchy(column_base_il_df, portfolio_num, acc_num, cond_num, loc_num, base_level):
    """create group of locations based on the condition number and condition hierarchy found in column_base_il_df

    simplified logic
    go through each record in column_base_il_df:
    if location is new:
        if condition is new:
            create a new group at base level
        if the condition group is already created:
             add the location to the group
    if the location is already part of a group:
        if condition is new:
            if layer is already in the location group:
               if new condition priority is smaller:
                   a new group is created and the location is move to it
                   the old location group is set as parent of the new group
            if new condition priority is bigger:
                a new group is created with the location and the condition of the old group
                the old group is set as parent of he new group
        if the condition group is already created:
            we add the condition if it correspond to a new layer
    """
    if 'condpriority' not in column_base_il_df.columns:
        column_base_il_df['condpriority'] = 0
    column_base_il_df['condpriority'].fillna(0)

    cond_hierachy = {}
    for rec in column_base_il_df[[portfolio_num, acc_num, 'layer_id', cond_num, 'condpriority']].to_dict(orient="records"):
        cond_hierachy[(rec[portfolio_num], rec[acc_num], rec[cond_num])] = (rec['layer_id'], rec['condpriority'])

    loc_to_group = {}
    cond_to_group = {}
    account_groups = {}
    last_group_id = 1
    max_level = base_level
    for rec in column_base_il_df[[portfolio_num, acc_num, cond_num, loc_num, 'layer_id', 'condpriority']].drop_duplicates().to_dict(orient="records"):
        groups = account_groups.setdefault((rec[portfolio_num], rec[acc_num]), {})
        cond_key = (rec[portfolio_num], rec[acc_num], rec[cond_num])
        loc_key = (rec[portfolio_num], rec[acc_num], rec[loc_num])
        if loc_key not in loc_to_group:
            if cond_key not in cond_to_group:
                # print("first time condition, first time location")
                group = {'locations': {loc_key},
                         'layers': {rec['layer_id']: (cond_key, rec['condpriority'])},
                         'needed': {rec['layer_id']: set()},
                         'level': base_level}
                groups[last_group_id] = group
                loc_to_group[loc_key] = last_group_id
                cond_to_group[cond_key] = last_group_id
                last_group_id += 1

            else:
                # print("first time location, already condition")
                group_id = cond_to_group[cond_key]
                group = groups[group_id]
                group['locations'].add(loc_key)
                loc_to_group[loc_key] = group_id

                # update the location needed for correct condition logic
                for layer, needed in group['needed'].items():
                    if layer != rec['layer_id']:
                        needed.add(loc_key)
                parent_group_id = group.get("parent")
                while parent_group_id:
                    parent_group = groups[parent_group_id]
                    for layer, needed in parent_group['needed'].items():
                        needed.add(loc_key)
                    parent_group_id = parent_group.get("parent")
        else:
            if cond_key not in cond_to_group:
                # print("first time condition, already location")
                prev_group_id = loc_to_group[loc_key]
                prev_group = groups[prev_group_id]
                if rec['layer_id'] in prev_group['layers']:
                    prev_cond_key, prev_CondPriority = prev_group['layers'][rec['layer_id']]
                    if rec['condpriority'] == prev_CondPriority:
                        raise OasisException(f'condition of the same priority and policy {cond_key} {prev_cond_key}')
                    elif rec['condpriority'] < prev_CondPriority:
                        group = {'locations': {loc_key},
                                 'layers': {rec['layer_id']: (cond_key, rec['condpriority'])},
                                 'needed': {rec['layer_id']: set()},
                                 'parent': prev_group_id,
                                 'level': base_level}
                        groups[last_group_id] = group
                        prev_group.setdefault('childs', set()).add(last_group_id)
                        prev_group['locations'].remove(loc_key)
                        max_level = max(max_level, __update_level(prev_group, base_level + 1, groups))

                        loc_to_group[loc_key] = last_group_id
                        cond_to_group[cond_key] = last_group_id
                        last_group_id += 1
                    else: # CondPriority > prev_CondPriority
                        group = {'locations': {loc_key},
                                 'layers': {rec['layer_id']: (prev_cond_key, prev_CondPriority)},
                                 'needed': {rec['layer_id']: set()},
                                 'parent': prev_group_id,
                                 'level': base_level}
                        groups[last_group_id] = group
                        loc_to_group[loc_key] = last_group_id
                        cond_to_group[prev_cond_key] = last_group_id

                        prev_group['layers'][rec['layer_id']] = (cond_key, rec['condpriority'])
                        prev_group['locations'].remove(loc_key)
                        cond_to_group[cond_key] = prev_group_id
                        prev_group.setdefault('childs', set()).add(last_group_id)
                        max_level = max(max_level, __update_level(prev_group, base_level + 1, groups))
                        last_group_id += 1
                else:
                    group['layers'][rec['layer_id']] = (cond_key, rec['condpriority'])
                    cond_to_group[cond_key] = prev_group_id

                    # update the location needed for correct condition logic
                    group['needed'][rec['layer_id']] = set(group['locations']) - set([loc_key])
                    parent_group_id = group.get("parent")
                    while parent_group_id:
                        parent_group = groups[parent_group_id]
                        if rec['layer_id'] not in parent_group['needed']:
                            parent_group['needed'][rec['layer_id']] = set(parent_group['locations'])
                        parent_group_id = group.get("parent")
            else:
                # print("already condition, already location")
                group = groups[loc_to_group[loc_key]]
                if rec['layer_id'] not in group['layers']:
                    group['layers'][rec['layer_id']] = (cond_key, rec['condpriority'])

                    # update the location needed for correct condition logic
                    group['needed'][rec['layer_id']] = set(group['locations']) - set([loc_key])
                    parent_group_id = group.get("parent")
                    while parent_group_id:
                        parent_group = groups[parent_group_id]
                        if rec['layer_id'] not in prev_group['needed']:
                            parent_group['needed'][rec['layer_id']] = set(parent_group['locations'])
                        parent_group_id = group.get("parent")

                else:
                    group['needed'][rec['layer_id']].discard(loc_key)

    missing_conditions = []
    for account, groups in account_groups.items():
        for group_id, group in groups.items():
            for layer_id, loc_keys in group['needed'].items():
                for loc_key in loc_keys:
                    missing_conditions.append(loc_key+(layer_id,))

    if missing_conditions:
        missing_conditions_df = pd.DataFrame(missing_conditions, columns=[portfolio_num, acc_num, loc_num, 'layer_id'])
        raise OasisException(f"missing conditions for :\n{missing_conditions_df.to_string(index=False)}")

    return account_groups, max_level


def __get_level_location_to_agg_cond(column_base_il_df, oed_hierarchy):
    """create a dataframe with the computed agg id base on the condition and condition priority
    """
    acc_num = oed_hierarchy['accnum']['ProfileElementName'].lower()
    loc_num = oed_hierarchy['locnum']['ProfileElementName'].lower()
    portfolio_num = oed_hierarchy['portnum']['ProfileElementName'].lower()
    cond_num = oed_hierarchy['condnum']['ProfileElementName'].lower()

    base_level = 0
    account_groups, max_level = __get_cond_grouping_hierarchy(column_base_il_df, portfolio_num, acc_num, cond_num, loc_num, base_level)
    level_location_to_agg_cond_dict = __extract_level_location_to_agg_cond_dict(account_groups, base_level, max_level)

    level_location_to_agg_cond = pd.DataFrame.from_records(
        [(ptf, account, level_id, layer_id, location, agg_id, condition)
         for (ptf, account, level_id, layer_id, location), (agg_id, condition)
         in level_location_to_agg_cond_dict.items()],
        columns=[portfolio_num, acc_num, 'level_id', 'layer_id', loc_num, 'agg_id', cond_num]
    )

    return level_location_to_agg_cond, max_level


def compute_agg_tiv(tiv_df, agg_key, bi_tiv_col, loc_num):
    """ compute the agg tiv depending on the agg_key"""
    agg_tiv_df = (tiv_df.drop_duplicates(agg_key + [loc_num], keep='first')[list(set(agg_key + ['tiv', 'tiv_sum', bi_tiv_col]))]
          .groupby(agg_key,  observed=True).sum().reset_index())
    if 'is_bi_coverage' in agg_key:
        # we need to separate bi coverage from the other tiv
        agg_tiv_df.loc[agg_tiv_df['is_bi_coverage']==False, 'agg_tiv'] = agg_tiv_df['tiv_sum'] - agg_tiv_df[bi_tiv_col]
        agg_tiv_df.loc[agg_tiv_df['is_bi_coverage']==True, 'agg_tiv'] = agg_tiv_df[bi_tiv_col]
    else:
        agg_tiv_df['agg_tiv'] = agg_tiv_df['tiv_sum']
    return agg_tiv_df[agg_key + ['agg_tiv']]


def __get_level_terms(column_base_il_df, column_mapper):
    """
    get the column to term dictionary base on a level column mapper (created from profile)
    """
    level_terms = {}
    for ProfileElementName, term_info in column_mapper.items():
        if (ProfileElementName in column_base_il_df.columns
                and column_base_il_df[ProfileElementName].any()):
            level_terms[ProfileElementName] = term_info['FMTermType'].lower()
    return level_terms


def __drop_duplicated_row(prev_level_df, level_df, level_terms, agg_key, sub_agg_key):
    """drop duplicated row base on the agg_key, sub_agg_key and layer_id"""
    sub_level_layer_needed = level_df['agg_id'].isin(prev_level_df.loc[prev_level_df["layer_id"] == 2]['to_agg_id'])

    level_value_count = level_df[['agg_id'] + list(level_terms.values())].drop_duplicates()['agg_id'].value_counts()
    this_level_layer_needed = level_df['agg_id'].isin(level_value_count[level_value_count > 1].index.values)
    if 'share' in level_df.columns:
        this_level_layer_needed |= ~level_df['share'].isin({0, 1})

    level_df['layer_id'] = np.where(sub_level_layer_needed | this_level_layer_needed,
                                    level_df['layer_id'],
                                    1)
    level_df.drop_duplicates(subset=agg_key + sub_agg_key + ['layer_id'], inplace=True)


def __process_standard_level_df(column_base_il_df,
                              prev_level_df,
                              tiv_df,
                              il_inputs_df_list,
                              level_id,
                              present_cols,
                              level_cols,
                              level_column_mapper,
                              bi_tiv_col,
                              oed_hierarchy,
                              fm_aggregation_profile,
                              fm_term_filters):

    # identify useful column name
    loc_num = oed_hierarchy['locnum']['ProfileElementName'].lower()

    # identify fm columns for this level
    level_terms = __get_level_terms(column_base_il_df, level_column_mapper[level_id])

    if level_terms: # if there is fm terms we create a new level and complete the previous level info
        level_df = column_base_il_df[set(present_cols).union(set(level_terms))]
        level_df['level_id'] = len(il_inputs_df_list) + 2
        level_df['orig_level_id'] = level_id
        if level_id in fm_term_filters:
            temp_df = pd.DataFrame(0, index=level_df.index, columns=sorted(set(level_terms.values())))
            for ProfileElementName, fm_term in level_terms.items():
                filter_df = fm_term_filters[level_id](level_df, ProfileElementName)
                temp_df.loc[filter_df, fm_term.lower()] = level_df.loc[filter_df, ProfileElementName.lower()]

            level_df[temp_df.columns] = temp_df
        else:
            for ProfileElementName, fm_term in level_terms.items():
                level_df[fm_term] = level_df[ProfileElementName]

        agg_key = [v['field'].lower() for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]
        sub_agg_key = [v['field'].lower() for v in fm_aggregation_profile[level_id].get('FMSubAggKey', {}).values()
                       if v['field'].lower() in level_df.columns]

        level_df['agg_id'] = factorize_ndarray(level_df.loc[:, agg_key].values, col_idxs=range(len(agg_key)))[0]
        prev_level_df['to_agg_id'] = factorize_ndarray(prev_level_df.loc[:, agg_key].values, col_idxs=range(len(agg_key)))[0]
        il_inputs_df_list.append(prev_level_df)

        __drop_duplicated_row(prev_level_df, level_df, level_terms, agg_key, sub_agg_key)

        # compute the aggregated tiv
        agg_tiv_df = compute_agg_tiv(tiv_df, agg_key, bi_tiv_col, loc_num)
        level_df = merge_dataframes(level_df, agg_tiv_df, on=agg_key, how='left')

        level_df = level_df[level_cols.union(set(level_terms.values()))]

        return level_df
    else:
        return prev_level_df


def __process_condition_level_df(column_base_il_df,
                               prev_level_df,
                               tiv_df,
                               il_inputs_df_list,
                               level_id,
                               present_cols,
                               level_cols,
                               level_column_mapper,
                               bi_tiv_col,
                               oed_hierarchy,
                               fm_aggregation_profile,
                               fm_term_filters):
    # identify useful column name
    acc_num = oed_hierarchy['accnum']['ProfileElementName'].lower()
    portfolio_num = oed_hierarchy['portnum']['ProfileElementName'].lower()
    cond_num = oed_hierarchy['condnum']['ProfileElementName'].lower()
    loc_num = oed_hierarchy['locnum']['ProfileElementName'].lower()
    level_location_to_agg_cond_df, cond_inter_level = __get_level_location_to_agg_cond(column_base_il_df, oed_hierarchy)

    # identify fm columns for this level
    level_terms = __get_level_terms(column_base_il_df, level_column_mapper[level_id])

    if level_terms: # if there is fm terms we create a new level and complete the previous level info
        agg_key = [v['field'].lower() for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]
        sub_agg_key = [v['field'].lower() for v in fm_aggregation_profile[level_id].get('FMSubAggKey', {}).values()
                       if v['field'].lower() in column_base_il_df.columns]

        for inter_level in range(cond_inter_level + 1):
            level_df = column_base_il_df[set(present_cols).union(set(level_terms))]
            level_df['level_id'] = len(il_inputs_df_list) + 2
            level_df['orig_level_id'] = level_id

            for ProfileElementName, fm_term in level_terms.items():
                level_df[fm_term] = level_df[ProfileElementName]

            this_level_location_to_agg_cond_df = level_location_to_agg_cond_df[level_location_to_agg_cond_df['level_id'] == inter_level]
            this_level_location_to_agg_cond_df.drop(columns=['level_id'], inplace=True)
            level_df = merge_dataframes(
                level_df,
                this_level_location_to_agg_cond_df,
                on=[portfolio_num, acc_num, 'layer_id', loc_num],
                how='inner',
                drop_duplicates=False
            )
            level_df.loc[level_df[cond_num] == 0, set(level_terms.values())] = 0

            this_level_location_to_agg_cond_df.rename(columns={'agg_id': 'to_agg_id'}, inplace=True)
            this_level_location_to_agg_cond_df.drop(columns=[cond_num], inplace=True)

            prev_level_df = merge_dataframes(
                prev_level_df,
                this_level_location_to_agg_cond_df,
                on=[portfolio_num, acc_num, 'layer_id', loc_num],
                how='inner',
                drop_duplicates=False
            )

            __drop_duplicated_row(prev_level_df, level_df, level_terms, agg_key, sub_agg_key)

            # compute the aggregated tiv
            agg_tiv_df = compute_agg_tiv(tiv_df, agg_key, bi_tiv_col, loc_num)
            level_df = merge_dataframes(level_df, agg_tiv_df, on=agg_key, how='left')

            level_df = level_df[level_cols.union(set(level_terms.values()))]
            il_inputs_df_list.append(prev_level_df)
            prev_level_df = level_df

    return prev_level_df


@oasis_log
def get_il_input_items(
    exposure_df,
    gul_inputs_df,
    accounts_df,
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
    profile = get_grouped_fm_profile_by_level_and_term_group(exposure_profile, accounts_profile)

    # Get the FM aggregation profile - this describes how the IL input
    # items are to be aggregated in the various FM levels
    fm_aggregation_profile = copy.deepcopy(fm_aggregation_profile)

    if not fm_aggregation_profile:
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


    # get column name to fm term
    fm_terms = get_grouped_fm_terms_by_level_and_term_group(grouped_profile_by_level_and_term_group=profile)

    gul_inputs_df = __merge_exposure_and_gul(exposure_df, gul_inputs_df, fm_terms, profile, oed_hierarchy)
    bi_tiv_col = 'bitiv'

    column_base_il_df = __merge_gul_and_account(gul_inputs_df, accounts_df, fm_terms, oed_hierarchy)

    # Profile dict are base on key that correspond to the fm term name.
    # this prevent multiple file column to point to the same fm term
    # which is necessary to have a generic logic that works with step policy
    # so we change the key to be the column and use FMTermType to store the term name
    level_column_mapper = {}
    for level_id, level_profile in profile.items():
        column_map = {}
        level_column_mapper[level_id] = column_map
        for term_name, term_info in level_profile[1].items():# for fm we only use term_id 1
            new_term_info = copy.deepcopy(term_info)
            new_term_info['FMTermType'] = term_name
            column_map[term_info['ProfileElementName'].lower()] = new_term_info

    # column dependent fm term (level, dependency column name , dependency name in profile, default support ids)
    fm_term_filters = {}

    def site_pd_term_filter(level_df, ProfileElementName):
        return level_df['coverage_type_id'].isin(
            (level_column_mapper[SUPPORTED_FM_LEVELS['site pd']['id']].get(ProfileElementName) or {}).get('CoverageTypeID') or supp_cov_types)

    fm_term_filters[SUPPORTED_FM_LEVELS['site pd']['id']] = site_pd_term_filter

    # column_base_il_df contains for each items, the complete list of fm term necessary for each level
    # up until the top account level. We are now going to pivot it to get for each line a node with
    # agg_id, parrent_agg_id, level, layer and all the fm term interpretable as a generic policy
    useful_cols = sorted(set(['layer_id', 'orig_level_id', 'level_id', 'agg_id', 'gul_input_id', 'agg_tiv']
                              + get_usefull_summary_cols(oed_hierarchy))
                              - {'policytc_id', 'item_id', 'output_id'})

    # Determine whether step policies are listed, are not full of nans and step
    # numbers are greater than zero
    step_policies_present = ('steptriggertype' in accounts_df and 'stepnumber' in accounts_df
                             and accounts_df['steptriggertype'].notnull().any()
                             and accounts_df[accounts_df['steptriggertype'].notnull()]['stepnumber'].gt(0).any())

    # If step policies listed, keep step trigger type and columns associated
    # with those step trigger types that are present
    if step_policies_present:
        # we happend the fm step policy term to policy layer
        step_policy_level_map = level_column_mapper[SUPPORTED_FM_LEVELS['policy layer']['id']]
        for col in ['steptriggertype', 'cov_agg_id', 'assign_step_calcrule']:
            step_policy_level_map[col] = {
                'ProfileElementName': col,
                'FMTermType': col,
            }
        for key, step_term in get_default_step_policies_profile().items():
            step_policy_level_map[step_term['Key'].lower()] = {
                'ProfileElementName': step_term['Key'],
                'FMTermType': step_term['FMProfileField'],
                'FMProfileStep' : step_term.get('FMProfileStep')
            }
        def assign_cov_agg_id(row):
            try:
                cov_agg_method = STEP_TRIGGER_TYPES[row['steptriggertype']]['coverage_aggregation_method']
                return COVERAGE_AGGREGATION_METHODS[cov_agg_method][row['coverage_type_id']]
            except KeyError:
                return 0

        column_base_il_df['cov_agg_id'] = column_base_il_df.apply(lambda row: assign_cov_agg_id(row), axis=1)

        def assign_calcrule_flag(row):
            try:
                calcrule_assign_method = STEP_TRIGGER_TYPES[row['steptriggertype']]['calcrule_assignment_method']
                return CALCRULE_ASSIGNMENT_METHODS[calcrule_assign_method][row['cov_agg_id']]

            except KeyError:
                return False

        column_base_il_df['assign_step_calcrule'] = column_base_il_df.apply(lambda row: assign_calcrule_flag(row), axis=1)

        fm_aggregation_profile[SUPPORTED_FM_LEVELS['policy layer']['id']]['FMAggKey']['cov_agg_id'] = {
                "src": "FM",
                "field": "cov_agg_id",
                "name": "coverage aggregation id"
            }

        all_steps = column_base_il_df['steptriggertype'].unique()

        def step_policy_term_filter(level_df, ProfileElementName):
            if 'FMProfileStep' not in level_column_mapper[SUPPORTED_FM_LEVELS['policy layer']['id']].get(ProfileElementName, {}):
                return pd.Series(True, index=level_df.index)
            else:
                return (level_df['steptriggertype'].isin((level_column_mapper[SUPPORTED_FM_LEVELS['policy layer']['id']].get(ProfileElementName) or {}).get('FMProfileStep') or all_steps)
                        & level_df['assign_step_calcrule'] > 0)

        fm_term_filters[SUPPORTED_FM_LEVELS['policy layer']['id']] = step_policy_term_filter


    agg_keys = set()
    for level_id in fm_aggregation_profile:
        agg_keys = agg_keys.union(set([v['field'].lower() for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]))

    level_cols = set(useful_cols).union(agg_keys)
    present_cols = [col for col in column_base_il_df.columns if col in set(useful_cols).union(agg_keys)]

    #get Tiv for each coverage
    tiv_df = column_base_il_df[sorted(set(agg_keys.union({'coverage_id', 'tiv', 'tiv_sum', bi_tiv_col, 'is_bi_coverage'})))].drop_duplicates(keep='first')

    #initialization
    level_id = SUPPORTED_FM_LEVELS['site coverage']['id']
    coverage_level_term = ['deductible', 'deductible_min', 'deductible_max', 'limit', 'ded_code', 'ded_type',
                           'lim_code', 'lim_type']
    prev_level_df = column_base_il_df[set(present_cols + coverage_level_term)]
    prev_agg_key = [v['field'].lower() for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]
    prev_level_df.drop_duplicates(subset=prev_agg_key, inplace=True)
    prev_level_df['agg_id'] = prev_level_df['coverage_id']
    prev_level_df['level_id'] = 1
    prev_level_df['orig_level_id'] = level_id
    prev_level_df['layer_id'] = 1
    prev_level_df['agg_tiv'] = prev_level_df['tiv']
    prev_level_df['attachment'] = 0
    prev_level_df['share'] = 0
    il_inputs_df_list = []

    # create level for each SUPPORTED_FM_LEVELS
    for level, level_info in list(SUPPORTED_FM_LEVELS.items())[1:]:
        if level == 'cond all': # special treatment for condition level
            process_level_df = __process_condition_level_df
        else:
            process_level_df = __process_standard_level_df

        prev_level_df = process_level_df(column_base_il_df,
                                         prev_level_df,
                                         tiv_df,
                                         il_inputs_df_list,
                                         level_info['id'],
                                         present_cols,
                                         level_cols,
                                         level_column_mapper,
                                         bi_tiv_col,
                                         oed_hierarchy,
                                         fm_aggregation_profile,
                                         fm_term_filters)

    # create account aggregation if necessary
    level = 'policy layer'
    level_id = SUPPORTED_FM_LEVELS[level]['id']
    agg_key = [v['field'].lower() for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]
    sub_agg_key = [v['field'].lower() for v in fm_aggregation_profile[level_id].get('FMSubAggKey', {}).values()
                   if v['field'].lower() in prev_level_df.columns]
    need_account_aggregation = prev_level_df[agg_key + sub_agg_key + ['layer_id']].groupby(agg_key + sub_agg_key + ['layer_id']).size().max() > 1

    if need_account_aggregation:
        level_df = column_base_il_df[set(present_cols)]
        level_df['orig_level_id'] = level_id
        level_df['level_id'] = len(il_inputs_df_list) + 2
        level_df['agg_id'] = factorize_ndarray(level_df.loc[:, agg_key].values, col_idxs=range(len(agg_key)))[0]
        prev_level_df['to_agg_id'] = factorize_ndarray(prev_level_df.loc[:, agg_key].values, col_idxs=range(len(agg_key)))[0]

        level_df.drop_duplicates(subset=agg_key + sub_agg_key + ['layer_id'], inplace=True)
        il_inputs_df_list.append(prev_level_df)
        prev_level_df = level_df

    prev_level_df['to_agg_id'] = 0
    il_inputs_df_list.append(prev_level_df)
    il_inputs_df = pd.concat(il_inputs_df_list)
    for col in il_inputs_df.columns:
        try:
            il_inputs_df[col].fillna(0, inplace=True)
        except:
            pass

    # set top agg_id for later xref computation
    il_inputs_df['top_agg_id'] = factorize_ndarray(il_inputs_df.loc[:, agg_key].values, col_idxs=range(len(agg_key)))[0]

    # Final setting of data types before returning the IL input items
    dtypes = {
        **{t: 'float64' for t in ['tiv', 'agg_tiv', 'deductible', 'deductible_min', 'deductible_max', 'limit', 'attachment', 'share',
                                  'deductible1', 'limit1', 'limit2','trigger_start', 'trigger_end', 'payout_start', 'payout_end',
                                  'scale1', 'scale2']},
        **{t: 'uint32' for t in ['agg_id', 'item_id', 'layer_id', 'level_id', 'orig_level_id', 'calcrule_id', 'policytc_id', 'steptriggertype', 'step_id']},
        # **{t: 'uint16' for t in [cond_num]},
        **{t: 'uint8' for t in ['ded_code', 'ded_type', 'lim_code', 'lim_type', 'trigger_type', 'payout_type']}
    }
    il_inputs_df = set_dataframe_column_dtypes(il_inputs_df, dtypes)


    # Assign default values to IL inputs
    il_inputs_df = assign_defaults_to_il_inputs(il_inputs_df)

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

    if step_policies_present:
        # Before assigning calc. rule IDs and policy TC IDs, the steptriggertype
        # should be split into its sub-types in cases where the associated
        # coverages are covered separately
        # For example, steptriggertype = 5 covers buildings and contents separately
        def assign_sub_step_trigger_type(row):
            try:
                step_trigger_type = STEP_TRIGGER_TYPES[row['steptriggertype']]['sub_step_trigger_types'][
                    row['coverage_type_id']]
                return step_trigger_type
            except KeyError:
                return row['steptriggertype']
        il_inputs_df['steptriggertype'] = il_inputs_df.apply(
            lambda row: assign_sub_step_trigger_type(row), axis=1
        )


    # Set the calc. rule IDs
    if step_policies_present:
        il_inputs_df.loc[
            ~(il_inputs_df['steptriggertype'] > 0), 'calcrule_id'
        ] = get_calc_rule_ids(
            il_inputs_df[~(il_inputs_df['steptriggertype'] > 0)]
        )

        il_inputs_df.loc[
            il_inputs_df['steptriggertype'] > 0, 'calcrule_id'
        ] = get_step_calc_rule_ids(
            il_inputs_df[il_inputs_df['steptriggertype'] > 0],
        )
    else:
        il_inputs_df['calcrule_id'] = get_calc_rule_ids(il_inputs_df)

    il_inputs_df['calcrule_id'] = il_inputs_df['calcrule_id'].astype('uint32')

    # Set the policy TC IDs
    if 'cov_agg_id' in il_inputs_df:
        il_inputs_df.loc[
            ~(il_inputs_df['assign_step_calcrule'] > 0), 'policytc_id'
        ] = get_policytc_ids(
            il_inputs_df[~(il_inputs_df['assign_step_calcrule'] > 0)]
        )

        il_inputs_df.loc[
            il_inputs_df['assign_step_calcrule'] > 0, 'policytc_id'
        ] = get_step_policytc_ids(
            il_inputs_df[il_inputs_df['assign_step_calcrule'] > 0],
            offset=il_inputs_df['policytc_id'].max(),
            idx_cols=[acc_num, policy_num, portfolio_num]
        )
    else:
        il_inputs_df['policytc_id'] = get_policytc_ids(il_inputs_df)
    il_inputs_df['policytc_id'] = il_inputs_df['policytc_id'].astype('uint32')

    il_inputs_df = set_dataframe_column_dtypes(il_inputs_df, dtypes)
    return il_inputs_df


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
            fm_profile_df = il_inputs_df[set(il_inputs_df.columns).intersection(set(step_profile_cols))]
            for col in step_profile_cols:
                if col not in fm_profile_df.columns:
                    fm_profile_df[col] = 0

            for non_step_name, step_name in profile_cols_map.items():
                fm_profile_df.loc[
                    ~(il_inputs_df['steptriggertype'] > 0), step_name
                ] = il_inputs_df.loc[
                    ~(il_inputs_df['steptriggertype'] > 0),
                    non_step_name
                ]
            fm_profile_df.fillna(0, inplace=True)
            fm_profile_df = fm_profile_df.drop_duplicates()

            # Ensure step_id is of int data type and set default value to 1
            dtypes = {t: 'int64' if t == 'step_id' else 'float64' for t in profile_cols_map.values()}
            fm_profile_df = set_dataframe_column_dtypes(fm_profile_df, dtypes)
            fm_profile_df.loc[fm_profile_df['step_id'] == 0, 'step_id'] = 1

            fm_profile_df.loc[:, step_profile_cols].to_csv(
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
        item_level = il_inputs_df.loc[il_inputs_df['level_id'] == 1, ['gul_input_id', 'level_id', 'agg_id']]
        item_level.rename(columns={'gul_input_id': 'from_agg_id',
                                   'agg_id': 'to_agg_id',
                                   }, inplace=True)
        fm_programme_df = il_inputs_df.loc[il_inputs_df['level_id'] < max_level, ['agg_id', 'level_id', 'to_agg_id']]
        fm_programme_df['level_id'] += 1
        fm_programme_df.rename(columns={'agg_id': 'from_agg_id'}, inplace=True)
        fm_programme_df.drop_duplicates(keep='first', inplace=True)

        fm_programme_df = pd.concat([item_level, fm_programme_df])
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
        xref_df = get_xref_df(il_inputs_df)
        pd.DataFrame(
            {
                'output': factorize_ndarray(xref_df.loc[:, ['gul_input_id', 'layer_id']].values, col_idxs=range(2))[0],
                'agg_id': xref_df['gul_input_id'],
                'layer_id': xref_df['layer_id']
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
