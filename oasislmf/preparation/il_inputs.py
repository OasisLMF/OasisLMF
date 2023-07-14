__all__ = [
    'set_calc_rule_ids',
    'get_calc_rule_ids',
    'get_grouped_fm_profile_by_level_and_term_group',
    'get_grouped_fm_terms_by_level_and_term_group',
    'get_oed_hierarchy',
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
import itertools
import os
import sys
import warnings
from ast import literal_eval

import numpy as np
import pandas as pd

from ods_tools.oed import fill_empty

from oasislmf.preparation.summaries import get_useful_summary_cols, get_xref_df
from oasislmf.utils.calc_rules import get_calc_rules, get_step_calc_rules
from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
from oasislmf.utils.data import (factorize_array, factorize_ndarray,
                                 fast_zip_arrays,
                                 merge_check, merge_dataframes,
                                 set_dataframe_column_dtypes)
from oasislmf.utils.defaults import (OASIS_FILES_PREFIXES, assign_defaults_to_il_inputs,
                                     get_default_accounts_profile, get_default_exposure_profile,
                                     get_default_fm_aggregation_profile)
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.fm import (CALCRULE_ASSIGNMENT_METHODS, COVERAGE_AGGREGATION_METHODS,
                               DEDUCTIBLE_AND_LIMIT_TYPES, FML_ACCALL, STEP_TRIGGER_TYPES,
                               SUPPORTED_FM_LEVELS)
from oasislmf.utils.log import oasis_log
from oasislmf.utils.path import as_path
from oasislmf.utils.profiles import (get_default_step_policies_profile, get_fm_terms_oed_columns,
                                     get_grouped_fm_profile_by_level_and_term_group,
                                     get_grouped_fm_terms_by_level_and_term_group,
                                     get_oed_hierarchy)

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
cross_layer_level = {FML_ACCALL, }

risk_disaggregation_term = {'deductible', 'deductible_min', 'deductible_max', 'attachment', 'limit'}


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
        t for t in fast_zip_arrays(*il_inputs_calc_rules_df.loc[:, terms_indicators + types_and_codes].transpose().values)
    ]
    il_inputs_calc_rules_df = merge_dataframes(
        il_inputs_calc_rules_df, calc_rules, how='left', on='id_key', drop_duplicates=False
    ).fillna(0)
    il_inputs_calc_rules_df['calcrule_id'] = il_inputs_calc_rules_df['calcrule_id'].astype('uint32')
    if 0 in il_inputs_calc_rules_df.calcrule_id.unique():
        err_msg = 'Calculation Rule mapping error, non-matching keys:\n'
        no_match_keys = il_inputs_calc_rules_df.loc[il_inputs_calc_rules_df.calcrule_id == 0].id_key.unique()

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
        il_inputs_calc_rules_df[il_inputs_calc_rules_df['orig_level_id'] != policy_layer_id],
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
        il_inputs_calc_rules_df[il_inputs_calc_rules_df['orig_level_id'] == policy_layer_id],
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

    cols = ['orig_level_id',
            'item_id', 'level_id', 'StepTriggerType', 'assign_step_calcrule',
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
        no_match_keys = il_inputs_calc_rules_df.loc[il_inputs_calc_rules_df.calcrule_id == 0].id_key.unique()

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
    return df.loc[:, list(set(fm_column_names).intersection(df.columns))].any().any()


def __get_bi_tiv_col_name(profile):
    """get the name of the bi tiv column
    :param profile: profile containing all the terms
    :type profile: dict

    :return: the name of the bi tiv column
    :rtype: str
    """
    cov_level_id = SUPPORTED_FM_LEVELS['site coverage']['id']
    bi_cov_id = SUPPORTED_COVERAGE_TYPES['bi']['id']
    return profile[cov_level_id][bi_cov_id]['tiv']['ProfileElementName']


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
    cond_tag = oed_hierarchy['condtag']['ProfileElementName']

    # get usefull term columns from exposure_df
    site_pd_and_site_all_term_cols = get_fm_terms_oed_columns(fm_terms, levels=['site pd', 'site all'])
    exposure_usefull_fm_cols = list(set(site_pd_and_site_all_term_cols).intersection(exposure_df.columns))

    # The coverage FM level (site coverage, #1) ID
    cov_level_id = SUPPORTED_FM_LEVELS['site coverage']['id']
    # Get the TIV column names and corresponding coverage types
    tiv_terms = {
        v['tiv']['CoverageTypeID']: v['tiv']['ProfileElementName']
        for v in profile[cov_level_id].values()
    }

    # Calculate sum of TIV columns
    exposure_df['tiv_sum'] = exposure_df[tiv_terms.values()].sum(axis=1)

    # set default cond_tag
    if cond_tag not in exposure_df.columns:
        exposure_df[cond_tag] = '0'
        exposure_df[cond_tag] = exposure_df[cond_tag].astype('category')
    else:
        fill_empty(exposure_df, cond_tag, '0')

    # Identify BI TIV column
    bi_tiv_col = __get_bi_tiv_col_name(profile)

    # merge exposure_df and gul_inputs_df #####
    return merge_dataframes(
        exposure_df.loc[:, exposure_usefull_fm_cols + [
            'loc_id', 'tiv_sum', bi_tiv_col, cond_tag
        ]],
        gul_inputs_df,
        join_on='loc_id',
        how='inner'
    ).rename(columns={'item_id': 'gul_input_id'})


def __merge_gul_and_account(gul_inputs_df, accounts_df, fm_terms, oed_hierarchy):
    """prepare gul and account df and merge them based on [portfolio_num, acc_num, cond_tag]"""

    acc_num = oed_hierarchy['accnum']['ProfileElementName']
    portfolio_num = oed_hierarchy['portnum']['ProfileElementName']
    policy_num = oed_hierarchy['polnum']['ProfileElementName']
    cond_tag = oed_hierarchy['condtag']['ProfileElementName']
    cond_num = oed_hierarchy['condnum']['ProfileElementName']
    cond_class = oed_hierarchy['condclass']['ProfileElementName']
    loc_num = oed_hierarchy['locnum']['ProfileElementName']

    ###### prepare accounts_df #####
    # add default cond_tag if needed
    if cond_tag in gul_inputs_df:
        def add_default_value(df, col, value):
            if col not in df:
                df[col] = value
                df[col] = df[col].astype('category')
            else:
                fill_empty(df, col, value)

        add_default_value(accounts_df, cond_tag, '0')
        add_default_value(accounts_df, cond_num, '')

    if cond_tag in accounts_df.columns:
        # Oed schema allows policy levels to contain location with and without specific cond class
        # for a specific layer, a location without cond class for this layer is pass through
        # if the cond class for other location is 0 and excluded if cond class is 1

        # first we determine if the layer has the pass through or the exclusion rule and store the result in polcondclass
        if cond_class in accounts_df.columns:  # cond_class is use to specify if a condition exclude other location (1) or not (0)
            # policy level cond class is set to 1 if some location are excluded
            accounts_df['PolCondClass'] = (accounts_df.groupby([portfolio_num, acc_num, policy_num, 'layer_id'], sort=False)
                                           [cond_class]
                                           .transform(max))
        else:
            # otherwise they are set to 0 (no exclusion)
            accounts_df[['PolCondClass', cond_class]] = 0

        if 'CondPriority' not in accounts_df.columns:
            accounts_df['CondPriority'] = 1
        fill_empty(accounts_df, ['CondPriority'], 1)

        # create a df all_cond_policy containing all the cond_tag for each policies
        policy_df = accounts_df.drop_duplicates(subset=[portfolio_num, acc_num, policy_num, 'layer_id']).drop(columns=[cond_tag, 'CondPriority'])
        cond_df = accounts_df[[portfolio_num, acc_num, cond_tag, 'CondPriority']].drop_duplicates()
        all_cond_policy = (pd.merge(policy_df, cond_df, on=[portfolio_num, acc_num])
                           .drop(columns=cond_class)
                           .rename(columns={'PolCondClass': cond_class}))

        # get which cond tag are not specified in which layer
        missing_cond_policy_df = pd.merge(accounts_df[[portfolio_num, acc_num, policy_num, 'layer_id', cond_tag]],
                                          all_cond_policy, how='right', indicator=True)
        missing_cond_policy_df = missing_cond_policy_df[missing_cond_policy_df['_merge'] == 'right_only'].drop(columns='_merge')

        # finally, we can create default cond that will be applied to the location that have no cond in some layer
        # depending on the cond_class, the default cond will be passing through or excluding the loss
        null_cond = accounts_df[accounts_df[cond_tag] != '0']
        null_cond[cond_tag] = '0'
        null_cond = pd.concat([null_cond, missing_cond_policy_df])
        null_cond[cond_num] = ''
        level_id = SUPPORTED_FM_LEVELS['cond all']['id']
        level_term_cols = get_fm_terms_oed_columns(fm_terms, level_ids=[level_id])
        null_cond[level_term_cols] = 0
        null_cond.drop_duplicates(subset=[portfolio_num, acc_num, cond_tag, 'layer_id'], inplace=True)
        if cond_class in null_cond.columns:
            filter_cond = (null_cond['PolCondClass'] == 1)
            if filter_cond.any():
                null_cond.loc[filter_cond, cond_num] = 'FullFilter'
                null_cond.loc[filter_cond, 'CondDed6All'] = 1
                null_cond.loc[filter_cond, 'CondDedType6All'] = 1

        accounts_df = pd.concat([accounts_df, null_cond])
        merge_col = [portfolio_num, acc_num, cond_tag]
    else:
        merge_col = [portfolio_num, acc_num]

    ##### merge accounts_df and gul_inputs_df #####
    # check for empty intersection between dfs
    merge_check(
        gul_inputs_df[merge_col],
        accounts_df[merge_col],
        on=merge_col
    )

    # Construct a basic IL inputs frame by merging the combined exposure +
    # GUL inputs frame above, with the accounts frame, on portfolio no.,
    # account no. and condition no. (by default items in the GUL inputs frame
    # are set with a condition no. of 0)
    column_base_il_df = merge_dataframes(
        gul_inputs_df,
        accounts_df,
        on=merge_col,
        how='left',
        drop_duplicates=True
    )
    missing_account_row = column_base_il_df.loc[column_base_il_df['layer_id'].isna()]
    if not missing_account_row.empty:
        raise OasisException("locations have policies, accounts combination not present in the account file \n" +
                             missing_account_row[[loc_num] + merge_col].drop_duplicates().to_string(index=False))

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

    def set_agg_cond(group_id, group, level_id, layer_id, condition):
        level_group_key = group_id + (level_id,)
        if level_group_key not in level_grouping:
            level_grouping[level_group_key] = agg_dict[level_id]
            agg_dict[level_id] += 1
        for loc_key in group['locations']:
            location = loc_key[-1]
            level_location_to_agg_cond_dict[group_id + (level_id, layer_id, location)] = (level_grouping[level_group_key], condition)

    for main_key, groups in account_groups.items():
        # first we iter through groups to create agg
        no_parent_not_top_groups = dict(groups)
        for group_id, group in groups.items():
            for layer_id, cond_num in group['layers'].items():
                for level_id in range(base_level, group['level']):
                    set_agg_cond(group_id, group, level_id, layer_id, '')

                if group['level'] == max_level:
                    no_parent_not_top_groups.pop(group_id, None)
                set_agg_cond(group_id, group, group['level'], layer_id, cond_num)

                for child_group_id in group.get('childs', []):
                    no_parent_not_top_groups.pop(child_group_id, None)
                    set_agg_cond(group_id, groups[child_group_id], group['level'], layer_id, cond_num)

        for group_id in no_parent_not_top_groups:
            for level_id in range(group['level'] + 1, max_level + 1):
                group = groups[group_id]
                for layer_id in group['layers']:
                    set_agg_cond(group_id, group, level_id, layer_id, '')

    return level_location_to_agg_cond_dict


def __get_cond_grouping_hierarchy(column_base_il_df, main_key, cond_tag, cond_num, loc_num, base_level):
    """create group of locations based on the condition tag, condition number and condition hierarchy found in column_base_il_df

    """

    def attach_cond(child_group, parent_group, cond_to_group):
        child_group['parent'] = parent_group['tag'][0]
        parent_group.setdefault('childs', {})[child_group['tag'][0]] = None

        cur_parent_cond = parent_group['tag'][0]
        min_level = child_group['level'] + 1
        max_loop = column_base_il_df['CondPriority'].max() + 1
        breaker = 0
        while cur_parent_cond:
            cur_parent_group = cond_to_group[cur_parent_cond]
            cur_parent_group['level'] = max(min_level, cur_parent_group['level'])
            min_level = cur_parent_group['level'] + 1
            for loc in child_group['needed_loc']:
                if loc not in cur_parent_group['needed_loc']:
                    cur_parent_group['needed_loc'][loc] = True
            cur_parent_cond = cur_parent_group.get('parent')

            if breaker > max_loop:
                raise RecursionError(f'Issue with condtag {cur_parent_cond} stuck looping through hierarchy')
            else:
                breaker += 1

    # determine the number of layers for each main_key
    main_key_layers = {}
    for rec in column_base_il_df[main_key + ['layer_id']].to_dict(orient="records"):
        main_key_layers.setdefault(tuple(rec[name] for name in main_key), set()).add(rec['layer_id'])

    loc_to_cond = {}
    account_groups = {}
    for rec in column_base_il_df[main_key + [cond_tag, cond_num, loc_num, 'layer_id', 'CondPriority']].drop_duplicates().to_dict(orient="records"):
        group_key = tuple(rec[name] for name in main_key)
        groups = account_groups.setdefault(group_key, {})
        cond_key = tuple(rec[name] for name in main_key + [cond_tag])
        loc_key = tuple(rec[name] for name in main_key + [loc_num])

        parent_cond = None
        child_cond = None

        # if location already exist we determine where the record condition is inside the location parent groups
        if loc_key in loc_to_cond:
            cur_cond = loc_to_cond[loc_key]
            while cur_cond:
                cur_group = groups[cur_cond]
                cur_condpriority = cur_group['tag'][1]
                if rec['CondPriority'] == cur_condpriority:
                    if cur_cond != cond_key:
                        raise OasisException(f'{loc_key} condition of the same priority and policy {cond_key} {cur_cond}')
                elif rec['CondPriority'] > cur_condpriority:
                    child_cond = cur_cond
                else:
                    parent_cond = cur_cond
                    break
                cur_cond = cur_group.get('parent')

        if cond_key not in groups:
            # new cond key we create a new group
            group = {'locations': {},
                     'tag': (cond_key, rec['CondPriority']),
                     'layers': {layer_id: '' for layer_id in main_key_layers[group_key]},
                     'needed_loc': {},
                     'level': base_level}
            groups[cond_key] = group
        else:
            # cond key already exist, if the parent of the cond_tag is not the same as the parent found within the location parents
            # we rearrange the hierarchy based on priority
            group = groups[cond_key]
            cond_parent_cond = group.get('parent')
            if cond_parent_cond and parent_cond and cond_parent_cond != parent_cond:
                cond_parent_group = groups[cond_parent_cond]
                loc_parent_group = groups[parent_cond]
                if cond_parent_group['tag'][1] == loc_parent_group['tag'][1]:
                    raise OasisException(f"condition of the same priority and policy {cond_parent_group['tag'][0]} {loc_parent_group['tag'][0]}")
                elif cond_parent_group['tag'][1] > loc_parent_group['tag'][1]:
                    attach_cond(loc_parent_group, cond_parent_group, groups)
                    cond_parent_group['childs'].pop(cond_key, None)
                else:
                    attach_cond(cond_parent_group, loc_parent_group, groups)
                    loc_parent_group['childs'].pop(child_cond, None)
                    parent_cond = group.get('parent')

        if child_cond:
            child_group = groups[child_cond]
            attach_cond(child_group, group, groups)
        else:
            group['locations'][loc_key] = None
            loc_to_cond[loc_key] = cond_key

        group['needed_loc'][loc_key] = False
        if group['layers'][rec['layer_id']]:
            if rec[cond_num] != group['layers'][rec['layer_id']]:
                raise ValueError(
                    f"account {group_key} has multiple cond_number ({rec[cond_num]}, {group['layers'][rec['layer_id']]})"
                    " for the same cond_tag and same layer_id")
        else:
            group['layers'][rec['layer_id']] = rec[cond_num]

        if parent_cond:
            parent_group = groups[parent_cond]
            parent_group['locations'].pop(loc_key, None)
            parent_group.get('childs', {}).pop(child_cond, None)
            attach_cond(group, parent_group, groups)

    missing_conditions = []
    max_level = base_level
    for account, groups in account_groups.items():
        for group_id, group in groups.items():
            if group['level'] > max_level:
                max_level = group['level']
            for loc_key, needed in group['needed_loc'].items():
                if needed:
                    missing_conditions.append(group['tag'][0] + ('location', loc_key[-1],))

    if missing_conditions:
        missing_conditions_df = pd.DataFrame(missing_conditions, columns=main_key + [cond_tag, 'type', 'missing'])
        raise OasisException(f"missing conditions for :\n{missing_conditions_df.to_string(index=False)}")

    return account_groups, max_level


def __get_level_location_to_agg_cond(column_base_il_df, oed_hierarchy, main_key):
    """create a dataframe with the computed agg id base on the condition and condition priority
    """
    loc_num = oed_hierarchy['locnum']['ProfileElementName']
    cond_num = oed_hierarchy['condnum']['ProfileElementName']
    cond_tag = oed_hierarchy['condtag']['ProfileElementName']

    base_level = 0
    account_groups, max_level = __get_cond_grouping_hierarchy(column_base_il_df, main_key, cond_tag, cond_num, loc_num, base_level)
    level_location_to_agg_cond_dict = __extract_level_location_to_agg_cond_dict(account_groups, base_level, max_level)

    level_location_to_agg_cond = pd.DataFrame.from_records(
        [level_location + agg_cond
         for level_location, agg_cond
         in level_location_to_agg_cond_dict.items()],
        columns=main_key + [cond_tag, 'level_id', 'layer_id', loc_num, 'agg_id', cond_num]
    )

    return level_location_to_agg_cond, max_level


def compute_agg_tiv(tiv_df, is_bi_coverage, bi_tiv_col, loc_num):
    """ compute the agg tiv depending on the agg_key"""
    agg_tiv_df = (tiv_df.drop_duplicates(['agg_id', loc_num], keep='first')[['agg_id', 'tiv', 'tiv_sum', 'is_bi_coverage', bi_tiv_col]]
                  .groupby('agg_id', observed=True).sum().reset_index())
    if is_bi_coverage:
        # we need to separate bi coverage from the other tiv
        agg_tiv_df.loc[agg_tiv_df['is_bi_coverage'] == False, 'agg_tiv'] = agg_tiv_df['tiv_sum'] - agg_tiv_df[bi_tiv_col]
        agg_tiv_df.loc[agg_tiv_df['is_bi_coverage'] == True, 'agg_tiv'] = agg_tiv_df[bi_tiv_col]
    else:
        agg_tiv_df['agg_tiv'] = agg_tiv_df['tiv_sum']
    return agg_tiv_df[['agg_id', 'agg_tiv']]


def __get_level_terms(column_base_il_df, column_mapper):
    """
    get the column to term dictionary base on a level column mapper (created from profile)
    """
    level_terms = {}
    for ProfileElementName, term_info in column_mapper.items():
        if (ProfileElementName in column_base_il_df.columns
                and column_base_il_df[ProfileElementName].any()):
            level_terms[ProfileElementName] = (term_info['FMTermType'].lower(), term_info.get('FMTermGroupID', 1))
    return level_terms


def __split_fm_terms_by_risk(df):
    """
    adjust the financial term to the number of risks
    for example deductible is split into each individul building risk

    Args:
         df (DataFrame): the DataFrame an FM level
    """

    for term in risk_disaggregation_term.intersection(set(df.columns)):
        if f'{term[:3]}_code' in df.columns:
            code_filter = df[f'{term[:3]}_code'] == 0
            df.loc[code_filter, term] /= df.loc[code_filter, 'NumberOfRisks']
        else:
            df[term] /= df['NumberOfRisks']


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
    level_df.drop_duplicates(subset=['agg_id'] + sub_agg_key + ['layer_id'], inplace=True)


def __process_standard_level_df(column_base_il_df,
                                prev_level_df,
                                prev_agg_key,
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
    # identify fm columns for this level
    level_terms_group = __get_level_terms(column_base_il_df, level_column_mapper[level_id])
    level_terms = {ProfileElementName: fm_term for (ProfileElementName, (fm_term, FMTermGroupID)) in level_terms_group.items()}

    if not level_terms:  # if no terms we skip the level
        return prev_level_df, prev_agg_key

    cur_level = len(il_inputs_df_list) + 2

    column_base_il_df['level_id'] = len(il_inputs_df_list) + 2
    column_base_il_df['orig_level_id'] = level_id

    column_base_il_df['has_terms'] = False
    for term in level_terms:
        column_base_il_df[term].fillna(0, inplace=True)
        column_base_il_df['has_terms'] |= column_base_il_df[term].astype(bool)
    agg_key = [v['field'] for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]

    level_df_with_term = column_base_il_df[column_base_il_df['has_terms']]
    il_df_no_term = column_base_il_df[~column_base_il_df['has_terms']]

    prev_level_df_with_next_term = prev_level_df[prev_level_df['gul_input_id'].isin(level_df_with_term['gul_input_id'])]
    prev_level_df_no_next_term = prev_level_df[prev_level_df['gul_input_id'].isin(il_df_no_term['gul_input_id'])]

    # identify useful column name
    loc_num = oed_hierarchy['locnum']['ProfileElementName']
    if level_id in fm_term_filters:
        temp_df = pd.DataFrame(0, index=level_df_with_term.index, columns=sorted(set(level_terms.values()), key=str.lower))
        temp_df[f'filter_agg_{level_id}'] = -1

        for i, (ProfileElementName, (fm_term, FMTermGroupID)) in enumerate(level_terms_group.items()):
            filter_df = ((fm_term_filters[level_id](level_df_with_term, ProfileElementName))
                         & (~level_df_with_term[ProfileElementName].isna())
                         & (level_df_with_term[ProfileElementName].astype(bool)))
            if not level_df_with_term.loc[filter_df & ~(temp_df[f'filter_agg_{level_id}'].isin([-1, FMTermGroupID]))].empty:
                raise OasisException(f"multiple terms {fm_term} for level {level_id} in location:\n"
                                     f"{level_df_with_term.loc[filter_df & ~(temp_df[f'filter_agg_{level_id}'].isin([-1, FMTermGroupID])), agg_key]}")
            temp_df.loc[filter_df, f'filter_agg_{level_id}'] = FMTermGroupID
            temp_df.loc[filter_df, fm_term.lower()] = level_df_with_term.loc[filter_df, ProfileElementName]

        agg_key.append(f'filter_agg_{level_id}')
        level_df_with_term[temp_df.columns] = temp_df
        column_base_il_df.loc[column_base_il_df['has_terms'], f'filter_agg_{level_id}'] = temp_df[f'filter_agg_{level_id}']
    else:
        for ProfileElementName, fm_term in level_terms.items():
            level_df_with_term[fm_term] = level_df_with_term[ProfileElementName]

    if 'risk_id' in agg_key:
        __split_fm_terms_by_risk(level_df_with_term)

    level_df_with_term['orig_level_id'] = level_id

    sub_agg_key = [v['field'] for v in fm_aggregation_profile[level_id].get('FMSubAggKey', {}).values()
                   if v['field'] in level_df_with_term.columns]

    level_df_with_term['agg_id'] = factorize_ndarray(level_df_with_term.loc[:, agg_key].values, col_idxs=range(len(agg_key)))[0]
    level_df_with_term['prev_agg_id'] = factorize_ndarray(level_df_with_term.loc[:, prev_agg_key].values, col_idxs=range(len(prev_agg_key)))[0]

    # check rows in prev df that are this level granularity (if prev_agg_id has multiple corresponding agg_id)
    need_root_start_df = level_df_with_term.groupby("prev_agg_id").agg({"agg_id": pd.Series.nunique})
    need_root_start_df = need_root_start_df[need_root_start_df['agg_id'] > 1].index

    # create new prev df for element that need to restart from items
    root_df = level_df_with_term[((level_df_with_term['prev_agg_id'].isin(need_root_start_df)) & (level_df_with_term['layer_id'] == 1))]

    root_df['to_agg_id'] = root_df['agg_id']
    root_df['agg_id'] = -root_df['gul_input_id']
    root_df.drop_duplicates(subset='agg_id', inplace=True)
    root_df['level_id'] = cur_level - 1

    # rows with no parent points to 0
    prev_level_df_no_parent = prev_level_df_with_next_term[prev_level_df_with_next_term['gul_input_id'].isin(root_df['gul_input_id'])]
    prev_level_df_no_parent['to_agg_id'] = 0

    # select good to_agg_id for rows with parents
    prev_level_df_with_parent = prev_level_df_with_next_term[~prev_level_df_with_next_term['gul_input_id'].isin(
        root_df['gul_input_id'])].sort_values(by='gul_input_id')
    prev_level_df_with_parent = prev_level_df_with_parent.merge(
        level_df_with_term[['gul_input_id', 'agg_id']].rename(columns={'agg_id': 'to_agg_id'}))

    # row with no term are simply copied from previous level, they will take no time or space in subsequent fm computation
    cur_max_agg_id = level_df_with_term['agg_id'].max()
    prev_agg = prev_level_df_no_next_term[['agg_id', ]].drop_duplicates()
    prev_agg['to_agg_id'] = np.arange(cur_max_agg_id + 1, cur_max_agg_id + 1 + prev_agg.shape[0])
    prev_level_df_no_next_term = pd.merge(prev_level_df_no_next_term, prev_agg)
    # we can now copy previous no term previous level to be use in this level
    il_df_no_term = prev_level_df_no_next_term[present_cols + ['orig_level_id', 'to_agg_id']]
    il_df_no_term.rename(columns={'to_agg_id': 'agg_id'}, inplace=True)
    il_df_no_term['level_id'] = cur_level

    il_inputs_df_list.append(pd.concat([prev_level_df_with_parent, prev_level_df_no_parent,
                                        root_df, prev_level_df_no_next_term]).sort_values(by=['agg_id']))

    level_df_with_term.drop(columns=['prev_agg_id'])
    level_df = pd.concat([level_df_with_term, il_df_no_term])
    level_df['level_id'] = cur_level
    level_df.drop(columns=['agg_tiv'], errors='ignore', inplace=True)
    level_tiv_df = tiv_df.merge(level_df[['coverage_id', 'agg_id']])
    __drop_duplicated_row(prev_level_df_with_parent, level_df, level_terms, agg_key, sub_agg_key)

    # compute the aggregated tiv
    agg_tiv_df = compute_agg_tiv(level_tiv_df, 'is_bi_coverage' in agg_key, bi_tiv_col, loc_num)
    level_df = merge_dataframes(level_df, agg_tiv_df, on='agg_id', how='left')

    level_df = level_df[list(level_cols.union(set(level_terms.values())))]

    return level_df, agg_key


def __process_condition_level_df(column_base_il_df,
                                 prev_level_df,
                                 prev_agg_key,
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
    acc_num = oed_hierarchy['accnum']['ProfileElementName']
    portfolio_num = oed_hierarchy['portnum']['ProfileElementName']
    policy_num = oed_hierarchy['polnum']['ProfileElementName']
    cond_tag = oed_hierarchy['condtag']['ProfileElementName']
    cond_num = oed_hierarchy['condnum']['ProfileElementName']
    loc_num = oed_hierarchy['locnum']['ProfileElementName']

    if "cov_agg_id" in column_base_il_df.columns:
        main_key = [portfolio_num, acc_num, "cov_agg_id"]
    else:
        main_key = [portfolio_num, acc_num]

    # identify fm columns for this level
    level_terms_group = __get_level_terms(column_base_il_df, level_column_mapper[level_id])
    level_terms = {ProfileElementName: fm_term for (ProfileElementName, (fm_term, FMTermGroupID)) in level_terms_group.items()}

    if level_terms:  # if there is fm terms we create a new level and complete the previous level info
        level_location_to_agg_cond_df, cond_inter_level = __get_level_location_to_agg_cond(column_base_il_df,
                                                                                           oed_hierarchy, main_key)
        agg_key = [v['field'] for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]
        sub_agg_key = [v['field'] for v in fm_aggregation_profile[level_id].get('FMSubAggKey', {}).values()
                       if v['field'] in column_base_il_df.columns]

        for inter_level in range(cond_inter_level + 1):
            level_df = column_base_il_df[list(set(present_cols).union(set(level_terms)))]
            level_df['level_id'] = len(il_inputs_df_list) + 2
            level_df['orig_level_id'] = level_id

            for ProfileElementName, fm_term in level_terms.items():
                level_df[fm_term] = level_df[ProfileElementName].fillna(0)

            this_level_location_to_agg_cond_df = level_location_to_agg_cond_df[level_location_to_agg_cond_df['level_id'] == inter_level]
            this_level_location_to_agg_cond_df.drop(columns=['level_id'], inplace=True)

            level_df = merge_dataframes(
                level_df,
                this_level_location_to_agg_cond_df,
                on=main_key + [loc_num, 'layer_id', cond_tag, ],
                how='inner',
                drop_duplicates=False
            )

            level_df.loc[level_df[cond_num] == '', list(set(level_terms.values()))] = 0

            this_level_location_to_agg_cond_df.rename(columns={'agg_id': 'to_agg_id'}, inplace=True)
            this_level_location_to_agg_cond_df.drop(columns=[cond_tag, cond_num], inplace=True)
            prev_level_df = merge_dataframes(
                prev_level_df,
                this_level_location_to_agg_cond_df,
                on=main_key + ['layer_id', loc_num],
                how='inner',
                drop_duplicates=False
            )
            level_tiv_df = tiv_df.merge(level_df[['coverage_id', 'agg_id']])
            __drop_duplicated_row(prev_level_df, level_df, level_terms, agg_key, sub_agg_key)

            # compute the aggregated tiv
            agg_tiv_df = compute_agg_tiv(level_tiv_df, 'is_bi_coverage' in agg_key, bi_tiv_col, loc_num)
            level_df = merge_dataframes(level_df, agg_tiv_df, on='agg_id', how='left')

            level_df = level_df[list(level_cols.union(set(level_terms.values())))]

            il_inputs_df_list.append(prev_level_df)
            prev_level_df = level_df
    else:
        agg_key = prev_agg_key

    return prev_level_df, agg_key


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
    acc_num = oed_hierarchy['accnum']['ProfileElementName']
    policy_num = oed_hierarchy['polnum']['ProfileElementName']
    portfolio_num = oed_hierarchy['portnum']['ProfileElementName']

    # get column name to fm term
    fm_terms = get_grouped_fm_terms_by_level_and_term_group(grouped_profile_by_level_and_term_group=profile, lowercase=False)
    gul_inputs_df = __merge_exposure_and_gul(exposure_df, gul_inputs_df, fm_terms, profile, oed_hierarchy)
    bi_tiv_col = 'BITIV'

    column_base_il_df = __merge_gul_and_account(gul_inputs_df, accounts_df, fm_terms, oed_hierarchy)

    # Profile dict are base on key that correspond to the fm term name.
    # this prevent multiple file column to point to the same fm term
    # which is necessary to have a generic logic that works with step policy
    # so we change the key to be the column and use FMTermType to store the term name
    level_column_mapper = {}
    for level_id, level_profile in profile.items():
        column_map = {}
        level_column_mapper[level_id] = column_map

        # for fm we only use term_id 1
        for term_name, term_info in itertools.chain.from_iterable(profile.items() for profile in level_profile.values()):
            new_term_info = copy.deepcopy(term_info)
            new_term_info['FMTermType'] = term_name
            column_map[term_info['ProfileElementName']] = new_term_info

    # column dependent fm term (level, dependency column name , dependency name in profile, default support ids)
    fm_term_filters = {}

    def site_pd_term_filter(level_df, ProfileElementName):
        return level_df['coverage_type_id'].isin(
            (level_column_mapper[SUPPORTED_FM_LEVELS['site pd']['id']].get(ProfileElementName) or {}).get('CoverageTypeID') or supp_cov_types)

    fm_term_filters[SUPPORTED_FM_LEVELS['site pd']['id']] = site_pd_term_filter

    def policy_term_filter(level_name):
        def filter(level_df, ProfileElementName):
            coverage_type_ids = (level_column_mapper[SUPPORTED_FM_LEVELS[level_name]['id']].get(
                ProfileElementName) or {}).get('CoverageTypeID') or supp_cov_types
            if isinstance(coverage_type_ids, int):
                return level_df['coverage_type_id'] == coverage_type_ids
            else:
                return level_df['coverage_type_id'].isin(coverage_type_ids)
        return filter

    fm_term_filters[SUPPORTED_FM_LEVELS['policy coverage']['id']] = policy_term_filter('policy coverage')
    fm_term_filters[SUPPORTED_FM_LEVELS['policy pd']['id']] = policy_term_filter('policy pd')

    # column_base_il_df contains for each items, the complete list of fm term necessary for each level
    # up until the top account level. We are now going to pivot it to get for each line a node with
    # agg_id, parrent_agg_id, level, layer and all the fm term interpretable as a generic policy
    useful_cols = sorted(set(['layer_id', 'orig_level_id', 'level_id', 'agg_id', 'gul_input_id', 'agg_tiv', 'NumberOfRisks']
                             + get_useful_summary_cols(oed_hierarchy))
                         - {'policytc_id', 'item_id', 'output_id'}, key=str.lower)

    # Determine whether step policies are listed, are not full of nans and step
    # numbers are greater than zero
    step_policies_present = ('StepTriggerType' in column_base_il_df and 'StepNumber' in column_base_il_df
                             and column_base_il_df['StepTriggerType'].notnull().any()
                             and column_base_il_df[column_base_il_df['StepTriggerType'].notnull()]['StepNumber'].gt(0).any())

    # If step policies listed, keep step trigger type and columns associated
    # with those step trigger types that are present
    if step_policies_present:
        # we happend the fm step policy term to policy layer
        step_policy_level_map = level_column_mapper[SUPPORTED_FM_LEVELS['policy layer']['id']]
        for col in ['StepTriggerType', 'cov_agg_id', 'assign_step_calcrule']:
            step_policy_level_map[col] = {
                'ProfileElementName': col,
                'FMTermType': col,
            }
        for key, step_term in get_default_step_policies_profile().items():
            step_policy_level_map[step_term['Key']] = {
                'ProfileElementName': step_term['Key'],
                'FMTermType': step_term['FMProfileField'],
                'FMProfileStep': step_term.get('FMProfileStep')
            }

        def assign_cov_agg_id(row):
            try:
                cov_agg_method = STEP_TRIGGER_TYPES[row['StepTriggerType']]['coverage_aggregation_method']
                return COVERAGE_AGGREGATION_METHODS[cov_agg_method][row['coverage_type_id']]
            except KeyError:
                return 0

        column_base_il_df['cov_agg_id'] = column_base_il_df.apply(lambda row: assign_cov_agg_id(row), axis=1)

        def assign_calcrule_flag(row):
            try:
                calcrule_assign_method = STEP_TRIGGER_TYPES[row['StepTriggerType']]['calcrule_assignment_method']
                return CALCRULE_ASSIGNMENT_METHODS[calcrule_assign_method][row['cov_agg_id']]

            except KeyError:
                return False

        column_base_il_df['assign_step_calcrule'] = column_base_il_df.apply(lambda row: assign_calcrule_flag(row), axis=1)

        for level_info in list(SUPPORTED_FM_LEVELS.values())[1:]:
            fm_aggregation_profile[level_info['id']]['FMAggKey']['cov_agg_id'] = {
                "src": "FM",
                "field": "cov_agg_id",
                "name": "coverage aggregation id"
            }
        all_steps = column_base_il_df['StepTriggerType'].unique()

        def step_policy_term_filter(level_df, ProfileElementName):
            if 'FMProfileStep' not in level_column_mapper[SUPPORTED_FM_LEVELS['policy layer']['id']].get(ProfileElementName, {}):
                return pd.Series(True, index=level_df.index)
            else:
                return ((level_df['StepTriggerType'].isin(
                    (level_column_mapper[SUPPORTED_FM_LEVELS['policy layer']['id']].get(ProfileElementName) or {}).get('FMProfileStep')
                    or all_steps))
                    & (level_df['assign_step_calcrule'] > 0))

        fm_term_filters[SUPPORTED_FM_LEVELS['policy layer']['id']] = step_policy_term_filter

    agg_keys = set()
    for level_id in fm_aggregation_profile:
        agg_keys = agg_keys.union(set([v['field'] for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]))

    column_base_il_df[['risk_id', 'NumberOfRisks']] = column_base_il_df[['building_id', 'NumberOfBuildings']]
    column_base_il_df.loc[column_base_il_df['IsAggregate'] == 0, ['risk_id', 'NumberOfRisks']] = 1, 1
    column_base_il_df.loc[column_base_il_df['NumberOfRisks'] == 0, 'NumberOfRisks'] = 1

    level_cols = set(useful_cols).union(agg_keys)
    present_cols = [col for col in column_base_il_df.columns if col in set(useful_cols).union(agg_keys)]

    # get Tiv for each coverage
    tiv_df = column_base_il_df[sorted(
        set(agg_keys.union({'coverage_id', 'tiv', 'tiv_sum', bi_tiv_col, 'is_bi_coverage'})), key=str.lower)].drop_duplicates(keep='first')

    # initialization
    level_id = SUPPORTED_FM_LEVELS['site coverage']['id']
    coverage_level_term = ['deductible', 'deductible_min', 'deductible_max', 'limit', 'ded_code', 'ded_type',
                           'lim_code', 'lim_type']
    for col in coverage_level_term:
        if col not in column_base_il_df:
            column_base_il_df[col] = 0
    prev_level_df = column_base_il_df[list(set(present_cols + coverage_level_term))]
    prev_agg_key = [v['field'] for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]
    prev_level_df.drop_duplicates(subset=prev_agg_key, inplace=True)
    __split_fm_terms_by_risk(prev_level_df)
    prev_level_df['agg_id'] = factorize_ndarray(prev_level_df.loc[:, ['loc_id', 'risk_id', 'coverage_type_id']].values, col_idxs=range(3))[0]
    prev_level_df['level_id'] = 1
    prev_level_df['orig_level_id'] = level_id
    prev_level_df['layer_id'] = 1
    prev_level_df['agg_tiv'] = prev_level_df['tiv']
    prev_level_df['attachment'] = 0
    prev_level_df['share'] = 0
    il_inputs_df_list = []
    prev_agg_key = ['coverage_id']
    # create level for each SUPPORTED_FM_LEVELS
    for level, level_info in list(SUPPORTED_FM_LEVELS.items())[1:]:
        if level == 'cond all':  # special treatment for condition level
            process_level_df = __process_condition_level_df
        else:
            process_level_df = __process_standard_level_df

        prev_level_df, prev_agg_key = process_level_df(column_base_il_df,
                                                       prev_level_df,
                                                       prev_agg_key,
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
    agg_key = [v['field'] for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]
    sub_agg_key = [v['field'] for v in fm_aggregation_profile[level_id].get('FMSubAggKey', {}).values()
                   if v['field'] in prev_level_df.columns]
    need_account_aggregation = prev_level_df[agg_key + sub_agg_key + ['layer_id']].groupby(agg_key + sub_agg_key + ['layer_id']).size().max() > 1

    if need_account_aggregation:
        level_df = column_base_il_df[list(set(present_cols))]
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
        except (TypeError, ValueError):
            pass

    # set top agg_id for later xref computation
    il_inputs_df['top_agg_id'] = factorize_ndarray(il_inputs_df.loc[:, agg_key].values, col_idxs=range(len(agg_key)))[0]

    # Final setting of data types before returning the IL input items
    dtypes = {
        **{t: 'float64' for t in ['tiv', 'agg_tiv', 'deductible', 'deductible_min', 'deductible_max', 'limit', 'attachment', 'share',
                                  'deductible1', 'limit1', 'limit2', 'trigger_start', 'trigger_end', 'payout_start', 'payout_end',
                                  'scale1', 'scale2']},
        **{t: 'int32' for t in
           ['agg_id', 'item_id', 'layer_id', 'level_id', 'orig_level_id', 'calcrule_id', 'policytc_id', 'steptriggertype', 'step_id']},
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
        # Before assigning calc. rule IDs and policy TC IDs, the StepTriggerType
        # should be split into its sub-types in cases where the associated
        # coverages are covered separately
        # For example, StepTriggerType = 5 covers buildings and contents separately
        def assign_sub_step_trigger_type(row):
            try:
                step_trigger_type = STEP_TRIGGER_TYPES[row['steptriggertype']]['sub_step_trigger_types'][
                    row['coverage_type_id']]
                return step_trigger_type
            except KeyError:
                return row['steptriggertype']

        il_inputs_df['StepTriggerType'] = il_inputs_df.apply(
            lambda row: assign_sub_step_trigger_type(row), axis=1
        )

    # Set the calc. rule IDs
    if step_policies_present:
        il_inputs_df.loc[
            ~(il_inputs_df['StepTriggerType'] > 0), 'calcrule_id'
        ] = get_calc_rule_ids(
            il_inputs_df[~(il_inputs_df['StepTriggerType'] > 0)]
        )

        il_inputs_df.loc[
            il_inputs_df['StepTriggerType'] > 0, 'calcrule_id'
        ] = get_step_calc_rule_ids(
            il_inputs_df[il_inputs_df['StepTriggerType'] > 0],
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
        fm_policytc_df = il_inputs_df.loc[il_inputs_df['agg_id'] > 0, ['layer_id', 'level_id', 'agg_id', 'policytc_id', 'orig_level_id']]
        fm_policytc_df.loc[fm_policytc_df['orig_level_id'].isin(cross_layer_level), 'layer_id'] = 1  # remove layer for cross layer level
        fm_policytc_df.drop(columns=['orig_level_id']).drop_duplicates().to_csv(
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
    il_inputs_df = il_inputs_df[il_inputs_df['agg_id'] > 0]
    try:
        # Step policies exist
        if 'cov_agg_id' in il_inputs_df:
            fm_profile_df = il_inputs_df[list(set(il_inputs_df.columns).intersection(set(step_profile_cols)))]
            for col in step_profile_cols:
                if col not in fm_profile_df.columns:
                    fm_profile_df[col] = 0

            for non_step_name, step_name in profile_cols_map.items():
                fm_profile_df.loc[
                    ~(il_inputs_df['StepTriggerType'] > 0), step_name
                ] = il_inputs_df.loc[
                    ~(il_inputs_df['StepTriggerType'] > 0),
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
        il_inputs_df = il_inputs_df[il_inputs_df['to_agg_id'] != 0]
        item_level = il_inputs_df.loc[il_inputs_df['level_id'] == 1, ['gul_input_id', 'level_id', 'agg_id']]
        item_level.rename(columns={'gul_input_id': 'from_agg_id',
                                   'agg_id': 'to_agg_id',
                                   }, inplace=True)
        item_level.drop_duplicates(keep='first', inplace=True)
        fm_programme_df = il_inputs_df[['agg_id', 'level_id', 'to_agg_id']]
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
        ).to_csv(
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
    chunksize = chunksize or min(2 * 10 ** 5, len(il_inputs_df))

    # A dict of IL input file names and file paths
    il_input_files = {
        fn: os.path.join(target_dir, '{}.csv'.format(oasis_files_prefixes[fn])) for fn in oasis_files_prefixes
    }

    this_module = sys.modules[__name__]
    # Write the files serially
    for fn in il_input_files:
        getattr(this_module, 'write_{}_file'.format(fn))(il_inputs_df.copy(deep=True), il_input_files[fn], chunksize)

    return il_input_files
