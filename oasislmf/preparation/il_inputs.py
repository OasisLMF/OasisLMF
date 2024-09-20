__all__ = [
    'get_calc_rule_ids',
    'get_grouped_fm_profile_by_level_and_term_group',
    'get_grouped_fm_terms_by_level_and_term_group',
    'get_oed_hierarchy',
    'get_il_input_items',
    'get_profile_ids',
    'get_step_profile_ids',
    'write_il_input_files',
    'write_fm_policytc_file',
    'write_fm_profile_file',
    'write_fm_programme_file',
    'write_fm_xref_file'
]

import copy
import gc
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from ods_tools.oed import fill_empty, BLANK_VALUES

from oasislmf.preparation.summaries import get_useful_summary_cols, get_xref_df
from oasislmf.utils.calc_rules import get_calc_rules
from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
from oasislmf.utils.data import (factorize_array, factorize_ndarray,
                                 set_dataframe_column_dtypes)
from oasislmf.utils.defaults import (OASIS_FILES_PREFIXES, SUMMARY_TOP_LEVEL_COLS, assign_defaults_to_il_inputs,
                                     get_default_accounts_profile, get_default_exposure_profile,
                                     get_default_fm_aggregation_profile, SOURCE_IDX)
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.fm import (CALCRULE_ASSIGNMENT_METHODS, COVERAGE_AGGREGATION_METHODS,
                               DEDUCTIBLE_AND_LIMIT_TYPES, FML_ACCALL, STEP_TRIGGER_TYPES,
                               SUPPORTED_FM_LEVELS, FM_TERMS, GROUPED_SUPPORTED_FM_LEVELS)
from oasislmf.utils.log import oasis_log
from oasislmf.utils.path import as_path
from oasislmf.utils.profiles import (get_default_step_policies_profile,
                                     get_grouped_fm_profile_by_level_and_term_group,
                                     get_grouped_fm_terms_by_level_and_term_group,
                                     get_oed_hierarchy)

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define a list of all supported OED coverage types in the exposure
supp_cov_type_ids = [v['id'] for v in SUPPORTED_COVERAGE_TYPES.values()]

policytc_cols = ['calcrule_id', 'limit', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'share']
step_profile_cols = [
    'profile_id', 'calcrule_id',
    'deductible1', 'deductible2', 'deductible3', 'attachment1',
    'limit1', 'share1', 'share2', 'share3', 'step_id',
    'trigger_start', 'trigger_end', 'payout_start', 'payout_end',
    'limit2', 'scale1', 'scale2'
]

profile_cols = ['profile_id', 'calcrule_id', 'deductible1', 'deductible2', 'deductible3', 'attachment1', 'limit1',
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

fm_term_ids = [fm_term['id'] for fm_term in FM_TERMS.values()]


def get_calc_rule_ids(il_inputs_calc_rules_df, calc_rule_type):
    """
    merge selected il_inputs with the correct calc_rule table and  return a pandas Series of calc. rule IDs

    Args:
     il_inputs_calc_rules_df (DataFrame):  IL input items dataframe
     calc_rule_type (str): type of calc_rule to look for

    Returns:
        pandas Series of calc. rule IDs
    """
    calc_rules_df, calc_rule_term_info = get_calc_rules(calc_rule_type)
    calc_rules_df = calc_rules_df.drop(columns=['desc', 'id_key'], axis=1, errors='ignore')

    terms = []
    terms_indicators = []
    for term in calc_rule_term_info['terms']:
        if term in il_inputs_calc_rules_df.columns:
            terms.append(term)
            terms_indicators.append('{}_gt_0'.format(term))
        else:
            calc_rules_df = calc_rules_df[calc_rules_df['{}_gt_0'.format(term)] == 0].drop(columns=['{}_gt_0'.format(term)])
    for term in calc_rule_term_info['types_and_codes']:
        if term in il_inputs_calc_rules_df.columns:
            il_inputs_calc_rules_df[term] = il_inputs_calc_rules_df[term].fillna(0).astype('uint8')
        else:
            calc_rules_df = calc_rules_df[calc_rules_df[term] == 0].drop(columns=[term])
    il_inputs_calc_rules_df.loc[:, terms_indicators] = np.where(il_inputs_calc_rules_df[terms] > 0, 1, 0)

    merge_col = list(set(il_inputs_calc_rules_df.columns).intersection(calc_rules_df.columns).difference({'calcrule_id'}))

    calcrule_ids = (
        il_inputs_calc_rules_df[merge_col].reset_index()
        .merge(calc_rules_df[merge_col + ['calcrule_id']].drop_duplicates(), how='left', on=merge_col)
    ).set_index('index')['calcrule_id'].fillna(0)

    if 0 in calcrule_ids.unique():
        no_match_keys = il_inputs_calc_rules_df.loc[calcrule_ids == 0, ['PortNumber', 'AccNumber', 'LocNumber'] + merge_col].drop_duplicates()
        err_msg = 'Calculation Rule mapping error, non-matching keys:\n{}'.format(no_match_keys)
        raise OasisException(err_msg)
    return calcrule_ids


def get_profile_ids(il_inputs_df):
    """
    Returns a Numpy array of policy TC IDs from a table of IL input items

    :param il_inputs_df: IL input items dataframe
    :type il_inputs_df: pandas.DataFrame

    :return: Numpy array of policy TC IDs
    :rtype: numpy.ndarray
    """
    factor_col = list(set(il_inputs_df.columns).intersection(policytc_cols + step_profile_cols))
    return factorize_ndarray(il_inputs_df.loc[:, factor_col].values,
                             col_idxs=range(len(factor_col)))[0]


def get_step_profile_ids(
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
    fm_policytc_df['profile_id'] = factorize_ndarray(fm_policytc_df.loc[:, ['layer_id', 'level_id', 'agg_id']].values, col_idxs=range(3))[0]
    fm_policytc_df['pol_id'] = factorize_ndarray(fm_policytc_df.loc[:, idx_cols + ['coverage_id']].values, col_idxs=range(len(idx_cols) + 1))[0]

    step_calcrule_policytc_agg = pd.DataFrame(
        fm_policytc_df[fm_policytc_df['step_id'] > 0]['profile_id'].to_list(),
        index=fm_policytc_df[fm_policytc_df['step_id'] > 0]['pol_id']
    ).groupby('pol_id').aggregate(list).to_dict()[0]

    fm_policytc_df.loc[
        fm_policytc_df['step_id'] > 0, 'profile_id'
    ] = fm_policytc_df.loc[fm_policytc_df['step_id'] > 0]['pol_id'].map(step_calcrule_policytc_agg)
    fm_policytc_df['profile_id'] = fm_policytc_df['profile_id'].apply(
        lambda x: tuple(x) if isinstance(x, list) else x
    )
    return factorize_array(fm_policytc_df['profile_id'])[0] + offset


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


def __drop_duplicated_row(prev_level_df, level_df, sub_agg_key):
    """drop duplicated row base on the agg_key, sub_agg_key and layer_id"""
    if prev_level_df is not None:
        sub_level_layer_needed = level_df['agg_id'].isin(prev_level_df.loc[prev_level_df["layer_id"] == 2]['to_agg_id'])
    else:
        sub_level_layer_needed = False

    level_value_count = level_df[['agg_id'] + list(set(level_df.columns).intersection(fm_term_ids))].drop_duplicates()['agg_id'].value_counts()
    this_level_layer_needed = level_df['agg_id'].isin(level_value_count[level_value_count > 1].index.values)
    if 'share' in level_df.columns:
        this_level_layer_needed |= ~level_df['share'].isin({0, 1})
    level_df['layer_id'] = np.where(sub_level_layer_needed | this_level_layer_needed,
                                    level_df['layer_id'],
                                    1)
    level_df.drop_duplicates(subset=['agg_id'] + sub_agg_key + ['layer_id'], inplace=True)


def get_cond_info(locations_df, accounts_df):
    level_conds = {}
    extra_accounts = []
    default_cond_tag = '0'
    if 'CondTag' in locations_df.columns:
        fill_empty(locations_df, 'CondTag', default_cond_tag)
        loc_condkey_df = locations_df.loc[locations_df['CondTag'] != default_cond_tag, ['PortNumber', 'AccNumber', 'CondTag']].drop_duplicates()
    else:
        loc_condkey_df = pd.DataFrame([], columns=['PortNumber', 'AccNumber', 'CondTag'])

    if 'CondTag' in accounts_df.columns:
        fill_empty(accounts_df, 'CondTag', default_cond_tag)
        acc_condkey_df = accounts_df.loc[accounts_df['CondTag'] != '', ['PortNumber', 'AccNumber', 'CondTag']].drop_duplicates()
        condkey_match_df = acc_condkey_df.merge(loc_condkey_df, how='outer', indicator=True)
        missing_condkey_df = condkey_match_df.loc[condkey_match_df['_merge'] == 'right_only', ['PortNumber', 'AccNumber', 'CondTag']]
    else:
        acc_condkey_df = pd.DataFrame([], columns=['PortNumber', 'AccNumber', 'CondTag'])
        missing_condkey_df = loc_condkey_df

    if missing_condkey_df.shape[0]:
        raise OasisException(f'Those condtag are present in locations but missing in the account file:\n{missing_condkey_df}')

    if acc_condkey_df.shape[0]:
        if 'CondTag' not in locations_df.columns:
            locations_df['CondTag'] = default_cond_tag
        # we get information about cond from accounts_df
        cond_tags = {}  # information about each cond tag
        account_layer_exclusion = {}  # for each account and layer, store info about cond class exclusion
        fill_empty(accounts_df, 'CondPriority', 1)
        fill_empty(accounts_df, 'CondPeril', '')
        for acc_rec in accounts_df.to_dict(orient="records"):
            cond_tag_key = (acc_rec['PortNumber'], acc_rec['AccNumber'], acc_rec['CondTag'])
            cond_number_key = (acc_rec['PortNumber'], acc_rec['AccNumber'], acc_rec['CondTag'], acc_rec['CondNumber'])
            cond_tag = cond_tags.setdefault(cond_tag_key, {'CondPriority': acc_rec['CondPriority'] or 1, 'CondPeril': acc_rec['CondPeril']})
            cond_tag.setdefault('layers', {})[acc_rec['layer_id']] = {'CondNumber': cond_number_key}
            exclusion_cond_tags = account_layer_exclusion.setdefault((acc_rec['PortNumber'], acc_rec['AccNumber']), {}).setdefault(acc_rec['layer_id'],
                                                                                                                                   set())
            if acc_rec.get('CondClass') == 1:
                exclusion_cond_tags.add(acc_rec['CondTag'])

        # we get the list of loc for each cond_tag
        loc_conds = {}
        KEY_INDEX = 0
        PRIORITY_INDEX = 1
        for loc_rec in locations_df.to_dict(orient="records"):
            loc_key = (loc_rec['PortNumber'], loc_rec['AccNumber'], loc_rec['LocNumber'])
            cond_key = (loc_rec['PortNumber'], loc_rec['AccNumber'], loc_rec.get('CondTag', default_cond_tag))
            if cond_key in cond_tags:
                cond_tag = cond_tags[cond_key]
            else:
                cond_tag = {'CondPriority': 1, 'layers': {}}
                cond_tags[cond_key] = cond_tag

            # if 'locations' not in cond_tag: # first time we see the cond tag in this loop, update the layers
            #     for layer_id in account_layers[loc_rec['PortNumber'], loc_rec['AccNumber']]:
            #         if layer_id not in cond_tag['CondNumber']:
            #             cond_tag['CondNumber'][layer_id] = {'CondNumber': tuple(list(cond_key)+[''])}

            cond_location = cond_tag.setdefault('locations', set())
            cond_location.add(loc_key)
            cond_tag_priority = cond_tag['CondPriority']
            conds = loc_conds.setdefault(loc_key, [])

            for i, cond in enumerate(conds):
                if cond_tag_priority < cond[PRIORITY_INDEX]:
                    conds.insert(i, (cond_key, cond_tag_priority))
                    break
                elif cond_tag_priority == cond[PRIORITY_INDEX] and cond_key != cond[KEY_INDEX]:
                    raise OasisException(f"{cond_key} and {cond[KEY_INDEX]} have same priority in {loc_key}")
            else:
                conds.append((cond_key, cond_tag_priority))

        # at first we just want condtag for each level
        for cond_key, cond_info in cond_tags.items():
            port_num, acc_num, cond_tag = cond_key
            cond_level_start = 1
            for loc_key in cond_info.get('locations', set()):
                for i, (cond_key_i, _) in enumerate(loc_conds[loc_key]):
                    if cond_key_i == cond_key:
                        cond_level_start = max(cond_level_start, i + 1)
                        break
            cond_info['cond_level_start'] = cond_level_start
            for layer_id, exclusion_conds in account_layer_exclusion[(cond_key[0], cond_key[1])].items():
                if layer_id not in cond_info['layers']:
                    if exclusion_conds:
                        extra_accounts.append({
                            'PortNumber': port_num,
                            'AccNumber': acc_num,
                            'CondTag': cond_tag,
                            'layer_id': layer_id,
                            'CondNumber': 'FullFilter',
                            'CondDed6All': 1,
                            'CondDedType6All': 1,
                            'CondPeril': 'AA1',
                        })
                    else:
                        extra_accounts.append({
                            'PortNumber': port_num,
                            'AccNumber': acc_num,
                            'CondTag': cond_tag,
                            'layer_id': layer_id,
                            'CondNumber': '',
                            'CondPeril': 'AA1',
                        })
            level_conds.setdefault(cond_level_start, set()).add(cond_key)
    return level_conds, extra_accounts


def get_levels(gul_inputs_df, locations_df, accounts_df):
    level_conds, extra_accounts = get_cond_info(locations_df, accounts_df)
    for group_name, group_info in copy.deepcopy(GROUPED_SUPPORTED_FM_LEVELS).items():
        if group_info['oed_source'] == 'location':
            locations_df['layer_id'] = 1
            yield locations_df, list(group_info['levels'].items())[1:], group_info['fm_peril_field']  # [1:] => 'site coverage' is already done
        elif group_info['oed_source'] == 'account':
            if group_name == 'cond' and level_conds:
                loc_conds_df = locations_df[['loc_id', 'PortNumber', 'AccNumber', 'CondTag']].drop_duplicates()
                for stage, cond_keys in level_conds.items():
                    cond_filter_df = pd.DataFrame(cond_keys, columns=['PortNumber', 'AccNumber', 'CondTag'])
                    loc_conds_df_filter = cond_filter_df[['PortNumber', 'AccNumber', 'CondTag']].drop_duplicates().merge(loc_conds_df, how='left')
                    gul_inputs_df.drop(columns=['CondTag'], inplace=True, errors='ignore')
                    gul_inputs_df['CondTag'] = gul_inputs_df[['loc_id', 'PortNumber', 'AccNumber']].merge(loc_conds_df_filter, how='left')['CondTag']
                    yield (cond_filter_df.merge(pd.concat([accounts_df, pd.DataFrame(extra_accounts)]), how='left'),
                           group_info['levels'].items(),
                           group_info['fm_peril_field'])
            else:
                yield accounts_df, group_info['levels'].items(), group_info.get('fm_peril_field')


def get_level_term_info(term_df_source, level_column_mapper, level_id, step_level, fm_peril_field, oed_schema):
    level_terms = set()
    terms_maps = {}
    coverage_group_map = {}
    fm_group_tiv = {}
    non_zero_default = {}
    for ProfileElementName, term_info in level_column_mapper[level_id].items():
        default_value = oed_schema.get_default(ProfileElementName)
        if ProfileElementName not in term_df_source.columns:
            if default_value == 0 or default_value in BLANK_VALUES:
                continue
            else:
                non_zero_default[ProfileElementName] = [term_info['FMTermType'].lower(), default_value]
                continue
        else:
            fill_empty(term_df_source, ProfileElementName, default_value)
        if 'FMProfileStep' in term_info:
            profile_steps = term_info["FMProfileStep"]
            if isinstance(profile_steps, int):
                profile_steps = [profile_steps]
            valid_step_trigger_types = term_df_source.loc[(term_df_source['StepTriggerType'].isin(profile_steps))
                                                          & (term_df_source[ProfileElementName] > 0), 'StepTriggerType'].unique()
            if len(valid_step_trigger_types):
                level_terms.add(term_info['FMTermType'].lower())
            for step_trigger_type in valid_step_trigger_types:
                coverage_aggregation_method = STEP_TRIGGER_TYPES[step_trigger_type]['coverage_aggregation_method']
                calcrule_assignment_method = STEP_TRIGGER_TYPES[step_trigger_type]['calcrule_assignment_method']
                for coverage_type_id in supp_cov_type_ids:
                    FMTermGroupID = COVERAGE_AGGREGATION_METHODS[coverage_aggregation_method][coverage_type_id]

                    if (step_trigger_type, coverage_type_id) in coverage_group_map:
                        if coverage_group_map[(step_trigger_type, coverage_type_id)] != FMTermGroupID:
                            raise OasisException(
                                f"multiple coverage_type_id {(step_trigger_type, coverage_type_id)} "
                                f"assigned to the different FMTermGroupID "
                                f"{(FMTermGroupID, coverage_group_map[(step_trigger_type, coverage_type_id)])}")
                    else:
                        coverage_group_map[(step_trigger_type, coverage_type_id)] = FMTermGroupID

                    terms_map = terms_maps.setdefault((FMTermGroupID, step_trigger_type), {fm_peril_field: 'fm_peril'} if fm_peril_field else {})
                    if CALCRULE_ASSIGNMENT_METHODS[calcrule_assignment_method][FMTermGroupID]:
                        terms_map[ProfileElementName] = term_info['FMTermType'].lower()
        else:
            if not (term_df_source[ProfileElementName] > 0).any():
                continue
            level_terms.add(term_info['FMTermType'].lower())
            coverage_type_ids = term_info.get("CoverageTypeID", supp_cov_type_ids)
            FMTermGroupID = term_info.get('FMTermGroupID', 1)
            if step_level:
                term_key = (FMTermGroupID, 0)
            else:
                term_key = FMTermGroupID

            if isinstance(coverage_type_ids, int):
                coverage_type_ids = [coverage_type_ids]
            fm_group_tiv[FMTermGroupID] = coverage_type_ids
            for coverage_type_id in coverage_type_ids:
                if step_level:
                    coverage_group_key = (0, coverage_type_id)
                else:
                    coverage_group_key = coverage_type_id
                if coverage_type_id in coverage_group_map:
                    if coverage_group_map[coverage_group_key] != FMTermGroupID:
                        raise OasisException(
                            f"multiple coverage_type_id {coverage_group_key}"
                            f"assigned to the different FMTermGroupID {(FMTermGroupID, coverage_group_map[coverage_group_key])}")
                else:
                    coverage_group_map[coverage_group_key] = FMTermGroupID
            terms_maps.setdefault(term_key, {fm_peril_field: 'fm_peril'} if fm_peril_field else {})[
                ProfileElementName] = term_info['FMTermType'].lower()
    if level_terms:
        for ProfileElementName, (term, default_value) in non_zero_default.items():
            term_df_source[ProfileElementName] = default_value
            level_terms.add(term)
            for terms_map in terms_maps.values():
                terms_map[ProfileElementName] = term

    return level_terms, terms_maps, coverage_group_map, fm_group_tiv


@oasis_log
def get_il_input_items(
        gul_inputs_df,
        locations_df,
        accounts_df,
        oed_schema,
        exposure_profile=get_default_exposure_profile(),
        accounts_profile=get_default_accounts_profile(),
        fm_aggregation_profile=get_default_fm_aggregation_profile(),
        do_disaggregation=True,
):
    """
    Generates and returns a Pandas dataframe of IL input items.

    :param locations_df: Source exposure
    :type locations_df: pandas.DataFrame

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

    :param do_disaggregation: whether to split terms and conditions for aggregate exposure (optional)
    :param do_disaggregation: bool

    :return: IL inputs dataframe
    :rtype: pandas.DataFrame

    :return Accounts dataframe
    :rtype: pandas.DataFrame
    """
    profile = get_grouped_fm_profile_by_level_and_term_group(exposure_profile, accounts_profile)
    tiv_terms = {v['tiv']['ProfileElementName']: str(v['tiv']['CoverageTypeID']) for k, v in profile[1].items()}

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
    # =====
    useful_cols = sorted(set(['layer_id', 'orig_level_id', 'level_id', 'agg_id', 'gul_input_id', 'agg_tiv', 'NumberOfRisks']
                             + get_useful_summary_cols(oed_hierarchy) + list(tiv_terms.values()))
                         - {'profile_id', 'item_id', 'output_id'}, key=str.lower)
    gul_inputs_df.rename(columns={'item_id': 'gul_input_id'}, inplace=True)
    # adjust tiv columns and name them as their coverage id
    gul_inputs_df.rename(columns=tiv_terms, inplace=True)
    gul_inputs_df[['risk_id', 'NumberOfRisks']] = gul_inputs_df[['building_id', 'NumberOfBuildings']]
    gul_inputs_df.loc[gul_inputs_df['IsAggregate'] == 0, ['risk_id', 'NumberOfRisks']] = 1, 1
    gul_inputs_df.loc[gul_inputs_df['NumberOfRisks'] == 0, 'NumberOfRisks'] = 1

    # initialization
    agg_keys = set()
    for level_id in fm_aggregation_profile:
        agg_keys = agg_keys.union(set([v['field'] for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]))

    present_cols = [col for col in gul_inputs_df.columns if col in set(useful_cols).union(agg_keys).union(fm_term_ids + ['LocPeril'])]
    gul_inputs_df = gul_inputs_df[present_cols]

    # Remove TIV col from location as this information is present in gul_input_df
    locations_df = locations_df.drop(columns=tiv_terms.keys())

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

    level_id = SUPPORTED_FM_LEVELS['site coverage']['id']

    prev_level_df = gul_inputs_df[present_cols]
    prev_agg_key = [v['field'] for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]

    # no duplicate, if we have, error will appear later for agg_id_1
    prev_level_df.drop_duplicates(subset=prev_agg_key, inplace=True, ignore_index=True)
    if 'LocPeril' in prev_level_df:
        peril_filter = oed_schema.peril_filtering(prev_level_df['peril_id'], prev_level_df['LocPeril'])
        prev_level_df.loc[~peril_filter, list(set(prev_level_df.columns).intersection(fm_term_ids))] = 0
    if do_disaggregation:
        __split_fm_terms_by_risk(prev_level_df)
    prev_level_df['agg_id'] = factorize_ndarray(prev_level_df.loc[:, ['loc_id', 'risk_id', 'coverage_type_id']].values, col_idxs=range(3))[0]
    prev_level_df['level_id'] = 1
    prev_level_df['orig_level_id'] = level_id
    prev_level_df['layer_id'] = 1
    prev_level_df['agg_tiv'] = prev_level_df['tiv']
    prev_level_df['attachment'] = 0
    prev_level_df['share'] = 0

    il_inputs_df_list = []
    gul_inputs_df = gul_inputs_df.drop(columns=fm_term_ids + ['tiv'], errors='ignore').reset_index(drop=True)
    prev_df_subset = prev_agg_key

    # Determine whether step policies are listed, are not full of nans and step
    # numbers are greater than zero
    step_policies_present = False
    extra_fm_col = ['layer_id']
    if 'StepTriggerType' in accounts_df and 'StepNumber' in accounts_df:
        fill_empty(accounts_df, ['StepTriggerType', 'StepNumber'], 0)
        step_account = (accounts_df[accounts_df[accounts_df['StepTriggerType'].notnull()]['StepNumber'].gt(0)][['PortNumber', 'AccNumber']]
                        .drop_duplicates())
        step_policies_present = bool(step_account.shape[0])
        # this is done only for fmcalc to make sure it program nodes stay as a tree, remove 'is_step' logic when fmcalc is dropped
        if step_policies_present:
            step_account['is_step'] = 1
            gul_inputs_df = gul_inputs_df.merge(step_account, how='left')
            gul_inputs_df['is_step'] = gul_inputs_df['is_step'].fillna(0).astype('int8')
            extra_fm_col = ['layer_id', 'is_step']
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

    for term_df_source, levels, fm_peril_field in get_levels(gul_inputs_df, locations_df, accounts_df):
        for level, level_info in levels:
            level_id = level_info['id']
            step_level = 'StepTriggerType' in level_column_mapper[level_id]  # only true is step policy are present
            level_terms, terms_maps, coverage_group_map, fm_group_tiv = get_level_term_info(
                term_df_source, level_column_mapper, level_id, step_level, fm_peril_field, oed_schema)
            if not terms_maps:  # no terms we skip this level
                continue
            agg_key = [v['field'] for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]
            # get all rows with terms in term_df_source and determine the correct FMTermGroupID
            level_df_list = []
            for group_key, terms in terms_maps.items():
                if step_level:
                    FMTermGroupID, step_trigger_type = group_key
                    group_df = term_df_source[term_df_source['StepTriggerType'] == step_trigger_type]
                    terms = {**terms_maps.get((1, 0), {}), **terms}  # take all common terms in (1,0) plus term with step_trigger_type filter
                else:
                    FMTermGroupID = group_key
                    group_df = term_df_source

                group_df = (group_df[list(set(agg_key + list(terms.keys()) + extra_fm_col)
                                          .union(set(useful_cols).difference(set(gul_inputs_df.columns)))
                                          .intersection(group_df.columns))]
                            .assign(FMTermGroupID=FMTermGroupID))
                numeric_terms = [term for term in terms.keys() if is_numeric_dtype(group_df[term])]
                keep_df = group_df[(group_df[numeric_terms] > 0).any(axis='columns')][list(
                    set(agg_key).intersection(group_df.columns))].drop_duplicates()
                group_df = keep_df.merge(group_df, how='left')

                # multiple ProfileElementName can have the same fm terms (ex: StepTriggerType 5), we take the max to have a unique one
                for ProfileElementName, term in terms.items():
                    if term in group_df:
                        group_df[term] = group_df[[term, ProfileElementName]].max(axis=1)
                        group_df.drop(columns=ProfileElementName, inplace=True)
                    else:
                        group_df.rename(columns={ProfileElementName: term}, inplace=True)
                level_df_list.append(group_df)
            level_df = pd.concat(level_df_list, copy=True)
            level_df = level_df.drop_duplicates(subset=set(level_df.columns) - set(SOURCE_IDX.values()))

            if step_level:
                # merge with gul_inputs_df needs to be based on 'steptriggertype' and 'coverage_type_id'
                gul_inputs_df.drop(columns=['FMTermGroupID'], errors='ignore', inplace=True)
                coverage_type_id_df = pd.DataFrame(
                    [[StepTriggerType, coverage_type_id, FMTermGroupID]
                        for (StepTriggerType, coverage_type_id), FMTermGroupID in coverage_group_map.items()],
                    columns=['steptriggertype', 'coverage_type_id', 'FMTermGroupID'])
                level_df = level_df.merge(coverage_type_id_df)
            else:
                # map the coverage_type_id to the correct FMTermGroupID for this level. coverage_type_id without term and therefor FMTermGroupID are map to 0
                gul_inputs_df['FMTermGroupID'] = gul_inputs_df['coverage_type_id'].map(coverage_group_map, na_action='ignore')

            # make sure correct tiv sum exist
            for FMTermGroupID, coverage_type_ids in fm_group_tiv.items():
                tiv_key = '_'.join(map(str, sorted(coverage_type_ids)))
                if tiv_key not in gul_inputs_df:
                    gul_inputs_df[tiv_key] = gul_inputs_df[map(str, sorted(coverage_type_ids))].sum(axis=1)

            # we have prepared FMTermGroupID on gul or level df (depending on  step_level) now we can merge the terms for this level to gul
            level_df = (gul_inputs_df.merge(level_df, how='left'))

            # assign correct tiv
            for FMTermGroupID, coverage_type_ids in fm_group_tiv.items():
                tiv_key = '_'.join(map(str, sorted(coverage_type_ids)))
                level_df['tiv'] = level_df.loc[level_df['FMTermGroupID'] == FMTermGroupID, tiv_key]

            # check that peril_id is part of the fm peril policy, if not we remove the terms
            if 'fm_peril' in level_df:
                level_df['fm_peril'] = level_df['fm_peril'].fillna('')
                peril_filter = oed_schema.peril_filtering(level_df['peril_id'], level_df['fm_peril'])
                level_df.loc[~peril_filter, list(level_terms) + ['FMTermGroupID']] = 0
                factorize_key = agg_key + ['FMTermGroupID', 'fm_peril']
            else:
                factorize_key = agg_key + ['FMTermGroupID']
            level_df['FMTermGroupID'] = level_df['FMTermGroupID'].astype('Int64')

            # make sure agg_id without term still have the same amount of layer
            level_df = level_df.merge(
                prev_level_df[['gul_input_id', 'layer_id', 'agg_id']]
                .rename(columns={'layer_id': 'layer_id_prev', 'agg_id': 'agg_id_prev'}),
                how='left'
            )

            level_df.loc[level_df['layer_id'].isna(), 'layer_id'] = level_df.loc[level_df['layer_id'].isna(), 'layer_id_prev']
            level_df['layer_id'] = level_df['layer_id'].astype('int32')
            level_df['agg_id_prev'] = level_df['agg_id_prev'].fillna(0).astype('int64')
            level_df.drop(columns=['layer_id_prev'], inplace=True)
            if do_disaggregation:
                if 'risk_id' in agg_key:
                    __split_fm_terms_by_risk(level_df)

            # for line that had no term FMTermGroupID == 0, we store in FMTermGroupID the previous agg_id to keep the same granularity
            if 'is_step' in level_df:
                level_df.loc[(level_df['FMTermGroupID'] == 0) & (level_df['is_step'] == 1), 'FMTermGroupID'] = -level_df['agg_id_prev']

            level_df['agg_id'] = factorize_ndarray(level_df.loc[:, factorize_key].values, col_idxs=range(len(factorize_key)))[0]

            # check rows in prev df that are this level granularity (if prev_agg_id has multiple corresponding agg_id)
            need_root_start_df = level_df.groupby("agg_id_prev")["agg_id"].nunique()
            need_root_start_df = need_root_start_df[need_root_start_df > 1].index

            # create new prev df for element that need to restart from items
            root_df = level_df[((level_df['agg_id_prev'].isin(need_root_start_df)) & (level_df['layer_id'] == 1))]

            root_df['to_agg_id'] = root_df['agg_id']
            root_df['agg_id'] = -root_df['gul_input_id']
            root_df.drop_duplicates(subset='agg_id', inplace=True)
            root_df['level_id'] = len(il_inputs_df_list) + 1

            max_agg_id = np.max(level_df['agg_id'])
            prev_level_df['to_agg_id'] = (prev_level_df[['gul_input_id']]
                                          .merge(level_df.drop_duplicates(subset=['gul_input_id'])[['gul_input_id', 'agg_id']])['agg_id'])

            prev_level_df.loc[prev_level_df['to_agg_id'].isna(), 'to_agg_id'] = (
                max_agg_id +
                factorize_ndarray(prev_level_df.loc[prev_level_df['to_agg_id'].isna(), ['agg_id']].values, col_idxs=range(len(['agg_id'])))[0])
            prev_level_df.loc[prev_level_df['agg_id'].isin(need_root_start_df), 'to_agg_id'] = 0

            prev_level_df['to_agg_id'] = prev_level_df['to_agg_id'].astype('int32')

            level_df.drop(columns=['agg_id_prev'], inplace=True)

            # we don't need coverage_type_id as an agg key as tiv already represent the sum of correct tiv values
            level_df = level_df.merge(
                level_df[['loc_id', 'building_id', 'agg_id', 'tiv']]
                .drop_duplicates()[['agg_id', 'tiv']]
                .groupby('agg_id', observed=True)['tiv']
                .sum()
                .reset_index(name='agg_tiv'), how='left'
            )

            __drop_duplicated_row(il_inputs_df_list[-1] if il_inputs_df_list else None, prev_level_df, prev_df_subset)
            il_inputs_df_list.append(pd.concat([df for df in [prev_level_df, root_df] if not df.empty]))

            level_df['level_id'] = len(il_inputs_df_list) + 1
            level_df['orig_level_id'] = level_id

            sub_agg_key = [v['field'] for v in fm_aggregation_profile[level_id].get('FMSubAggKey', {}).values()
                           if v['field'] in level_df.columns]

            prev_level_df = level_df
            prev_df_subset = ['agg_id'] + sub_agg_key + ['layer_id']

    # create account aggregation if necessary
    if prev_level_df['orig_level_id'].max() < SUPPORTED_FM_LEVELS['policy layer']['id']:
        level = 'policy layer'
        level_id = SUPPORTED_FM_LEVELS[level]['id']

        agg_key = [v['field'] for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]
        sub_agg_key = [v['field'] for v in fm_aggregation_profile[level_id].get('FMSubAggKey', {}).values()
                       if v['field'] in prev_level_df.columns]

        need_account_aggregation = (prev_level_df[agg_key + sub_agg_key + ['layer_id', 'agg_id']]
                                    .drop_duplicates()
                                    .groupby(agg_key + sub_agg_key + ['layer_id'], observed=True)
                                    .size()
                                    .max() > 1) or set(SUMMARY_TOP_LEVEL_COLS).difference(set(prev_level_df.columns))

        if need_account_aggregation:
            level_df = gul_inputs_df.merge(accounts_df[list(set(agg_key + sub_agg_key + ['layer_id'])
                                                            .union(set(useful_cols).difference(set(gul_inputs_df.columns)))
                                                            .intersection(accounts_df.columns))])
            level_df['orig_level_id'] = level_id
            level_df['level_id'] = len(il_inputs_df_list) + 2
            level_df['agg_id'] = factorize_ndarray(level_df.loc[:, agg_key].values, col_idxs=range(len(agg_key)))[0]
            prev_level_df['to_agg_id'] = (prev_level_df[['gul_input_id']]
                                          .merge(level_df.drop_duplicates(subset=['gul_input_id'])[['gul_input_id', 'agg_id']])['agg_id'])

            il_inputs_df_list.append(prev_level_df)
            prev_level_df = level_df

    prev_level_df['to_agg_id'] = 0
    il_inputs_df_list.append(prev_level_df.drop_duplicates(subset=sub_agg_key + ['agg_id', 'layer_id']))

    il_inputs_df = pd.concat(il_inputs_df_list)
    for col in set(list(il_inputs_df.columns)):
        try:
            il_inputs_df[col] = il_inputs_df[col].fillna(0)
        except (TypeError, ValueError):
            pass
    for col in fm_term_ids:
        if col not in il_inputs_df.columns:
            il_inputs_df[col] = 0

    del prev_level_df
    del il_inputs_df_list
    gc.collect()

    # set top agg_id for later xref computation
    il_inputs_df['top_agg_id'] = factorize_ndarray(il_inputs_df.loc[:, agg_key].values, col_idxs=range(len(agg_key)))[0]

    # Final setting of data types before returning the IL input items
    dtypes = {
        **{t: 'float64' for t in ['tiv', 'agg_tiv', 'deductible', 'deductible_min', 'deductible_max', 'limit', 'attachment', 'share',
                                  'deductible1', 'limit1', 'limit2', 'trigger_start', 'trigger_end', 'payout_start', 'payout_end',
                                  'scale1', 'scale2']},
        **{t: 'int32' for t in
           ['agg_id', 'to_agg_id', 'item_id', 'layer_id', 'level_id', 'orig_level_id', 'calcrule_id', 'profile_id', 'steptriggertype', 'step_id']},
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
    il_inputs_df.loc[il_inputs_df['ded_type'] == DEDUCTIBLE_AND_LIMIT_TYPES['pctiv']['id'], 'ded_type'] = DEDUCTIBLE_AND_LIMIT_TYPES['flat']['id']

    il_inputs_df['limit'] = np.where(
        il_inputs_df['lim_type'] == DEDUCTIBLE_AND_LIMIT_TYPES['pctiv']['id'],
        il_inputs_df['limit'] * il_inputs_df['agg_tiv'],
        il_inputs_df['limit']
    )
    il_inputs_df.loc[il_inputs_df['lim_type'] == DEDUCTIBLE_AND_LIMIT_TYPES['pctiv']['id'], 'lim_type'] = DEDUCTIBLE_AND_LIMIT_TYPES['flat']['id']

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
    il_inputs_df.reset_index(drop=True, inplace=True)
    policy_layer_filter = il_inputs_df['orig_level_id'] == SUPPORTED_FM_LEVELS['policy layer']['id']
    if step_policies_present:
        step_filter = (il_inputs_df['StepTriggerType'] > 0)
        base_filter = (~step_filter) & (~policy_layer_filter)
        policy_layer_filter = (~step_filter) & policy_layer_filter
        il_inputs_df.loc[step_filter, 'calcrule_id'] = get_calc_rule_ids(il_inputs_df[step_filter], calc_rule_type='step')
    else:
        base_filter = ~policy_layer_filter
    il_inputs_df.loc[base_filter, 'calcrule_id'] = get_calc_rule_ids(il_inputs_df[base_filter], calc_rule_type='base')
    il_inputs_df.loc[policy_layer_filter, 'calcrule_id'] = get_calc_rule_ids(il_inputs_df[policy_layer_filter], calc_rule_type='policy_layer')
    il_inputs_df['calcrule_id'] = il_inputs_df['calcrule_id'].astype('uint32')

    # Set the policy TC IDs
    if 'StepTriggerType' in il_inputs_df:
        il_inputs_df.loc[
            ~(il_inputs_df['StepTriggerType'] > 0), 'profile_id'
        ] = get_profile_ids(
            il_inputs_df[~(il_inputs_df['StepTriggerType'] > 0)]
        )

        il_inputs_df.loc[
            il_inputs_df['StepTriggerType'] > 0, 'profile_id'
        ] = get_step_profile_ids(
            il_inputs_df[il_inputs_df['StepTriggerType'] > 0],
            offset=il_inputs_df['profile_id'].max(),
            idx_cols=['AccNumber', 'PolNumber', 'PortNumber']
        )
    else:
        il_inputs_df['profile_id'] = get_profile_ids(il_inputs_df)
    il_inputs_df['profile_id'] = il_inputs_df['profile_id'].astype('uint32')

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
        fm_policytc_df = il_inputs_df.loc[il_inputs_df['agg_id'] > 0, ['layer_id', 'level_id', 'agg_id', 'profile_id', 'orig_level_id']]
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
        if 'StepTriggerType' in il_inputs_df:
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
            # make sure there is no step file in the folder
            cols = ['profile_id', 'calcrule_id', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'limit', 'share']
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

            cols = ['profile_id', 'calcrule_id', 'deductible1', 'deductible2', 'deductible3', 'attachment1', 'limit1', 'share1', 'share2', 'share3']
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
        item_level = il_inputs_df.loc[(il_inputs_df['level_id'] == 1) & (il_inputs_df['agg_id'] > 0), ['gul_input_id', 'level_id', 'agg_id']]
        item_level.rename(columns={'gul_input_id': 'from_agg_id',
                                   'agg_id': 'to_agg_id',
                                   }, inplace=True)
        item_level.drop_duplicates(keep='first', inplace=True)
        il_inputs_df = il_inputs_df[il_inputs_df['to_agg_id'] != 0]
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
        xref_df = get_xref_df(il_inputs_df)[['gul_input_id', 'layer_id']].rename(columns={'gul_input_id': 'agg_id'})
        xref_df['output'] = xref_df.reset_index().index + 1
        xref_df[['output', 'agg_id', 'layer_id']].to_csv(
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
        oasis_files_prefixes=OASIS_FILES_PREFIXES['il'],
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
    :type locations_df: pandas.DataFrame

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
    oasis_files_prefixes = copy.deepcopy(oasis_files_prefixes)

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
