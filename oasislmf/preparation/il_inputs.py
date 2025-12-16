__all__ = [
    'get_calc_rule_ids',
    'get_grouped_fm_profile_by_level_and_term_group',
    'get_grouped_fm_terms_by_level_and_term_group',
    'get_oed_hierarchy',
    'get_il_input_items',
]

import contextlib
import copy
import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from ods_tools.oed import fill_empty, BLANK_VALUES

from oasislmf.preparation.summaries import get_useful_summary_cols
from oasislmf.pytools.common.data import (fm_policytc_headers, fm_policytc_dtype,
                                          fm_profile_headers, fm_profile_dtype,
                                          fm_profile_step_headers, fm_profile_step_dtype,
                                          fm_programme_headers, fm_programme_dtype,
                                          fm_xref_headers, fm_xref_dtype
                                          )
from oasislmf.utils.calc_rules import get_calc_rules
from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
from oasislmf.utils.data import factorize_ndarray, get_ids
from oasislmf.utils.defaults import (OASIS_FILES_PREFIXES,
                                     get_default_accounts_profile, get_default_exposure_profile,
                                     get_default_fm_aggregation_profile, SOURCE_IDX)
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.fm import (CALCRULE_ASSIGNMENT_METHODS, COVERAGE_AGGREGATION_METHODS,
                               DEDUCTIBLE_AND_LIMIT_TYPES, FM_LEVELS, FML_ACCALL, STEP_TRIGGER_TYPES,
                               SUPPORTED_FM_LEVELS, FM_TERMS, GROUPED_SUPPORTED_FM_LEVELS)
from oasislmf.utils.log import oasis_log
from oasislmf.utils.path import as_path
from oasislmf.utils.profiles import (get_default_step_policies_profile,
                                     get_grouped_fm_profile_by_level_and_term_group,
                                     get_grouped_fm_terms_by_level_and_term_group,
                                     get_oed_hierarchy)

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

# convertion from np dtype to pandas dtype
fm_policytc_dtype = {col: dtype for col, (dtype, _) in fm_policytc_dtype.fields.items()}
fm_profile_dtype = {col: dtype for col, (dtype, _) in fm_profile_dtype.fields.items()}
fm_profile_step_dtype = {col: dtype for col, (dtype, _) in fm_profile_step_dtype.fields.items()}
fm_programme_dtype = {col: dtype for col, (dtype, _) in fm_programme_dtype.fields.items()}
fm_xref_dtype = {col: dtype for col, (dtype, _) in fm_xref_dtype.fields.items()}


# Define a list of all supported OED coverage types in the exposure
supp_cov_type_ids = [v['id'] for v in SUPPORTED_COVERAGE_TYPES.values()]

policytc_cols = ['profile_id', 'calcrule_id', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'limit', 'share']

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


def prepare_ded_and_limit(level_df):
    simplify_no_terms = {
        'deductible': ['ded_type', 'ded_code'],
        'limit': ['lim_type', 'lim_code']
    }
    level_df['need_tiv'] = False
    for term, type_and_code in simplify_no_terms.items():
        if term in level_df.columns:
            for elm in type_and_code:
                if elm in level_df.columns:
                    level_df.loc[level_df[term] == 0, elm] = 0
                    if elm[-5:] == '_type':
                        level_df['need_tiv'] |= level_df[elm].isin({2, 3})  # 2 or 3 type means term depend on tiv
        else:
            level_df = level_df.drop(columns=type_and_code, errors='ignore')

    return level_df


def get_calc_rule_ids(il_inputs_calc_rules_df, calc_rule_type):
    """
    merge selected il_inputs with the correct calc_rule table and  return a pandas Series of calc. rule IDs

    Args:
     il_inputs_calc_rules_df (DataFrame):  IL input items dataframe
     calc_rule_type (str): type of calc_rule to look for

    Returns:
        pandas Series of calc. rule IDs
    """
    il_inputs_calc_rules_df = il_inputs_calc_rules_df.copy()
    calc_rules_df, calc_rule_term_info = get_calc_rules(calc_rule_type)
    calc_rules_df = calc_rules_df.drop(columns=['desc', 'id_key'], axis=1, errors='ignore')

    terms = []
    terms_indicators = []
    no_terms = {
        'deductible': ['ded_type', 'ded_code'],
        'limit': ['lim_type', 'lim_code']
    }
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

    for term, type_and_code in no_terms.items():
        if f'{term}_gt_0' in calc_rules_df.columns:
            for elm in type_and_code:
                if elm in il_inputs_calc_rules_df.columns:
                    il_inputs_calc_rules_df.loc[il_inputs_calc_rules_df[term] == 0, elm] = 0
        else:
            for elm in type_and_code:
                if elm in calc_rules_df.columns:
                    calc_rules_df = calc_rules_df[calc_rules_df[elm] == 0].drop(columns=[elm])

    il_inputs_calc_rules_df.loc[:, terms_indicators] = np.where(il_inputs_calc_rules_df[terms] > 0, 1, 0)

    merge_col = list(set(il_inputs_calc_rules_df.columns).intersection(calc_rules_df.columns).difference({'calcrule_id'}))
    if len(merge_col):
        calcrule_ids = (
            il_inputs_calc_rules_df[merge_col].reset_index()
            .merge(calc_rules_df[merge_col + ['calcrule_id']].drop_duplicates(), how='left', on=merge_col)
        ).set_index('index')['calcrule_id'].fillna(0)
    else:
        return 100  # no term we return pass through
    if 0 in calcrule_ids.unique():
        _cols = list(set(['PortNumber', 'AccNumber', 'LocNumber'] + merge_col).intersection(il_inputs_calc_rules_df.columns))
        no_match_keys = il_inputs_calc_rules_df.loc[calcrule_ids == 0, _cols].drop_duplicates()
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
    factor_col = list(set(il_inputs_df.columns).intersection(policytc_cols).difference({'profile_id', }))
    return factorize_ndarray(il_inputs_df.loc[:, factor_col].values, col_idxs=range(len(factor_col)))[0]


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


def get_cond_info(locations_df, accounts_df):
    pol_info = {}
    level_conds = {}
    extra_accounts = []
    default_cond_tag = '0'
    if 'CondTag' in locations_df.columns:
        fill_empty(locations_df, 'CondTag', default_cond_tag)
        loc_condkey_df = locations_df.loc[locations_df['CondTag'] != default_cond_tag, ['acc_id', 'CondTag']].drop_duplicates()
    else:
        loc_condkey_df = pd.DataFrame([], columns=['acc_id', 'CondTag'])

    if 'CondTag' in accounts_df.columns:
        fill_empty(accounts_df, 'CondTag', default_cond_tag)
        acc_condkey_df = accounts_df.loc[accounts_df['CondTag'] != '', ['acc_id', 'CondTag']].drop_duplicates()
        condkey_match_df = acc_condkey_df.merge(loc_condkey_df, how='outer', indicator=True)
        missing_condkey_df = condkey_match_df.loc[condkey_match_df['_merge'] == 'right_only', ['acc_id', 'CondTag']]
    else:
        acc_condkey_df = pd.DataFrame([], columns=['acc_id', 'CondTag'])
        missing_condkey_df = loc_condkey_df

    if missing_condkey_df.shape[0]:
        raise OasisException(f'Those condtag are present in locations but missing in the account file:\n{missing_condkey_df}')

    if acc_condkey_df.shape[0]:
        if 'CondTag' not in locations_df.columns:
            locations_df['CondTag'] = default_cond_tag
        # we get information about cond from accounts_df
        cond_tags = {}  # information about each cond tag
        account_layer_exclusion = {}  # for each account and layer, store info about cond class exclusion
        if 'CondPriority' in accounts_df.columns:
            fill_empty(accounts_df, 'CondPriority', 1)
        else:
            accounts_df['CondPriority'] = 1
        if 'CondPeril' in accounts_df.columns:
            fill_empty(accounts_df, 'CondPeril', '')
        else:
            accounts_df['CondPeril'] = ''
        for acc_rec in accounts_df.to_dict(orient="records"):
            cond_tag_key = (acc_rec['acc_id'], acc_rec['CondTag'])
            cond_number_key = (acc_rec['acc_id'], acc_rec['CondTag'], acc_rec['CondNumber'])
            cond_tag = cond_tags.setdefault(cond_tag_key, {'CondPriority': acc_rec['CondPriority'] or 1, 'CondPeril': acc_rec['CondPeril']})
            cond_tag.setdefault('layers', {})[acc_rec['layer_id']] = {'CondNumber': cond_number_key}
            exclusion_cond_tags = account_layer_exclusion.setdefault(acc_rec['acc_id'], {}).setdefault(acc_rec['layer_id'],
                                                                                                       set())
            pol_info[(acc_rec['acc_id'], acc_rec['layer_id'])] = [acc_rec['PolNumber'], acc_rec['LayerNumber'], acc_rec['acc_idx']]
            if acc_rec.get('CondClass') == 1:
                exclusion_cond_tags.add(acc_rec['CondTag'])

        # we get the list of loc for each cond_tag
        loc_conds = {}
        KEY_INDEX = 0
        PRIORITY_INDEX = 1
        for loc_rec in locations_df.to_dict(orient="records"):
            loc_key = loc_rec['loc_id']
            cond_key = (loc_rec['acc_id'], loc_rec.get('CondTag', default_cond_tag))
            if cond_key in cond_tags:
                cond_tag = cond_tags[cond_key]
            else:
                cond_tag = {'CondPriority': 1, 'layers': {}}
                cond_tags[cond_key] = cond_tag

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
            acc_id, cond_tag = cond_key
            cond_level_start = 1
            for loc_key in cond_info.get('locations', set()):
                for i, (cond_key_i, _) in enumerate(loc_conds[loc_key]):
                    if cond_key_i == cond_key:
                        cond_level_start = max(cond_level_start, i + 1)
                        break
            cond_info['cond_level_start'] = cond_level_start
            for layer_id, exclusion_conds in account_layer_exclusion[acc_id].items():
                if layer_id not in cond_info['layers']:
                    PolNumber, LayerNumber, acc_idx = pol_info[(acc_id, layer_id)]
                    if exclusion_conds:
                        extra_accounts.append({
                            'acc_idx': acc_idx,
                            'acc_id': acc_id,
                            'PolNumber': PolNumber,
                            'LayerNumber': LayerNumber,
                            'CondTag': cond_tag,
                            'layer_id': layer_id,
                            'CondNumber': 'FullFilter',
                            'CondDed6All': 1,
                            'CondDedType6All': 1,
                            'CondPeril': 'AA1',
                        })
                    else:
                        extra_accounts.append({
                            'acc_idx': acc_idx,
                            'acc_id': acc_id,
                            'PolNumber': PolNumber,
                            'LayerNumber': LayerNumber,
                            'CondTag': cond_tag,
                            'layer_id': layer_id,
                            'CondNumber': '',
                            'CondPeril': 'AA1',
                        })
            level_conds.setdefault(cond_level_start, set()).add(cond_key)
    return level_conds, extra_accounts


def get_levels(locations_df, accounts_df):
    if locations_df is not None and not {"CondTag", "CondNumber"}.difference(accounts_df.columns):
        level_conds, extra_accounts = get_cond_info(locations_df, accounts_df)
    else:
        level_conds = []
    for group_name, group_info in copy.deepcopy(GROUPED_SUPPORTED_FM_LEVELS).items():
        if group_info['oed_source'] == 'location':
            if locations_df is not None:
                locations_df['layer_id'] = 1
                yield locations_df, list(group_info['levels'].items()), group_info.get('fm_peril_field'), None
        elif group_info['oed_source'] == 'account':
            if group_name == 'cond' and level_conds:
                loc_conds_df = locations_df[['loc_id', 'acc_id', 'CondTag']].drop_duplicates()
                for stage, cond_keys in level_conds.items():
                    cond_filter_df = (pd.DataFrame(cond_keys, columns=['acc_id', 'CondTag'])
                                      .sort_values(by=['acc_id', 'CondTag']))
                    loc_conds_df_filter = cond_filter_df[['acc_id', 'CondTag']].drop_duplicates().merge(loc_conds_df, how='left')
                    yield (cond_filter_df.merge(pd.concat([accounts_df, pd.DataFrame(extra_accounts)]), how='left'),
                           group_info['levels'].items(),
                           group_info['fm_peril_field'],
                           loc_conds_df_filter[['loc_id', 'CondTag']])
            else:
                yield accounts_df, group_info['levels'].items(), group_info.get('fm_peril_field'), None


def get_level_term_info(term_df_source, level_column_mapper, level_id, step_level, fm_peril_field, oed_schema):
    level_terms = set()
    terms_maps = {}
    coverage_group_map = {}
    fm_group_tiv = {}
    non_zero_default = {}
    for ProfileElementName, term_info in level_column_mapper[level_id].items():
        if term_info.get("FMTermType") == "tiv":
            continue
        default_value = oed_schema.get_default(ProfileElementName)
        if ProfileElementName not in term_df_source.columns:
            if default_value == 0 or default_value in BLANK_VALUES:
                continue
            else:
                non_zero_default[ProfileElementName] = [term_info['FMTermType'].lower(), default_value]
                continue
        else:
            fill_empty(term_df_source, ProfileElementName, default_value)

        if pd.isna(default_value):
            non_default_val = ~term_df_source[ProfileElementName].isna()
        else:
            non_default_val = (term_df_source[ProfileElementName] != default_value)
        if 'FMProfileStep' in term_info:
            profile_steps = term_info["FMProfileStep"]
            if isinstance(profile_steps, int):
                profile_steps = [profile_steps]
            valid_step_trigger_types = term_df_source.loc[(term_df_source['StepTriggerType'].isin(profile_steps))
                                                          & non_default_val, 'StepTriggerType'].unique()
            if len(valid_step_trigger_types):
                level_terms.add(term_info['FMTermType'].lower())
            for step_trigger_type in valid_step_trigger_types:
                coverage_aggregation_method = STEP_TRIGGER_TYPES[step_trigger_type]['coverage_aggregation_method']
                calcrule_assignment_method = STEP_TRIGGER_TYPES[step_trigger_type]['calcrule_assignment_method']
                for coverage_type_id in supp_cov_type_ids:
                    FMTermGroupID = COVERAGE_AGGREGATION_METHODS[coverage_aggregation_method].get(coverage_type_id)
                    if FMTermGroupID is None:  # step policy not supported for this coverage
                        continue

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
            if not (non_default_val).any():
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


def associate_items_peril_to_policy_peril(item_perils, policy_df, fm_peril_col, oed_schema):
    """
    for each peril_id in item_peril_list we map it to each string representing policy perils
    we then merge this mapping to the initial policies so that each line will have a peril_id
    that can be use directly as key when merging with the gul_input_df
    Args:
        item_perils: all peril id from gul_input_df
        policy_df: the df with the policies
        fm_peril_col: the name of the column to use that define the policy peril filter
        oed_schema: the schema object we use to map peril and subperils
    Returns:

    """
    fm_perils = policy_df[[fm_peril_col]].drop_duplicates()
    peril_map_df = pd.merge(fm_perils, item_perils, how='cross')
    peril_map_df = peril_map_df[oed_schema.peril_filtering(peril_map_df['peril_id'], peril_map_df[fm_peril_col])]
    return policy_df.merge(peril_map_df)


@oasis_log
def get_il_input_items(
        gul_inputs_df,
        exposure_data,
        target_dir,
        logger,
        exposure_profile=get_default_exposure_profile(),
        accounts_profile=get_default_accounts_profile(),
        fm_aggregation_profile=get_default_fm_aggregation_profile(),
        do_disaggregation=True,
        oasis_files_prefixes=OASIS_FILES_PREFIXES['il'],
        chunksize=(2 * 10 ** 5),
):
    """
    Generates and returns a Pandas dataframe of IL input items.

    :param gul_inputs_df: GUL input items
    :type gul_inputs_df: pandas.DataFrame

    :param exposure_data: object containing all information about the insurance policies

    :param target_dir: path to the directory used to write the fm files

    :param logger: logger object to trace progress

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
    target_dir = as_path(target_dir, 'Target IL input files directory', is_dir=True, preexists=False)
    with contextlib.ExitStack() as stack:
        fm_policytc_file = stack.enter_context(open(os.path.join(target_dir, f"{oasis_files_prefixes['fm_policytc']}.csv"), 'w'))
        fm_policytc_file.write(f"{','.join(fm_policytc_headers)}{os.linesep}")

        fm_profile_file = stack.enter_context(open(os.path.join(target_dir, f"{oasis_files_prefixes['fm_profile']}.csv"), 'w'))

        fm_programme_file = stack.enter_context(open(os.path.join(target_dir, f"{oasis_files_prefixes['fm_programme']}.csv"), 'w'))
        fm_programme_file.write(f"{','.join(fm_programme_headers)}{os.linesep}")

        fm_xref_file = stack.enter_context(open(os.path.join(target_dir, f"{oasis_files_prefixes['fm_xref']}.csv"), 'w'))

        if exposure_data.location is not None:
            locations_df = exposure_data.location.dataframe
            accounts_df = exposure_data.account.dataframe
            if 'acc_id' not in accounts_df:
                accounts_df['acc_id'] = get_ids(exposure_data.account.dataframe, ['PortNumber', 'AccNumber'])
            acc_id_map = accounts_df[['PortNumber', 'AccNumber', 'acc_id']].drop_duplicates()
            gul_inputs_df = gul_inputs_df.merge(acc_id_map, how='left')
            locations_df = locations_df.merge(acc_id_map, how='left')
            locations_df = locations_df.drop(columns=['PortNumber', 'AccNumber', 'LocNumber'])
            accounts_df = accounts_df[accounts_df['acc_id'].isin(locations_df['acc_id'].unique())].drop(columns=['PortNumber', 'AccNumber'])

        else:  # no location, case for cyber, marine ...
            locations_df = None
            gul_inputs_df['acc_id'] = gul_inputs_df['loc_id']
            accounts_df = exposure_data.account.dataframe
            accounts_df['acc_id'] = accounts_df['loc_id']
            accounts_df = accounts_df.drop(columns=['PortNumber', 'AccNumber'])

        oed_schema = exposure_data.oed_schema

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
        tiv_terms = {v['tiv']['ProfileElementName']: str(v['tiv']['CoverageTypeID']) for k, v in
                     profile[FM_LEVELS['site coverage']['id']].items()}
        useful_cols = sorted(set(['layer_id', 'orig_level_id', 'level_id', 'agg_id', 'gul_input_id', 'tiv', 'NumberOfRisks']
                                 + get_useful_summary_cols(oed_hierarchy)).union(tiv_terms)
                             - {'profile_id', 'item_id', 'output_id'}, key=str.lower)
        gul_inputs_df = gul_inputs_df.rename(columns={'item_id': 'gul_input_id'})
        # adjust tiv columns and name them as their coverage id
        gul_inputs_df[['risk_id', 'NumberOfRisks']] = gul_inputs_df[['building_id', 'NumberOfBuildings']]
        gul_inputs_df.loc[gul_inputs_df['IsAggregate'] == 0, 'risk_id'] = 1
        gul_inputs_df.loc[gul_inputs_df['IsAggregate'] == 0, 'NumberOfRisks'] = 1
        gul_inputs_df.loc[gul_inputs_df['NumberOfRisks'] == 0, 'NumberOfRisks'] = 1

        # initialization
        agg_keys = set()
        for level_id in fm_aggregation_profile:
            agg_keys = agg_keys.union(set([v['field'] for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]))

        present_cols = [col for col in gul_inputs_df.columns if col in set(useful_cols).union(agg_keys).union(fm_term_ids)]
        gul_inputs_df = gul_inputs_df[present_cols].rename(columns=tiv_terms)

        # Profile dict are base on key that correspond to the fm term name.
        # this prevents multiple file columns to point to the same fm term
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

        gul_inputs_df = gul_inputs_df.drop(columns=fm_term_ids, errors='ignore').reset_index(drop=True)
        gul_inputs_df['agg_id_prev'] = gul_inputs_df['gul_input_id']
        gul_inputs_df['layer_id'] = 1
        gul_inputs_df['PolNumber'] = pd.NA
        gul_inputs_df = gul_inputs_df.rename(columns={})
        item_perils = gul_inputs_df[['peril_id']].drop_duplicates()

        # Determine whether step policies are listed, are not full of nans and step
        # numbers are greater than zero
        step_policies_present = False

        extra_fm_col = ['layer_id', 'PolNumber', 'NumberOfBuildings', 'IsAggregate']
        if 'StepTriggerType' in accounts_df and 'StepNumber' in accounts_df:
            fill_empty(accounts_df, ['StepTriggerType', 'StepNumber'], 0)
            step_account = (accounts_df[accounts_df[accounts_df['StepTriggerType'].notnull()]['StepNumber'].gt(0)][['acc_id']]
                            .drop_duplicates())
            step_policies_present = bool(step_account.shape[0])
            # this is done only for fmcalc to make sure it program nodes stay as a tree, remove 'is_step' logic when fmcalc is dropped
            if step_policies_present:
                step_account['is_step'] = 1
                gul_inputs_df = gul_inputs_df.merge(step_account, how='left')
                gul_inputs_df['is_step'] = gul_inputs_df['is_step'].fillna(0).astype('int8')
                extra_fm_col.append('is_step')
        # If step policies listed, keep step trigger type and columns associated
        # with those step trigger types that are present
        if step_policies_present:
            # we happend the fm step policy term to policy layer
            step_policy_level_map = level_column_mapper[SUPPORTED_FM_LEVELS['policy all']['id']]
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
            fm_profile_cols = fm_profile_step_headers
        else:
            fm_profile_cols = fm_profile_headers

        fm_profile_file.write(f"{','.join(fm_profile_cols)}{os.linesep}")
        pass_through_profile = pd.DataFrame({col: [0] for col in fm_profile_cols})
        pass_through_profile['profile_id'] = 1
        pass_through_profile['calcrule_id'] = 100
        write_fm_profile_level(pass_through_profile, fm_profile_file, step_policies_present, chunksize)

        profile_id_offset = 1  # profile_id 1 is the passthrough policy 100
        cur_level_id = 0
        for term_df_source, levels, fm_peril_field, CondTag_merger in get_levels(locations_df, accounts_df):
            t0 = time.time()
            if CondTag_merger is not None:
                gul_inputs_df = gul_inputs_df.drop(columns='CondTag', errors='ignore').merge(CondTag_merger, how='left')

            for level, level_info in levels:
                level_id = level_info['id']
                is_policy_layer_level = level_id == SUPPORTED_FM_LEVELS['policy layer']['id']
                step_level = 'StepTriggerType' in level_column_mapper[level_id]  # only true is step policy are present
                level_terms, terms_maps, coverage_group_map, fm_group_tiv = get_level_term_info(
                    term_df_source, level_column_mapper, level_id, step_level, fm_peril_field, oed_schema)
                agg_key = [v['field'] for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]
                if not terms_maps:  # no terms we skip this level
                    if is_policy_layer_level:  # for policy layer we group all to make sure we can have a0 ALLOCATION_RULE
                        cur_level_id += 1
                        write_empty_policy_layer(gul_inputs_df, cur_level_id, agg_key, fm_policytc_file,
                                                 fm_programme_file, chunksize)
                        gul_inputs_df = reset_gul_inputs(gul_inputs_df)
                        logger.info(f"level {cur_level_id} {level_info} took {time.time() - t0}")
                        t0 = time.time()
                    continue

                # reset non layered agg_id
                layered_agg_id = gul_inputs_df[gul_inputs_df['layer_id'] > 1]["agg_id_prev"].unique()
                gul_inputs_df['layer_id'] = gul_inputs_df['layer_id'].where(gul_inputs_df['agg_id_prev'].isin(layered_agg_id), 0)

                # get all rows with terms in term_df_source and determine the correct FMTermGroupID
                level_df_list = []
                valid_term_default = {}
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

                    # only keep policy with non default values, remove duplicate
                    numeric_terms = [term for term in terms.keys() if is_numeric_dtype(group_df[term])]
                    term_filter = pd.Series(data=False, index=group_df.index)
                    for term in numeric_terms:
                        if pd.isna(oed_schema.get_default(term)):
                            term_filter |= ~group_df[term].isna()
                            valid_term_default[terms[term]] = 0
                        else:
                            term_filter |= (group_df[term] != oed_schema.get_default(term))
                            valid_term_default[terms[term]] = oed_schema.get_default(term)
                        term_filter |= (group_df[term] != oed_schema.get_default(term))
                    keep_df = group_df[term_filter][list(
                        set(agg_key).intersection(group_df.columns))].drop_duplicates()

                    group_df = group_df.merge(keep_df, how='inner')

                    # multiple ProfileElementName can have the same fm terms (ex: StepTriggerType 5), we take the max to have a unique one
                    for ProfileElementName, term in terms.items():
                        if term in group_df:
                            group_df[term] = group_df[[term, ProfileElementName]].max(axis=1)
                            group_df.drop(columns=ProfileElementName, inplace=True)
                        else:
                            group_df.rename(columns={ProfileElementName: term}, inplace=True)
                    level_df_list.append(group_df)
                level_df = pd.concat(level_df_list, copy=True, ignore_index=True)

                for term, default in valid_term_default.items():
                    level_df[term] = level_df[term].fillna(default)

                if do_disaggregation and 'risk_id' in agg_key:
                    level_df['NumberOfRisks'] = level_df['NumberOfBuildings'].mask(level_df['IsAggregate'] == 0, 1)
                    __split_fm_terms_by_risk(level_df)
                    level_df = level_df.drop(columns=['NumberOfBuildings', 'IsAggregate', 'NumberOfRisks'])

                level_df = level_df.drop_duplicates(subset=set(level_df.columns) - set(SOURCE_IDX.values()))
                agg_id_merge_col = agg_key + ['FMTermGroupID']
                agg_id_merge_col_extra = ['need_tiv']
                if step_level:
                    # merge with gul_inputs_df needs to be based on 'steptriggertype' and 'coverage_type_id'
                    coverage_type_id_df = pd.DataFrame(
                        [[StepTriggerType, coverage_type_id, FMTermGroupID]
                            for (StepTriggerType, coverage_type_id), FMTermGroupID in coverage_group_map.items()],
                        columns=['steptriggertype', 'coverage_type_id', 'FMTermGroupID'])
                    level_df = level_df.merge(coverage_type_id_df)
                    level_df["is_step"] = level_df['steptriggertype'] > 0
                    agg_id_merge_col_extra.append('coverage_type_id')
                else:
                    # map the coverage_type_id to the correct FMTermGroupID for this level. coverage_type_id without term and therefor FMTermGroupID are map to 0
                    gul_inputs_df['FMTermGroupID'] = (gul_inputs_df['coverage_type_id']
                                                      .map(coverage_group_map, na_action='ignore')
                                                      .astype('Int32'))

                # check that peril_id is part of the fm peril policy, if not we remove the terms
                if 'fm_peril' in level_df:
                    level_df = associate_items_peril_to_policy_peril(item_perils, level_df, 'fm_peril', oed_schema)
                    factorize_key = agg_key + ['FMTermGroupID', 'fm_peril']
                    agg_id_merge_col += ['fm_peril']
                    agg_id_merge_col_extra.append('peril_id')
                else:
                    factorize_key = agg_key + ['FMTermGroupID']

                if level_df.empty:  # No actual terms for this level
                    if is_policy_layer_level:  # for policy layer we group all to make sure we can have a0 ALLOCATION_RULE
                        cur_level_id += 1
                        write_empty_policy_layer(gul_inputs_df, cur_level_id, agg_key, fm_policytc_file,
                                                 fm_programme_file, chunksize)
                        gul_inputs_df = reset_gul_inputs(gul_inputs_df)
                        logger.info(f"level {cur_level_id} {level_info} took {time.time() - t0}")
                        t0 = time.time()
                    continue

                level_df = prepare_ded_and_limit(level_df)

                agg_id_merge_col = list(set(agg_id_merge_col).intersection(level_df.columns))
                gul_inputs_df = gul_inputs_df.merge(
                    level_df[agg_id_merge_col + agg_id_merge_col_extra].drop_duplicates(), how='left', validate='many_to_one')
                if is_policy_layer_level:  # we merge all on account at this level even if there is no policy
                    gul_inputs_df["FMTermGroupID"] = gul_inputs_df["FMTermGroupID"].fillna(-1).astype('i4')
                else:
                    gul_inputs_df["FMTermGroupID"] = gul_inputs_df["FMTermGroupID"].mask(
                        gul_inputs_df["need_tiv"].isna(), -gul_inputs_df["agg_id_prev"]).astype('i4')
                gul_inputs_df["agg_id"] = factorize_ndarray(gul_inputs_df.loc[:, factorize_key].values, col_idxs=range(len(factorize_key)))[0]

                # merge Tiv in level_df when needed
                # make sure correct tiv sum exist
                tiv_df_list = []
                for FMTermGroupID, coverage_type_ids in fm_group_tiv.items():
                    tiv_key = '_'.join(map(str, sorted(coverage_type_ids)))
                    if tiv_key not in gul_inputs_df:
                        gul_inputs_df[tiv_key] = gul_inputs_df[list(
                            set(gul_inputs_df.columns).intersection(map(str, sorted(coverage_type_ids))))].sum(axis=1)

                    tiv_df_list.append(gul_inputs_df[(gul_inputs_df["need_tiv"] == True) &
                                                     (gul_inputs_df["FMTermGroupID"] == FMTermGroupID)]
                                       .drop_duplicates(subset=['agg_id', 'loc_id', 'building_id'])
                                       .rename(columns={tiv_key: 'agg_tiv'})
                                       .groupby("agg_id", observed=True)
                                       .agg({**{col: 'first' for col in agg_id_merge_col}, **{'agg_tiv': 'sum'}})
                                       )
                tiv_df = pd.concat(tiv_df_list)

                gul_inputs_df = gul_inputs_df.merge(tiv_df['agg_tiv'], left_on='agg_id', right_index=True, how='left')
                gul_inputs_df['agg_tiv'] = gul_inputs_df['agg_tiv'].fillna(0)

                level_df = level_df.merge(tiv_df.drop_duplicates(), how='left').drop(columns=['need_tiv'])
                level_df['agg_tiv'] = level_df['agg_tiv'].fillna(0)
                # Apply rule to convert type 2 deductibles and limits to TIV shares
                if 'deductible' in level_df.columns and 'ded_type' in level_df.columns:
                    level_df['deductible'] = level_df['deductible'].fillna(0.)
                    level_df['ded_type'] = level_df['ded_type'].infer_objects(copy=False).fillna(0.).astype('i4')
                    level_df['deductible'] = np.where(
                        level_df['ded_type'] == DEDUCTIBLE_AND_LIMIT_TYPES['pctiv']['id'],
                        level_df['deductible'] * level_df['agg_tiv'],
                        level_df['deductible']
                    )
                    level_df.loc[level_df['ded_type'] == DEDUCTIBLE_AND_LIMIT_TYPES['pctiv']
                                 ['id'], 'ded_type'] = DEDUCTIBLE_AND_LIMIT_TYPES['flat']['id']

                    type_filter = level_df['ded_type'] == DEDUCTIBLE_AND_LIMIT_TYPES['bi']['id']
                    level_df.loc[type_filter, 'deductible'] = level_df.loc[type_filter, 'deductible'] * level_df.loc[type_filter, 'agg_tiv'] / 365.

                if 'limit' in level_df.columns and 'lim_type' in level_df.columns:
                    level_df['limit'] = level_df['limit'].fillna(0.)
                    level_df['lim_type'] = level_df['lim_type'].infer_objects(copy=False).fillna(0.).astype('i4')
                    level_df['limit'] = np.where(
                        level_df['lim_type'] == DEDUCTIBLE_AND_LIMIT_TYPES['pctiv']['id'],
                        level_df['limit'] * level_df['agg_tiv'],
                        level_df['limit']
                    )
                    level_df.loc[level_df['lim_type'] == DEDUCTIBLE_AND_LIMIT_TYPES['pctiv']
                                 ['id'], 'lim_type'] = DEDUCTIBLE_AND_LIMIT_TYPES['flat']['id']

                    type_filter = level_df['lim_type'] == DEDUCTIBLE_AND_LIMIT_TYPES['bi']['id']
                    level_df.loc[type_filter, 'limit'] = level_df.loc[type_filter, 'limit'] * level_df.loc[type_filter, 'agg_tiv'] / 365.

                if level_id == SUPPORTED_FM_LEVELS['policy layer']['id']:
                    level_df['calcrule_id'] = get_calc_rule_ids(level_df, calc_rule_type='policy_layer')
                    level_df['profile_id'] = get_profile_ids(level_df) + profile_id_offset
                    profile_id_offset = level_df['profile_id'].max() if not level_df.empty else profile_id_offset
                elif "is_step" in level_df.columns:
                    step_filter = level_df["is_step"]
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
                    level_df.loc[step_filter, 'steptriggertype'] = level_df[step_filter].apply(
                        lambda row: assign_sub_step_trigger_type(row), axis=1
                    )
                    final_step_filter = level_df['steptriggertype'] > 0
                    # step part
                    level_df.loc[final_step_filter, 'calcrule_id'] = get_calc_rule_ids(level_df[final_step_filter], calc_rule_type='step')
                    has_step = final_step_filter & (~level_df["step_id"].isna())
                    level_df.loc[has_step, 'profile_id'] = factorize_ndarray(level_df.loc[has_step, ['layer_id'] + factorize_key].values,
                                                                             col_idxs=range(len(['layer_id'] + factorize_key)))[0] + profile_id_offset
                    profile_id_offset = level_df.loc[has_step, 'profile_id'].max() if has_step.any() else profile_id_offset

                    # non step part
                    level_df.loc[~final_step_filter, 'calcrule_id'] = get_calc_rule_ids(level_df[~final_step_filter], calc_rule_type='base')
                    level_df.loc[~has_step, 'profile_id'] = (get_profile_ids(level_df[~has_step]) + profile_id_offset)
                    profile_id_offset = level_df.loc[~has_step, 'profile_id'].max() if (~has_step).any() else profile_id_offset
                else:
                    level_df['calcrule_id'] = get_calc_rule_ids(level_df, calc_rule_type='base')
                    level_df['profile_id'] = get_profile_ids(level_df) + profile_id_offset
                    profile_id_offset = level_df['profile_id'].max() if not level_df.empty else profile_id_offset

                write_fm_profile_level(level_df, fm_profile_file, step_policies_present, chunksize=chunksize)

                # =============================================================
                # we have prepared FMTermGroupID on gul or level df (depending on  step_level) now we can merge the terms for this level to gul
                merge_col = list(set(level_df.columns).intersection(set(gul_inputs_df)))
                level_df = level_df[merge_col + ['profile_id']].drop_duplicates()
                # gul_inputs that don't have layer yet must not be merge using layer_id
                non_layered_inputs_df = (
                    gul_inputs_df[gul_inputs_df['layer_id'] == 0]
                    .drop(columns=['layer_id'])
                    .rename(columns={'PolNumber': 'PolNumber_temp'})
                    .merge(level_df, how='left')
                )
                if 'PolNumber' in level_df.columns:
                    # gul_inputs['PolNumber'] is set to level_df['PolNumber'] if empty or if this is the first time layer appear
                    new_layered_gul_input_id = non_layered_inputs_df[non_layered_inputs_df['layer_id'] > 1]['gul_input_id'].unique()
                    non_layered_inputs_df['PolNumber'] = non_layered_inputs_df['PolNumber'].where(
                        (non_layered_inputs_df['gul_input_id'].isin(new_layered_gul_input_id)) | (non_layered_inputs_df['PolNumber_temp'].isna()),
                        non_layered_inputs_df['PolNumber_temp']
                    )
                    non_layered_inputs_df = non_layered_inputs_df.drop(columns=['PolNumber_temp'])
                else:
                    non_layered_inputs_df = non_layered_inputs_df.rename(columns={'PolNumber_temp': 'PolNumber'})

                layered_inputs_df = gul_inputs_df[gul_inputs_df['layer_id'] > 0].merge(level_df, how='left')
                # drop premature layering (no difference of policy between layers)
                if not is_policy_layer_level and level_id not in cross_layer_level:
                    non_layered_inputs_df['layered_id'] = (non_layered_inputs_df['layer_id']
                                                           .where(non_layered_inputs_df['agg_id'].isin(layered_inputs_df['agg_id']), 0))
                    non_layered_inputs_df = (non_layered_inputs_df
                                             .drop_duplicates(subset=['gul_input_id', 'agg_id', 'profile_id', 'layered_id'])
                                             .drop(columns=['layered_id']))

                gul_inputs_df = pd.concat(df for df in [layered_inputs_df, non_layered_inputs_df] if not df.empty)

                gul_inputs_df['layer_id'] = gul_inputs_df['layer_id'].fillna(1).astype('i4')
                gul_inputs_df.sort_values(by=['gul_input_id', 'layer_id'])
                gul_inputs_df["profile_id"] = gul_inputs_df["profile_id"].fillna(1).astype('i4')

                # check rows in prev df that are this level granularity (if prev_agg_id has multiple corresponding agg_id)
                need_root_start_df = gul_inputs_df.groupby("agg_id_prev", observed=True)["agg_id"].nunique()
                need_root_start_df = need_root_start_df[need_root_start_df > 1].index

                gul_inputs_df = gul_inputs_df.drop(columns=["root_start"], errors='ignore')  # clean up
                if need_root_start_df.shape[0]:
                    gul_inputs_df["root_start"] = gul_inputs_df["agg_id"].isin(
                        set(gul_inputs_df.loc[gul_inputs_df["agg_id_prev"].isin(need_root_start_df), 'agg_id'])
                    )

                cur_level_id += 1
                gul_inputs_df['level_id'] = cur_level_id
                # write fm_policytc
                if level_id in cross_layer_level:
                    fm_policytc_df = gul_inputs_df.loc[gul_inputs_df['layer_id'] == 1, fm_policytc_headers]
                else:
                    fm_policytc_df = gul_inputs_df.loc[:, fm_policytc_headers]
                fm_policytc_df.drop_duplicates().astype(fm_policytc_dtype).to_csv(fm_policytc_file, index=False,
                                                                                  header=False, chunksize=chunksize)

                # write programe
                fm_programe_df = gul_inputs_df[['level_id', 'agg_id']].rename(columns={'agg_id': 'to_agg_id'})
                if "root_start" in gul_inputs_df:
                    fm_programe_df['from_agg_id'] = gul_inputs_df['agg_id_prev'].where(~gul_inputs_df["root_start"], -gul_inputs_df['gul_input_id'])
                else:
                    fm_programe_df['from_agg_id'] = gul_inputs_df['agg_id_prev']
                (fm_programe_df[fm_programme_headers].drop_duplicates().astype(fm_programme_dtype)
                 .to_csv(fm_programme_file, index=False, header=False, chunksize=chunksize))

                # reset gul_inputs_df level columns
                gul_inputs_df = reset_gul_inputs(gul_inputs_df)
                logger.info(f"level {cur_level_id} {level_info} took {time.time()-t0}")
                t0 = time.time()

        gul_inputs_df = gul_inputs_df.sort_values(['gul_input_id', 'layer_id'], kind='stable')
        gul_inputs_df['output_id'] = gul_inputs_df.reset_index().index + 1

        default_pol_agg_key = [v['field'] for v in fm_aggregation_profile[SUPPORTED_FM_LEVELS['policy layer']['id']]['FMAggKey'].values()]
        no_polnumber_df = gul_inputs_df.loc[gul_inputs_df['PolNumber'].isna(), default_pol_agg_key + ['output_id']]
        if not no_polnumber_df.empty:  # empty polnumber, we use accounts_df to set PolNumber based on policy layer agg_key
            gul_inputs_df.loc[gul_inputs_df['PolNumber'].isna(), 'PolNumber'] = no_polnumber_df.merge(
                accounts_df[default_pol_agg_key + ['PolNumber']].drop_duplicates(subset=default_pol_agg_key)
            ).set_index(no_polnumber_df['output_id'] - 1)['PolNumber']

        # write fm_xref
        (gul_inputs_df
         .rename(columns={'gul_input_id': 'agg_id', 'output_id': 'output'})[fm_xref_headers]
         .astype(fm_xref_dtype)
         .to_csv(fm_xref_file, index=False, header=True, chunksize=chunksize))

        # merge acc_idx
        acc_idx_col = list(set(gul_inputs_df.columns).intersection(accounts_df.columns))
        gul_inputs_df_col = list(gul_inputs_df.columns)
        if 'CondTag' in acc_idx_col:
            gul_inputs_df = (
                gul_inputs_df
                .merge(accounts_df[acc_idx_col + ['acc_idx']].drop_duplicates(subset=acc_idx_col),
                       how='left', validate='many_to_one'))
            acc_idx_col.remove('CondTag')
            gul_inputs_df.loc[gul_inputs_df['acc_idx'].isna(), 'acc_idx'] = (gul_inputs_df.loc[gul_inputs_df['acc_idx'].isna(), gul_inputs_df_col].reset_index()
                                                                             .merge(accounts_df[acc_idx_col + ['acc_idx']].drop_duplicates(subset=acc_idx_col),
                                                                                    how='left', validate='many_to_one').set_index('index')['acc_idx'])

        else:
            gul_inputs_df = (
                gul_inputs_df
                .merge(accounts_df[acc_idx_col + ['acc_idx']].drop_duplicates(subset=acc_idx_col),
                       how='left', validate='many_to_one'))

        return gul_inputs_df


def reset_gul_inputs(gul_inputs_df):
    return (
        gul_inputs_df.drop(columns=["root_start", "agg_id_prev", "is_step", "FMTermGroupID",
                                    "profile_id", "level_id", 'fm_peril', 'need_tiv',
                                    'agg_tiv'], errors='ignore')
        .rename(columns={"agg_id": "agg_id_prev"}))


def write_empty_policy_layer(gul_inputs_df, cur_level_id, agg_key, fm_policytc_file, fm_programme_file, chunksize):
    gul_inputs_df["agg_id"] = factorize_ndarray(gul_inputs_df.loc[:, agg_key].values, col_idxs=range(len(agg_key)))[0]
    gul_inputs_df["profile_id"] = 1
    gul_inputs_df["level_id"] = cur_level_id
    fm_policytc_df = gul_inputs_df.loc[:, fm_policytc_headers]
    fm_policytc_df.drop_duplicates().astype(fm_policytc_dtype).to_csv(fm_policytc_file, index=False, header=False,
                                                                      chunksize=chunksize)
    fm_programe_df = gul_inputs_df[['level_id', 'agg_id']].rename(columns={'agg_id': 'to_agg_id'})
    fm_programe_df['from_agg_id'] = gul_inputs_df['agg_id_prev']
    fm_programe_df[fm_programme_headers].drop_duplicates().astype(fm_programme_dtype).to_csv(fm_programme_file, index=False,
                                                                                             header=False, chunksize=chunksize)


def write_fm_profile_level(level_df, fm_profile_file, step_policies_present, chunksize=100000):
    """
    Writes an FM profile file.

    :param level_df: level_df (fm terms) dataframe
    :type level_df: pandas.DataFrame

    :param fm_profile_file: open file to write to
    :type fm_profile_file: fileObj

    :param step_policies_present: flag to know which type of file to write
    :type step_policies_present: bool

    :param chunksize: chunksize
    :type chunksize: int

    :return: FM profile file path
    :rtype: str
    """
    level_df = level_df.astype({'calcrule_id': 'i4', 'profile_id': 'i4'})
    # Step policies exist
    if step_policies_present:
        fm_profile_df = level_df[list(set(level_df.columns).intersection(set(fm_profile_step_headers + ['steptriggertype'])))].copy()
        for col in fm_profile_step_headers + ['steptriggertype']:
            if col not in fm_profile_df.columns:
                fm_profile_df[col] = 0
        for non_step_name, step_name in profile_cols_map.items():
            if step_name not in fm_profile_df.columns:
                fm_profile_df[step_name] = 0
            fm_profile_df[step_name] = fm_profile_df[step_name].astype(object)
            if non_step_name in level_df.columns:
                fm_profile_df.loc[
                    ~(fm_profile_df['steptriggertype'] > 0), step_name
                ] = level_df.loc[
                    ~(fm_profile_df['steptriggertype'] > 0),
                    non_step_name
                ]
        fm_profile_df.fillna(0, inplace=True)
        fm_profile_df = fm_profile_df.drop_duplicates()

        # Ensure step_id is of int data type and set default value to 1
        fm_profile_df = fm_profile_df.astype(fm_profile_step_dtype)
        fm_profile_df.loc[fm_profile_df['step_id'] == 0, 'step_id'] = 1

        fm_profile_df = fm_profile_df[fm_profile_step_headers].sort_values(by=["profile_id", 'step_id']).drop_duplicates()
    # No step policies
    else:
        # make sure there is no step file in the folder
        fm_profile_df = level_df[list(set(level_df.columns).intersection(set(policytc_cols)))].copy()
        for col in policytc_cols[2:]:
            if col not in fm_profile_df.columns:
                fm_profile_df[col] = 0.

        fm_profile_df = (
            fm_profile_df
            .rename(columns=profile_cols_map)
            .drop_duplicates()
            .assign(share2=0.0, share3=0.0)
            .astype(fm_profile_dtype)[fm_profile_headers]
        )
    try:
        fm_profile_df.to_csv(
            fm_profile_file,
            index=False,
            header=False,
            chunksize=chunksize,
        )
    except (IOError, OSError) as e:
        raise OasisException("Exception raised in 'write_fm_profile_file'", e)
