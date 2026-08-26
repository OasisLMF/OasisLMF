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
                                          fm_xref_headers, fm_xref_dtype,
                                          DTYPE_IDX, calcrule_id, profile_id, layer_id)
from oasislmf.pytools.converters.csvtobin.utils.common import df_to_ndarray
from oasislmf.utils.calc_rules import get_calc_rules
from oasislmf.utils.coverages import SUPPORTED_COVERAGE_TYPES
from oasislmf.utils.data import assign_risk_ids, get_ids, structured_dtype_to_pandas, DEFAULT_LOC_FIELD_TYPES
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
fm_policytc_pd_dtype = structured_dtype_to_pandas(fm_policytc_dtype)
fm_profile_pd_dtype = structured_dtype_to_pandas(fm_profile_dtype)
fm_profile_step_pd_dtype = structured_dtype_to_pandas(fm_profile_step_dtype)
fm_programme_pd_dtype = structured_dtype_to_pandas(fm_programme_dtype)
fm_xref_pd_dtype = structured_dtype_to_pandas(fm_xref_dtype)


# Define a list of all supported OED coverage types in the exposure
supp_cov_type_ids = [v['id'] for v in SUPPORTED_COVERAGE_TYPES.values()]

# peril applied to a condition that does not name one; 'AA1' is the OED all-perils code
DEFAULT_COND_PERIL = 'AA1'

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

BITYPE_columns = {"BIWaitingPeriodType", "BIPOIType"}
default_value_overrides = {
    'BIPOI': 0.,
}

# Calcrule ID for passthrough (no financial terms applied)
PASSTHROUGH_CALCRULE_ID = 100


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
    """Merges IL inputs with the correct calc_rule table and returns calc rule IDs.

    Args:
        il_inputs_calc_rules_df (pandas.DataFrame): IL input items dataframe.
        calc_rule_type (str): Type of calc_rule to look for.

    Returns:
        pandas.Series: Series of calculation rule IDs.
    """
    il_inputs_calc_rules_df = il_inputs_calc_rules_df.copy()
    calc_rules_df, calc_rule_term_info = get_calc_rules(calc_rule_type)
    calc_rules_df = calc_rules_df.drop(columns=['desc', 'id_key'], errors='ignore')

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
        return PASSTHROUGH_CALCRULE_ID  # no term we return pass through
    if 0 in calcrule_ids.unique():
        _cols = list(set(['PortNumber', 'AccNumber', 'LocNumber'] + merge_col).intersection(il_inputs_calc_rules_df.columns))
        no_match_keys = il_inputs_calc_rules_df.loc[calcrule_ids == 0, _cols].drop_duplicates()
        err_msg = 'Calculation Rule mapping error, non-matching keys:\n{}'.format(no_match_keys)
        raise OasisException(err_msg)
    return calcrule_ids


def get_profile_ids(il_inputs_df):
    """Returns a Numpy array of policy TC IDs from a table of IL input items.

    Args:
        il_inputs_df (pandas.DataFrame): IL input items dataframe.

    Returns:
        pandas.Series: Series of profile IDs.
    """
    factor_col = list(set(il_inputs_df.columns).intersection(policytc_cols).difference({'profile_id', }))
    return il_inputs_df.groupby(factor_col, sort=False, observed=True, dropna=False).ngroup().astype('int32') + 1


def __split_fm_terms_by_risk(df):
    """Adjusts financial terms by the number of risks.

    For example, deductible is split into each individual building risk.

    Args:
        df (pandas.DataFrame): The dataframe for an FM level.
    """
    for term in risk_disaggregation_term.intersection(set(df.columns)):
        if f'{term[:3]}_type' in df.columns:
            code_filter = df[f'{term[:3]}_type'] == 0
            df.loc[code_filter, term] /= df.loc[code_filter, 'NumberOfRisks']
        else:
            df[term] /= df['NumberOfRisks']


def get_cond_info(locations_df, accounts_df):
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
        if 'CondPriority' in accounts_df.columns:
            if not is_numeric_dtype(accounts_df['CondPriority']):
                accounts_df['CondPriority'] = accounts_df['CondPriority'].astype('object')
            fill_empty(accounts_df, 'CondPriority', 1)
        else:
            accounts_df['CondPriority'] = 1
        if 'CondPeril' in accounts_df.columns:
            fill_empty(accounts_df, 'CondPeril', '')
        else:
            accounts_df['CondPeril'] = ''

        # --- per (acc_id, CondTag) attributes taken from the first account row (0/blank priority -> 1) ---
        tag_first = accounts_df.groupby(['acc_id', 'CondTag'], sort=False, observed=True).agg(
            CondPriority=('CondPriority', 'first'),
            CondPeril=('CondPeril', 'first'),
        ).reset_index()
        # fill_empty above already mapped every blank to 1, so anything left that will not convert is
        # not an empty priority but a malformed one, and is rejected rather than silently defaulted
        prio = pd.to_numeric(tag_first['CondPriority'], errors='coerce')
        if prio.isna().any():
            bad_priority_df = tag_first.loc[prio.isna(), ['acc_id', 'CondTag', 'CondPriority']]
            raise OasisException(f'Those CondPriority values in the account file are not numeric:\n{bad_priority_df}')
        tag_first['priority'] = prio.mask(prio == 0, 1)
        peril = tag_first['CondPeril'].astype('object').fillna('')
        tag_first['cond_peril'] = peril.mask(peril == '', DEFAULT_COND_PERIL)
        peril_map = tag_first.set_index(['acc_id', 'CondTag'])['cond_peril'].to_dict()

        # per (acc_id, layer_id) policy info + exclusion flag
        # has_excl is resolved before the groupby so 'max' can take the cython path; a lambda here
        # forces pandas' per-group pure-python aggregation and dominates the whole function
        layer_cols = accounts_df[['acc_id', 'layer_id', 'PolNumber', 'LayerNumber', 'acc_idx']].assign(
            has_excl=(accounts_df['CondClass'].eq(1).fillna(False).astype(bool)
                      if 'CondClass' in accounts_df.columns else False),
        )
        # the exclusion flag is an 'any' over the layer, but the policy columns are literally the
        # last row's, nulls included, as the row loop overwrote them per row. a groupby 'last' is
        # last *non-null*, which would put a value on a filler row where the loop emitted NaN, so
        # they are taken by drop_duplicates and joined onto the groups in first-appearance order.
        account_layers = layer_cols.groupby(
            ['acc_id', 'layer_id'], sort=False, observed=True)['has_excl'].max().reset_index()
        account_layers = account_layers.merge(
            layer_cols.drop_duplicates(['acc_id', 'layer_id'], keep='last').drop(columns='has_excl'),
            on=['acc_id', 'layer_id'], how='left')

        # --- resolve each location's cond priority, detect same-priority conflicts, rank within the location ---
        loc_conds_df = locations_df[['loc_id', 'acc_id', 'CondTag']].drop_duplicates()
        loc_conds_df = loc_conds_df.merge(tag_first[['acc_id', 'CondTag', 'priority']], on=['acc_id', 'CondTag'], how='left')
        # location cond tags absent from accounts (e.g. default '0') -> priority 1
        loc_conds_df['priority'] = loc_conds_df['priority'].fillna(1)

        # loc_conds_df is unique on (loc_id, acc_id, CondTag), so a repeated (loc_id, priority) is
        # by construction two different cond tags claiming one priority on one location. taking the
        # first repeat in row order reports the same pair, in the same order, as the row loop did:
        # the arriving tag first, then the one already held for that location.
        collisions = loc_conds_df.duplicated(['loc_id', 'priority'], keep='first').to_numpy()
        if collisions.any():
            arriving = int(np.argmax(collisions))
            loc_key = loc_conds_df['loc_id'].iloc[arriving]
            prio_val = loc_conds_df['priority'].iloc[arriving]
            held = int(np.argmax(((loc_conds_df['loc_id'] == loc_key) & (loc_conds_df['priority'] == prio_val)).to_numpy()))
            keys = [tuple(row) for row in loc_conds_df[['acc_id', 'CondTag']].iloc[[arriving, held]].to_numpy().tolist()]
            raise OasisException(f"{keys[0]} and {keys[1]} have same priority in {loc_key}")

        # cond_level_start = the longest chain of strictly-inner conditions ending at this cond_tag,
        # taken over every location it sits on. Ranking within a location and maxing the rank does
        # not compose across locations: a tag can rank first where it is innermost and second
        # elsewhere, which lands it on the same level as a lower-priority tag it shares a location
        # with -- a pair the FM has to nest and cannot express as one node. Priorities are distinct
        # within a location (checked above), so ordering a location by priority gives a strict
        # chain, and the longest path over those chains is the shallowest assignment that keeps
        # every such pair on separate levels. Sorted on a copy: loc_conds_df's row order is
        # load-bearing, it feeds all_cond_keys below and thence the level_conds and
        # extra_accounts orderings.
        chain = loc_conds_df.sort_values(['loc_id', 'priority'], kind='stable')
        if chain.empty:
            # an account file can carry cond tags with no locations referencing them; guarded
            # explicitly because MultiIndex.factorize raises on an empty frame in older pandas
            level_map = {}
        else:
            tag_code, tag_keys = pd.MultiIndex.from_frame(chain[['acc_id', 'CondTag']]).factorize()
            chain = chain.assign(tag_code=tag_code,
                                 inner_code=pd.Series(tag_code, index=chain.index)
                                 .groupby(chain['loc_id'], observed=True).shift())

            levels = np.ones(len(tag_keys), dtype='int64')
            # every edge runs from a lower priority to a higher one, so ascending priority is a
            # topological order: a tag's inner neighbours are final by the time it is reached
            edges = chain[chain['inner_code'].notna()]
            for _, step in edges.groupby('priority', sort=True, observed=True):
                np.maximum.at(levels, step['tag_code'].to_numpy(),
                              levels[step['inner_code'].to_numpy().astype('int64')] + 1)

            level_map = {tuple(key): int(level) for key, level in zip(tag_keys, levels)}

        # every cond_tag key to emit results for: those defined on accounts plus those referenced by
        # locations (the latter adds default-'0' tags, which are synthetic priority-1 conds)
        all_cond_keys = pd.concat([tag_first[['acc_id', 'CondTag']],
                                   loc_conds_df[['acc_id', 'CondTag']]]).drop_duplicates()

        # a location acc_id with no account rows has no layers to attach its conds to. the row loop
        # raised KeyError on account_layer_exclusion[acc_id]; the frame version would drop it on the
        # merge below and leave the key in level_conds, which get_levels turns into an all-null row
        missing_acc_df = all_cond_keys.loc[~all_cond_keys['acc_id'].isin(accounts_df['acc_id']), ['acc_id', 'CondTag']]
        if missing_acc_df.shape[0]:
            raise OasisException(f'Those acc_id are present in locations but missing in the account file:\n{missing_acc_df}')

        for r in all_cond_keys.itertuples():
            cond_key = (int(r.acc_id), r.CondTag)
            level_conds.setdefault(level_map.get((r.acc_id, r.CondTag), 1), set()).add(cond_key)

        # get_levels iterates level_conds and never reads the key, so insertion order *is* the FM
        # nesting order. The loop above inserts a level the first time a cond_tag carrying it is
        # seen, which is account-file order, not depth order -- emit innermost level first instead.
        level_conds = dict(sorted(level_conds.items()))

        # extra 'filler' account rows for every (cond_tag, layer) the cond_tag does not already cover.
        # the cond_tag x layer coverage check is an anti-join against the account rows; only the
        # surviving fillers are built row-by-row, since the exclusion ones carry extra keys
        cond_tag_layers_df = all_cond_keys.merge(account_layers, on='acc_id', how='inner').merge(
            accounts_df[['acc_id', 'CondTag', 'layer_id']].drop_duplicates(),
            on=['acc_id', 'CondTag', 'layer_id'], how='left', indicator=True)
        for r in cond_tag_layers_df.loc[cond_tag_layers_df['_merge'] == 'left_only'].itertuples():
            extra = {
                'acc_idx': r.acc_idx, 'acc_id': int(r.acc_id), 'PolNumber': r.PolNumber,
                'LayerNumber': r.LayerNumber, 'CondTag': r.CondTag, 'layer_id': r.layer_id,
            }
            if r.has_excl:
                extra.update({'CondNumber': 'FullFilter', 'CondDed6All': 1, 'CondDedType6All': 1})
            else:
                extra['CondNumber'] = ''
            extra['CondPeril'] = peril_map.get((r.acc_id, r.CondTag), DEFAULT_COND_PERIL)
            extra_accounts.append(extra)
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
    missing_share_default = []
    for ProfileElementName, term_info in level_column_mapper[level_id].items():
        if term_info.get("FMTermType") == "tiv":
            continue
        if ProfileElementName in default_value_overrides:
            default_value = default_value_overrides.get(ProfileElementName)
        else:
            default_value = oed_schema.get_default(ProfileElementName)

        if ProfileElementName not in term_df_source.columns:
            if default_value == 0 or default_value in BLANK_VALUES:
                continue
            else:
                non_zero_default[ProfileElementName] = [term_info['FMTermType'].lower(), default_value]
                continue
        elif ProfileElementName not in BITYPE_columns:
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
            FMTermGroupID = term_info.get('FMTermGroupID', 1)
            if step_level:
                term_key = (FMTermGroupID, 0)
            else:
                term_key = FMTermGroupID

            if not (non_default_val).any():
                if term_info['FMTermType'] == 'share':
                    missing_share_default.append(((term_key, term_info)))
                continue

            level_terms.add(term_info['FMTermType'].lower())
            coverage_type_ids = term_info.get("CoverageTypeID", supp_cov_type_ids)
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

    # Only include missing share if other terms exist ('limit', 'attachment')
    for term_key, term_info in missing_share_default:
        if terms_maps.get(term_key):
            level_terms.add(term_info['FMTermType'].lower())
            terms_maps[term_key][ProfileElementName] = term_info['FMTermType'].lower()

    if level_terms:
        for ProfileElementName, (term, default_value) in non_zero_default.items():
            term_df_source[ProfileElementName] = default_value
            level_terms.add(term)
            for terms_map in terms_maps.values():
                terms_map[ProfileElementName] = term

    return level_terms, terms_maps, coverage_group_map, fm_group_tiv


def associate_items_peril_to_policy_peril(item_perils, policy_df, fm_peril_col, oed_schema):
    """Maps item perils to policy perils and merges the mapping with policies.

    For each peril_id in item_perils, maps it to policy perils so that each line
    will have a peril_id that can be used directly as a key when merging with gul_input_df.

    Args:
        item_perils (pandas.DataFrame): All peril IDs from gul_input_df.
        policy_df (pandas.DataFrame): The dataframe with the policies.
        fm_peril_col (str): The name of the column that defines the policy peril filter.
        oed_schema: The schema object used to map perils and subperils.

    Returns:
        pandas.DataFrame: Policy dataframe merged with peril mapping.
    """
    fm_perils = policy_df[[fm_peril_col]].drop_duplicates()
    peril_map_df = pd.merge(fm_perils, item_perils, how='cross')
    peril_map_df = peril_map_df[oed_schema.peril_filtering(peril_map_df['peril_id'], peril_map_df[fm_peril_col])]
    return policy_df.merge(peril_map_df)


def _check_unique_merge_keys(level_df, agg_id_merge_col, agg_id_merge_col_extra,
                             level_info, gul_inputs_columns,
                             acc_id_map=None, max_groups=5):
    """Pre-validate that the right side of the upcoming
    `validate='many_to_one'` merge into gul_inputs_df is unique on the actual
    join keys.

    Pandas' MergeError ("Merge keys are not unique in right dataset") gives the
    user no clue which key/row is conflicting, so we surface that here. The
    join keys for the merge are the intersection of right-side columns
    (``agg_id_merge_col + agg_id_merge_col_extra``) and ``gul_inputs_df``'s
    columns -- pandas merges on the column-name intersection when ``on=`` is
    omitted -- which is what we replicate here.

    Args:
        level_df (pandas.DataFrame): Holds the current level's terms (right
            side of the merge).
        agg_id_merge_col (list[str]): Aggregation key columns kept on the
            right side. Always part of the join.
        agg_id_merge_col_extra (list[str]): Extra columns kept on the right
            side. Some of these (e.g. peril_id) are also present in
            gul_inputs_df and therefore become part of the join key; the rest
            are pure "brought across" columns whose disagreement on a
            non-unique join key is what triggers the MergeError.
        level_info (dict): Level metadata containing 'id' and 'desc'.
        gul_inputs_columns (Iterable[str]): Columns of the left-hand
            gul_inputs_df, used to determine which extras participate in the
            join.
        acc_id_map (pandas.DataFrame, optional): Mapping acc_id ->
            (PortNumber, AccNumber) used to translate the diagnostic back to
            user-facing identifiers. May be None (e.g. cyber/marine).
        max_groups (int, optional): Cap on conflicting groups shown in the
            error message. Default 5.

    Raises:
        OasisException: When two or more rows share the same join keys but
            disagree on a non-join right-side column. The message lists the
            conflicting groups (translated to PortNumber/AccNumber when
            possible) and the column values that disagree.

    Examples:
        Unique join keys -> silent pass:

        >>> import pandas as pd
        >>> level_df = pd.DataFrame({
        ...     'agg_id': [1, 2],
        ...     'peril_id': ['WTC', 'WEC'],
        ...     'deductible': [100.0, 200.0],
        ... })
        >>> _check_unique_merge_keys(
        ...     level_df, ['agg_id'], ['peril_id', 'deductible'],
        ...     {'id': 1, 'desc': 'site coverage'},
        ...     ['agg_id', 'peril_id'],
        ... )

        Same (agg_id, peril_id) with disagreeing deductible -> raises:

        >>> bad = pd.DataFrame({
        ...     'agg_id': [1, 1],
        ...     'peril_id': ['WTC', 'WTC'],
        ...     'deductible': [100.0, 200.0],
        ... })
        >>> _check_unique_merge_keys(
        ...     bad, ['agg_id'], ['peril_id', 'deductible'],
        ...     {'id': 1, 'desc': 'site coverage'},
        ...     ['agg_id', 'peril_id'],
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        OasisException: Inconsistent FM terms at level 1 ...
    """
    right_cols = list(agg_id_merge_col) + list(agg_id_merge_col_extra)
    gul_cols = set(gul_inputs_columns)
    # Replicate pandas' implicit `on=intersection(...)` behaviour.
    join_keys = [c for c in right_cols if c in gul_cols]
    if not join_keys:
        return

    right_df = level_df[right_cols].drop_duplicates()
    dup_mask = right_df.duplicated(subset=join_keys, keep=False)
    if not dup_mask.any():
        return

    dup_keys = right_df.loc[dup_mask, join_keys].drop_duplicates()
    n_groups = len(dup_keys)
    sample_keys = dup_keys.head(max_groups)

    # pull the original level_df rows for the offending keys, so the user sees
    # the actual conflicting term values (limits, deductibles, ded_type, ...)
    conflicts = level_df.merge(sample_keys, how='inner', on=join_keys)
    if acc_id_map is not None and 'acc_id' in conflicts.columns:
        conflicts = conflicts.merge(acc_id_map, how='left', on='acc_id')
        # surface PortNumber/AccNumber first for readability
        front = [c for c in ['PortNumber', 'AccNumber'] if c in conflicts.columns]
        conflicts = conflicts[front + [c for c in conflicts.columns if c not in front]]

    truncation_note = (f"; first {max_groups} of {n_groups} shown"
                       if n_groups > max_groups else "")
    raise OasisException(
        f"Inconsistent FM terms at level {level_info['id']} "
        f"({level_info['desc']}).\n"
        f"The merge join keys {join_keys} map to more than one distinct set "
        f"of financial terms in the source data, so the IL input merge "
        f"cannot be resolved unambiguously.\n"
        f"{n_groups} conflicting key(s) found{truncation_note}:\n"
        f"{conflicts.to_string(max_rows=60)}\n\n"
        f"Fix: each unique join key must declare the same financial terms "
        f"on every row."
    )


@oasis_log
def prepare_il_source_dataframes(gul_inputs_df, exposure_data):
    """Resolve the acc_id key across gul/location/account frames and fill location defaults.

    When a location file is present, assigns acc_id (PortNumber+AccNumber) and merges it onto the
    gul and location frames, trims the accounts to those referenced by locations, and fills the
    default location field types/values. When there is no location file (cyber, marine, ...),
    acc_id falls back to loc_id.

    Args:
        gul_inputs_df (pandas.DataFrame): GUL input items.
        exposure_data: OedExposure object (location / account dataframes, oed_schema).

    Returns:
        tuple: (gul_inputs_df, locations_df, accounts_df, acc_id_map). locations_df and acc_id_map
        are None in the no-location case.
    """
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

        # fill default types
        for field_type in DEFAULT_LOC_FIELD_TYPES:
            if field_type['field_col'] not in locations_df.columns:
                continue
            else:
                locations_df[field_type['field_col']] = locations_df[field_type['field_col']].fillna(
                    0.)  # set default to 0 to ignore term if empty
            if field_type['type_col'] not in locations_df.columns:
                locations_df[field_type['type_col']] = field_type['type_value']
            else:
                locations_df[field_type['type_col']] = locations_df[field_type['type_col']].fillna(
                    field_type['type_value'])

    else:  # no location, case for cyber, marine ...
        locations_df = None
        acc_id_map = None
        gul_inputs_df['acc_id'] = gul_inputs_df['loc_id']
        accounts_df = exposure_data.account.dataframe
        accounts_df['acc_id'] = accounts_df['loc_id']
        accounts_df = accounts_df.drop(columns=['PortNumber', 'AccNumber'])

    return gul_inputs_df, locations_df, accounts_df, acc_id_map


@oasis_log
def build_level_column_mapper(profile):
    """Build the per-level column -> term-info mapping used to extract terms from source files.

    The profile is keyed by fm term name; this re-keys it by the source column
    (ProfileElementName) and stores the term name under FMTermType, so the extraction logic can
    stay generic (and work with step policies) without hard-coding column names.

    Args:
        profile (dict): grouped FM profile by level and term group.

    Returns:
        dict: level_id -> {ProfileElementName: term_info-with-FMTermType}.
    """
    level_column_mapper = {}
    for level_id, level_profile in profile.items():
        column_map = {}
        level_column_mapper[level_id] = column_map

        # for fm we only use term_id 1
        for term_name, term_info in itertools.chain.from_iterable(profile.items() for profile in level_profile.values()):
            new_term_info = copy.deepcopy(term_info)
            new_term_info['FMTermType'] = term_name
            column_map[term_info['ProfileElementName']] = new_term_info
    return level_column_mapper


@oasis_log
def configure_step_policies(accounts_df, gul_inputs_df, level_column_mapper, oasis_files_prefixes):
    """Detect step policies and, if present, wire the step terms into the policy-all level.

    Determines whether any step policies are listed (non-null StepTriggerType with StepNumber > 0),
    marks the affected gul rows with is_step, augments the policy-all level_column_mapper with the
    step-policy term columns, and selects the step vs non-step fm_profile headers / bin name.
    ``level_column_mapper`` is mutated in place when step policies are present.

    Args:
        accounts_df (pandas.DataFrame): accounts data (StepTriggerType / StepNumber).
        gul_inputs_df (pandas.DataFrame): GUL input items (gets an is_step column if stepped).
        level_column_mapper (dict): per-level column mapper (mutated in place if stepped).
        oasis_files_prefixes (dict): output file name prefixes.

    Returns:
        tuple: (gul_inputs_df, extra_fm_col, step_policies_present, fm_profile_cols, profile_bin_name).
    """
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
        profile_bin_name = f"{oasis_files_prefixes['fm_profile']}_step"
    else:
        fm_profile_cols = fm_profile_headers
        profile_bin_name = f"{oasis_files_prefixes['fm_profile']}"

    return gul_inputs_df, extra_fm_col, step_policies_present, fm_profile_cols, profile_bin_name


@oasis_log
def build_level_terms_df(term_df_source, terms_maps, step_level, agg_key, extra_fm_col,
                         useful_cols, gul_inputs_columns, oed_schema, do_disaggregation):
    """Build the level_df of financial terms for one FM level from its source rows.

    For each term group, filters the source rows to those carrying non-default term values,
    deduplicates, renames each ProfileElementName to its FM term (taking the max where several
    columns map to the same term), and concatenates the groups. Missing term values are filled
    with their defaults, aggregate exposure is optionally disaggregated per risk, and the result
    is deduplicated.

    Args:
        term_df_source (pandas.DataFrame): source rows for this level (locations_df or accounts_df).
        terms_maps (dict): group_key -> {ProfileElementName: term} mapping for this level.
        step_level (bool): whether this level carries step-policy terms.
        agg_key (list): aggregation key columns for this level.
        extra_fm_col (list): extra FM columns to retain (layer_id, PolNumber, ...).
        useful_cols (list): the union of columns useful downstream.
        gul_inputs_columns (pandas.Index): columns already present on gul_inputs_df.
        oed_schema: OED schema (for per-term default lookups).
        do_disaggregation (bool): if True, split aggregate terms by NumberOfRisks.

    Returns:
        pandas.DataFrame: the level's terms (level_df).
    """
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
                                  .union(set(useful_cols).difference(set(gul_inputs_columns)))
                                  .intersection(group_df.columns))]
                    .assign(FMTermGroupID=FMTermGroupID))

        # only keep policy with non default values, remove duplicate
        numeric_terms = [term for term in terms.keys() if is_numeric_dtype(group_df[term])]
        term_filter = pd.Series(data=False, index=group_df.index)
        for term in numeric_terms:
            if pd.isna(oed_schema.get_default(term)):
                term_filter |= ~group_df[term].isna()
                valid_term_default[terms[term]] = 0
            elif term in default_value_overrides:
                term_filter |= (group_df[term] != default_value_overrides.get(term))
                valid_term_default[terms[term]] = default_value_overrides.get(term)
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
    return level_df


@oasis_log
def apply_percent_tiv_terms(gul_inputs_df, level_df, fm_group_tiv, agg_id_merge_col):
    """Compute aggregate TIV per agg_id and convert percentage-of-TIV terms to absolute values.

    Some deductibles/limits are specified as a percentage of TIV (type 2) or as business
    interruption (BI) terms. This computes the aggregate TIV for each agg_id and rewrites those
    terms into absolute (flat) deductibles/limits on ``level_df``.

    Args:
        gul_inputs_df (pandas.DataFrame): current working IL inputs (gets an agg_tiv column).
        level_df (pandas.DataFrame): current level's terms (ded/lim converted in place).
        fm_group_tiv (dict): FMTermGroupID -> coverage_type_ids contributing to TIV.
        agg_id_merge_col (list): aggregation merge columns retained during TIV aggregation.

    Returns:
        tuple: (gul_inputs_df, level_df) with agg_tiv populated and pctiv/BI terms made absolute.
    """
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

    return gul_inputs_df, level_df


@oasis_log
def merge_level_terms_into_gul(gul_inputs_df, level_df, is_policy_layer_level, level_id):
    """Merge a level's terms (with profile_id) into gul_inputs_df, handling layered rows.

    Non-layered rows (layer_id == 0) must not merge on layer_id so they can pick up layer info from
    level_df; layered rows (layer_id > 0) merge on layer_id to preserve their assignment. After the
    merge, "premature layering" (rows differing only by layer_id but with identical profile_id) is
    removed. PolNumber is propagated from level_df where appropriate. A fast path is used when no
    rows are layered yet.

    Args:
        gul_inputs_df (pandas.DataFrame): current working IL inputs.
        level_df (pandas.DataFrame): current level's terms, restricted to merge cols + profile_id.
        is_policy_layer_level (bool): whether this is the policy-layer level.
        level_id (int): the FM level id.

    Returns:
        pandas.DataFrame: gul_inputs_df with profile_id/layer_id merged in.
    """
    merge_col = list(set(level_df.columns).intersection(set(gul_inputs_df)))
    level_df = level_df[merge_col + ['profile_id']].drop_duplicates()

    # Check if there are any layered inputs (rows that already have a layer assigned)
    layered_mask = gul_inputs_df['layer_id'] > 0
    has_layered = layered_mask.any()

    if has_layered:
        # Original path: separate merges for layered and non-layered
        # gul_inputs that don't have layer yet must not be merge using layer_id
        non_layered_inputs_df = (
            gul_inputs_df[~layered_mask]
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

        layered_inputs_df = gul_inputs_df[layered_mask].merge(level_df, how='left')

        # PREMATURE LAYERING REMOVAL:
        # After merge, non-layered rows may have expanded into multiple rows
        # (one per layer from level_df). If these layers have identical terms
        # (same profile_id), we should keep only one to avoid unnecessary duplication.
        # The 'layered_id' trick: set to layer_id only if agg_id exists in layered
        # inputs (meaning there's a reason to keep layers separate), else set to 0.
        # Then dedup on (gul_input_id, agg_id, profile_id, layered_id).
        if not is_policy_layer_level and level_id not in cross_layer_level:
            # Keep rows per layer when the (gul_input_id, agg_id) group spans more than
            # one profile_id: those layers carry genuinely different terms, so collapsing
            # them would drop layers and make the survivor depend on account row order (issue #2040).
            layer_specific = (non_layered_inputs_df
                              .groupby(['gul_input_id', 'agg_id'])['profile_id'].transform('nunique') > 1)
            non_layered_inputs_df['layered_id'] = (non_layered_inputs_df['layer_id']
                                                   .where(layer_specific
                                                          | non_layered_inputs_df['agg_id'].isin(layered_inputs_df['agg_id']), 0))
            non_layered_inputs_df = (non_layered_inputs_df
                                     .drop_duplicates(subset=['gul_input_id', 'agg_id', 'profile_id', 'layered_id'])
                                     .drop(columns=['layered_id']))

        gul_inputs_df = pd.concat([layered_inputs_df, non_layered_inputs_df], ignore_index=True)
    else:
        # Optimized path: all rows have layer_id == 0
        # No need for separate layered/non-layered split and concat
        gul_inputs_df = (gul_inputs_df
                         .drop(columns=['layer_id'])
                         .rename(columns={'PolNumber': 'PolNumber_temp'})
                         .merge(level_df, how='left'))

        # Handle PolNumber (same as original)
        if 'PolNumber' in level_df.columns:
            new_layered_gul_input_id = gul_inputs_df[gul_inputs_df['layer_id'] > 1]['gul_input_id'].unique()
            gul_inputs_df['PolNumber'] = gul_inputs_df['PolNumber'].where(
                (gul_inputs_df['gul_input_id'].isin(new_layered_gul_input_id)) | (gul_inputs_df['PolNumber_temp'].isna()),
                gul_inputs_df['PolNumber_temp']
            )
            gul_inputs_df = gul_inputs_df.drop(columns=['PolNumber_temp'])
        else:
            gul_inputs_df = gul_inputs_df.rename(columns={'PolNumber_temp': 'PolNumber'})

        # Drop premature layering (no difference of policy between layers).
        # Keep one row per layer when the (gul_input_id, agg_id) group spans more than one
        # profile_id (see the layered path above): otherwise layers sharing a profile_id
        # collapse to a single row whose identity depends on account row order (issue #2040).
        if not is_policy_layer_level and level_id not in cross_layer_level:
            layer_specific = (gul_inputs_df
                              .groupby(['gul_input_id', 'agg_id'])['profile_id'].transform('nunique') > 1)
            gul_inputs_df['layered_id'] = gul_inputs_df['layer_id'].where(layer_specific, 0)
            gul_inputs_df = (gul_inputs_df
                             .drop_duplicates(subset=['gul_input_id', 'agg_id', 'profile_id', 'layered_id'])
                             .drop(columns=['layered_id']))

    gul_inputs_df['layer_id'] = gul_inputs_df['layer_id'].fillna(1).astype(layer_id[DTYPE_IDX])
    gul_inputs_df["profile_id"] = gul_inputs_df["profile_id"].fillna(1).astype(profile_id[DTYPE_IDX])

    return gul_inputs_df


@oasis_log
def write_level_policytc_and_programme(gul_inputs_df, level_id, fm_policytc_bin, fm_policytc_csv,
                                       fm_programme_bin, fm_programme_csv, chunksize):
    """Write the fm_policytc and fm_programme records for the current level.

    fm_policytc maps (level_id, agg_id, layer_id) -> profile_id (only layer_id==1 for cross-layer
    levels); fm_programme defines the hierarchical from_agg_id -> to_agg_id links (using a negative
    gul_input_id as the starting point where root_start is set).

    Args:
        gul_inputs_df (pandas.DataFrame): current working IL inputs for this level.
        level_id (int): the FM level id.
        fm_policytc_bin (file): open binary handle for fm_policytc.
        fm_policytc_csv (file or None): open csv handle for fm_policytc, or None.
        fm_programme_bin (file): open binary handle for fm_programme.
        fm_programme_csv (file or None): open csv handle for fm_programme, or None.
        chunksize (int): rows per chunk when writing CSVs.
    """
    # fm_policytc: Maps (level_id, agg_id, layer_id) -> profile_id
    # For cross_layer levels, only write layer_id=1 (terms apply across all layers)
    if level_id in cross_layer_level:
        fm_policytc_df = gul_inputs_df.loc[gul_inputs_df['layer_id'] == 1, fm_policytc_headers]
    else:
        fm_policytc_df = gul_inputs_df.loc[:, fm_policytc_headers]
    fm_policytc_dedup = fm_policytc_df.drop_duplicates()
    df_to_ndarray(fm_policytc_dedup, fm_policytc_dtype).tofile(fm_policytc_bin)
    if fm_policytc_csv is not None:
        fm_policytc_dedup.astype(fm_policytc_pd_dtype).to_csv(fm_policytc_csv, index=False,
                                                              header=False, chunksize=chunksize)

    # fm_programme: Defines hierarchical links between levels
    # from_agg_id (previous level) -> to_agg_id (current level)
    # When root_start is True, use negative gul_input_id to indicate starting point
    fm_programe_df = gul_inputs_df[['level_id', 'agg_id']].rename(columns={'agg_id': 'to_agg_id'})
    if "root_start" in gul_inputs_df:
        fm_programe_df['from_agg_id'] = gul_inputs_df['agg_id_prev'].where(~gul_inputs_df["root_start"], -gul_inputs_df['gul_input_id'])
    else:
        fm_programe_df['from_agg_id'] = gul_inputs_df['agg_id_prev']
    fm_programe_dedup = fm_programe_df[fm_programme_headers].drop_duplicates()
    df_to_ndarray(fm_programe_dedup, fm_programme_dtype).tofile(fm_programme_bin)
    if fm_programme_csv is not None:
        fm_programe_dedup.astype(fm_programme_pd_dtype).to_csv(fm_programme_csv, index=False,
                                                               header=False, chunksize=chunksize)


@oasis_log
def assign_level_calcrule_and_profile_ids(level_df, level_id, factorize_key, profile_id_offset):
    """Assign calcrule_id and profile_id to a level's terms.

    Handles the three cases separately: the policy-layer level, step policies (splitting the
    StepTriggerType into its per-coverage sub-types and assigning step vs non-step profiles), and
    the base case. ``profile_id_offset`` is advanced so profile ids stay globally unique across
    levels.

    Args:
        level_df (pandas.DataFrame): the current level's terms.
        level_id (int): the FM level id.
        factorize_key (list): grouping columns used to assign step profile ids.
        profile_id_offset (int): running maximum profile_id assigned so far.

    Returns:
        tuple: (level_df, profile_id_offset) with calcrule_id/profile_id columns populated.
    """
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

        level_df.loc[has_step, 'profile_id'] = level_df.loc[has_step, ['layer_id'] + factorize_key].groupby(['layer_id'] + factorize_key, sort=False, observed=True, dropna=False).ngroup().astype(
            'int32') + profile_id_offset + 1
        profile_id_offset = level_df.loc[has_step, 'profile_id'].max() if has_step.any() else profile_id_offset

        # non step part
        level_df.loc[~final_step_filter, 'calcrule_id'] = get_calc_rule_ids(level_df[~final_step_filter], calc_rule_type='base')
        level_df.loc[~has_step, 'profile_id'] = (get_profile_ids(level_df[~has_step]) + profile_id_offset)
        profile_id_offset = level_df.loc[~has_step, 'profile_id'].max() if (~has_step).any() else profile_id_offset
    else:
        level_df['calcrule_id'] = get_calc_rule_ids(level_df, calc_rule_type='base')
        level_df['profile_id'] = get_profile_ids(level_df) + profile_id_offset
        profile_id_offset = level_df['profile_id'].max() if not level_df.empty else profile_id_offset

    return level_df, profile_id_offset


@oasis_log
def finalize_il_inputs(gul_inputs_df, fm_aggregation_profile, accounts_df, fm_xref_bin, fm_xref_csv, chunksize):
    """Assign output ids, backfill PolNumber, write fm_xref, and merge acc_idx back on.

    Sorts by (gul_input_id, layer_id) for deterministic output ordering, assigns a sequential
    output_id, fills any missing PolNumber from accounts using the policy-layer aggregation keys,
    writes the fm_xref file, and merges the account index (acc_idx) back onto the result.

    Args:
        gul_inputs_df (pandas.DataFrame): the accumulated IL inputs.
        fm_aggregation_profile (dict): FM aggregation profile (for the policy-layer agg keys).
        accounts_df (pandas.DataFrame): accounts data (PolNumber / acc_idx source).
        fm_xref_bin (file): open binary handle for fm_xref.
        fm_xref_csv (file or None): open csv handle for fm_xref, or None.
        chunksize (int): rows per chunk when writing CSVs.

    Returns:
        pandas.DataFrame: the finalized IL inputs.
    """
    # Sort by gul_input_id and layer_id for consistent output ordering
    gul_inputs_df = gul_inputs_df.sort_values(['gul_input_id', 'layer_id'], kind='stable')
    gul_inputs_df['output_id'] = gul_inputs_df.reset_index().index + 1

    # Fill missing PolNumber values from accounts_df using policy layer agg keys
    default_pol_agg_key = [v['field'] for v in fm_aggregation_profile[SUPPORTED_FM_LEVELS['policy layer']['id']]['FMAggKey'].values()]
    no_polnumber_df = gul_inputs_df.loc[gul_inputs_df['PolNumber'].isna(), default_pol_agg_key + ['output_id']]
    if not no_polnumber_df.empty:  # empty polnumber, we use accounts_df to set PolNumber based on policy layer agg_key
        polnumber_lookup = no_polnumber_df.reset_index().merge(
            accounts_df[default_pol_agg_key + ['PolNumber']].drop_duplicates(subset=default_pol_agg_key),
            how='left',
            on=default_pol_agg_key,
        ).set_index('index')['PolNumber']
        gul_inputs_df.loc[polnumber_lookup.index, 'PolNumber'] = polnumber_lookup

    # fm_xref: Maps GUL item IDs (agg_id) to FM output IDs
    # This is the final cross-reference between GUL and IL outputs
    fm_xref_df = gul_inputs_df.rename(columns={'gul_input_id': 'agg_id', 'output_id': 'output'})[fm_xref_headers]
    df_to_ndarray(fm_xref_df, fm_xref_dtype).tofile(fm_xref_bin)
    if fm_xref_csv is not None:
        fm_xref_df.astype(fm_xref_pd_dtype).to_csv(fm_xref_csv, index=False, header=True, chunksize=chunksize)

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
        intermediary_csv=False,
):
    """Generates IL (Insured Loss) input items by applying financial terms to GUL items.

    This function builds the Financial Module (FM) structure by processing insurance
    policy terms across multiple hierarchical levels. It takes GUL (Ground-Up Loss)
    items and enriches them with policy terms (deductibles, limits, shares) from
    location and account data.

    Overview of Processing Flow:
    ===========================
    1. SETUP: Prepare data structures, merge account IDs, handle step policies
    2. LEVEL PROCESSING: For each FM level (site coverage -> policy layer):
       a. Extract terms from location/account data for this level
       b. Compute aggregation IDs (agg_id) based on aggregation keys
       c. Calculate TIV (Total Insured Value) for percentage-based terms
       d. Assign calculation rules and profile IDs
       e. Merge terms back to gul_inputs_df (handling layered/non-layered separately)
       f. Write level data to FM output files
    3. FINALIZE: Assign output IDs, fill missing PolNumbers, write fm_xref

    Key Data Structures:
    ===================
    - gul_inputs_df: Main working DataFrame, starts with GUL items and accumulates
      FM columns (agg_id, layer_id, profile_id, level_id) as levels are processed.
      Each row represents an item that will flow through the FM calculation.

    - level_df: Terms extracted from location/account data for the current level.
      Contains financial terms (deductible, limit, share) and is merged into
      gul_inputs_df after profile IDs are assigned.

    - agg_id: Aggregation ID - groups items that share the same financial terms
      at each level. Computed using groupby().ngroup() on aggregation keys.

    - layer_id: Distinguishes policy layers (1 = base, 2+ = excess layers).
      Rows start with layer_id=0 (unassigned) and get layer_id from level_df merge.

    Layered vs Non-Layered Processing:
    ==================================
    When merging level_df terms into gul_inputs_df, rows are handled differently:

    - Non-layered rows (layer_id == 0): Have not yet been assigned to a layer.
      Must NOT merge on layer_id column, as they need to pick up layer info from
      level_df. After merge, they may expand into multiple rows (one per layer).

    - Layered rows (layer_id > 0): Already assigned to a specific layer from a
      previous level. Must merge on layer_id to preserve their layer assignment.

    After merging, "premature layering" is removed: if rows differ only by layer_id
    but have identical financial terms (same profile_id), duplicates are dropped.
    This prevents unnecessary row multiplication when layers don't differ.

    Output Files Written:
    ====================
    - fm_policytc.csv: Maps (level_id, agg_id, layer_id) -> profile_id
    - fm_profile.csv: Profile definitions with calculation rules and term values
    - fm_programme.csv: Hierarchical structure linking levels (from_agg_id -> to_agg_id)
    - fm_xref.csv: Maps GUL item IDs to FM output IDs

    Args:
        gul_inputs_df (pandas.DataFrame): GUL input items with columns including
            item_id, loc_id, coverage_type_id, tiv, peril_id, building_id, etc.
        exposure_data: OedExposure object containing location and account DataFrames
            with policy terms (deductibles, limits, shares, layers).
        target_dir (str): Directory path where FM output files will be written.
        logger: Logger object for progress messages.
        exposure_profile (dict, optional): Maps OED fields to FM term types for locations.
        accounts_profile (dict, optional): Maps OED fields to FM term types for accounts.
        fm_aggregation_profile (dict, optional): Defines aggregation keys for each FM level.
        do_disaggregation (bool, optional): If True, split aggregate exposure terms
            by NumberOfRisks. Default True.
        oasis_files_prefixes (dict, optional): File name prefixes for output files.
        chunksize (int, optional): Rows per chunk when writing CSVs. Default 200,000.
        intermediary_csv (bool, optional): If True, also write CSV files alongside
            binary for debugging. Defaults to False.

    Returns:
        pandas.DataFrame: IL inputs with columns including output_id, layer_id,
            gul_input_id, agg_id, PolNumber, acc_idx, and summary columns.
    """
    # =========================================================================
    # SETUP PHASE: Open output files and prepare input data
    # =========================================================================
    target_dir = as_path(target_dir, 'Target IL input files directory', is_dir=True, preexists=False)
    il_input_files = {}
    with contextlib.ExitStack() as stack:
        gul_inputs_df, locations_df, accounts_df, acc_id_map = prepare_il_source_dataframes(gul_inputs_df, exposure_data)

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
        gul_inputs_df = assign_risk_ids(gul_inputs_df)

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
        level_column_mapper = build_level_column_mapper(profile)

        gul_inputs_df = gul_inputs_df.drop(columns=fm_term_ids, errors='ignore').reset_index(drop=True)
        gul_inputs_df['agg_id_prev'] = gul_inputs_df['gul_input_id']
        gul_inputs_df['layer_id'] = 1
        gul_inputs_df['PolNumber'] = pd.NA
        item_perils = gul_inputs_df[['peril_id']].drop_duplicates()

        gul_inputs_df, extra_fm_col, step_policies_present, fm_profile_cols, profile_bin_name = configure_step_policies(
            accounts_df, gul_inputs_df, level_column_mapper, oasis_files_prefixes)

        # Binary file handles (always written)
        il_input_files['fm_policytc'] = os.path.join(target_dir, f"{oasis_files_prefixes['fm_policytc']}.bin")
        fm_policytc_bin = stack.enter_context(open(il_input_files['fm_policytc'], 'wb'))

        il_input_files['fm_programme'] = os.path.join(target_dir, f"{oasis_files_prefixes['fm_programme']}.bin")
        fm_programme_bin = stack.enter_context(open(il_input_files['fm_programme'], 'wb'))

        il_input_files['fm_xref'] = os.path.join(target_dir, f"{oasis_files_prefixes['fm_xref']}.bin")
        fm_xref_bin = stack.enter_context(open(il_input_files['fm_xref'], 'wb'))

        il_input_files['fm_profile'] = os.path.join(target_dir, f"{profile_bin_name}.bin")
        fm_profile_bin = stack.enter_context(open(il_input_files['fm_profile'], 'wb'))

        # CSV file handles (only when intermediary_csv=True)
        if intermediary_csv:
            fm_policytc_csv = stack.enter_context(open(os.path.join(target_dir, f"{oasis_files_prefixes['fm_policytc']}.csv"), 'w'))
            fm_policytc_csv.write(f"{','.join(fm_policytc_headers)}{os.linesep}")
            fm_profile_csv = stack.enter_context(open(os.path.join(target_dir, f"{oasis_files_prefixes['fm_profile']}.csv"), 'w'))
            fm_profile_csv.write(f"{','.join(fm_profile_cols)}{os.linesep}")
            fm_programme_csv = stack.enter_context(open(os.path.join(target_dir, f"{oasis_files_prefixes['fm_programme']}.csv"), 'w'))
            fm_programme_csv.write(f"{','.join(fm_programme_headers)}{os.linesep}")
            fm_xref_csv = stack.enter_context(open(os.path.join(target_dir, f"{oasis_files_prefixes['fm_xref']}.csv"), 'w'))
        else:
            fm_policytc_csv = None
            fm_profile_csv = None
            fm_programme_csv = None
            fm_xref_csv = None

        pass_through_profile = pd.DataFrame({col: [0] for col in fm_profile_cols})
        pass_through_profile['profile_id'] = 1
        pass_through_profile['calcrule_id'] = PASSTHROUGH_CALCRULE_ID
        write_fm_profile_level(pass_through_profile, fm_profile_csv, fm_profile_bin, step_policies_present, chunksize)

        profile_id_offset = 1  # profile_id 1 is the passthrough policy (calcrule 100)
        cur_level_id = 0

        # =========================================================================
        # LEVEL PROCESSING LOOP: Process each FM level from bottom to top
        # =========================================================================
        # get_levels() yields (term_df_source, levels, fm_peril_field, CondTag_merger):
        #   - term_df_source: DataFrame containing terms (locations_df or accounts_df)
        #   - levels: Iterator of (level_name, level_info) for levels using this source
        #   - fm_peril_field: Field name for FM peril filtering (if applicable)
        #   - CondTag_merger: DataFrame for conditional tag merging (accounts only)
        for term_df_source, levels, fm_peril_field, CondTag_merger in get_levels(locations_df, accounts_df):
            t0 = time.time()
            if CondTag_merger is not None:
                gul_inputs_df = gul_inputs_df.drop(columns='CondTag', errors='ignore').merge(CondTag_merger, how='left')

            for level, level_info in levels:
                level_id = level_info['id']
                is_policy_layer_level = level_id == SUPPORTED_FM_LEVELS['policy layer']['id']
                step_level = 'StepTriggerType' in level_column_mapper[level_id]  # only true is step policy are present
                fm_peril_field = fm_peril_field if fm_peril_field in term_df_source.columns else None
                level_terms, terms_maps, coverage_group_map, fm_group_tiv = get_level_term_info(
                    term_df_source, level_column_mapper, level_id, step_level, fm_peril_field, oed_schema)
                agg_key = [v['field'] for v in fm_aggregation_profile[level_id]['FMAggKey'].values()]
                if not terms_maps:  # no terms we skip this level
                    if is_policy_layer_level:  # for policy layer we group all to make sure we can have a0 ALLOCATION_RULE
                        cur_level_id += 1
                        write_empty_policy_layer(gul_inputs_df, cur_level_id, agg_key, fm_policytc_csv, fm_policytc_bin,
                                                 fm_programme_csv, fm_programme_bin, chunksize)
                        gul_inputs_df = reset_gul_inputs(gul_inputs_df)
                        logger.info(f"level {cur_level_id} {level_info} took {time.time() - t0}")
                        t0 = time.time()
                    continue

                # get all rows with terms in term_df_source and determine the correct FMTermGroupID
                level_df = build_level_terms_df(term_df_source, terms_maps, step_level, agg_key, extra_fm_col,
                                                useful_cols, gul_inputs_df.columns, oed_schema, do_disaggregation)
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
                        write_empty_policy_layer(gul_inputs_df, cur_level_id, agg_key, fm_policytc_csv, fm_policytc_bin,
                                                 fm_programme_csv, fm_programme_bin, chunksize)
                        gul_inputs_df = reset_gul_inputs(gul_inputs_df)
                        logger.info(f"level {cur_level_id} {level_info} took {time.time() - t0}")
                        t0 = time.time()
                    continue

                # reset non layered agg_id when level has term to merge
                layered_agg_id = gul_inputs_df[gul_inputs_df['layer_id'] > 1]["agg_id_prev"].unique()
                gul_inputs_df['layer_id'] = gul_inputs_df['layer_id'].where(gul_inputs_df['agg_id_prev'].isin(layered_agg_id), 0)

                level_df = prepare_ded_and_limit(level_df)

                agg_id_merge_col = list(set(agg_id_merge_col).intersection(level_df.columns))

                join_keys = [c for c in agg_id_merge_col + agg_id_merge_col_extra
                             if c in gul_inputs_df.columns]
                if 'need_tiv' in level_df.columns and join_keys:  # make sure only 1 need_tiv value per join_keys
                    level_df['need_tiv'] = (level_df
                                            .groupby(join_keys, observed=True, dropna=False)['need_tiv']
                                            .transform('any'))

                _check_unique_merge_keys(
                    level_df, agg_id_merge_col, agg_id_merge_col_extra, level_info,
                    gul_inputs_columns=gul_inputs_df.columns,
                    acc_id_map=acc_id_map if locations_df is not None else None,
                )
                gul_inputs_df = gul_inputs_df.merge(
                    level_df[agg_id_merge_col + agg_id_merge_col_extra].drop_duplicates(), how='left', validate='many_to_one')
                if is_policy_layer_level:  # we merge all on account at this level even if there is no policy
                    gul_inputs_df["FMTermGroupID"] = gul_inputs_df["FMTermGroupID"].fillna(-1).astype('i4')
                else:
                    gul_inputs_df["FMTermGroupID"] = gul_inputs_df["FMTermGroupID"].mask(
                        gul_inputs_df["need_tiv"].isna(), -gul_inputs_df["agg_id_prev"]).astype('i4')

                # Assign agg_id: items with identical factorize_key values get the same agg_id
                # This groups items that will share the same financial terms at this level
                gul_inputs_df["agg_id"] = gul_inputs_df.groupby(factorize_key, sort=False, observed=True, dropna=False).ngroup().astype('int32') + 1

                # =====================================================================
                # TIV PROCESSING: Calculate aggregate TIV for percentage-based terms
                # =====================================================================
                # Some terms (ded_type=2, lim_type=2) are specified as % of TIV
                # We need the aggregate TIV for each agg_id to convert to absolute values
                gul_inputs_df, level_df = apply_percent_tiv_terms(gul_inputs_df, level_df, fm_group_tiv, agg_id_merge_col)

                level_df, profile_id_offset = assign_level_calcrule_and_profile_ids(
                    level_df, level_id, factorize_key, profile_id_offset)

                write_fm_profile_level(level_df, fm_profile_csv,
                                       fm_profile_bin, step_policies_present,
                                       chunksize=chunksize)

                # =====================================================================
                # MERGE LEVEL TERMS INTO GUL_INPUTS_DF
                # =====================================================================
                # level_df has financial terms with profile_id; gul_inputs_df has agg_id and needs
                # the profile_id merged in, handling layered vs non-layered rows differently.
                gul_inputs_df = merge_level_terms_into_gul(gul_inputs_df, level_df, is_policy_layer_level, level_id)

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

                # =====================================================================
                # WRITE FM OUTPUT FILES FOR THIS LEVEL
                # =====================================================================
                write_level_policytc_and_programme(gul_inputs_df, level_id, fm_policytc_bin, fm_policytc_csv,
                                                   fm_programme_bin, fm_programme_csv, chunksize)

                # reset gul_inputs_df level columns
                gul_inputs_df = reset_gul_inputs(gul_inputs_df)
                logger.info(f"level {cur_level_id} {level_info} took {time.time() - t0}")
                t0 = time.time()

        # =========================================================================
        # FINALIZATION: Assign output IDs and write fm_xref
        # =========================================================================
        gul_inputs_df = finalize_il_inputs(gul_inputs_df, fm_aggregation_profile, accounts_df,
                                           fm_xref_bin, fm_xref_csv, chunksize)

        return gul_inputs_df, il_input_files


def reset_gul_inputs(gul_inputs_df):
    return (
        gul_inputs_df.drop(columns=["root_start", "agg_id_prev", "is_step", "FMTermGroupID",
                                    "profile_id", "level_id", 'fm_peril', 'need_tiv',
                                    'agg_tiv'], errors='ignore')
        .rename(columns={"agg_id": "agg_id_prev"}))


def write_empty_policy_layer(gul_inputs_df, cur_level_id, agg_key, fm_policytc_csv, fm_policytc_bin,
                             fm_programme_csv, fm_programme_bin, chunksize):
    gul_inputs_df["agg_id"] = gul_inputs_df.groupby(agg_key, sort=False, observed=True).ngroup().astype('int32') + 1
    gul_inputs_df["profile_id"] = 1
    gul_inputs_df["level_id"] = cur_level_id
    fm_policytc_df = gul_inputs_df.loc[:, fm_policytc_headers]
    fm_policytc_dedup = fm_policytc_df.drop_duplicates()
    df_to_ndarray(fm_policytc_dedup, fm_policytc_dtype).tofile(fm_policytc_bin)
    if fm_policytc_csv is not None:
        fm_policytc_dedup.astype(fm_policytc_pd_dtype).to_csv(fm_policytc_csv, index=False, header=False,
                                                              chunksize=chunksize)
    fm_programe_df = gul_inputs_df[['level_id', 'agg_id']].rename(columns={'agg_id': 'to_agg_id'})
    fm_programe_df['from_agg_id'] = gul_inputs_df['agg_id_prev']
    fm_programe_dedup = fm_programe_df[fm_programme_headers].drop_duplicates()
    df_to_ndarray(fm_programe_dedup, fm_programme_dtype).tofile(fm_programme_bin)
    if fm_programme_csv is not None:
        fm_programe_dedup.astype(fm_programme_pd_dtype).to_csv(fm_programme_csv, index=False,
                                                               header=False, chunksize=chunksize)


def write_fm_profile_level(level_df, fm_profile_csv, fm_profile_bin_file,
                           step_policies_present, chunksize=100000):
    """Writes an FM profile file.

    Args:
        level_df (pandas.DataFrame): FM terms dataframe.
        fm_profile_csv (file): Open CSV file object to write to, or None to skip CSV.
        fm_profile_bin_file (file): Open binary file object to write to.
        step_policies_present (bool): Flag to determine which type of file to write.
        chunksize (int, optional): Number of rows to write per chunk.

    Raises:
        OasisException: If writing to the file fails.
    """
    level_df = level_df.astype({'calcrule_id': calcrule_id[DTYPE_IDX], 'profile_id': profile_id[DTYPE_IDX]})
    # Step policies exist
    if step_policies_present:
        fm_profile_df = level_df[list(set(level_df.columns).intersection(set(fm_profile_step_headers + ['steptriggertype'])))].copy()
        for col in fm_profile_step_headers + ['steptriggertype']:
            if col not in fm_profile_df.columns:
                fm_profile_df[col] = 0.
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
        fm_profile_df = fm_profile_df.astype(fm_profile_step_pd_dtype)
        fm_profile_df.loc[fm_profile_df['step_id'] == 0, 'step_id'] = 1

        fm_profile_df = fm_profile_df[fm_profile_step_headers].sort_values(by=["profile_id", 'step_id']).drop_duplicates()
    # No step policies
    else:
        # make sure there is no step file in the folder
        fm_profile_df = level_df[list(set(level_df.columns).intersection(set(policytc_cols)))].copy()
        for col in policytc_cols[2:]:
            if col not in fm_profile_df.columns:
                fm_profile_df[col] = 0.0

        fm_profile_df = (
            fm_profile_df
            .rename(columns=profile_cols_map)
            .drop_duplicates()
            .assign(share2=0.0, share3=0.0)
            .astype(fm_profile_pd_dtype)[fm_profile_headers]
        )
    try:
        bin_dtype = fm_profile_step_dtype if step_policies_present else fm_profile_dtype
        df_to_ndarray(fm_profile_df, bin_dtype).tofile(fm_profile_bin_file)
        if fm_profile_csv is not None:
            fm_profile_df.to_csv(
                fm_profile_csv,
                index=False,
                header=False,
                chunksize=chunksize,
            )
    except (IOError, OSError) as e:
        raise OasisException("Exception raised in 'write_fm_profile_level'") from e
