# -*- coding: utf-8 -*-

__all__ = [
    'generate_il_input_items',
    'get_coverage_level_il_terms',
    'get_il_input_items',
    'get_il_terms_by_level_as_list',
    'get_layer_calcrule_id',
    'get_layer_level_il_terms',
    'get_policytc_ids',
    'get_sub_layer_calcrule_id',
    'get_sub_layer_non_coverage_level_il_terms',
    'unified_fm_profile_by_level',
    'unified_fm_profile_by_level_and_term_group',
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

from ..utils.concurrency import (
    multiprocess,
    multithread,
    Task,
)
from ..utils.data import get_dataframe
from ..utils.exceptions import OasisException
from ..utils.metadata import OED_FM_LEVELS
from ..utils.oed_profiles import (
    get_default_accounts_profile,
    get_default_exposure_profile,
    get_default_fm_aggregation_profile,
)


def get_coverage_level_il_terms(level_unified_profile, level_fm_agg_profile, level_il_items, exposure_df, accounts_df):

    lid = level_il_items[0]['level_id']

    cov_level = OED_FM_LEVELS['site coverage']['id']

    if lid != cov_level:
        raise OasisException('Invalid FM level ID {} for generating coverage level FM terms - expected to be {}'.format(lid, cov_level))

    lfmap = level_fm_agg_profile

    agg_key = tuple(v['field'].lower() for v in viewvalues(lfmap['FMAggKey']))

    li = sorted([it for it in viewvalues(level_il_items)], key=lambda it: tuple(it[k] for k in agg_key))

    comb_df = pd.merge(exposure_df, accounts_df, left_on='accnumber', right_on='accnumber')

    def get_combined_item(loc_id, acc_id, policy_num):
        return comb_df[(comb_df['locnumber'] == loc_id + 1) & (comb_df['index_y'] == acc_id) & (comb_df['polnumber'] == policy_num)].iloc[0]

    for it, i in chain((it, i) for i, (key, group) in enumerate(groupby(li, key=lambda it: tuple(it[k] for k in agg_key))) for it in group):
        it['agg_id'] = i + 1

        comb_item = get_combined_item(it['loc_id'], it['acc_id'], it['policy_num'])

        item_ded = comb_item.get(it['ded_elm']) or 0.0
        it['deductible'] = (item_ded if item_ded >= 1 else it['tiv'] * item_ded) or 0.0

        item_ded_min = comb_item.get(it['ded_min_elm']) or 0.0
        it['deductible_min'] = (item_ded_min if item_ded_min >= 1 else it['tiv'] * item_ded_min) or 0.0

        item_ded_max = comb_item.get(it['ded_max_elm']) or 0.0
        it['deductible_max'] = (item_ded_max if item_ded_max >= 1 else it['tiv'] * item_ded_max) or 0.0

        item_lim = comb_item.get(it['lim_elm']) or 0.0
        it['limit'] = (item_lim if item_lim >= 1 else it['tiv'] * item_lim) or 0.0

        it['share'] = comb_item.get(it['shr_elm']) or 0.0

        it['calcrule_id'] = get_sub_layer_calcrule_id(it['deductible'], it['deductible_min'], it['deductible_max'], it['limit'])

        yield it


def get_layer_level_il_terms(level_unified_profile, level_fm_agg_profile, level_il_items, exposure_df, accounts_df):

    lid = level_il_items[0]['level_id']

    layer_level = OED_FM_LEVELS['policy layer']['id']

    if lid != layer_level:
        raise OasisException('Invalid FM level ID {} for generating coverage level FM terms - expected to be {}'.format(lid, layer_level))

    lup = level_unified_profile

    lfmap = level_fm_agg_profile

    agg_key = tuple(v['field'].lower() for v in viewvalues(lfmap['FMAggKey']))

    li = sorted([it for it in viewvalues(level_il_items)], key=lambda it: tuple(it[k] for k in agg_key))

    comb_df = pd.merge(exposure_df, accounts_df, left_on='accnumber', right_on='accnumber')

    def get_combined_item(loc_id, acc_id, policy_num):
        return comb_df[(comb_df['locnumber'] == loc_id + 1) & (comb_df['index_y'] == acc_id) & (comb_df['polnumber'] == policy_num)].iloc[0]

    ded_fld = lup[1].get('deductible') or {}
    ded_elm = ded_fld['ProfileElementName'].lower() if ded_fld else None

    lim_fld = lup[1].get('limit') or {}
    lim_elm = lim_fld['ProfileElementName'].lower() if lim_fld else None

    shr_fld = lup[1].get('share') or {}
    shr_elm = shr_fld['ProfileElementName'].lower() if shr_fld else None

    for it, i in chain((it, i) for i, (key, group) in enumerate(groupby(li, key=lambda it: tuple(it[k] for k in agg_key))) for it in group):
        it['agg_id'] = i + 1

        comb_item = get_combined_item(it['loc_id'], it['acc_id'], it['policy_num'])

        it['ded_elm'] = ded_elm
        it['deductible'] = comb_item.get(ded_elm) or 0.0
        it['attachment'] = it['deductible']
        it['deductible_min'] = it['deductible_max'] = 0.0

        it['lim_elm'] = lim_elm
        it['limit'] = comb_item.get(lim_elm) or 9999999999

        it['shr_elm'] = shr_elm
        it['share'] = comb_item.get(shr_elm) or 1.0

        it['calcrule_id'] = get_layer_calcrule_id(it['attachment'], it['limit'], it['share'])

        yield it


def get_sub_layer_non_coverage_level_il_terms(level_unified_profile, level_fm_agg_profile, level_il_items, exposure_df, accounts_df):

    lid = level_il_items[0]['level_id']

    sub_layer_non_coverage_levels = (OED_FM_LEVELS[level]['id'] for level in ['site pd', 'site all', 'cond all', 'policy all'])

    if lid not in sub_layer_non_coverage_levels:
        raise OasisException('Invalid FM level ID {} for generating sub-layer non-coverage level FM terms - expected to be in the set {}'.format(lid, set(sub_layer_non_coverage_levels)))

    lup = level_unified_profile

    lfmap = level_fm_agg_profile

    agg_key = tuple(v['field'].lower() for v in viewvalues(lfmap['FMAggKey']))

    li = sorted([it for it in viewvalues(level_il_items)], key=lambda it: tuple(it[k] for k in agg_key))

    comb_df = pd.merge(exposure_df, accounts_df, left_on='accnumber', right_on='accnumber')

    def get_combined_item(loc_id, acc_id, policy_num):
        return comb_df[(comb_df['locnumber'] == loc_id + 1) & (comb_df['index_y'] == acc_id) & (comb_df['polnumber'] == policy_num)].iloc[0]

    ded_fld = lup[1].get('deductible') or {}
    ded_elm = ded_fld['ProfileElementName'].lower() if ded_fld else None

    ded_min_fld = lup[1].get('deductiblemin') or {}
    ded_min_elm = ded_min_fld['ProfileElementName'].lower() if ded_min_fld else None

    ded_max_fld = lup[1].get('deductiblemax') or {}
    ded_max_elm = ded_max_fld['ProfileElementName'].lower() if ded_max_fld else None

    lim_fld = lup[1].get('limit') or {}
    lim_elm = lim_fld['ProfileElementName'].lower() if lim_fld else None

    for it, i in chain((it, i) for i, (key, group) in enumerate(groupby(li, key=lambda it: tuple(it[k] for k in agg_key))) for it in group):
        it['agg_id'] = i + 1

        comb_item = get_combined_item(it['loc_id'], it['acc_id'], it['policy_num'])

        it['ded_elm'] = ded_elm if (not ded_fld.get('CoverageTypeID') or it['coverage_type_id'] in (ded_fld['CoverageTypeID'] or [])) else None
        item_ded = comb_item.get(it['ded_elm']) or 0.0
        it['deductible'] = (item_ded if item_ded >= 1 else it['tiv'] * item_ded) or 0.0

        it['ded_min_elm'] = ded_min_elm if (not ded_min_fld.get('CoverageTypeID') or it['coverage_type_id'] in (ded_min_fld['CoverageTypeID'] or [])) else None
        item_ded_min = comb_item.get(it['ded_min_elm']) or 0.0
        it['deductible_min'] = (item_ded_min if item_ded_min >= 1 else it['tiv'] * item_ded_min) or 0.0

        it['ded_max_elm'] = ded_max_elm if (not ded_max_fld.get('CoverageTypeID') or it['coverage_type_id'] in (ded_max_fld['CoverageTypeID'] or [])) else None
        item_ded_max = comb_item.get(it['ded_max_elm']) or 0.0
        it['deductible_max'] = (item_ded_max if item_ded_max >= 1 else it['tiv'] * item_ded_max) or 0.0

        it['lim_elm'] = lim_elm if (not lim_fld.get('CoverageTypeID') or it['coverage_type_id'] in (lim_fld['CoverageTypeID'] or [])) else None
        item_lim = comb_item.get(it['lim_elm']) or 0.0
        it['limit'] = (item_lim if item_lim >= 1 else it['tiv'] * item_lim) or 0.0

        it['calcrule_id'] = get_sub_layer_calcrule_id(it['deductible'], it['deductible_min'], it['deductible_max'], it['limit'])

        yield it


def get_il_terms_by_level_as_list(level_unified_profile, level_fm_agg_profile, level_il_items, exposure_df, accounts_df):

    level_id = level_il_items[0]['level_id']

    cov_level = OED_FM_LEVELS['site coverage']['id']

    sub_layer_non_coverage_levels = (OED_FM_LEVELS[level]['id'] for level in ['site pd', 'site all', 'cond all', 'policy all'])

    layer_level = OED_FM_LEVELS['policy layer']['id']

    if level_id == cov_level:
        return [it for it in get_coverage_level_il_terms(level_unified_profile, level_fm_agg_profile, level_il_items, exposure_df, accounts_df)]
    elif level_id in sub_layer_non_coverage_levels:
        return [it for it in get_sub_layer_non_coverage_level_il_terms(level_unified_profile, level_fm_agg_profile, level_il_items, exposure_df, accounts_df)]
    elif level_id == layer_level:
        return [it for it in get_layer_level_il_terms(level_unified_profile, level_fm_agg_profile, level_il_items, exposure_df, accounts_df)]


def get_policytc_ids(il_inputs_df):

    columns = [
        col for col in il_inputs_df.columns if col not in ('limit', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'share', 'calcrule_id',)
    ]

    policytc_df = il_inputs_df.drop(columns, axis=1).drop_duplicates()

    for col in policytc_df.columns:
        policytc_df[col] = policytc_df[col].astype(float) if col != 'calcrule_id' else policytc_df[col].astype(int)

    policytc_df['index'] = range(1, len(policytc_df) + 1)

    policytc_ids = {
        i: {
            'limit': policytc_df.iloc[i - 1]['limit'],
            'deductible': policytc_df.iloc[i - 1]['deductible'],
            'deductible_min': policytc_df.iloc[i - 1]['deductible_min'],
            'deductible_max': policytc_df.iloc[i - 1]['deductible_max'],
            'attachment': policytc_df.iloc[i - 1]['attachment'],
            'share': policytc_df.iloc[i - 1]['share'],
            'calcrule_id': int(policytc_df.iloc[i - 1]['calcrule_id'])
        } for i in policytc_df['index']
    }

    return policytc_ids


def get_layer_calcrule_id(att=0, lim=9999999999, shr=1):

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


def unified_fm_profile_by_level(profiles=[], profile_paths=[]):

    if not (profiles or profile_paths):
        raise OasisException('A list of source profiles (loc. or acc.) or a list of source profile paths must be provided')

    if not profiles:
        for pp in profile_paths:
            with io.open(pp, 'r', encoding='utf-8') as f:
                profiles.append(json.load(f))

    comb_prof = {k: v for p in profiles for k, v in ((k, v) for k, v in viewitems(p) if 'FMLevel' in v)}

    return {
        int(k): {v['ProfileElementName']: v for v in g} for k, g in groupby(sorted(viewvalues(comb_prof), key=lambda v: v['FMLevel']), key=lambda v: v['FMLevel'])
    }


def unified_fm_profile_by_level_and_term_group(profiles=[], profile_paths=[]):

    if not (profiles or profile_paths):
        raise OasisException('A list of source profiles (loc. or acc.) or a list of source profile paths must be provided')

    comb_prof = unified_fm_profile_by_level(profiles=profiles, profile_paths=profile_paths)

    return {
        k: {
            _k: {v['FMTermType'].lower(): v for v in g} for _k, g in groupby(sorted(viewvalues(comb_prof[k]), key=lambda v: v['FMTermGroupID']), key=lambda v: v['FMTermGroupID'])
        } for k in comb_prof
    }


def generate_il_input_items(
    exposure_df,
    accounts_df,
    gul_inputs_df,
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile(),
    aggregation_profile=get_default_fm_aggregation_profile()
):
    """
    Generates FM input items.

    :param exposure_df: OED source exposure
    :type exposure_df: pandas.DataFrame

    :param accounts_df: Canonical accounts
    :param accounts_df: pandas.DataFrame

    :param gul_inputs_df: GUL input items
    :type gul_inputs_df: pandas.DataFrame

    :param exposure_profile: Source exposure profile
    :type exposure_profile: dict

    :param accounts_profile: Source accounts profile
    :type accounts_profile: dict

    :param aggregation_profile: FM aggregation profile
    :param aggregation_profile: dict
    """
    cep = exposure_profile
    cap = accounts_profile
    fmap = aggregation_profile

    for df in [exposure_df, gul_inputs_df, accounts_df]:
        if not df.columns.contains('index'):
            df['index'] = pd.Series(data=range(len(df)))

    expgul_df = pd.merge(exposure_df, gul_inputs_df, left_on='index', right_on='loc_id')
    expgul_df['index'] = pd.Series(data=expgul_df.index)

    keys = (
        'item_id', 'gul_item_id', 'peril_id', 'coverage_type_id', 'coverage_id',
        'is_bi_coverage', 'loc_id', 'acc_id', 'policy_num', 'level_id', 'layer_id',
        'agg_id', 'policytc_id', 'deductible', 'deductible_min',
        'deductible_max', 'attachment', 'limit', 'share', 'calcrule_id', 'tiv_elm',
        'tiv', 'tiv_tgid', 'ded_elm', 'ded_min_elm', 'ded_max_elm',
        'lim_elm', 'shr_elm',
    )

    try:
        ufp = unified_fm_profile_by_level_and_term_group(profiles=(cep, cap,))

        if not ufp:
            raise OasisException(
                'Canonical loc. and/or acc. profiles are possibly missing FM term information: '
                'FM term definitions for TIV, limit, deductible and/or share.'
            )

        if not fmap:
            raise OasisException(
                'FM aggregation profile is empty - this is required to perform aggregation'
            )

        fm_levels = tuple(ufp.keys())

        cov_level_id = fm_levels[0]

        coverage_level_preset_data = [t for t in zip(
            tuple(expgul_df['item_id'].values),           # 1 - FM item ID
            tuple(expgul_df['item_id'].values),           # 2 - GUL item ID
            tuple(expgul_df['peril_id'].values),          # 3 - peril ID
            tuple(expgul_df['coverage_type_id'].values),  # 4 - coverage type ID
            tuple(expgul_df['coverage_id'].values),       # 5 - coverage ID
            tuple(expgul_df['is_bi_coverage'].values),    # 6 - is BI coverage?
            tuple(expgul_df['loc_id'].values),            # 7 - src. exp. DF index
            (-1,) * len(expgul_df),                       # 8 - src. acc. DF index
            (-1,) * len(expgul_df),                       # 9 - src. acc. policy num.
            (cov_level_id,) * len(expgul_df),             # 10 - coverage level ID
            (1,) * len(expgul_df),                        # 11 - layer ID
            (-1,) * len(expgul_df),                       # 12 - agg. ID
            tuple(expgul_df['tiv_elm'].values),           # 13 - TIV element
            tuple(expgul_df['tiv'].values),               # 14 -TIV value
            tuple(expgul_df['tiv_tgid'].values),          # 15 -coverage element/term group ID
            tuple(expgul_df['ded_elm'].values),           # 16 -deductible element
            tuple(expgul_df['ded_min_elm'].values),       # 17 -deductible min. element
            tuple(expgul_df['ded_max_elm'].values),       # 18 -deductible max. element
            tuple(expgul_df['lim_elm'].values),           # 19 -limit element
            tuple(expgul_df['shr_elm'].values)            # 20 -share element
        )]

        def get_acc_item(i):
            return accounts_df[(accounts_df['accnumber'] == expgul_df[expgul_df['loc_id'] == coverage_level_preset_data[i][6]].iloc[0]['accnumber'])].iloc[0]

        def get_acc_id(i):
            return int(get_acc_item(i)['index'])

        coverage_level_preset_items = {
            i: {
                k: v for k, v in zip(
                    keys,
                    [i + 1, gul_item_id, peril_id, coverage_type_id, coverage_id, is_bi_coverage, loc_id, get_acc_id(i), policy_num, level_id, layer_id, agg_id, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12, tiv_elm, tiv, tiv_tgid, ded_elm, ded_min_elm, ded_max_elm, lim_elm, shr_elm]
                )
            } for i, (item_id, gul_item_id, peril_id, coverage_type_id, coverage_id, is_bi_coverage, loc_id, _, policy_num, level_id, layer_id, agg_id, tiv_elm, tiv, tiv_tgid, ded_elm, ded_min_elm, ded_max_elm, lim_elm, shr_elm) in enumerate(coverage_level_preset_data)
        }

        num_cov_items = len(coverage_level_preset_items)

        preset_items = {
            level_id: (coverage_level_preset_items if level_id == cov_level_id else copy.deepcopy(coverage_level_preset_items)) for level_id in fm_levels
        }

        for i, (level_id, item_id, it) in enumerate(chain((level_id, k, v) for level_id in fm_levels[1:] for k, v in preset_items[level_id].items())):
            it['level_id'] = level_id
            it['item_id'] = num_cov_items + i + 1
            it['ded_elm'] = it['ded_min_elm'] = it['ded_max_elm'] = it['lim_elm'] = it['shr_elm'] = None

        num_sub_layer_level_items = sum(len(preset_items[level_id]) for level_id in fm_levels[:-1])
        layer_level = max(fm_levels)
        layer_level_items = copy.deepcopy(preset_items[layer_level])
        layer_level_min_idx = min(layer_level_items)

        def layer_id(i):
            return list(
                accounts_df[accounts_df['accnumber'] == accounts_df.iloc[i]['accnumber']]['polnumber'].values
            ).index(accounts_df.iloc[i]['polnumber']) + 1

        for i, (loc_id, acc_id) in enumerate(
            chain((loc_id, acc_id) for loc_id in layer_level_items for loc_id, acc_id in product(
                [loc_id],
                accounts_df[accounts_df['accnumber'] == accounts_df.iloc[layer_level_items[loc_id]['acc_id']]['accnumber']]['index'].values)
            )
        ):
            it = copy.deepcopy(layer_level_items[loc_id])
            it['item_id'] = num_sub_layer_level_items + i + 1
            it['layer_id'] = layer_id(acc_id)
            it['acc_id'] = acc_id
            preset_items[layer_level][layer_level_min_idx + i] = it

        for it in (it for c in chain(viewvalues(preset_items[k]) for k in preset_items) for it in c):
            it['policy_num'] = accounts_df.iloc[it['acc_id']]['polnumber']
            lfmaggkey = fmap[it['level_id']]['FMAggKey']
            for v in viewvalues(lfmaggkey):
                src = v['src'].lower()
                if src in ['loc', 'acc']:
                    f = v['field'].lower()
                    it[f] = exposure_df.iloc[it['loc_id']][f] if src == 'loc' else accounts_df.iloc[it['acc_id']][f]

        #import ipdb; ipdb.set_trace()

        concurrent_tasks = (
            Task(get_il_terms_by_level_as_list, args=(ufp[level_id], fmap[level_id], preset_items[level_id], exposure_df.copy(deep=True), accounts_df.copy(deep=True),), key=level_id)
            for level_id in fm_levels
        )
        num_ps = min(len(fm_levels), multiprocessing.cpu_count())
        for it in multiprocess(concurrent_tasks, pool_size=num_ps):
            yield it
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
        raise OasisException(e)

def get_il_input_items(
    exposure_df,
    accounts_df,
    gul_inputs_df,
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile(),
    aggregation_profile=get_default_fm_aggregation_profile(),
    reduced=True
):
    """
    Loads FM input items generated by ``generate_il_input_items`` into a static
    structure such as a pandas dataframe.

    :param exposure_df: OED source exposure
    :type exposure_df: pandas.DataFrame

    :param accounts_df: OED source accounts
    :type accounts_df: pandas.DataFrame

    :param gul_inputs_df: GUL input items
    :type gul_inputs_df: pandas.DataFrame

    :param exposure_profile: OED source exposure profile
    :type exposure_profile: dict

    :param accounts_profile: OED source accounts profile
    :type accounts_profile: dict

    :param aggregation_profile: FM aggregation profile
    :param aggregation_profile: dict

    :param reduced: Whether to reduce the FM input items table by removing any
                    items with zero financial terms
    :param reduced: bool
    """
    cep = exposure_profile
    cap = accounts_profile
    fmap = aggregation_profile

    try:
        il_items = [
            it for it in generate_il_input_items(
                exposure_df,
                accounts_df,
                gul_inputs_df,
                exposure_profile=cep,
                accounts_profile=cap,
                aggregation_profile=fmap
            )
        ]
        il_items.sort(key=lambda it: it['item_id'])

        il_inputs_df = pd.DataFrame(data=il_items, dtype=object)
        il_inputs_df['index'] = pd.Series(data=il_inputs_df.index, dtype=int)

        if reduced:
            def is_zero_terms_level(level_id):
                return not any(
                    it['deductible'] != 0 or
                    it['deductible_min'] != 0 or
                    it['deductible_max'] != 0 or
                    it['limit'] != 0 or
                    it['share'] != 0
                    for _, it in il_inputs_df[il_inputs_df['level_id'] == level_id].iterrows()
                )

            levels = sorted([l for l in set(il_inputs_df['level_id'])])
            non_zero_terms_levels = [lid for lid in levels if lid in [levels[0], levels[-1]] or not is_zero_terms_level(lid)]

            il_inputs_df = il_inputs_df[(il_inputs_df['level_id'].isin(non_zero_terms_levels))]

            il_inputs_df['index'] = range(len(il_inputs_df))

            il_inputs_df['item_id'] = range(1, len(il_inputs_df) + 1)

            levels = sorted([l for l in set(il_inputs_df['level_id'])])

            def level_id(i):
                return levels.index(il_inputs_df.iloc[i]['level_id']) + 1

            il_inputs_df['level_id'] = il_inputs_df['index'].apply(level_id)

        policytc_ids = get_policytc_ids(il_inputs_df)

        def get_policytc_id(i):
            return [
                k for k in viewkeys(policytc_ids) if policytc_ids[k] == {k: il_inputs_df.iloc[i][k] for k in ('limit', 'deductible', 'attachment', 'deductible_min', 'deductible_max', 'share', 'calcrule_id',)}
            ][0]

        il_inputs_df['policytc_id'] = il_inputs_df['index'].apply(lambda i: get_policytc_id(i))

        for col in il_inputs_df.columns:
            if col == 'peril_id':
                il_inputs_df[col] = il_inputs_df[col].astype(object)
            elif col.endswith('id'):
                il_inputs_df[col] = il_inputs_df[col].astype(int)
            elif col in ('tiv', 'limit', 'deductible', 'deductible_min', 'deductible_max', 'share',):
                il_inputs_df[col] = il_inputs_df[col].astype(float)
    except (IOError, MemoryError, OasisException, OSError, TypeError, ValueError) as e:
        raise OasisException(e)

    return il_inputs_df, accounts_df


def write_fm_policytc_file(il_inputs_df, fm_policytc_fp):
    """
    Writes an FM policy T & C file.
    """
    try:
        fm_policytc_df = pd.DataFrame(
            columns=['layer_id', 'level_id', 'agg_id', 'policytc_id'],
            data=[key[:4] for key, _ in il_inputs_df.groupby(['layer_id', 'level_id', 'agg_id', 'policytc_id', 'limit', 'deductible', 'share'])],
            dtype=object
        )
        fm_policytc_df.to_csv(
            path_or_buf=fm_policytc_fp,
            encoding='utf-8',
            chunksize=1000,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fm_policytc_fp

def write_fm_profile_file(il_inputs_df, fm_profile_fp):
    """
    Writes an FM profile file.
    """
    try:
        cols = ['policytc_id', 'calcrule_id', 'limit', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'share']

        fm_profile_df = il_inputs_df[cols]

        fm_profile_df = pd.DataFrame(
            columns=cols,
            data=[key for key, _ in fm_profile_df.groupby(cols)]
        )

        col_repl = [
            {'deductible': 'deductible1'},
            {'deductible_min': 'deductible2'},
            {'deductible_max': 'deductible3'},
            {'attachment': 'attachment1'},
            {'limit': 'limit1'},
            {'share': 'share1'}
        ]
        for repl in col_repl:
            fm_profile_df.rename(columns=repl, inplace=True)

        n = len(fm_profile_df)

        fm_profile_df['index'] = range(n)

        fm_profile_df['share2'] = fm_profile_df['share3'] = [0] * n

        fm_profile_df.to_csv(
            columns=['policytc_id', 'calcrule_id', 'deductible1', 'deductible2', 'deductible3', 'attachment1', 'limit1', 'share1', 'share2', 'share3'],
            path_or_buf=fm_profile_fp,
            encoding='utf-8',
            chunksize=1000,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fm_profile_fp

def write_fm_programme_file(il_inputs_df, fm_programme_fp):
    """
    Writes a FM programme file.
    """
    try:
        cov_level = il_inputs_df['level_id'].min()
        fm_programme_df = pd.DataFrame(
            pd.concat([il_inputs_df[il_inputs_df['level_id'] == cov_level], il_inputs_df])[['level_id', 'agg_id']],
            dtype=int
        ).reset_index(drop=True)

        num_cov_items = len(il_inputs_df[il_inputs_df['level_id'] == cov_level])

        for i in range(num_cov_items):
            fm_programme_df.at[i, 'level_id'] = 0

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
            (from_agg_id, level_id, to_agg_id) for from_level_id, to_level_id in zip(levels, levels[1:]) for from_agg_id, level_id, to_agg_id in from_agg_id_to_agg_id(from_level_id, to_level_id)
        ]

        fm_programme_df = pd.DataFrame(columns=['from_agg_id', 'level_id', 'to_agg_id'], data=data, dtype=int).drop_duplicates()

        fm_programme_df.to_csv(
            path_or_buf=fm_programme_fp,
            encoding='utf-8',
            chunksize=1000,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fm_programme_fp

def write_fm_xref_file(il_inputs_df, fm_xref_fp):
    """
    Writes a FM xref file.
    """
    try:
        data = [
            (i + 1, agg_id, layer_id) for i, (agg_id, layer_id) in enumerate(product(set(il_inputs_df['agg_id']), set(il_inputs_df['layer_id'])))
        ]

        fm_xref_df = pd.DataFrame(columns=['output', 'agg_id', 'layer_id'], data=data, dtype=int)

        fm_xref_df.to_csv(
            path_or_buf=fm_xref_fp,
            encoding='utf-8',
            chunksize=1000,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fm_xref_fp

def write_fmsummaryxref_file(il_inputs_df, fmsummaryxref_fp):
    """
    Writes an FM summaryxref file.
    """
    try:
        data = [
            (i + 1, 1, 1) for i, _ in enumerate(product(set(il_inputs_df['agg_id']), set(il_inputs_df['layer_id'])))
        ]

        fmsummaryxref_df = pd.DataFrame(columns=['output', 'summary_id', 'summaryset_id'], data=data, dtype=int)

        fmsummaryxref_df.to_csv(
            path_or_buf=fmsummaryxref_fp,
            encoding='utf-8',
            chunksize=1000,
            index=False
        )
    except (IOError, OSError) as e:
        raise OasisException(e)

    return fmsummaryxref_fp


def write_il_input_files(
    exposure_df,
    gul_inputs_df,
    accounts_fp,
    target_dir,
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile(),
    aggregation_profile=get_default_fm_aggregation_profile(),
    fname_prefixes={
        'fm_policytc': 'fm_policytc',
        'fm_profile': 'fm_profile',
        'fm_programme': 'fm_programme',
        'fm_xref': 'fm_xref',
        'fmsummaryxref': 'fmsummaryxref'
    }
):
    """
    Generate standard Oasis FM input files, namely::

        fm_policytc.csv
        fm_profile.csv
        fm_programme.csv
        fm_xref.csv
        fmsummaryxref.csv
    """
    accounts_df = get_dataframe(src_fp=accounts_fp, empty_data_error_msg='No accounts data found in the source accounts (acc.) file')

    il_inputs_df, _ = get_il_input_items(
        exposure_df,
        accounts_df,
        gul_inputs_df,
        exposure_profile=exposure_profile,
        accounts_profile=accounts_profile,
        aggregation_profile=aggregation_profile
    )

    il_input_files = {
        k: os.path.join(target_dir, '{}.csv'.format(fname_prefixes[k])) for k in viewkeys(fname_prefixes)
    }

    concurrent_tasks = (
        Task(getattr(sys.modules[__name__], 'write_{}_file'.format(f)), args=(il_inputs_df.copy(deep=True), il_input_files[f],), key=f)
        for f in il_input_files
    )
    num_ps = min(len(il_input_files), multiprocessing.cpu_count())
    for _, _ in multithread(concurrent_tasks, pool_size=num_ps):
        pass

    return il_input_files, il_inputs_df, accounts_df
