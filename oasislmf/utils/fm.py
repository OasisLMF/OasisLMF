# -*- coding: utf-8 -*-

__all__ = [
    'get_coverage_level_fm_terms',
    'get_fm_terms_by_level_as_list',
    'get_layer_calcrule_id', 
    'get_layer_level_fm_terms',
    'get_policytc_ids',
    'get_sub_layer_calcrule_id',
    'get_sub_layer_non_coverage_level_fm_terms',
    'unified_canonical_fm_profile_by_level',
    'unified_canonical_fm_profile_by_level_and_term_group'
]

import io
import itertools
import json
import six

import pandas as pd

from .exceptions import OasisException
from .metadata import (
    OASIS_FM_LEVELS,
    OED_FM_LEVELS,
)


def get_coverage_level_fm_terms(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df, oed=True):

    lid = level_fm_items[0]['level_id']

    cov_level = OASIS_FM_LEVELS['coverage']['id'] if not oed else OED_FM_LEVELS['site coverage']['id']

    if lid != cov_level:
        raise OasisException('Invalid FM level ID {} for generating coverage level FM terms - expected to be {}'.format(lid, cov_level))

    lufcp = level_unified_canonical_profile

    lfmap = level_fm_agg_profile

    agg_key = tuple(v['field'].lower() for v in six.itervalues(lfmap['FMAggKey']))

    li = sorted([it for it in six.itervalues(level_fm_items)], key=lambda it: tuple(it[k] for k in agg_key))

    can_df = pd.merge(canexp_df, canacc_df, left_on='accntnum', right_on='accntnum')

    def get_can_item(canexp_id, canacc_id, policy_num):
        return can_df[(can_df['row_id_x']==canexp_id+1) & (can_df['row_id_y']==canacc_id+1) & (can_df['policynum']==policy_num)].iloc[0]

    for it, i in itertools.chain((it, i) for i, (key, group) in enumerate(itertools.groupby(li, key=lambda it: tuple(it[k] for k in agg_key))) for it in group):
        it['agg_id'] = i + 1

        can_item = get_can_item(it['canexp_id'], it['canacc_id'], it['policy_num'])

        can_item_ded = can_item.get(it['ded_elm']) or 0.0
        it['deductible'] = (can_item_ded if can_item_ded >= 1 else it['tiv']*can_item_ded) or 0.0

        can_item_ded_min = can_item.get(it['ded_min_elm']) or 0.0
        it['deductible_min'] = (can_item_ded_min if can_item_ded_min >= 1 else it['tiv']*can_item_ded_min) or 0.0

        can_item_ded_max = can_item.get(it['ded_max_elm']) or 0.0
        it['deductible_max'] = (can_item_ded_max if can_item_ded_max >= 1 else it['tiv']*can_item_ded_max) or 0.0

        can_item_lim = can_item.get(it['lim_elm']) or 0.0
        it['limit'] = (can_item_lim if can_item_lim >= 1 else it['tiv']*can_item_lim) or 0.0

        it['share'] = can_item.get(it['shr_elm']) or 0.0

        it['calcrule_id'] = get_sub_layer_calcrule_id(it['deductible'], it['deductible_min'], it['deductible_max'], it['limit'])

        yield it


def get_layer_level_fm_terms(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df, oed=True):

    lid = level_fm_items[0]['level_id']

    layer_level = OASIS_FM_LEVELS['layer']['id'] if not oed else OED_FM_LEVELS['policy layer']['id']

    if lid != layer_level:
        raise OasisException('Invalid FM level ID {} for generating coverage level FM terms - expected to be {}'.format(lid, layer_level))

    lufcp = level_unified_canonical_profile

    lfmap = level_fm_agg_profile

    agg_key = tuple(v['field'].lower() for v in six.itervalues(lfmap['FMAggKey']))

    li = sorted([it for it in six.itervalues(level_fm_items)], key=lambda it: tuple(it[k] for k in agg_key))

    can_df = pd.merge(canexp_df, canacc_df, left_on='accntnum', right_on='accntnum')

    def get_can_item(canexp_id, canacc_id, policy_num):
        return can_df[(can_df['row_id_x']==canexp_id+1) & (can_df['row_id_y']==canacc_id+1) & (can_df['policynum']==policy_num)].iloc[0]

    ded_fld = lufcp[1].get('deductible') or {}
    ded_elm = ded_fld['ProfileElementName'].lower() if ded_fld else None

    lim_fld = lufcp[1].get('limit') or {}
    lim_elm = lim_fld['ProfileElementName'].lower() if lim_fld else None

    shr_fld = lufcp[1].get('share') or {}
    shr_elm = shr_fld['ProfileElementName'].lower() if shr_fld else None

    for it, i in itertools.chain((it, i) for i, (key, group) in enumerate(itertools.groupby(li, key=lambda it: tuple(it[k] for k in agg_key))) for it in group):
        it['agg_id'] = i + 1

        can_item = get_can_item(it['canexp_id'], it['canacc_id'], it['policy_num'])

        it['ded_elm'] = ded_elm
        it['deductible'] = can_item.get(ded_elm) or 0.0
        it['attachment'] = it['deductible']
        it['deductible_min'] = it['deductible_max'] = 0.0

        it['lim_elm'] = lim_elm
        it['limit'] = can_item.get(lim_elm) or 9999999999

        it['shr_elm'] = shr_elm
        it['share'] = can_item.get(shr_elm) or 1.0

        it['calcrule_id'] = get_layer_calcrule_id(it['attachment'], it['limit'], it['share'])

        yield it


def get_sub_layer_non_coverage_level_fm_terms(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df, oed=True):

    lid = level_fm_items[0]['level_id']

    sub_layer_non_coverage_levels = (
        range(OASIS_FM_LEVELS['combined']['id'], OASIS_FM_LEVELS['layer']['id']) if not oed else
        (OED_FM_LEVELS[level]['id'] for level in ['site pd', 'site all', 'cond all', 'policy all'])
    )

    if lid not in sub_layer_non_coverage_levels:
        raise OasisException('Invalid FM level ID {} for generating sub-layer non-coverage level FM terms - expected to be in the set {}'.format(lid, set(sub_layer_non_coverage_levels)))

    lufcp = level_unified_canonical_profile

    lfmap = level_fm_agg_profile

    agg_key = tuple(v['field'].lower() for v in six.itervalues(lfmap['FMAggKey']))

    li = sorted([it for it in six.itervalues(level_fm_items)], key=lambda it: tuple(it[k] for k in agg_key))

    can_df = pd.merge(canexp_df, canacc_df, left_on='accntnum', right_on='accntnum')

    def get_can_item(canexp_id, canacc_id, policy_num):
        return can_df[(can_df['row_id_x']==canexp_id+1) & (can_df['row_id_y']==canacc_id+1) & (can_df['policynum']==policy_num)].iloc[0]

    ded_fld = lufcp[1].get('deductible') or {}
    ded_elm = ded_fld['ProfileElementName'].lower() if ded_fld else None

    ded_min_fld = lufcp[1].get('deductiblemin') or {}
    ded_min_elm = ded_min_fld['ProfileElementName'].lower() if ded_min_fld else None

    ded_max_fld = lufcp[1].get('deductiblemax') or {}
    ded_max_elm = ded_max_fld['ProfileElementName'].lower() if ded_max_fld else None

    lim_fld = lufcp[1].get('limit') or {}
    lim_elm = lim_fld['ProfileElementName'].lower() if lim_fld else None

    for it, i in itertools.chain((it, i) for i, (key, group) in enumerate(itertools.groupby(li, key=lambda it: tuple(it[k] for k in agg_key))) for it in group):
        it['agg_id'] = i + 1

        can_item = get_can_item(it['canexp_id'], it['canacc_id'], it['policy_num'])

        it['ded_elm'] = ded_elm if (not ded_fld.get('CoverageTypeID') or it['coverage_type_id'] in (ded_fld['CoverageTypeID'] or [])) else None
        can_item_ded = can_item.get(it['ded_elm']) or 0.0
        it['deductible'] = (can_item_ded if can_item_ded >= 1 else it['tiv']*can_item_ded) or 0.0

        it['ded_min_elm'] = ded_min_elm  if (not ded_min_fld.get('CoverageTypeID') or it['coverage_type_id'] in (ded_min_fld['CoverageTypeID'] or [])) else None
        can_item_ded_min = can_item.get(it['ded_min_elm']) or 0.0
        it['deductible_min'] = (can_item_ded_min if can_item_ded_min >= 1 else it['tiv']*can_item_ded_min) or 0.0

        it['ded_max_elm'] = ded_max_elm  if (not ded_max_fld.get('CoverageTypeID') or it['coverage_type_id'] in (ded_max_fld['CoverageTypeID'] or [])) else None
        can_item_ded_max = can_item.get(it['ded_max_elm']) or 0.0
        it['deductible_max'] = (can_item_ded_max if can_item_ded_max >= 1 else it['tiv']*can_item_ded_max) or 0.0

        it['lim_elm'] = lim_elm  if (not lim_fld.get('CoverageTypeID') or it['coverage_type_id'] in (lim_fld['CoverageTypeID'] or [])) else None
        can_item_lim = can_item.get(it['lim_elm']) or 0.0
        it['limit'] = (can_item_lim if can_item_lim >= 1 else it['tiv']*can_item_lim) or 0.0

        it['calcrule_id'] = get_sub_layer_calcrule_id(it['deductible'], it['deductible_min'], it['deductible_max'], it['limit'])

        yield it


def get_fm_terms_by_level_as_list(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df, oed=True):

    level_id = level_fm_items[0]['level_id']

    cov_level = OASIS_FM_LEVELS['coverage']['id'] if not oed else OED_FM_LEVELS['site coverage']['id']

    sub_layer_non_coverage_levels = (
        range(OASIS_FM_LEVELS['combined']['id'], OASIS_FM_LEVELS['layer']['id']) if not oed else
        (OED_FM_LEVELS[level]['id'] for level in ['site pd', 'site all', 'cond all', 'policy all'])
    )

    layer_level = OASIS_FM_LEVELS['layer']['id'] if not oed else OED_FM_LEVELS['policy layer']['id']

    if level_id == cov_level:
        return [it for it in get_coverage_level_fm_terms(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df, oed)]
    elif level_id in sub_layer_non_coverage_levels:
        return [it for it in get_sub_layer_non_coverage_level_fm_terms(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df, oed)]
    elif level_id == layer_level:
        return [it for it in get_layer_level_fm_terms(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df, oed)]


def get_policytc_ids(fm_items_df):

    columns = [
        col for col in fm_items_df.columns if not col in ('limit', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'share', 'calcrule_id',)
    ]

    policytc_df = fm_items_df.drop(columns, axis=1).drop_duplicates()
    
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
    elif (ded > 0 and ded_code == 2) and (ded_min > 0 and ded_max > 0) and (lim == lim_code == 0):
        return 21


def unified_canonical_fm_profile_by_level(profiles=[], profile_paths=[]):

    if not (profiles or profile_paths):
        raise OasisException('A list of canonical profiles (loc. or acc.) or a list of canonical profiles paths must be provided')

    if not profiles:
        for pp in profile_paths:
            with io.open(pp, 'r', encoding='utf-8') as f:
                profiles.append(json.load(f))

    comb_prof = {k:v for p in profiles for k, v in ((k, v) for k, v in six.iteritems(p) if 'FMLevel' in v)}
    
    return {
        int(k):{v['ProfileElementName']:v for v in g} for k, g in itertools.groupby(sorted(six.itervalues(comb_prof), key=lambda v: v['FMLevel']), key=lambda v: v['FMLevel'])
    }


def unified_canonical_fm_profile_by_level_and_term_group(profiles=[], profile_paths=[]):

    if not (profiles or profile_paths):
        raise OasisException('A list of canonical profiles (loc. or acc.) or a list of canonical profiles paths must be provided')

    comb_prof = unified_canonical_fm_profile_by_level(profiles=profiles, profile_paths=profile_paths)

    return {
        k:{
            _k:{v['FMTermType'].lower():v for v in g} for _k, g in itertools.groupby(sorted(six.itervalues(comb_prof[k]), key=lambda v: v['FMTermGroupID']), key=lambda v: v['FMTermGroupID'])
        } for k in comb_prof
    }
