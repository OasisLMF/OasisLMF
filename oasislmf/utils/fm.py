# -*- coding: utf-8 -*-

__all__ = [
    'get_calcrule_id',
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
from .metadata import OASIS_FM_LEVELS


def get_calcrule_id(limit, share, ded_type):

    if limit == share == 0 and ded_type == DEDUCTIBLE_TYPES['blanket']['id']:
        return 12
    elif limit == 0 and share > 0 and ded_type == DEDUCTIBLE_TYPES['blanket']['id']:
        return 15
    elif limit > 0 and share == 0 and ded_type == DEDUCTIBLE_TYPES['blanket']['id']:
        return 1
    elif ded_type == DEDUCTIBLE_TYPES['minimum']['id']:
        return 11
    elif ded_type == DEDUCTIBLE_TYPES['maximum']['id']:
        return 10
    else:
        return 2


def get_coverage_level_fm_terms(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df):

    lid = level_fm_items[0]['level_id']

    if lid != OASIS_FM_LEVELS['coverage']['id']:
        raise OasisException('Invalid FM level ID {} for generating coverage level FM terms - expected to be {}'.format(lid, OASIS_FM_LEVELS['coverage']['id']))

    lufcp = level_unified_canonical_profile

    lfmap = level_fm_agg_profile

    agg_key = tuple(v['field'].lower() for v in six.itervalues(lfmap['FMAggKey']))

    li = sorted([it for it in six.itervalues(level_fm_items)], key=lambda it: tuple(it[k] for k in agg_key))

    can_df = pd.merge(canexp_df, canacc_df, left_on='accntnum', right_on='accntnum')

    get_can_item = lambda canexp_id, canacc_id, policy_num: can_df[(can_df['row_id_x']==canexp_id+1) & (can_df['row_id_y']==canacc_id+1) & (can_df['policynum']==policy_num)].iloc[0]

    for it, i in itertools.chain((it, i) for i, (key, group) in enumerate(itertools.groupby(li, key=lambda it: tuple(it[k] for k in agg_key))) for it in group):
        it['agg_id'] = i + 1

        can_item = get_can_item(it['canexp_id'], it['canacc_id'], it['policy_num'])

        it['deductible'] = can_item.get(it['ded_elm']) or 0.0
        it['deductible_min'] = can_item.get(it['ded_min_elm']) or 0.0
        it['deductible_max'] = can_item.get(it['ded_max_elm']) or 0.0

        it['limit'] = can_item.get(it['lim_elm']) or 0.0

        it['share'] = can_item.get(it['shr_elm']) or 0.0

        it['calcrule_id'] = get_sub_layer_calcrule_id(it['deductible'], it['deductible_min'], it['deductible_max'], it['limit'])

        yield it


def get_layer_level_fm_terms(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df):

    lid = level_fm_items[0]['level_id']

    if lid != OASIS_FM_LEVELS['layer']['id']:
        raise OasisException('Invalid FM level ID {} for generating coverage level FM terms - expected to be {}'.format(lid, OASIS_FM_LEVELS['layer']['id']))

    lufcp = level_unified_canonical_profile

    lfmap = level_fm_agg_profile

    agg_key = tuple(v['field'].lower() for v in six.itervalues(lfmap['FMAggKey']))

    li = sorted([it for it in six.itervalues(level_fm_items)], key=lambda it: tuple(it[k] for k in agg_key))

    can_df = pd.merge(canexp_df, canacc_df, left_on='accntnum', right_on='accntnum')

    get_can_item = lambda canexp_id, canacc_id, policy_num: can_df[(can_df['row_id_x']==canexp_id+1) & (can_df['row_id_y']==canacc_id+1) & (can_df['policynum']==policy_num)].iloc[0]

    ded_fld = lufcp[1].get('deductible')
    ded_elm = ded_fld['ProfileElementName'].lower() if ded_fld else None

    lim_fld = lufcp[1].get('limit')
    lim_elm = lim_fld['ProfileElementName'].lower() if lim_fld else None

    shr_fld = lufcp[1].get('share')
    shr_elm = shr_fld['ProfileElementName'].lower() if shr_fld else None

    it['calcrule_id'] = get_layer_calcrule_id(it['deductible'], it['limit'], it['share'])

    yield it

def get_sub_layer_non_coverage_level_fm_terms(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df):

    lid = level_fm_items[0]['level_id']

    r = range(OASIS_FM_LEVELS['combined']['id'], OASIS_FM_LEVELS['layer']['id'])
    if lid not in r:
        raise OasisException('Invalid FM level ID {} for generating sub-layer non-coverage level FM terms - expected to be in the range {}...{}'.format(lid, r.start, r.stop - 1))

    lufcp = level_unified_canonical_profile

    lfmap = level_fm_agg_profile

    agg_key = tuple(v['field'].lower() for v in six.itervalues(lfmap['FMAggKey']))

    li = sorted([it for it in six.itervalues(level_fm_items)], key=lambda it: tuple(it[k] for k in agg_key))

    can_df = pd.merge(canexp_df, canacc_df, left_on='accntnum', right_on='accntnum')

    get_can_item = lambda canexp_id, canacc_id, policy_num: can_df[(can_df['row_id_x']==canexp_id+1) & (can_df['row_id_y']==canacc_id+1) & (can_df['policynum']==policy_num)].iloc[0]

    ded_fld = lufcp[1].get('deductible')
    ded_elm = ded_fld['ProfileElementName'].lower() if ded_fld else None

    ded_min_fld = lufcp[1].get('deductiblemin')
    ded_min_elm = ded_fld['ProfileElementName'].lower() if ded_min_fld else None

    ded_max_fld = lufcp[1].get('deductiblemax')
    ded_max_elm = ded_fld['ProfileElementName'].lower() if ded_max_fld else None

    lim_fld = lufcp[1].get('limit')
    lim_elm = lim_fld['ProfileElementName'].lower() if lim_fld else None

    shr_fld = lufcp[1].get('share')
    shr_elm = shr_fld['ProfileElementName'].lower() if shr_fld else None

    for it, i in itertools.chain((it, i) for i, (key, group) in enumerate(itertools.groupby(li, key=lambda it: tuple(it[k] for k in agg_key))) for it in group):
        it['agg_id'] = i + 1

        can_item = get_can_item(it['canexp_id'], it['canacc_id'], it['policy_num'])

        it['ded_elm'] = ded_elm
        can_item_ded = can_item.get(ded_elm) or 0.0
        it['deductible'] = (can_item_ded if can_item_ded >= 1 else it['tiv']*can_item_ded) or 0.0

        it['ded_min_elm'] = ded_min_elm
        can_item_ded_min = can_item.get(ded_min_elm) or 0.0
        it['deductible_min'] = (can_item_min_ded if can_item_min_ded >= 1 else it['tiv']*can_item_min_ded) or 0.0

        it['ded_max_elm'] = ded_max_elm
        can_item_ded_max = can_item.get(ded_max_elm) or 0.0
        it['deductible_max'] = (can_item_max_ded if can_item_max_ded >= 1 else it['tiv']*can_item_max_ded) or 0.0

        it['lim_elm'] = lim_elm
        can_item_lim = can_item.get(lim_elm) or 0.0
        it['limit'] = (can_item_lim if can_item_lim >= 1 else it['tiv']*can_item_lim) or 0.0

        it['calcrule_id'] = get_sub_layer_calcrule_id(it['deductible'], it['deductible_min'], it['deductible_max'], it['limit'])

        yield it


def get_fm_terms_by_level_as_list(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df):

    level_id = level_fm_items[0]['level_id']

    if level_id == OASIS_FM_LEVELS['coverage']['id']:
        return get_coverage_level_fm_terms(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df)
    elif level_id in range(OASIS_FM_LEVELS['combined']['id'], OASIS_FM_LEVELS['layer']['id']):
        return get_sub_layer_non_coverage_level_fm_terms(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df)
    elif level_id == OASIS_FM_LEVELS['layer']['id']:
        return get_layer_level_fm_terms(level_unified_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df)

def get_policytc_ids(fm_items_df):

    columns = [
        col for col in fm_items_df.columns if not col in ('limit', 'deductible', 'share', 'calcrule_id',)
    ]

    policytc_df = fm_items_df.drop(columns, axis=1).drop_duplicates()
    
    for col in policytc_df.columns:
        policytc_df[col] = policytc_df[col].astype(float) if col != 'calcrule_id' else policytc_df[col].astype(int)

    policytc_df['index'] = range(1, len(policytc_df) + 1)

    policytc_ids = {
        i: {
            'limit': policytc_df.iloc[i - 1]['limit'],
            'deductible': policytc_df.iloc[i - 1]['deductible'],
            'share': policytc_df.iloc[i - 1]['share'],
            'calcrule_id': int(policytc_df.iloc[i - 1]['calcrule_id'])
        } for i in policytc_df['index']
    }

    return policytc_ids


def get_layer_calcrule_id(ded=0, lim=9999999999, shr=1):

    if lim > 0 or ded > 0 or shr > 0:
        return 2


def get_sub_layer_calcrule_id(ded, ded_min, ded_max, lim, ded_code=0, lim_code=0):

    if ded == ded_code == ded_min == ded_max == lim == lim_code == 0:
        return 12
    elif (ded > 0 and ded_code == 0) and (ded_min == ded_max == 0) and (lim > 0 and lim_code == 0):
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
    elif (ded > 0 and ded_code == 0) and (ded_min == ded_max == 0) and (lim == lim_code == 0):
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
