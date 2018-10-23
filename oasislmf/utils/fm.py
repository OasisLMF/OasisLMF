# -*- coding: utf-8 -*-

__all__ = [
    'unified_canonical_fm_profile_by_level',
    'unified_canonical_fm_profile_by_level_and_term_group',
    'get_calcrule_id',
    'get_fm_terms_by_level_as_list',
    'get_coverage_level_fm_terms',
    'get_non_coverage_level_fm_terms',
    'get_policytc_ids'
]

import io
import itertools
import json
import six

import pandas as pd

from .exceptions import OasisException
from .metadata import (
    DEDUCTIBLE_TYPES,
    FM_TERMS,
)


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


def get_coverage_level_fm_terms(level_grouped_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df):

    lid = 1

    lgcp = level_grouped_canonical_profile

    lfmap = level_fm_agg_profile

    agg_key = tuple(v['field'].lower() for v in six.itervalues(lfmap['FMAggKey']))

    li = sorted([it for it in six.itervalues(level_fm_items)], key=lambda it: tuple(it[k] for k in agg_key))

    can_df = pd.merge(canexp_df, canacc_df, left_on='accntnum', right_on='accntnum')

    get_can_item = lambda canexp_id, canacc_id, policy_num: can_df[(can_df['row_id_x']==canexp_id+1) & (can_df['row_id_y']==canacc_id+1) & (can_df['policynum']==policy_num)].iloc[0]

    for it, i in itertools.chain((it, i) for i, (key, group) in enumerate(itertools.groupby(li, key=lambda it: tuple(it[k] for k in agg_key))) for it in group):
        it['agg_id'] = i + 1

        can_item = get_can_item(it['canexp_id'], it['canacc_id'], it['policy_num'])

        limit = can_item.get(it['lim_elm']) or 0.0
        it['limit'] = limit

        deductible = can_item.get(it['ded_elm']) or 0.0
        it['deductible'] = deductible
    
        share = can_item.get(it['shr_elm']) or 0.0
        it['share'] = share

        it['calcrule_id'] = get_calcrule_id(it['limit'], it['share'], it['deductible_type'])

        yield it


def get_non_coverage_level_fm_terms(level_grouped_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df):

    lid = level_fm_items[0]['level_id']

    lgcp = level_grouped_canonical_profile

    lfmap = level_fm_agg_profile

    agg_key = tuple(v['field'].lower() for v in six.itervalues(lfmap['FMAggKey']))

    li = sorted([it for it in six.itervalues(level_fm_items)], key=lambda it: tuple(it[k] for k in agg_key))

    can_df = pd.merge(canexp_df, canacc_df, left_on='accntnum', right_on='accntnum')

    get_can_item = lambda canexp_id, canacc_id, policy_num: can_df[(can_df['row_id_x']==canexp_id+1) & (can_df['row_id_y']==canacc_id+1) & (can_df['policynum']==policy_num)].iloc[0]

    lim_fld = lgcp[1].get('limit')
    lim_elm = lim_fld['ProfileElementName'].lower() if lim_fld else None
    ded_fld = lgcp[1].get('deductible')
    ded_elm = ded_fld['ProfileElementName'].lower() if ded_fld else None
    ded_type = ded_fld['DeductibleType'] if ded_fld else 'B'
    shr_fld = lgcp[1].get('share')
    shr_elm = shr_fld['ProfileElementName'].lower() if shr_fld else None

    for it, i in itertools.chain((it, i) for i, (key, group) in enumerate(itertools.groupby(li, key=lambda it: tuple(it[k] for k in agg_key))) for it in group):
        it['agg_id'] = i + 1

        can_item = get_can_item(it['canexp_id'], it['canacc_id'], it['policy_num'])

        it['lim_elm'] = lim_elm
        can_item_lim = can_item.get(lim_elm) or 0.0
        it['limit'] = (can_item_lim if can_item_lim >= 1 else it['tiv']*can_item_lim) or 0.0

        it['ded_elm'] = ded_elm
        can_item_ded = can_item.get(ded_elm) or 0.0
        it['deductible'] = (can_item_ded if can_item_ded >= 1 else it['tiv']*can_item_ded) or 0.0
        it['deductible_type'] = ded_type

        it['shr_elm'] = shr_elm
        it['share'] = can_item.get(shr_elm) or 0.0

        it['calcrule_id'] = get_calcrule_id(it['limit'], it['share'], it['deductible_type'])

        yield it


def get_fm_terms_by_level_as_list(level_grouped_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df):

    level_id = level_fm_items[0]['level_id']

    return (
        list(get_coverage_level_fm_terms(level_grouped_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df)) if level_id == 1
        else list(get_non_coverage_level_fm_terms(level_grouped_canonical_profile, level_fm_agg_profile, level_fm_items, canexp_df, canacc_df))
    )


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
