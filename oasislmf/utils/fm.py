# -*- coding: utf-8 -*-

__all__ = [
    'canonical_profiles_fm_terms_grouped_by_level',
    'canonical_profiles_fm_terms_grouped_by_level_and_term_type',
    'get_calcrule_id',
    'get_fm_terms_by_level',
    'get_fm_terms_by_level_as_list',
    'get_fm_terms_by_level2',
    'get_fm_terms_by_level_as_list2',
    'get_policytc_id',
    'get_policytc_ids'
]

import io
import itertools
import json

import pandas as pd

from .exceptions import OasisException


def canonical_profiles_fm_terms_grouped_by_level(canonical_profiles=[], canonical_profiles_paths=[]):

    if not (canonical_profiles or canonical_profiles_paths):
        raise OasisException('A list of canonical profiles (loc. or acc.) or a list of canonical profiles paths must be provided')

    if not canonical_profiles:
        for p in canonical_profiles_paths:
            with io.open(p, 'r', encoding='utf-8') as f:
                canonical_profiles.append(json.load(f))

    cp = dict((k,v) for p in canonical_profiles for k, v in p.items())

    return {
        level_id: {
            gi['ProfileElementName'].lower(): gi for gi in list(g)
        } for level_id, g in itertools.groupby(
            sorted(
               [v for v in cp.values() if 'FMLevel' in v and v['FMTermType'].lower() in ['tiv', 'deductible', 'limit', 'share']], 
                key=lambda d: d['FMLevel']
            ),
            key=lambda f: f['FMLevel']
        )
    }


def canonical_profiles_fm_terms_grouped_by_level_and_term_type(canonical_profiles=[], canonical_profiles_paths=[]):

    if not (canonical_profiles or canonical_profiles_paths):
        raise OasisException('A list of canonical profiles (loc. or acc.) or a list of canonical profiles paths must be provided')

    fm_terms = canonical_profiles_fm_terms_grouped_by_level(canonical_profiles=canonical_profiles, canonical_profiles_paths=canonical_profiles_paths)

    fm_levels = sorted(fm_terms.keys())

    return {
        level_id: {
            k:{gi['FMTermType'].lower():gi for gi in list(g)} for k, g in itertools.groupby(sorted(fm_terms[level_id].values(), key=lambda f: f['ProfileElementName']), key=lambda f: f['FMTermGroupID'])
        } for level_id in fm_levels
    }


def get_calcrule_id(limit, share, ded_type):

    if limit == share == 0 and ded_type == 'B':
        return 12
    elif limit == 0 and share > 0 and ded_type == 'B':
        return 15
    elif limit > 0 and share == 0 and ded_type == 'B':
        return 1
    elif ded_type == 'MI':
        return 11
    elif ded_type == 'MA':
        return 10
    else:
        return 2


def get_coverage_level_terms(coverage_level_id, coverage_level_grouped_fm_terms, canexp_df, canacc_df, level_fm_df):

    clgfmt = coverage_level_grouped_fm_terms

    get_canexp_item = lambda i: canexp_df.iloc[int(level_fm_df.loc[i]['canexp_id'])]

    get_canacc_item = lambda i: canacc_df.iloc[int(level_fm_df.loc[i]['canacc_id'])]

    limit_field = None
    ded_field = None
    ded_type = None
    share_field = None

    for i, fm_item in level_fm_df.iterrows():
        tiv = fm_item['tiv']

        canexp_item = get_canexp_item(i)
        canacc_item = get_canacc_item(i)

        cf = [v for v in clgfmt.values() if canexp_item.get(v['tiv']['ProfileElementName'].lower()) and canexp_item[v['tiv']['ProfileElementName'].lower()] == tiv][0]

        fm_terms = {
            'level_id': coverage_level_id,
            'index': int(fm_item['index']),
            'item_id': int(fm_item['item_id']),
            'tiv': tiv,
            'limit': 0.0,
            'deductible': 0.0,
            'deductible_type': u'B',
            'share': 0.0,
            'calcrule_id': 2
        }

        if cf.get('limit'):
            limit_field_name = cf['limit']['ProfileElementName'].lower()
            limit_val = float(canexp_item.get(limit_field_name)) if canexp_item.get(limit_field_name) else 0.0
            fm_terms['limit'] = limit_val

        if cf.get('deductible'):
            ded_field_name = cf['deductible']['ProfileElementName'].lower()
            ded_val = float(canexp_item.get(ded_field_name)) if canexp_item.get(ded_field_name) else 0.0
            fm_terms['deductible'] = ded_val

            fm_terms['deductible_type'] = cf['deductible']['DeductibleType']

        if cf.get('share'):
            share_field_name = cf['share']['ProfileElementName'].lower()
            share_val = float(canexp_item.get(share_field_name)) if canexp_item.get(share_field_name) else 0.0
            fm_terms['share'] = share_val

        fm_terms['calcrule_id'] = get_calcrule_id(fm_terms['limit'], fm_terms['share'], fm_terms['deductible_type'])

        yield fm_terms


def get_fm_terms_by_level(level_id, level_grouped_fm_terms, canexp_df, canacc_df, level_fm_df):

    lgfmt = level_grouped_fm_terms

    if level_id == 1:
        for fmt in get_coverage_level_terms(level_id, lgfmt, canexp_df, canacc_df, level_fm_df):
            yield fmt
    else:
        limit_field = lgfmt[1].get('limit')
        limit_field_name = limit_field['ProfileElementName'].lower() if limit_field else None

        ded_field = lgfmt[1].get('deductible')
        ded_field_name = ded_field['ProfileElementName'].lower() if ded_field else None

        ded_type = ded_field['DeductibleType'] if ded_field else u'B'

        share_field = lgfmt[1].get('share')
        share_field_name = share_field['ProfileElementName'].lower() if share_field else None

        get_canexp_item = lambda i: canexp_df.iloc[int(level_fm_df.loc[i]['canexp_id'])]

        get_canacc_item = lambda i: canacc_df.iloc[int(level_fm_df.loc[i]['canacc_id'])]

        for i, fm_item in level_fm_df.iterrows():

            canexp_item = get_canexp_item(i)
            canacc_item = get_canacc_item(i)

            can_item = None

            fm_terms = {
                'level_id': level_id,
                'index': int(fm_item['index']),
                'item_id': int(fm_item['item_id']),
                'tiv': fm_item['tiv'],
                'limit': 0.0,
                'deductible': 0.0,
                'deductible_type': u'B',
                'share': 0.0,
                'calcrule_id': 3
            }

            if limit_field:
                can_item = canexp_item if limit_field['ProfileType'].lower() == 'loc' else canacc_item
                limit_val = float(can_item[limit_field_name])
                fm_terms['limit'] = limit_val

            if ded_field:
                can_item = canexp_item if ded_field['ProfileType'].lower() == 'loc' else canacc_item
                ded_val = float(can_item[ded_field_name])
                fm_terms['deductible'] = ded_val

            fm_terms['deductible_type'] = ded_type

            if share_field:
                can_item = canexp_item if share_field['ProfileType'].lower() == 'loc' else canacc_item
                share_val = float(can_item[share_field_name])
                fm_terms['share'] = share_val

            fm_terms['calcrule_id'] = get_calcrule_id(fm_terms['limit'], fm_terms['share'], fm_terms['deductible_type'])

            yield fm_terms


def get_fm_terms_by_level_as_list(level_id, level_grouped_fm_terms, canexp_df, canacc_df, level_fm_df):

    return list(get_fm_terms_by_level(level_id, level_grouped_fm_terms, canexp_df, canacc_df, level_fm_df))


def get_fm_terms_by_level2(combined_grouped_canonical_profile, level_fm_items, canexp_df, canacc_df):

    lid = level_fm_items[0]['level_id']

    lp = combined_grouped_canonical_profile[lid]

    can_df = pd.merge(canexp_df, canacc_df, left_on='accntnum', right_on='accntnum')

    get_can_item = lambda canexp_id, canacc_id, layer_id: can_df[(can_df['row_id_x']==canexp_id+1) & (can_df['row_id_y']==canacc_id+1) & (can_df['policynum']=='Layer{}'.format(layer_id))].iloc[0]

    lim = 0
    ded = 0
    ded_type = 'B'
    shr = 0
    calcrule_id = 2

    for _, it in enumerate(level_fm_items):
        can_item = get_can_item(it['canexp_id'], it['canacc_id'], it['layer_id'])
        tgid = it['tiv_tgid'] if lid == 1 else 1

        lim_elm = lp[tgid].get('limit')
        if lim_elm:
            lim = it['limit'] = float(can_item[lim_elm['ProfileElementName'].lower()])

        ded_elm = lp[tgid].get('deductible')
        if ded_elm:
            ded = it['deductible'] = float(can_item[ded_elm['ProfileElementName'].lower()])
            ded_type = it['deductible_type'] = ded_elm['DeductibleType']

        shr_elm = lp[tgid].get('share')
        if shr_elm:
            shr = it['share'] = float(can_item[shr_elm['ProfileElementName'].lower()])

        it['calcrule_id'] = get_calcrule_id(lim, shr, ded_type)

        yield it


def get_fm_terms_by_level_as_list2(combined_grouped_canonical_profile, level_fm_items, canexp_df, canacc_df):
    return list(get_fm_terms_by_level2(combined_grouped_canonical_profile, level_fm_items, canexp_df, canacc_df))


def get_policytc_id(fm_item, policytc_ids):

    terms = {
        'limit': fm_item['limit'],
        'deductible': fm_item['deductible'],
        'share': fm_item['share'],
        'calcrule_id': fm_item['calcrule_id']
    }

    for i, v in policytc_ids.items():
        if terms == v:
            return i

def get_policytc_ids(fm_items_df):

    columns = [
        col for col in fm_items_df.columns if not col in ('limit', 'deductible', 'share', 'calcrule_id',)
    ]

    policytc_df = fm_items_df.drop(columns, axis=1).drop_duplicates()
    
    for col in policytc_df.columns:
        policytc_df[col] = policytc_df[col].astype(float) if col != 'calcrule_id' else policytc_df[col].astype(int)

    policytc_df['index'] = list(range(1, len(policytc_df) + 1))

    policytc_ids = {
        i: {
            'limit': policytc_df.iloc[i - 1]['limit'],
            'deductible': policytc_df.iloc[i - 1]['deductible'],
            'share': policytc_df.iloc[i - 1]['share'],
            'calcrule_id': int(policytc_df.iloc[i - 1]['calcrule_id'])
        } for i in policytc_df['index']
    }

    return policytc_ids
