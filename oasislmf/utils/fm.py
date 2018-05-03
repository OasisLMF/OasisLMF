# -*- coding: utf-8 -*-

__all__ = [
    'canonical_profiles_fm_terms',
    'canonical_profiles_grouped_fm_terms',
    'get_calc_rule',
    'get_deductible',
    'get_deductible_type',
    'get_fm_terms',
    'get_fm_terms_by_level',
    'get_fm_terms_by_level_as_list',
    'get_limit',
    'get_policytc_id',
    'get_policytc_ids',
    'get_share'
]

import io
import itertools
import json

from .exceptions import OasisException


def canonical_profiles_fm_terms(canonical_profiles=(), canonical_profiles_paths=()):
    if not (canonical_profiles or canonical_profiles_paths):
        raise OasisException('A list of canonical profiles (loc. or acc.) or a list of canonical profiles paths must be provided')

    if not canonical_profiles:
        for f in canonical_profiles_paths:
            with io.open(canonical_profile_path, 'r', encoding='utf-8') as f:
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


def canonical_profiles_grouped_fm_terms(canonical_profiles=(), canonical_profiles_paths=()):
    if not (canonical_profiles or canonical_profiles_paths):
        raise OasisException('A list of canonical profiles (loc. or acc.) or a list of canonical profiles paths must be provided')

    fm_terms = canonical_profiles_fm_terms(canonical_profiles=canonical_profiles, canonical_profiles_paths=canonical_profiles_paths)

    fm_levels = sorted(fm_terms.keys())

    return {
        level_id: {
            k:{gi['FMTermType'].lower():gi for gi in list(g)} for k, g in itertools.groupby(sorted(fm_terms[level_id].values(), key=lambda f: f['ProfileElementName']), key=lambda f: f['FMTermGroupID'])
        } for level_id in fm_levels
    }


def get_calc_rule(limit, share, ded_type):

    if limit == share == 0.0 and ded_type == 'B':
        return 12
    elif limit == 0.0 and share > 0 and ded_type == 'B':
        return 15
    elif limit > 0 and share == 0 and ded_type == 'B':
        return 1
    elif ded_type == 'MI':
        return 11
    elif ded_type == 'MA':
        return 10
    else:
        return 2


def get_deductible(canonical_profiles_grouped_fm_terms, canexp_item, canacc_item, fm_item):

    gfmt = canonical_profiles_grouped_fm_terms

    level_id = fm_item['level_id']
    tiv = fm_item['tiv']

    is_coverage_level = any('tiv' in gfmt[level_id][gid] for gid in gfmt[level_id])

    can_item = None
    ded_field = None

    if is_coverage_level:
        for gid in gfmt[level_id]:
            tiv_field = gfmt[level_id][gid]['tiv']
            tiv_field_name = tiv_field['ProfileElementName'].lower()

            if tiv_field_name in canexp_item and float(canexp_item[tiv_field_name]) == float(tiv):
                ded_field = gfmt[level_id][gid]['deductible'] if 'deductible' in gfmt[level_id][gid] else None
                break
    else:
        ded_field = gfmt[level_id][1]['deductible'] if 'deductible' in gfmt[level_id][1] else None

    if not ded_field:
        return 0.0

    can_item = canexp_item if ded_field['ProfileType'].lower() == 'loc' else canacc_item
    
    ded_field_name = ded_field['ProfileElementName'].lower()
    ded_val = float(can_item[ded_field_name]) if ded_field_name in can_item else 0.0

    return ded_val


def get_deductible_type(canonical_profiles_grouped_fm_terms, canexp_item, canacc_item, fm_item):

    gfmt = canonical_profiles_grouped_fm_terms

    level_id = fm_item['level_id']
    tiv = fm_item['tiv']

    is_coverage_level = any('tiv' in gfmt[level_id][gid] for gid in gfmt[level_id])

    can_item = None
    ded_field = None

    if is_coverage_level:
        for gid in gfmt[level_id]:
            tiv_field = gfmt[level_id][gid]['tiv']
            tiv_field_name = tiv_field['ProfileElementName'].lower()

            if tiv_field_name in canexp_item and float(canexp_item[tiv_field_name]) == float(tiv):
                ded_field = gfmt[level_id][gid]['deductible'] if 'deductible' in gfmt[level_id][gid] else None
                break
    else:
        ded_field = gfmt[level_id][1]['deductible'] if 'deductible' in gfmt[level_id][1] else None

    return ded_field['DeductibleType'] if ded_field else u'B'


def get_fm_terms(canonical_profiles_grouped_fm_terms, canexp_item, canacc_item, fm_item):

    gfmt = canonical_profiles_grouped_fm_terms

    level_id = fm_item['level_id']
    tiv = fm_item['tiv']

    is_coverage_level = any('tiv' in gfmt[level_id][gid] for gid in gfmt[level_id])
    
    if is_coverage_level:
        for gid in gfmt[level_id]:
            tiv_field = gfmt[level_id][gid]['tiv']
            tiv_field_name = tiv_field['ProfileElementName'].lower()

            if tiv_field_name in canexp_item and float(canexp_item[tiv_field_name]) == float(tiv):
                limit_field = gfmt[level_id][gid]['limit'] if 'limit' in gfmt[level_id][gid] else None
                ded_field = gfmt[level_id][gid]['deductible'] if 'deductible' in gfmt[level_id][gid] else None
                ded_type = ded_field['DeductibleType'] if ded_field else u'B'
                share_field = gfmt[level_id][gid]['share'] if 'share' in gfmt[level_id][gid] else None
                break
    else:
            limit_field = gfmt[level_id][1]['limit'] if 'limit' in gfmt[level_id][1] else None
            ded_field = gfmt[level_id][1]['deductible'] if 'deductible' in gfmt[level_id][1] else None
            ded_type = ded_field['DeductibleType'] if ded_field else u'B'
            share_field = gfmt[level_id][1]['share'] if 'share' in gfmt[level_id][1] else None

    can_item = None

    fm_terms = {
        'index': fm_item['index'],
        'item_id': fm_item['item_id'],
        'tiv': tiv,
        'limit': 0.0,
        'deductible': 0.0,
        'deductible_type': u'B',
        'share': 0.0,
        'calcrule_id': 2
    }

    if limit_field:
        can_item = canexp_item if limit_field['ProfileType'].lower() == 'loc' else canacc_item
        limit_field_name = limit_field['ProfileElementName'].lower()
        limit_val = float(can_item[limit_field_name]) if limit_field_name in can_item else 0.0
        fm_terms['limit'] = limit_val

    if ded_field:
        can_item = canexp_item if ded_field['ProfileType'].lower() == 'loc' else canacc_item
        ded_field_name = ded_field['ProfileElementName'].lower()
        ded_val = float(can_item[ded_field_name]) if ded_field_name in can_item else 0.0
        fm_terms['deductible'] = ded_val

    fm_terms['deductible_type'] = ded_type

    if share_field:
        can_item = canexp_item if share_field['ProfileType'].lower() == 'loc' else canacc_item
        share_field_name = share_field['ProfileElementName'].lower()
        share_val = float(can_item[share_field_name]) if share_field_name in can_item else 0.0
        fm_terms['share'] = share_val

    fm_terms['calcrule_id'] = get_calc_rule(fm_terms['limit'], fm_terms['share'], fm_terms['deductible_type'])

    return fm_terms


def get_coverage_level_terms(coverage_level_id, coverage_level_grouped_fm_terms, canexp_df, canacc_df, fm_df):

    clgfmt = coverage_level_grouped_fm_terms

    get_canexp_item = lambda i: canexp_df.iloc[int(fm_df.iloc[i]['canexp_id'])]

    get_item_layer = lambda i: list(canacc_df[canacc_df['accntnum'] == get_canexp_item(i)['accntnum']]['policynum'].values)[int(fm_df.iloc[i]['layer_id']) - 1]

    get_canacc_item = lambda i: canacc_df[canacc_df['accntnum'] == get_canexp_item(i)['accntnum']][canacc_df['policynum'] == get_item_layer(i)]

    coverage_level_fm_df = pd.DataFrame(fm_df.where(fm_df['level_id'] == coverage_level_id).drop_duplicates().dropna(), dtype=object)

    for i in range(len(coverage_level_fm_df)):
        fm_item = coverage_level_fm_df.iloc[i]

        tiv = fm_item['tiv']

        canexp_item = get_canexp_item(i)
        canacc_item = get_canacc_item(i)

        for gid in clgfmt:
                tiv_field = clgfmt[gid]['tiv']
                tiv_field_name = tiv_field['ProfileElementName'].lower()

                if tiv_field_name in canexp_item and float(canexp_item[tiv_field_name]) == float(tiv):
                    limit_field = clgfmt[gid]['limit'] if 'limit' in clgfmt[gid] else None
                    ded_field = clgfmt[gid]['deductible'] if 'deductible' in clgfmt[gid] else None
                    ded_type = ded_field['DeductibleType'] if ded_field else u'B'
                    share_field = clgfmt[gid]['share'] if 'share' in clgfmt[gid] else None
                    break

        can_item = None

        fm_terms = {
            'index': int(fm_item['index']),
            'item_id': int(fm_item['item_id']),
            'tiv': tiv,
            'limit': 0.0,
            'deductible': 0.0,
            'deductible_type': u'B',
            'share': 0.0,
            'calcrule_id': 2
        }

        if limit_field:
            can_item = canexp_item if limit_field['ProfileType'].lower() == 'loc' else canacc_item
            limit_field_name = limit_field['ProfileElementName'].lower()
            limit_val = float(can_item[limit_field_name]) if limit_field_name in can_item else 0.0
            fm_terms['limit'] = limit_val

        if ded_field:
            can_item = canexp_item if ded_field['ProfileType'].lower() == 'loc' else canacc_item
            ded_field_name = ded_field['ProfileElementName'].lower()
            ded_val = float(can_item[ded_field_name]) if ded_field_name in can_item else 0.0
            fm_terms['deductible'] = ded_val

        fm_terms['deductible_type'] = ded_type

        if share_field:
            can_item = canexp_item if share_field['ProfileType'].lower() == 'loc' else canacc_item
            share_field_name = share_field['ProfileElementName'].lower()
            share_val = float(can_item[share_field_name]) if share_field_name in can_item else 0.0
            fm_terms['share'] = share_val

        fm_terms['calcrule_id'] = get_calc_rule(fm_terms['limit'], fm_terms['share'], fm_terms['deductible_type'])

        yield fm_terms


def get_fm_terms_by_level(level_id, canonical_profiles_grouped_fm_terms, canexp_df, canacc_df, fm_df):

    gfmt = canonical_profiles_grouped_fm_terms

    is_coverage_level = lambda level_id: any('tiv' in gfmt[level_id][gid] for gid in gfmt[level_id])

    if is_coverage_level(level_id):
        for fmt in get_coverage_level_terms(level_id, gfmt[level_id], canexp_df, canacc_df, fm_df):
            yield fmt
    else:
        limit_field = gfmt[level_id][1]['limit'] if 'limit' in gfmt[level_id][1] else None
        ded_field = gfmt[level_id][1]['deductible'] if 'deductible' in gfmt[level_id][1] else None
        ded_type = ded_field['DeductibleType'] if ded_field else u'B'
        share_field = gfmt[level_id][1]['share'] if 'share' in gfmt[level_id][1] else None

        level_fm_df = pd.DataFrame(fm_df.where(fm_df['level_id'] == level_id).drop_duplicates().dropna(), dtype=object)

        get_canexp_item = lambda i: canexp_df.iloc[int(level_fm_df.iloc[i]['canexp_id'])]

        get_item_layer = lambda i: list(canacc_df[canacc_df['accntnum'] == get_canexp_item(i)['accntnum']]['policynum'].values)[int(level_fm_df.iloc[i]['layer_id']) - 1]

        get_canacc_item = lambda i: canacc_df[canacc_df['accntnum'] == get_canexp_item(i)['accntnum']][canacc_df['policynum'] == get_item_layer(i)]

        for i in range(len(level_fm_df)):

            fm_item = level_fm_df.iloc[i]

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
                limit_field_name = limit_field['ProfileElementName'].lower()
                limit_val = float(can_item[limit_field_name]) if limit_field_name in can_item else 0.0
                fm_terms['limit'] = limit_val

            if ded_field:
                can_item = canexp_item if ded_field['ProfileType'].lower() == 'loc' else canacc_item
                ded_field_name = ded_field['ProfileElementName'].lower()
                ded_val = float(can_item[ded_field_name]) if ded_field_name in can_item else 0.0
                fm_terms['deductible'] = ded_val

            fm_terms['deductible_type'] = ded_type

            if share_field:
                can_item = canexp_item if share_field['ProfileType'].lower() == 'loc' else canacc_item
                share_field_name = share_field['ProfileElementName'].lower()
                share_val = float(can_item[share_field_name]) if share_field_name in can_item else 0.0
                fm_terms['share'] = share_val

            fm_terms['calcrule_id'] = get_calc_rule(fm_terms['limit'], fm_terms['share'], fm_terms['deductible_type'])

            yield fm_terms


def get_fm_terms_by_level_as_list(level_id, canonical_profiles_grouped_fm_terms, canexp_df, canacc_df, fm_df):
    return list(get_fm_terms_by_level(level_id, canonical_profiles_grouped_fm_terms, canexp_df, canacc_df, fm_df))


def get_limit(canonical_profiles_grouped_fm_terms, canexp_item, canacc_item, fm_item):

    gfmt = canonical_profiles_grouped_fm_terms

    level_id = fm_item['level_id']
    tiv = fm_item['tiv']

    is_coverage_level = any('tiv' in gfmt[level_id][gid] for gid in gfmt[level_id])

    can_item = None
    limit_field = None

    if is_coverage_level:
        for gid in gfmt[level_id]:
            tiv_field  = gfmt[level_id][gid]['tiv']
            tiv_field_name = tiv_field['ProfileElementName'].lower()

            if tiv_field_name in canexp_item and float(canexp_item[tiv_field_name]) == float(tiv):
                limit_field = gfmt[level_id][gid]['limit'] if 'limit' in gfmt[level_id][gid] else None
                break
    else:
        limit_field = gfmt[level_id][1]['limit'] if 'limit' in gfmt[level_id][1] else None

    if not limit_field:
        return 0.0

    can_item = canexp_item if limit_field['ProfileType'].lower() == 'loc' else canacc_item
    
    limit_field_name = limit_field['ProfileElementName'].lower()
    limit_val = float(can_item[limit_field_name]) if limit_field_name in can_item else 0.0

    return limit_val


def get_policytc_id(fm_item, policytc_ids):

    for i in policytc_ids:
        if (
            fm_item['limit'] == policytc_ids[i]['limit'] and
            fm_item['deductible'] == policytc_ids[i]['deductible'] and
            fm_item['share'] == policytc_ids[i]['share'] and
            fm_item['calcrule_id'] == policytc_ids[i]['calcrule_id']
        ):
            return i

    return

def get_policytc_ids(fm_df):

    columns = [
        col for col in fm_df.columns if not col in ['limit', 'deductible', 'share', 'calcrule_id']
    ]

    policytc_df = fm_df.drop(columns, axis=1).drop_duplicates()
    
    for col in policytc_df.columns:
        policytc_df[col] = policytc_df[col].astype(float) if col != 'calcrule_id' else policytc_df[col].astype(int)

    policytc_df['index'] = range(1, len(policytc_df) + 1)

    policytc_ids = {
        i: {
            'limit': policytc_df.iloc[i - 1]['limit'],
            'deductible': policytc_df.iloc[i - 1]['deductible'],
            'share': policytc_df.iloc[i - 1]['share'],
            'calcrule_id': policytc_df.iloc[i - 1]['calcrule_id']
        } for i in policytc_df['index']
    }

    return policytc_ids

def get_share(canonical_profiles_grouped_fm_terms, canexp_item, canacc_item, fm_item):

    gfmt = canonical_profiles_grouped_fm_terms

    level_id = fm_item['level_id']
    tiv = fm_item['tiv']

    is_coverage_level = any('tiv' in gfmt[level_id][gid] for gid in gfmt[level_id])

    can_item = None
    share_field = None

    if is_coverage_level:
        for gid in gfmt[level_id]:
            tiv_field = gfmt[level_id][gid]['tiv']
            tiv_field_name = tiv_field['ProfileElementName'].lower()

            if tiv_field_name in canexp_item and float(canexp_item[tiv_field_name]) == float(tiv):
                share_field = gfmt[level_id][gid]['share'] if 'share' in gfmt[level_id][gid] else None
                break
    else:
        share_field = gfmt[level_id][1]['share'] if 'share' in gfmt[level_id][1] else None

    if not share_field:
        return 0.0

    can_item = canexp_item if share_field['ProfileType'].lower() == 'loc' else canacc_item
    
    share_field_name = share_field['ProfileElementName'].lower()
    share_val = float(can_item[share_field_name]) if share_field_name in can_item else 0.0

    return share_val
