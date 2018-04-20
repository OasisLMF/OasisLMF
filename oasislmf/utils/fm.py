# -*- coding: utf-8 -*-

__all__ = [
    'canonical_profiles_fm_terms',
    'canonical_profiles_grouped_fm_terms',
    'get_calc_rule',
    'get_deductible',
    'get_deductible_type',
    'get_fm_terms',
    'get_limit',
    'get_share'
]

import io
import itertools
import json

from .exceptions import OasisException


def canonical_profiles_fm_terms(canonical_profiles=[], canonical_profiles_paths=[]):
    if not (canonical_profiles or canonical_profiles_paths):
        raise OasisException('A list of canonical profiles (loc. or acc.) or a list of canonical profiles paths must be provided')

    if not canonical_profiles:
        for f in canonical_profiles_paths:
            with io.open(canonical_profile_path, 'r', encoding='utf-8') as f:
                canonical_profiles.append(json.load(f))

    cp = dict((k,v) for p in canonical_profiles for k, v in p.items())

    return {
        fm_level: {
            gi['ProfileElementName'].lower(): gi for gi in list(g)
        } for fm_level, g in itertools.groupby(
            sorted(
               [v for v in cp.values() if 'FMLevel' in v and v['FMTermType'].lower() in ['tiv', 'deductible', 'limit', 'share']], 
                key=lambda d: d['FMLevel']
            ),
            key=lambda f: f['FMLevel']
        )
    }


def canonical_profiles_grouped_fm_terms(canonical_profiles=[], canonical_profiles_paths=[]):
    if not (canonical_profiles or canonical_profiles_paths):
        raise OasisException('A list of canonical profiles (loc. or acc.) or a list of canonical profiles paths must be provided')

    fm_terms = canonical_profiles_fm_terms(canonical_profiles=canonical_profiles, canonical_profiles_paths=canonical_profiles_paths)

    fm_levels = sorted(fm_terms.keys())

    return {
        fm_level: {
            k:{gi['FMTermType'].lower():gi for gi in list(g)} for k, g in itertools.groupby(sorted(fm_terms[fm_level].values(), key=lambda f: f['ProfileElementName']), key=lambda f: f['FMTermGroupID'])
        } for fm_level in fm_levels
    }


def get_limit(canexp_item, canacc_item, fm_item, canonical_profiles_grouped_fm_terms):

    gfmt = canonical_profiles_grouped_fm_terms

    fm_level = fm_item['fm_level']
    tiv = fm_item['tiv']

    is_coverage_level = any('tiv' in gfmt[fm_level][gid] for gid in gfmt[fm_level])

    can_item = None
    limit_field = None

    if is_coverage_level:
        for gid in gfmt[fm_level]:
            tiv_field = gfmt[fm_level][gid]['tiv']
            tiv_field_name = tiv_field['ProfileElementName'].lower()

            if tiv_field_name in canexp_item and float(canexp_item[tiv_field_name]) == float(tiv):
                limit_field = gfmt[fm_level][gid]['limit'] if 'limit' in gfmt[fm_level][gid] else None
                break
    else:
        limit_field = gfmt[fm_level][1]['limit'] if 'limit' in gfmt[fm_level][1] else None

    if not limit_field:
        return 0.0

    can_item = canexp_item if limit_field['ProfileType'].lower() == 'loc' else canacc_item
    
    limit_field_name = limit_field['ProfileElementName'].lower()
    limit_val = float(can_item[limit_field_name]) if limit_field_name in can_item else 0.0

    return limit_val


def get_deductible(canexp_item, canacc_item, fm_item, canonical_profiles_grouped_fm_terms):

    gfmt = canonical_profiles_grouped_fm_terms

    fm_level = fm_item['fm_level']
    tiv = fm_item['tiv']

    is_coverage_level = any('tiv' in gfmt[fm_level][gid] for gid in gfmt[fm_level])

    can_item = None
    ded_field = None

    if is_coverage_level:
        for gid in gfmt[fm_level]:
            tiv_field = gfmt[fm_level][gid]['tiv']
            tiv_field_name = tiv_field['ProfileElementName'].lower()

            if tiv_field_name in canexp_item and float(canexp_item[tiv_field_name]) == float(tiv):
                ded_field = gfmt[fm_level][gid]['deductible'] if 'deductible' in gfmt[fm_level][gid] else None
                break
    else:
        ded_field = gfmt[fm_level][1]['deductible'] if 'deductible' in gfmt[fm_level][1] else None

    if not ded_field:
        return 0.0

    can_item = canexp_item if ded_field['ProfileType'].lower() == 'loc' else canacc_item
    
    ded_field_name = ded_field['ProfileElementName'].lower()
    ded_val = float(can_item[ded_field_name]) if ded_field_name in can_item else 0.0

    return ded_val


def get_deductible_type(canexp_item, canacc_item, fm_item, canonical_profiles_grouped_fm_terms):

    gfmt = canonical_profiles_grouped_fm_terms

    fm_level = fm_item['fm_level']
    tiv = fm_item['tiv']

    is_coverage_level = any('tiv' in gfmt[fm_level][gid] for gid in gfmt[fm_level])

    can_item = None
    ded_field = None

    if is_coverage_level:
        for gid in gfmt[fm_level]:
            tiv_field = gfmt[fm_level][gid]['tiv']
            tiv_field_name = tiv_field['ProfileElementName'].lower()

            if tiv_field_name in canexp_item and float(canexp_item[tiv_field_name]) == float(tiv):
                ded_field = gfmt[fm_level][gid]['deductible'] if 'deductible' in gfmt[fm_level][gid] else None
                break
    else:
        ded_field = gfmt[fm_level][1]['deductible'] if 'deductible' in gfmt[fm_level][1] else None

    return ded_field['DeductibleType'] if ded_field else u'B'


def get_share(canexp_item, canacc_item, fm_item, canonical_profiles_grouped_fm_terms):

    gfmt = canonical_profiles_grouped_fm_terms

    fm_level = fm_item['fm_level']
    tiv = fm_item['tiv']

    is_coverage_level = any('tiv' in gfmt[fm_level][gid] for gid in gfmt[fm_level])

    can_item = None
    share_field = None

    if is_coverage_level:
        for gid in gfmt[fm_level]:
            tiv_field = gfmt[fm_level][gid]['tiv']
            tiv_field_name = tiv_field['ProfileElementName'].lower()

            if tiv_field_name in canexp_item and float(canexp_item[tiv_field_name]) == float(tiv):
                share_field = gfmt[fm_level][gid]['share'] if 'share' in gfmt[fm_level][gid] else None
                break
    else:
        share_field = gfmt[fm_level][1]['share'] if 'share' in gfmt[fm_level][1] else None

    if not share_field:
        return 0.0

    can_item = canexp_item if share_field['ProfileType'].lower() == 'loc' else canacc_item
    
    share_field_name = share_field['ProfileElementName'].lower()
    share_val = float(can_item[share_field_name]) if share_field_name in can_item else 0.0

    return share_val


def get_calc_rule(limit_val, share_val, ded_type):
    if limit_val == share_val == 0 and ded_type == 'BB':
        return 12
    elif limit_val == 0 and share_val > 0 and ded_type == 'B':
        return 15
    elif limit_val > 0 and share_val == 0 and ded_type == 'B':
        return 1
    elif ded_type == 'MI':
        return 11
    elif ded_type == 'MA':
        return 10
    else:
        return 2


def get_fm_terms(canexp_item, canacc_item, fm_item, canonical_profiles_grouped_fm_terms):

    gfmt = canonical_profiles_grouped_fm_terms

    fm_level = fm_item['fm_level']
    tiv = fm_item['tiv']

    is_coverage_level = any('tiv' in gfmt[fm_level][gid] for gid in gfmt[fm_level])

    can_item = None
    
    fm_terms = {
        'tiv': tiv,
        'limit': 0.0,
        'deductible': 0.0,
        'deductible_type': u'B',
        'share': 0.0,
        'calc_rule': 2,
    }
    
    if is_coverage_level:
        for gid in gfmt[fm_level]:
            tiv_field = gfmt[fm_level][gid]['tiv']
            tiv_field_name = tiv_field['ProfileElementName'].lower()

            if tiv_field_name in canexp_item and float(canexp_item[tiv_field_name]) == float(tiv):
                limit_field = gfmt[fm_level][gid]['limit'] if 'limit' in gfmt[fm_level][gid] else None
                ded_field = gfmt[fm_level][gid]['deductible'] if 'deductible' in gfmt[fm_level][gid] else None
                ded_type = ded_field['DeductibleType'] if ded_field else u'B'
                share_field = gfmt[fm_level][gid]['share'] if 'share' in gfmt[fm_level][gid] else None
                break
    else:
            limit_field = gfmt[fm_level][1]['limit'] if 'limit' in gfmt[fm_level][1] else None
            ded_field = gfmt[fm_level][1]['deductible'] if 'deductible' in gfmt[fm_level][1] else None
            ded_type = ded_field['DeductibleType'] if ded_field else u'B'
            share_field = gfmt[fm_level][1]['share'] if 'share' in gfmt[fm_level][1] else None

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

    calc_rule = get_calc_rule(fm_terms['limit'], fm_terms['share'], fm_terms['deductible_type'])
    fm_terms['calc_rule'] = calc_rule

    return fm_terms


