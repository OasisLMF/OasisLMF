__all__ = [
    'get_fm_level_term_oed_columns',
    'get_grouped_fm_profile_by_level',
    'get_grouped_fm_profile_by_level_and_term_group',
    'get_grouped_fm_terms_by_level_and_term_group',
    'get_oed_hierarchy_terms'
]


from collections import OrderedDict
from itertools import groupby

from .defaults import (
    FM_LEVELS,
    get_default_exposure_profile,
    get_default_accounts_profile,
)
from .exceptions import OasisException


def get_fm_level_term_oed_columns(level_keys=[], level_ids=[]):
    if not (level_keys or level_ids):
        raise OasisException('An iterable of either FM level keys or IDs is required')

    _level_ids = level_ids.copy() or [FM_LEVELS[k]['id'] for k in level_keys]
    fm_terms = get_grouped_fm_terms_by_level_and_term_group()
    _level_ids = [l for l in _level_ids if l in fm_terms.keys()]

    if not _level_ids:
        raise OasisException('Level keys/IDs provided are not contained in the default FM profiles')

    return [
        t for l in _level_ids for t in fm_terms[l][1].values() if t
    ]


def get_grouped_fm_profile_by_level(
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile()
):
    exp_prof_fm_keys = {k: v for k, v in exposure_profile.items() if 'FMLevel' in v}
    acc_prof_fm_keys = {k: v for k, v in accounts_profile.items() if 'FMLevel' in v}

    comb_prof = {**exp_prof_fm_keys, **acc_prof_fm_keys}

    return OrderedDict({
        int(k): {v['ProfileElementName']: v for v in g}
        for k, g in groupby(sorted(comb_prof.values(), key=lambda v: v['FMLevel']), key=lambda v: v['FMLevel'])
    })


def get_grouped_fm_profile_by_level_and_term_group(
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile(),
    grouped_profile_by_level=None
):
    grouped = grouped_profile_by_level or get_grouped_fm_profile_by_level(exposure_profile, accounts_profile)

    grouped_fm_term_types = {
        'deductible': 'deductible',
        'deductiblemin': 'deductible_min',
        'deductiblemax': 'deductible_max',
        'limit': 'limit',
        'share': 'share'
    }

    return OrderedDict({
        k: {
            _k: {
                (grouped_fm_term_types.get(v['FMTermType'].lower()) or v['FMTermType'].lower()): v for v in g
            } for _k, g in groupby(sorted(grouped[k].values(), key=lambda v: v['FMTermGroupID']), key=lambda v: v['FMTermGroupID'])
        } for k in sorted(grouped)
    })


def get_grouped_fm_terms_by_level_and_term_group(
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile(),
    grouped_profile_by_level=None,
    grouped_profile_by_level_and_term_group=None,
    lowercase=True
):

    grouped = (
        grouped_profile_by_level_and_term_group or
        get_grouped_fm_profile_by_level_and_term_group(exposure_profile, accounts_profile, grouped_profile_by_level)
    )

    return OrderedDict({
        level_id: {
            tgid: {
                term_type: (
                    (
                        grouped[level_id][tgid][term_type]['ProfileElementName'].lower() if lowercase
                        else grouped[level_id][tgid][term_type]['ProfileElementName']
                    ) if grouped[level_id][tgid].get(term_type) else None
                ) for term_type in ('deductible', 'deductible_min', 'deductible_max', 'limit', 'share',)
            } for tgid in grouped[level_id]
        } for level_id in sorted(grouped)[1:]
    })


def get_oed_hierarchy_terms(
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile(),
    grouped_profile_by_level=None,
    grouped_profile_by_level_and_term_group=None,
    lowercase=True
):
    grouped = (
        grouped_profile_by_level_and_term_group or
        get_grouped_fm_profile_by_level_and_term_group(exposure_profile, accounts_profile, grouped_profile_by_level)
    )

    hierarchy_terms = OrderedDict({
        k.lower(): (v['ProfileElementName'].lower() if lowercase else v['ProfileElementName'])
        for k, v in sorted(grouped[0][1].items())
    })
    hierarchy_terms.setdefault('locid', ('locnumber' if lowercase else 'LocNumber'))
    hierarchy_terms.setdefault('accid', 'accnumber' if lowercase else 'AccNumber')
    hierarchy_terms.setdefault('polid', 'polnumber' if lowercase else 'PolNumber')
    hierarchy_terms.setdefault('portid', 'portnumber' if lowercase else 'PortNumber')
    hierarchy_terms.setdefault('condid', 'condnumber' if lowercase else 'CondNumber')

    return hierarchy_terms
