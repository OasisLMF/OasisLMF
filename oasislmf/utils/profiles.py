__all__ = [
    'get_fm_terms_oed_columns',
    'get_grouped_fm_profile_by_level',
    'get_grouped_fm_profile_by_level_and_term_group',
    'get_grouped_fm_terms_by_level_and_term_group',
    'get_oed_hierarchy_terms'
]


from collections import OrderedDict
from itertools import groupby

from .defaults import (
    SUPPORTED_FM_LEVELS,
    FM_TERMS,
    get_default_exposure_profile,
    get_default_accounts_profile,
)


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

    grouped_fm_term_types = OrderedDict({
        'deductible': FM_TERMS['deductible']['id'],
        'deductiblemin': FM_TERMS['min deductible']['id'],
        'deductiblemax': FM_TERMS['max deductible']['id'],
        'limit': FM_TERMS['limit']['id'],
        'share': FM_TERMS['share']['id']
    })

    return OrderedDict({
        k: OrderedDict({
            _k: OrderedDict({
                (grouped_fm_term_types.get(v['FMTermType'].lower()) or v['FMTermType'].lower()): v for v in g
            }) for _k, g in groupby(sorted(grouped[k].values(), key=lambda v: v['FMTermGroupID']), key=lambda v: v['FMTermGroupID'])
        }) for k in sorted(grouped)
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
        level_id: OrderedDict({
            tgid: OrderedDict({
                term_type: (
                    (
                        grouped[level_id][tgid][term_type]['ProfileElementName'].lower() if lowercase
                        else grouped[level_id][tgid][term_type]['ProfileElementName']
                    ) if grouped[level_id][tgid].get(term_type) else None
                ) for term_type in [v['id'] for v in FM_TERMS.values()]
            }) for tgid in grouped[level_id]
        }) for level_id in sorted(grouped)[1:]
    })


def get_fm_terms_oed_columns(
    fm_terms=get_grouped_fm_terms_by_level_and_term_group(),
    levels=list(SUPPORTED_FM_LEVELS.keys()),
    level_ids=None,
    term_group_ids=[1],
    terms=[v['id'] for v in FM_TERMS.values()],
    remove_nulls=True
):
    level_ids = level_ids or [SUPPORTED_FM_LEVELS[k]['id'] for k in levels]
    _fm_terms = OrderedDict({
        level_id: level_terms_dict
        for level_id, level_terms_dict in fm_terms.items()
        if level_id in level_ids
    })

    cols = [
        _fm_terms[level_id][tgid].get(term)
        for level_id in level_ids
        for tgid in term_group_ids
        for term in terms
    ]

    return cols if not remove_nulls else [col for col in cols if col]


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
    hierarchy_terms.setdefault('locgrp', 'locgroup' if lowercase else 'LocGroup')
    hierarchy_terms.setdefault('accid', 'accnumber' if lowercase else 'AccNumber')
    hierarchy_terms.setdefault('polid', 'polnumber' if lowercase else 'PolNumber')
    hierarchy_terms.setdefault('portid', 'portnumber' if lowercase else 'PortNumber')
    hierarchy_terms.setdefault('condid', 'condnumber' if lowercase else 'CondNumber')
    hierarchy_terms.setdefault('locperilscovered', 'locperilscovered' if lowercase else 'LocPerilsCovered')

    return hierarchy_terms
