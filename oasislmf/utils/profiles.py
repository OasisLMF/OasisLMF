__all__ = [
    'get_fm_terms_oed_columns',
    'get_grouped_fm_profile_by_level',
    'get_grouped_fm_profile_by_level_and_term_group',
    'get_grouped_fm_terms_by_level_and_term_group',
    'get_oed_hierarchy',
    'get_step_policies_oed_mapping'
]


from collections import OrderedDict
from itertools import groupby

from .defaults import (
    get_default_exposure_profile,
    get_default_accounts_profile,
    get_default_step_policies_profile,
)
from .fm import (
    SUPPORTED_FM_LEVELS,
    FM_TERMS,
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
        'deductiblecode': FM_TERMS['deductible code']['id'],
        'deductibletype': FM_TERMS['deductible type']['id'],
        'deductiblemin': FM_TERMS['min deductible']['id'],
        'deductiblemax': FM_TERMS['max deductible']['id'],
        'limit': FM_TERMS['limit']['id'],
        'limitcode': FM_TERMS['limit code']['id'],
        'limittype': FM_TERMS['limit type']['id'],
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
        }) for level_id in sorted(grouped)
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


def get_oed_hierarchy(
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile()
):
    return {v['Key'].lower(): v for k, v in {**exposure_profile, **accounts_profile}.items() if v.get('OEDHierarchy')}


def get_step_policies_oed_mapping(step_trigger_type, only_cols=False):

    step_policies_profile = get_default_step_policies_profile()

    if only_cols is True:
        cols = []
        for k, v in step_policies_profile.items():
            if step_trigger_type in v['FMProfileStep']:
                cols.append(v['Key'].lower())
        return cols
    else:
        oed_mapping = {}
        for k, v in step_policies_profile.items():
            if step_trigger_type in v['FMProfileStep']:
                oed_mapping[v['FMProfileField']] = v['Key'].lower()
        return oed_mapping
