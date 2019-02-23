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
    'unified_fm_terms_by_level_and_term_group',
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

from collections import OrderedDict
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
from ..utils.data import (
    get_dataframe,
    merge_dataframes,
)
from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.metadata import OED_FM_LEVELS
from ..utils.defaults import (
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


def unified_fm_profile_by_level_and_term_group(profiles=[], profile_paths=[], unified_profile_by_level=None):

    if not (profiles or profile_paths or unified_profile_by_level):
        raise OasisException('A list of source profiles (loc. or acc.), or source profile paths, or a unified FM profile grouped by level, must be provided')

    ufp = unified_profile_by_level

    if ufp:
        return {
            k: {
                _k: {v['FMTermType'].lower(): v for v in g} for _k, g in groupby(sorted(viewvalues(ufp[k]), key=lambda v: v['FMTermGroupID']), key=lambda v: v['FMTermGroupID'])
            } for k in ufp
        }

    ufp = unified_fm_profile_by_level(profiles=profiles, profile_paths=profile_paths)

    return {
        k: {
            _k: {v['FMTermType'].lower(): v for v in g} for _k, g in groupby(sorted(viewvalues(ufp[k]), key=lambda v: v['FMTermGroupID']), key=lambda v: v['FMTermGroupID'])
        } for k in ufp
    }


def unified_fm_terms_by_level_and_term_group(profiles=[], profile_paths=[], unified_profile_by_level_and_term_group=None):

    ufp = unified_profile_by_level_and_term_group

    if not (profiles or profile_paths or unified_profile_by_level_and_term_group):
        raise OasisException('A list of source profiles (loc. or acc.), or source profile paths, or a unified FM profile grouped by level and term group, must be provided')

    if ufp:
        return OrderedDict({
            level: {
                tiv_tgid: {
                    term_type: (
                        ufp[level][tiv_tgid][term_type]['ProfileElementName'].lower() if ufp[level][tiv_tgid].get(term_type) else None
                    ) for term_type in ('deductible', 'deductiblemin', 'deductiblemax', 'limit', 'share',)
                } for tiv_tgid in ufp[level]
            } for level in ufp
        })

    ufp = unified_fm_profile_by_level_and_term_group(profiles=profiles, profile_paths=profile_paths)

    return OrderedDict({
        level: {
            tiv_tgid: {
                term_type: (
                    ufp[level][tiv_tgid][term_type]['ProfileElementName'].lower() if ufp[level][tiv_tgid].get(term_type) else None
                ) for term_type in ('deductible', 'deductiblemin', 'deductiblemax', 'limit', 'share',)
            } for tiv_tgid in ufp[level]
        } for level in ufp
    })


@oasis_log
def generate_il_input_items(
    exposure_df,
    accounts_df,
    gul_inputs_df,
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile(),
    fm_aggregation_profile=get_default_fm_aggregation_profile()
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

    :param fm_aggregation_profile: FM aggregation profile
    :param fm_aggregation_profile: dict
    """
    cep = exposure_profile
    cap = accounts_profile

    ufp = unified_fm_profile_by_level_and_term_group(profiles=(cep, cap,))

    if not ufp:
        raise OasisException(
            'Canonical loc. and/or acc. profiles are possibly missing FM term information: '
            'FM term definitions for TIV, limit, deductible and/or share.'
        )

    fmap = fm_aggregation_profile

    if not fmap:
        raise OasisException(
            'FM aggregation profile is empty - this is required to perform aggregation'
        )

    fm_levels = sorted(ufp.keys())

    cov_level = min(fm_levels)
    layer_level = max(fm_levels)

    try:
        for df in [exposure_df, gul_inputs_df, accounts_df]:
            df['index'] = df.get('index', range(len(df)))

        expgul_df = merge_dataframes(
            exposure_df,
            gul_inputs_df,
            left_on='locnumber',
            right_on='loc_id',
            how='outer'
        )

        il_inputs_df = merge_dataframes(
            expgul_df,
            accounts_df,
            left_on='acc_id',
            right_on='accnumber',
            how='outer'
        )

        layers = OrderedDict({
            (k, p): tuple(v['polnumber'].unique()).index(p) + 1 for k, v in il_inputs_df.groupby(['accnumber']) for p in v['polnumber'].unique()
        })

        il_inputs_df['layer_id'] = il_inputs_df['index'].apply(lambda i: layers[(il_inputs_df.iloc[i]['accnumber'], il_inputs_df.iloc[i]['polnumber'])])

        il_inputs_df = il_inputs_df[il_inputs_df['layer_id'] == 1].reset_index()
        il_inputs_df['index'] = il_inputs_df.index

        il_inputs_df['item_id'] = il_inputs_df['gul_item_id'] = il_inputs_df['index'].apply(lambda i: i + 1)

        def convert_fractional_terms_to_wholes(df, terms):
            for term in terms:
                def prod(_df):
                    return _df['tiv'] * _df[term]
                df['temp'] = prod(df[['tiv', term]].where(df[term] < 1))
                df[term] = df[[term, 'temp']].max(axis=1)

            return df.drop('temp', axis=1)

        terms = ['deductible', 'deductible_min', 'deductible_max', 'limit']

        il_inputs_df = convert_fractional_terms_to_wholes(il_inputs_df, terms)

        n = len(il_inputs_df)

        il_inputs_df['level_id'] = [cov_level] * n

        il_inputs_df['agg_id'] = [-1] * n
        il_inputs_df['attachment'] = [0] * n
        il_inputs_df['calcrule_id'] = [-1] * n

        il_inputs_df['share'] = [0] * n
        il_inputs_df['policytc_id'] = [-1] * n

        fm_terms = unified_fm_terms_by_level_and_term_group(unified_profile_by_level_and_term_group=ufp)

        def set_non_coverage_level_fm_terms(df, level, terms, term_defaults=None):
            term_defaults = term_defaults or {t:0.0 for t in terms}
            for term in terms:
                df[term] = df.get(fm_terms[level][1].get(term), [term_defaults.get(term) or 0.0] * n)
            return df

        for level in fm_levels[1:-1]:
            level_df = il_inputs_df[il_inputs_df['level_id'] == 1].copy(deep=True)
            level_df['level_id'] = [level] * n
            set_non_coverage_level_fm_terms(level_df, level, terms)
            convert_fractional_terms_to_wholes(level_df, terms)
            il_inputs_df = pd.concat([il_inputs_df, level_df], ignore_index=True)


        il_inputs_df['index'] = il_inputs_df.index
        il_inputs_df['item_id'] = il_inputs_df['index'].apply(lambda i: i + 1)

        layer_df = merge_dataframes(
            il_inputs_df[il_inputs_df['level_id'] == 1],
            accounts_df,
            left_on='acc_id',
            right_on='accnumber',
            how='outer'
        )        
        layer_df['level_id'] = [layer_level] * len(layer_df)
        layer_df['layer_id'] = layer_df['index'].apply(lambda i: layers[(layer_df.iloc[i]['accnumber'], layer_df.iloc[i]['polnumber'])])

        n = len(layer_df)

        terms += ['share']
        set_non_coverage_level_fm_terms(layer_df, layer_level, terms, term_defaults={'limit': 9999999999, 'share': 1.0})
        terms.remove('share')
        layer_df = convert_fractional_terms_to_wholes(layer_df, terms)

        il_inputs_df = pd.concat([il_inputs_df, layer_df], ignore_index=True)
        il_inputs_df['index'] = il_inputs_df.index
        il_inputs_df['item_id'] = il_inputs_df['index'].apply(lambda i: i + 1)
        il_inputs_df['policy_num'] = il_inputs_df['polnumber']

        import ipdb; ipdb.set_trace()

        agg_keys = {
            level: tuple(v['field'].lower() for v in viewvalues(fmap[level]['FMAggKey']))
            for level in fm_levels
        }


    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
        raise OasisException(e)

        """
            for it in (it for c in chain(viewvalues(preset_items[k]) for k in preset_items) for it in c):
                it['policy_num'] = accounts_df.iloc[it['acc_id']]['polnumber']
                lfmaggkey = fmap[it['level_id']]['FMAggKey']
                for v in viewvalues(lfmaggkey):
                    src = v['src'].lower()
                    if src in ['loc', 'acc']:
                        f = v['field'].lower()
                        it[f] = exposure_df.iloc[it['loc_id']][f] if src == 'loc' else accounts_df.iloc[it['acc_id']][f]

            concurrent_tasks = (
                Task(get_il_terms_by_level_as_list, args=(ufp[level_id], fmap[level_id], preset_items[level_id], exposure_df.copy(deep=True), accounts_df.copy(deep=True),), key=level_id)
                for level_id in fm_levels
            )
            num_ps = min(len(fm_levels), multiprocessing.cpu_count())
            for it in multiprocess(concurrent_tasks, pool_size=num_ps):
                yield it
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
            raise OasisException(e)
        """


@oasis_log
def get_il_input_items(
    exposure_df,
    accounts_df,
    gul_inputs_df,
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile(),
    fm_aggregation_profile=get_default_fm_aggregation_profile(),
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

    :param fm_aggregation_profile: FM aggregation profile
    :param fm_aggregation_profile: dict

    :param reduced: Whether to reduce the FM input items table by removing any
                    items with zero financial terms
    :param reduced: bool
    """
    cep = exposure_profile
    cap = accounts_profile
    fmap = fm_aggregation_profile

    try:
        il_items = [
            it for it in generate_il_input_items(
                exposure_df,
                accounts_df,
                gul_inputs_df,
                exposure_profile=cep,
                accounts_profile=cap,
                fm_aggregation_profile=fmap
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


@oasis_log
def write_il_input_files(
    exposure_df,
    gul_inputs_df,
    accounts_fp,
    target_dir,
    exposure_profile=get_default_exposure_profile(),
    accounts_profile=get_default_accounts_profile(),
    fm_aggregation_profile=get_default_fm_aggregation_profile(),
    oasis_files_prefixes={
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
        fm_aggregation_profile=fm_aggregation_profile
    )

    il_input_files = {
        k: os.path.join(target_dir, '{}.csv'.format(oasis_files_prefixes[k])) for k in viewkeys(oasis_files_prefixes)
    }

    concurrent_tasks = (
        Task(getattr(sys.modules[__name__], 'write_{}_file'.format(f)), args=(il_inputs_df.copy(deep=True), il_input_files[f],), key=f)
        for f in il_input_files
    )
    num_ps = min(len(il_input_files), multiprocessing.cpu_count())
    for _, _ in multithread(concurrent_tasks, pool_size=num_ps):
        pass

    return il_input_files, il_inputs_df, accounts_df
