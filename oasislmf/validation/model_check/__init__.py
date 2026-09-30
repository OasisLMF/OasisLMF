__all__ = [
    'CheckReport',
    'run_model_check',
]

import os

import numpy as np

from ...pytools.getmodel.manager import get_conditional_vuln_ids
from ...pytools.gulmc.aggregate import read_aggregate_vulnerability
from .inputs import check_amplifications, check_items, check_keys, check_lookup_dicts, load_portfolio
from .model_files import (check_aggregate_vulnerability, check_damage_bins, check_event_files, check_vulnerability,
                          load_vulnerability, read_lossfactor_amplification_ids, scan_footprint)
from .report import CheckReport
from .settings import check_settings, resolve_run_files


def _vulnerability_ids_to_load(items, model_storage):
    if items is None:
        return None
    ids = set(np.unique(items['vulnerability_id']).tolist())
    agg = read_aggregate_vulnerability(model_storage)
    if agg is not None:
        ids |= set(agg['vulnerability_id'][np.isin(agg['aggregate_vulnerability_id'], list(ids))].tolist())
    return sorted(ids)


def run_model_check(report, run_dir, model_storage, analysis_settings, model_settings, merged_settings,
                    lookup_config_json=None, dynamic_footprint=False, max_events=None, full_model=False, check_portfolio=True):
    """Check that the model data, settings and generated inputs in ``run_dir`` form a runnable analysis.

    ``run_dir`` must be laid out like a loss run: oasis files in ``input/`` and the model data linked in ``static/``.
    """
    input_dir = os.path.join(run_dir, 'input')
    static_dir = os.path.join(run_dir, 'static')

    check_settings(report, analysis_settings, model_settings, merged_settings)
    run_files = resolve_run_files(report, merged_settings, run_dir, model_storage)
    damage_bins = check_damage_bins(report, model_storage)

    portfolio = {'items': None, 'coverages': None}
    if check_portfolio:
        portfolio = load_portfolio(report, input_dir)
        check_keys(report, portfolio['keys'], portfolio['keys_errors'], merged_settings)
    items = portfolio['items']

    event_ids = check_event_files(report, run_files)
    footprint = scan_footprint(report, model_storage, run_dir, event_ids,
                               portfolio_areaperils=None if items is None else np.unique(items['areaperil_id']),
                               dynamic=dynamic_footprint, max_events=max_events)

    vuln = load_vulnerability(report, model_storage, None if full_model else _vulnerability_ids_to_load(items, model_storage))
    agg_map = {}
    if vuln is not None:
        if vuln.num_intensity_bins is not None and footprint.num_intensity_bins is not None \
                and vuln.num_intensity_bins != footprint.num_intensity_bins:
            report.error('vulnerability.intensity_bins', f'vulnerability parquet meta has {vuln.num_intensity_bins} intensity bins '
                         f'but the footprint has {footprint.num_intensity_bins} (reshape fails at run time)')
        zero_bin_is_point = damage_bins is not None and len(damage_bins) and damage_bins['bin_to'][0] == 0
        check_vulnerability(report, vuln, damage_bins, footprint.num_intensity_bins, zero_bin_is_point=zero_bin_is_point)
        agg_map = check_aggregate_vulnerability(report, model_storage, vuln.available_ids,
                                                None if items is None else np.unique(items['vulnerability_id']))

    cond_ids = get_conditional_vuln_ids(model_storage)
    amp_ids = read_lossfactor_amplification_ids(static_dir)
    if check_portfolio:
        check_items(report, items, portfolio['coverages'], vuln, agg_map, footprint, cond_ids)
        check_amplifications(report, input_dir, amp_ids, bool(merged_settings.get('pla')))
    check_lookup_dicts(report, lookup_config_json, None if vuln is None else vuln.available_ids, agg_map, amp_ids, cond_ids)
    return report
