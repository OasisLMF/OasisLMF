__all__ = [
    'load_portfolio',
    'check_keys',
    'check_items',
    'check_amplifications',
    'check_lookup_dicts',
]

import json
import os

import numpy as np
import pandas as pd

from ...pytools.common.data import areaperil_int
from ...pytools.common.input_files import read_amplifications, read_coverages
from ...pytools.gulmc.items import read_items
from .report import WARNING

ID_COLUMN_CHECKS = ('vulnerability_id', 'area_peril_id', 'areaperil_id', 'amplification_id')


def _read_keys_file(input_dir, name):
    for ext, reader in (('parquet', pd.read_parquet), ('csv', pd.read_csv)):
        path = os.path.join(input_dir, f'{name}.{ext}')
        if os.path.exists(path):
            return reader(path)
    return None


def load_portfolio(report, input_dir):
    portfolio = {
        'keys': _read_keys_file(input_dir, 'keys'),
        'keys_errors': _read_keys_file(input_dir, 'keys-errors'),
        'items': None,
        'coverages': None,
    }
    try:
        items = np.array(read_items(input_dir))
        coverages = np.asarray(read_coverages(input_dir))
        portfolio['items'], portfolio['coverages'] = items, coverages
    except (FileNotFoundError, RuntimeError) as e:
        report.error('inputs.exists', str(e))
    return portfolio


def check_keys(report, keys, keys_errors, model_settings):
    if keys_errors is not None and len(keys_errors):
        status = keys_errors.groupby(['PerilID', 'Status']).size()
        top_messages = keys_errors['Message'].value_counts().head(5)
        report.warning('keys.lookup_failures', f'{len(keys_errors)} location/peril/coverage rows failed the lookup; they are excluded from the run',
                       [f'{p} {s}: {n}' for (p, s), n in status.items()] + [f'"{m}": {n}' for m, n in top_messages.items()],
                       count=len(keys_errors))
    else:
        report.ok('keys.lookup_failures')
    if keys is None or not len(keys) or not {'AreaPerilID', 'VulnerabilityID'}.issubset(keys.columns):
        return

    max_ap = np.iinfo(areaperil_int).max
    report.missing('keys.areaperil_range', f'AreaPerilID outside 1..{max_ap} (OASIS_AREAPERIL_TYPE={areaperil_int}); items.bin truncates it',
                   keys.loc[(keys['AreaPerilID'] < 1) | (keys['AreaPerilID'] > max_ap), 'AreaPerilID'].unique().tolist())
    report.missing('keys.vulnerability_range', 'VulnerabilityID must be positive',
                   keys.loc[keys['VulnerabilityID'] < 1, 'VulnerabilityID'].unique().tolist())

    supported = {p['id'] for p in model_settings.get('lookup_settings', {}).get('supported_perils', [])}
    if supported:
        unknown = set(keys['PerilID'].unique()) - supported
        if model_settings.get('model_settings', {}).get('correlation_settings') or model_settings.get('correlation_settings'):
            report.missing('keys.supported_perils', 'keys contain perils not in lookup_settings.supported_perils; '
                           'with correlation_settings these items are silently dropped from the run', unknown)
        else:
            report.missing('keys.supported_perils', 'keys contain perils not in lookup_settings.supported_perils', unknown, level=WARNING)


def check_items(report, items, coverages, vuln, agg_map, footprint, cond_ids=()):
    if items is None or not len(items):
        return

    report.missing('items.coverage_range', f'coverage_id outside 1..{len(coverages)} (out-of-bounds write in gulmc)',
                   np.unique(items['coverage_id'][(items['coverage_id'] < 1) | (items['coverage_id'] > len(coverages))]).tolist())
    report.missing('items.areaperil_zero', 'items with areaperil_id 0 are never mapped (silent zero loss)',
                   items['item_id'][items['areaperil_id'] == 0].tolist())
    report.missing('coverages.tiv', 'coverages with TIV <= 0 produce zero loss', (np.flatnonzero(coverages <= 0) + 1).tolist(), level=WARNING)

    if vuln is not None:
        item_vulns = np.unique(items['vulnerability_id'])
        known = np.union1d(np.union1d(vuln.available_ids, np.array(sorted(agg_map), dtype=np.int64)), cond_ids)
        report.missing('items.vulnerability_exists', f'item vulnerability ids missing from {vuln.source} (the run raises)',
                       np.setdiff1d(item_vulns, known).tolist())

    if footprint is not None and footprint.hit_areaperils is not None and footprint.events_checked:
        not_hit = ~np.isin(items['areaperil_id'], footprint.hit_areaperils)
        if not_hit.all():
            if footprint.sampled:
                report.warning('items.hit_by_events', f'no item areaperil is hit by any of the {footprint.events_checked} sampled events '
                               '(--check-max-events 0 checks every event)')
            else:
                report.error('items.hit_by_events', 'no item areaperil is hit by any checked event: every loss will be zero')
        elif not_hit.any():
            valid_cov = (items['coverage_id'] >= 1) & (items['coverage_id'] <= len(coverages))
            tiv = coverages[items['coverage_id'][valid_cov] - 1]
            unhit_tiv = coverages[items['coverage_id'][valid_cov & not_hit] - 1].sum()
            report.warning('items.hit_by_events',
                           f'{int(not_hit.sum())} of {len(items)} items ({unhit_tiv / max(tiv.sum(), 1e-12):.1%} of item TIV) '
                           f'are not hit by any of the {footprint.events_checked} checked events',
                           np.unique(items['areaperil_id'][not_hit]).tolist(), count=int(not_hit.sum()))
        else:
            report.ok('items.hit_by_events')


def check_amplifications(report, input_dir, lossfactor_amp_ids, pla_enabled):
    amp_fp = os.path.join(input_dir, 'amplifications.bin')
    if not os.path.exists(amp_fp):
        if pla_enabled:
            report.error('amplifications.exists', 'pla is enabled in analysis settings but no amplifications.bin was generated '
                         '(keys have no AmplificationID?)')
        return
    try:
        amps = read_amplifications(input_dir)
    except ValueError as e:
        report.error('amplifications.format', str(e))
        return
    if not pla_enabled:
        return
    if lossfactor_amp_ids is None:
        report.error('amplifications.lossfactors', 'pla is enabled but the model has no lossfactors.bin')
        return
    used = np.unique(amps[amps > 0])
    report.missing('amplifications.lossfactors', 'amplification ids never appear in lossfactors.bin (factor 1.0 applied silently)',
                   np.setdiff1d(used, lossfactor_amp_ids).tolist(), level=WARNING)


def _lookup_dict_files(lookup_config, config_dir):
    keys_data_path = os.path.join(config_dir, lookup_config.get('keys_data_path', ''))
    for step_name, step in lookup_config.get('step_definition', {}).items():
        params = step.get('parameters', {})
        if 'file_path' not in params:
            continue
        id_columns = [c.lower() for c in params.get('id_columns', [])]
        if not any(c in ID_COLUMN_CHECKS for c in id_columns):
            continue
        path = params['file_path'].replace('%%KEYS_DATA_PATH%%', keys_data_path)
        if not os.path.isabs(path):
            path = os.path.join(config_dir, path)
        yield step_name, os.path.normpath(path), params.get('file_type', os.path.splitext(path)[1].lstrip('.')), id_columns


def check_lookup_dicts(report, lookup_config_json, vuln_available_ids, agg_ids, lossfactor_amp_ids, cond_ids=()):
    if not lookup_config_json or not os.path.exists(lookup_config_json):
        report.info('lookup.dicts', 'no built-in lookup config; lookup dictionaries not checked (generated keys are still checked)')
        return
    with open(lookup_config_json) as f:
        lookup_config = json.load(f)

    for step_name, path, file_type, id_columns in _lookup_dict_files(lookup_config, os.path.dirname(os.path.abspath(lookup_config_json))):
        check = f'lookup.{step_name}'
        if not os.path.exists(path):
            report.error(f'{check}.exists', f'lookup file not found: {path}')
            continue
        df = pd.read_parquet(path) if file_type == 'parquet' else pd.read_csv(path)
        df.columns = df.columns.str.lower()
        for col in id_columns:
            if col not in df.columns:
                report.error(f'{check}.{col}', f'{os.path.basename(path)} has no {col} column')
                continue
            ids = df[col].dropna()
            if col == 'vulnerability_id' and vuln_available_ids is not None:
                known = np.union1d(np.union1d(vuln_available_ids, np.asarray(sorted(agg_ids), dtype=np.int64)), cond_ids)
                report.missing(f'{check}.vulnerability_id',
                               f'{os.path.basename(path)} maps to vulnerability ids missing from the vulnerability file; '
                               'any location that hits them fails the loss run', np.setdiff1d(ids.unique(), known).tolist(), level=WARNING)
            elif col in ('area_peril_id', 'areaperil_id'):
                max_ap = np.iinfo(areaperil_int).max
                report.missing(f'{check}.areaperil_range', f'{os.path.basename(path)} has area peril ids outside 1..{max_ap}',
                               ids[(ids < 1) | (ids > max_ap)].unique().tolist())
            elif col == 'amplification_id' and lossfactor_amp_ids is not None:
                report.missing(f'{check}.amplification_id',
                               f'{os.path.basename(path)} has amplification ids that never appear in lossfactors.bin (factor 1.0)',
                               np.setdiff1d(ids.unique(), lossfactor_amp_ids).tolist(), level=WARNING)
