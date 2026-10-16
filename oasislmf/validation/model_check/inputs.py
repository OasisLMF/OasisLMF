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

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from ...pytools.common.data import areaperil_int, vulnerability_dtype
from ...pytools.common.input_files import read_amplifications, read_coverages
from ...pytools.gulmc.items import read_items
from .report import WARNING

ID_COLUMN_CHECKS = ('vulnerability_id', 'area_peril_id', 'areaperil_id', 'amplification_id')
LOOKUP_FILE_PARAMS = ('file_path', 'file_type', 'id_columns')


def _read_keys_file(input_dir, name):
    for ext, reader in (('parquet', pd.read_parquet), ('csv', pd.read_csv)):
        path = os.path.join(input_dir, f'{name}.{ext}')
        if os.path.exists(path):
            return reader(path)
    return None


def load_portfolio(report, input_dir):
    """Load the generated keys, keys errors, items and coverages from ``input_dir``.

    Reports:
        ``inputs.exists``: items or coverages are missing or unreadable (error).

    Args:
        report (CheckReport): report the findings are added to.
        input_dir (str): the run's ``input/`` directory.

    Returns:
        dict: ``keys``, ``keys_errors``, ``items`` and ``coverages``; each is None if not available.
    """
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
    """Check the generated keys and keys errors.

    Reports:
        ``keys.lookup_failures``: rows that failed the lookup and are excluded from the run, grouped by peril and status (warning).
        ``keys.areaperil_range``: ``AreaPerilID`` outside the ``OASIS_AREAPERIL_TYPE`` range (error).
        ``keys.vulnerability_range``: ``VulnerabilityID`` < 1 (error).
        ``keys.supported_perils``: perils not in ``lookup_settings.supported_perils``; error with correlation settings
        (items are dropped), warning otherwise.

    Args:
        report (CheckReport): report the findings are added to.
        keys (pd.DataFrame | None): the generated keys.
        keys_errors (pd.DataFrame | None): the generated keys errors.
        model_settings (dict): the model settings file contents.
    """
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
    """Check items and coverages against the model data.

    Reports:
        ``items.coverage_range``: ``coverage_id`` outside the coverages file (error).
        ``items.areaperil_zero``: items with areaperil 0, which give zero loss (error).
        ``coverages.tiv``: coverages with TIV <= 0 (warning).
        ``items.vulnerability_exists``: item vulnerability ids in no vulnerability, aggregate or conditional file (error).
        ``items.hit_by_events``: items whose areaperil no checked event hits; error if none are hit, warning otherwise.

    Args:
        report (CheckReport): report the findings are added to.
        items (np.ndarray | None): the items.
        coverages (np.ndarray | None): TIV per coverage.
        vuln (VulnerabilityData | None): the loaded vulnerability data.
        agg_map (dict): aggregate id -> sub-vulnerability ids.
        footprint (FootprintSummary | None): result of ``scan_footprint``.
        cond_ids (array-like): conditional vulnerability ids.
    """
    if items is None or not len(items):
        return

    report.missing('items.coverage_range', f'coverage_id outside 1..{len(coverages)} (out-of-bounds write in gulmc)',
                   np.unique(items['coverage_id'][(items['coverage_id'] < 1) | (items['coverage_id'] > len(coverages))]).tolist())
    report.missing('items.areaperil_zero', 'items with areaperil_id 0 are never mapped (silent zero loss)',
                   items['item_id'][items['areaperil_id'] == 0].tolist())
    report.missing('coverages.tiv', 'coverages with TIV <= 0 produce zero loss', (np.flatnonzero(coverages <= 0) + 1).tolist(), level=WARNING)

    if vuln is not None:
        item_vulns = np.unique(items['vulnerability_id'])
        known = np.union1d(np.union1d(vuln.available_ids, np.array(sorted(agg_map), dtype=vulnerability_dtype['vulnerability_id'])), cond_ids)
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
    """Check the amplifications file against the model's loss factors when post loss amplification is on.

    Reports:
        ``amplifications.exists``: pla is enabled but no ``amplifications.bin`` was generated (error).
        ``amplifications.format``: ``amplifications.bin`` cannot be read (error).
        ``amplifications.lossfactors``: pla is enabled but the model has no ``lossfactors.bin`` (error), or amplification
        ids are missing from it, so a factor of 1.0 is applied (warning).

    Args:
        report (CheckReport): report the findings are added to.
        input_dir (str): the run's ``input/`` directory.
        lossfactor_amp_ids (np.ndarray | None): amplification ids in ``lossfactors.bin``.
        pla_enabled (bool): whether post loss amplification is enabled.
    """
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


def _resolve_lookup_path(lookup_config, config_dir, file_path):
    """Resolve a lookup step file path as the built-in lookup does: placeholders and relative paths use ``keys_data_path``."""
    keys_data_path = os.path.join(config_dir, lookup_config.get('keys_data_path', ''))
    path = file_path.replace('%%KEYS_DATA_PATH%%', keys_data_path)
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(keys_data_path, path))


def _lookup_dict_files(lookup_config, config_dir):
    for step_name, step in lookup_config.get('step_definition', {}).items():
        params = step.get('parameters', {})
        if 'file_path' not in params:
            continue
        id_columns = [c.lower() for c in params.get('id_columns', [])]
        if not any(c in ID_COLUMN_CHECKS for c in id_columns):
            continue
        yield step_name, step.get('type'), params, _resolve_lookup_path(lookup_config, config_dir, params['file_path']), id_columns


def _read_lookup_file(step_type, params, path):
    """Read a lookup step's file with the reader and options the step itself uses.

    ``rtree`` reads with ``geopandas.read_<file_type>`` and ``file_read_params``; other steps (``merge``) read with
    ``pandas.read_<file_type>`` and their remaining parameters, falling back to ``read_csv``.
    """
    file_type = params.get('file_type', os.path.splitext(path)[1].lstrip('.'))
    if step_type == 'rtree' and gpd is not None and hasattr(gpd, f'read_{file_type}'):
        read_params = params.get('file_read_params') or params.get('area_peril_read_params') or {}
        return getattr(gpd, f'read_{file_type}')(path, **read_params)
    read_params = {} if step_type == 'rtree' else {k: v for k, v in params.items() if k not in LOOKUP_FILE_PARAMS}
    reader = getattr(pd, f'read_{file_type}', None)
    return (reader if callable(reader) else pd.read_csv)(path, **read_params)


def check_lookup_dicts(report, lookup_config_json, vuln_available_ids, agg_ids, lossfactor_amp_ids, cond_ids=()):
    """Check the files that built-in lookup steps read ids from against the model data.

    Only steps with a ``file_path`` and an ``id_columns`` entry the model data can be checked against are covered
    (in practice ``merge``, ``rtree`` and ``h3``); computed ids are covered by ``check_keys``.

    Reports:
        ``lookup.dicts``: no built-in lookup config, so nothing is checked (info).
        ``lookup.<step>.exists``: the step's file is missing (error).
        ``lookup.<step>.read``: the step's file cannot be read with the step's reader and options (error).
        ``lookup.<step>.<column>``: the file lacks a declared id column (error).
        ``lookup.<step>.vulnerability_id``: vulnerability ids missing from the model (warning).
        ``lookup.<step>.areaperil_range``: area peril ids outside the ``OASIS_AREAPERIL_TYPE`` range (error).
        ``lookup.<step>.amplification_id``: amplification ids missing from ``lossfactors.bin`` (warning).

    Args:
        report (CheckReport): report the findings are added to.
        lookup_config_json (str | None): path to the built-in lookup config.
        vuln_available_ids (np.ndarray | None): ids present in the vulnerability file.
        agg_ids (iterable): aggregate vulnerability ids.
        lossfactor_amp_ids (np.ndarray | None): amplification ids in ``lossfactors.bin``.
        cond_ids (array-like): conditional vulnerability ids.
    """
    if not lookup_config_json or not os.path.exists(lookup_config_json):
        report.info('lookup.dicts', 'no built-in lookup config; lookup dictionaries not checked (generated keys are still checked)')
        return
    with open(lookup_config_json) as f:
        lookup_config = json.load(f)

    config_dir = os.path.dirname(os.path.abspath(lookup_config_json))
    for step_name, step_type, params, path, id_columns in _lookup_dict_files(lookup_config, config_dir):
        check = f'lookup.{step_name}'
        if not os.path.exists(path):
            report.error(f'{check}.exists', f'lookup file not found: {path}')
            continue
        try:
            df = _read_lookup_file(step_type, params, path)
        except Exception as e:
            report.error(f'{check}.read', f'{os.path.basename(path)} could not be read as the {step_type} step reads it: {type(e).__name__}: {e}')
            continue
        df.columns = df.columns.str.lower()
        for col in id_columns:
            if col not in df.columns:
                report.error(f'{check}.{col}', f'{os.path.basename(path)} has no {col} column')
                continue
            ids = df[col].dropna()
            if col == 'vulnerability_id' and vuln_available_ids is not None:
                agg_id_array = np.asarray(sorted(agg_ids), dtype=vulnerability_dtype['vulnerability_id'])
                known = np.union1d(np.union1d(vuln_available_ids, agg_id_array), cond_ids)
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
