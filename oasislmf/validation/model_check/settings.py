__all__ = [
    'check_settings',
    'resolve_run_files',
]

import os

from ...execution.bin import OPTIONAL_MODEL_SETS, prepare_run_inputs
from ...utils.exceptions import OasisException


def _declared_options(model_settings, key):
    info = model_settings.get('model_settings', {}).get(key)
    if isinstance(info, dict) and 'options' in info:
        return [opt['id'] if isinstance(opt, dict) else opt for opt in info['options']]
    return None


def check_settings(report, analysis_settings, model_settings, merged_settings):
    """Check the analysis settings choices against the options the model settings declare.

    Reports:
        ``settings.analysis``: no analysis settings were given, so model defaults apply (warning).
        ``settings.declared_options``: a ``model_settings`` value is not one of the options the model declares (error).
        ``settings.declared_options_case``: a value only matches a declared option ignoring case (warning).
        ``settings.event_occurrence_pair``: ``event_occurrence_id`` is not in the chosen event set's ``valid_occurrence_ids`` (error).
        ``settings.peril_filter``: ``peril_filter`` names perils the model does not support (error).

    Args:
        report (CheckReport): report the findings are added to.
        analysis_settings (dict): the user's analysis settings, or None.
        model_settings (dict): the model settings file contents.
        merged_settings (dict): analysis settings merged over the model defaults, as a run sees them.
    """
    if not analysis_settings:
        report.warning('settings.analysis', 'no analysis settings provided, model defaults will be used')

    effective = merged_settings.get('model_settings', {})
    requested = (analysis_settings or {}).get('model_settings', {})
    not_declared, case_only = [], []
    for key, value in requested.items():
        options = _declared_options(model_settings, key)
        if options is None or value in options:
            continue
        msg = f'{key}: {value!r} not in declared options {options}'
        lower_options = {str(o).lower() for o in options}
        (case_only if isinstance(value, str) and value.lower() in lower_options else not_declared).append(msg)
    report.missing('settings.declared_options', 'analysis settings choose values the model settings do not declare', not_declared)
    if case_only:
        report.warning('settings.declared_options_case',
                       'analysis settings values only match the model options ignoring case (file lookup falls back to lower case)', case_only)

    event_set = str(effective.get('event_set')).lower()
    occurrence_id = effective.get('event_occurrence_id')
    for opt in model_settings.get('model_settings', {}).get('event_set', {}).get('options', []):
        if isinstance(opt, dict) and str(opt.get('id')).lower() == event_set and 'valid_occurrence_ids' in opt and occurrence_id is not None:
            if str(occurrence_id).lower() not in {str(o).lower() for o in opt['valid_occurrence_ids']}:
                report.error('settings.event_occurrence_pair',
                             f'event_occurrence_id {occurrence_id!r} is not valid for event_set {event_set!r}',
                             opt['valid_occurrence_ids'])
            else:
                report.ok('settings.event_occurrence_pair')

    model_perils = {p['id'] for p in model_settings.get('lookup_settings', {}).get('supported_perils', [])}
    if model_perils and (analysis_settings or {}).get('peril_filter'):
        report.missing('settings.peril_filter', 'peril_filter contains perils the model does not support',
                       set(analysis_settings['peril_filter']) - model_perils)


def resolve_run_files(report, merged_settings, run_dir, model_storage):
    """Select event/occurrence/period files and optional model sets exactly as a loss run would.

    Reports:
        ``settings.run_inputs``: ``prepare_run_inputs`` could not resolve the event set / occurrence files (error).
        ``settings.<set>``: an optional model set (``footprint_set``, ``vulnerability_set``, ``pla_loss_factors_set``,
        ``hazard_case_set``) could not be applied (error).

    Args:
        report (CheckReport): report the findings are added to.
        merged_settings (dict): analysis settings merged over the model defaults.
        run_dir (str): run directory laid out like a loss run.
        model_storage (BaseStorage): storage for the model data.

    Returns:
        dict: ``events``, ``occurrence``, ``periods`` and ``returnperiods`` mapped to the resolved ``input/*.bin`` paths that exist.
    """
    try:
        prepare_run_inputs(merged_settings, run_dir, model_storage)
        report.ok('settings.run_inputs')
    except OasisException as e:
        report.error('settings.run_inputs', str(e))

    model_sets = merged_settings.get('model_settings', {})
    for key, setter in OPTIONAL_MODEL_SETS.items():
        value = model_sets.get(key)
        if not value:
            continue
        try:
            setter(value, run_dir)
            report.ok(f'settings.{key}')
        except (OasisException, OSError) as e:
            report.error(f'settings.{key}', f'{type(e).__name__}: {e}')

    return {name: os.path.join(run_dir, 'input', f'{name}.bin')
            for name in ('events', 'occurrence', 'periods', 'returnperiods')
            if os.path.exists(os.path.join(run_dir, 'input', f'{name}.bin'))}
