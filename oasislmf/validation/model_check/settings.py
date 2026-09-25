__all__ = [
    'check_settings',
    'resolve_run_files',
]

import os

from ...execution.bin import (prepare_run_inputs, set_footprint_set, set_hazard_case_set,
                              set_loss_factors_set, set_vulnerability_set)
from ...utils.exceptions import OasisException

OPTIONAL_MODEL_SETS = {
    'footprint_set': set_footprint_set,
    'vulnerability_set': set_vulnerability_set,
    'pla_loss_factors_set': set_loss_factors_set,
    'hazard_case_set': set_hazard_case_set,
}


def _declared_options(model_settings, key):
    info = model_settings.get('model_settings', {}).get(key)
    if isinstance(info, dict) and 'options' in info:
        return [opt['id'] if isinstance(opt, dict) else opt for opt in info['options']]
    return None


def check_settings(report, analysis_settings, model_settings, merged_settings):
    """Check the analysis settings choices against the options the model settings declare."""
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
    """Select event/occurrence/period files and optional model sets exactly as a loss run would."""
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
        except OasisException as e:
            report.error(f'settings.{key}', str(e))

    return {name: os.path.join(run_dir, 'input', f'{name}.bin')
            for name in ('events', 'occurrence', 'periods', 'returnperiods')
            if os.path.exists(os.path.join(run_dir, 'input', f'{name}.bin'))}
