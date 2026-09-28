import json
import os

import numpy as np
import pandas as pd
import pytest
from oasis_data_manager.filestore.backends.local import LocalStorage

import oasislmf.computation.run.model as model_run
from oasislmf.computation.run.check import CheckModel
from oasislmf.pytools.common.data import damagebin_dtype, items_dtype, occurrence_dtype, oasis_float, vulnerability_dtype
from oasislmf.pytools.getmodel.common import Event_dtype, EventIndexBin_dtype, FootprintHeader
from oasislmf.utils.exceptions import OasisException
from oasislmf.validation.model_check import CheckReport, run_model_check

MODEL_SETTINGS = {
    'model_settings': {
        'event_set': {'name': 'Event Set', 'desc': 'Event set', 'default': 'p', 'options': [
            {'id': 'p', 'desc': 'Probabilistic', 'valid_occurrence_ids': ['lt']}]},
        'event_occurrence_id': {'name': 'Occurrence', 'desc': 'Occurrence', 'default': 'lt', 'options': [{'id': 'lt', 'desc': 'Long term'}]},
    },
    'lookup_settings': {'supported_perils': [{'id': 'WTC', 'desc': 'Wind'}]},
}
ANALYSIS_SETTINGS = {
    'model_settings': {'event_set': 'p', 'event_occurrence_id': 'lt'},
    'gul_output': True,
    'gul_summaries': [{'id': 1, 'ord_output': {'alt_period': True}}],
}


def default_model():
    return {
        'damage_bins': [(1, 0., 0., 0., 1), (2, 0., .5, .25, 1), (3, .5, 1., .75, 1)],
        'vulnerability': [(1, 1, 1, .8), (1, 1, 2, .2), (1, 2, 2, .5), (1, 2, 3, .5)],
        'num_damage_bins': 3,
        'num_intensity_bins': 2,
        'footprint': {1: [(10, 1, 1.)], 2: [(10, 2, .6), (10, 1, .4), (20, 2, 1.)]},
        'footprint_order': None,
        'events_p': [1, 2],
        'occurrence_lt': [(1, 1, 0), (2, 2, 0)],
        'no_of_periods': 2,
        'items': [(1, 1, 10, 1, 1), (2, 2, 20, 1, 2)],
        'coverages': [100., 200.],
        'keys': [(1, 'WTC', 1, 10, 1), (2, 'WTC', 1, 20, 1)],
    }


def write_run_dir(root, model):
    static, inputs = root / 'static', root / 'input'
    static.mkdir()
    inputs.mkdir()

    np.array(model['damage_bins'], dtype=damagebin_dtype).tofile(static / 'damage_bin_dict.bin')
    with open(static / 'vulnerability.bin', 'wb') as f:
        np.array([model['num_damage_bins']], dtype=np.int32).tofile(f)
        np.array(model['vulnerability'], dtype=vulnerability_dtype).tofile(f)

    index = []
    with open(static / 'footprint.bin', 'wb') as f:
        np.array([(model['num_intensity_bins'], 1)], dtype=FootprintHeader).tofile(f)
        offset = FootprintHeader.itemsize
        for event_id in model['footprint_order'] or sorted(model['footprint']):
            rows = np.array(model['footprint'][event_id], dtype=Event_dtype)
            rows.tofile(f)
            index.append((event_id, offset, rows.nbytes))
            offset += rows.nbytes
    np.array(index, dtype=EventIndexBin_dtype).tofile(static / 'footprint.idx')

    np.array(model['events_p'], dtype=np.int32).tofile(static / 'events_p.bin')
    with open(static / 'occurrence_lt.bin', 'wb') as f:
        np.array([0, model['no_of_periods']], dtype=np.int32).tofile(f)
        np.array(model['occurrence_lt'], dtype=occurrence_dtype).tofile(f)

    np.array(model['items'], dtype=items_dtype).tofile(inputs / 'items.bin')
    np.array(model['coverages'], dtype=oasis_float).tofile(inputs / 'coverages.bin')
    pd.DataFrame(model['keys'], columns=['LocID', 'PerilID', 'CoverageTypeID', 'AreaPerilID', 'VulnerabilityID']).to_csv(
        inputs / 'keys.csv', index=False)
    return static


def run_check(tmp_path, model=None, analysis_settings=None, model_settings=None, **kwargs):
    static = write_run_dir(tmp_path, model or default_model())
    analysis_settings = ANALYSIS_SETTINGS if analysis_settings is None else analysis_settings
    model_settings = MODEL_SETTINGS if model_settings is None else model_settings
    merged = {**model_settings, **analysis_settings}
    return run_model_check(CheckReport(), str(tmp_path), LocalStorage(str(static)), analysis_settings, model_settings, merged, **kwargs)


def checks(report, level=None):
    return {f.check for f in report.findings if level is None or f.level == level}


def test_clean_model_passes(tmp_path):
    report = run_check(tmp_path)
    assert not report.errors and not report.warnings, report.format()
    assert 'items.vulnerability_exists' in report.passed
    assert 'footprint.probability_sum' in report.passed


def test_item_vulnerability_missing_from_model(tmp_path):
    model = default_model()
    model['items'][1] = (2, 2, 20, 99, 2)
    report = run_check(tmp_path, model)
    finding = next(f for f in report.errors if f.check == 'items.vulnerability_exists')
    assert finding.examples == [99]


def test_footprint_intensity_beyond_header(tmp_path):
    model = default_model()
    model['footprint'][1] = [(10, 3, 1.)]
    assert 'footprint.intensity_range' in checks(run_check(tmp_path, model), 'ERROR')


def test_footprint_non_contiguous_areaperil_and_bad_sum(tmp_path):
    model = default_model()
    model['footprint'][2] = [(10, 2, .6), (20, 2, 1.), (10, 1, .3)]
    errors = checks(run_check(tmp_path, model), 'ERROR')
    assert {'footprint.areaperil_contiguous', 'footprint.probability_sum'} <= errors


def test_unsorted_footprint_index(tmp_path):
    model = default_model()
    model['footprint_order'] = [2, 1]
    assert 'footprint.index_sorted' in checks(run_check(tmp_path, model), 'ERROR')


def test_vulnerability_damage_bins_and_probabilities(tmp_path):
    model = default_model()
    model['vulnerability'] = [(1, 1, 1, .8), (1, 1, 4, .1), (1, 2, 2, .5), (1, 2, 3, .5)]
    model['num_damage_bins'] = 4
    errors = checks(run_check(tmp_path, model), 'ERROR')
    assert {'vulnerability.damage_bin_dict', 'vulnerability.probability_sum'} <= errors


def test_damage_bins_out_of_order(tmp_path):
    model = default_model()
    model['damage_bins'] = [model['damage_bins'][0], model['damage_bins'][2], model['damage_bins'][1]]
    assert 'damage_bin_dict.contiguous' in checks(run_check(tmp_path, model), 'ERROR')


def test_occurrence_period_out_of_range(tmp_path):
    model = default_model()
    model['occurrence_lt'] = [(1, 1, 0), (2, 5, 0)]
    assert 'occurrence.period_range' in checks(run_check(tmp_path, model), 'ERROR')


def test_event_set_file_missing(tmp_path):
    settings = {**ANALYSIS_SETTINGS, 'model_settings': {'event_set': 'h', 'event_occurrence_id': 'lt'}}
    errors = checks(run_check(tmp_path, analysis_settings=settings), 'ERROR')
    assert {'settings.run_inputs', 'settings.declared_options'} <= errors


def test_option_case_mismatch_is_warning(tmp_path):
    settings = {**ANALYSIS_SETTINGS, 'model_settings': {'event_set': 'P', 'event_occurrence_id': 'lt'}}
    report = run_check(tmp_path, analysis_settings=settings)
    assert 'settings.declared_options_case' in checks(report, 'WARNING')
    assert 'settings.run_inputs' not in checks(report, 'ERROR')


def test_items_not_hit_by_any_event(tmp_path):
    model = default_model()
    model['footprint'] = {1: [(10, 1, 1.)], 2: [(10, 2, 1.)]}
    finding = next(f for f in run_check(tmp_path, model).warnings if f.check == 'items.hit_by_events')
    assert finding.examples == [20]
    assert '66.7% of item TIV' in finding.message


def test_keys_peril_not_supported(tmp_path):
    model = default_model()
    model['keys'][0] = (1, 'ORF', 1, 10, 1)
    assert 'keys.supported_perils' in checks(run_check(tmp_path, model), 'WARNING')


def test_lookup_dict_vulnerability_missing(tmp_path):
    keys_dir = tmp_path / 'keys_data'
    keys_dir.mkdir()
    pd.DataFrame({'PERIL_ID': ['WTC', 'WTC'], 'CLASS': [1, 2], 'VULNERABILITY_ID': [1, 7]}).to_csv(keys_dir / 'vulnerability_dict.csv', index=False)
    lookup = {'keys_data_path': './', 'step_definition': {'vulnerability': {
        'type': 'merge', 'columns': ['peril_id', 'class'],
        'parameters': {'file_path': '%%KEYS_DATA_PATH%%/vulnerability_dict.csv', 'id_columns': ['vulnerability_id']}}}}
    with open(keys_dir / 'lookup.json', 'w') as f:
        json.dump(lookup, f)

    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    report = run_check(run_dir, lookup_config_json=str(keys_dir / 'lookup.json'))
    finding = next(f for f in report.warnings if f.check == 'lookup.vulnerability.vulnerability_id')
    assert finding.examples == [7]


@pytest.mark.parametrize('max_events', [None, 1])
def test_event_sampling(tmp_path, max_events):
    report = run_check(tmp_path, max_events=max_events)
    assert not report.errors


def write_settings(tmp_path):
    analysis_fp, model_fp = tmp_path / 'analysis_settings.json', tmp_path / 'model_settings.json'
    analysis_fp.write_text(json.dumps({**ANALYSIS_SETTINGS, 'model_name_id': 'test', 'model_supplier_id': 'test', 'number_of_samples': 1}))
    model_fp.write_text(json.dumps({**MODEL_SETTINGS, 'name': 'test', 'description': 'test'}))
    return str(analysis_fp), str(model_fp)


def test_check_model_step_on_existing_inputs(tmp_path):
    model = default_model()
    model['items'][1] = (2, 2, 20, 99, 2)
    write_run_dir(tmp_path, model)
    analysis_fp, model_fp = write_settings(tmp_path)
    step = CheckModel(model_data_dir=str(tmp_path / 'static'), check_inputs_dir=str(tmp_path / 'input'),
                      analysis_settings_json=analysis_fp, model_settings_json=model_fp, check_report_json=str(tmp_path / 'report.json'))
    with pytest.raises(OasisException, match='1 errors'):
        step.run()
    report = json.loads((tmp_path / 'report.json').read_text())
    assert [f['check'] for f in report['findings'] if f['level'] == 'ERROR'] == ['items.vulnerability_exists']
    assert sorted(os.listdir(tmp_path / 'input')) == ['coverages.bin', 'items.bin', 'keys.csv']


@pytest.mark.parametrize('model_check, check_fails, expected', [
    (False, False, ['GenerateFiles', 'GenerateLosses']),
    (True, False, ['GenerateFiles', 'CheckModel', 'GenerateLosses']),
    (True, True, ['GenerateFiles', 'CheckModel']),
])
def test_run_model_check_option(tmp_path, monkeypatch, model_check, check_fails, expected):
    calls = []

    def fake_run(name, fail=False):
        def run(self):
            calls.append(name)
            if name == 'CheckModel':
                assert self.check_inputs_dir == os.path.join(str(tmp_path / 'run'), 'input')
            if fail:
                raise OasisException('Model check failed')
        return run

    monkeypatch.setattr(model_run, 'get_exposure_data', lambda *args, **kwargs: None)
    monkeypatch.setattr(model_run.GenerateFiles, 'run', fake_run('GenerateFiles'))
    monkeypatch.setattr(model_run.CheckModel, 'run', fake_run('CheckModel', check_fails))
    monkeypatch.setattr(model_run.GenerateLosses, 'run', fake_run('GenerateLosses'))
    (tmp_path / 'model_data').mkdir()
    analysis_fp, model_fp = write_settings(tmp_path)

    run = model_run.RunModel(model_run_dir=str(tmp_path / 'run'), model_data_dir=str(tmp_path / 'model_data'),
                             analysis_settings_json=analysis_fp, model_settings_json=model_fp, model_check=model_check)
    if check_fails:
        with pytest.raises(OasisException):
            run.run()
    else:
        run.run()
    assert calls == expected
