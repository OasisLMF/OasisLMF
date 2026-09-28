import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from oasis_data_manager.filestore.backends.local import LocalStorage
from ods_tools.oed.settings import Settings

import oasislmf.computation.run.model as model_run
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.getmodel.vulnerability import vulnerability_to_parquet
from oasislmf.computation.run.check import CheckModel
from oasislmf.pytools.common.data import damagebin_dtype, items_dtype, occurrence_dtype, oasis_float, vulnerability_dtype
from oasislmf.pytools.getmodel.common import Event_dtype, EventIndexBin_dtype, FootprintHeader
from oasislmf.pytools.getmodel.manager import VulnerabilityIndex_dtype
from oasislmf.utils.exceptions import OasisException
from oasislmf.validation.model_check import CheckReport, run_model_check
from oasislmf.validation.model_check.model_files import load_vulnerability
from tests.computation.data.common import MIN_LOC

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

    if model.get('damage_bins') is not None:
        np.array(model['damage_bins'], dtype=damagebin_dtype).tofile(static / 'damage_bin_dict.bin')
    write_vulnerability(static, model, model.get('vulnerability_format', 'bin'))

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

    for name, file_type in [('aggregate_vulnerability', 'aggregatevulnerability'), ('weights', 'weights'), ('lossfactors', 'lossfactors'),
                            ('returnperiods', 'returnperiods'), ('periods', 'periods')]:
        if model.get(name) is not None:
            write_with_converter(root, static / f'{name}.bin', file_type, model[name])
    if model.get('amplifications') is not None:
        write_with_converter(root, inputs / 'amplifications.bin', 'amplifications', model['amplifications'])

    if model.get('items') is not None:
        np.array(model['items'], dtype=items_dtype).tofile(inputs / 'items.bin')
    np.array(model['coverages'], dtype=oasis_float).tofile(inputs / 'coverages.bin')
    pd.DataFrame(model['keys'], columns=['LocID', 'PerilID', 'CoverageTypeID', 'AreaPerilID', 'VulnerabilityID']).to_csv(
        inputs / 'keys.csv', index=False)
    if model.get('keys_errors') is not None:
        pd.DataFrame(model['keys_errors'], columns=['LocID', 'PerilID', 'CoverageTypeID', 'Status', 'Message']).to_csv(
            inputs / 'keys-errors.csv', index=False)
    for src, dst in model.get('rename_files', {}).items():
        os.rename(root / src, root / dst)
    for name, data in model.get('raw_files', {}).items():
        if data is None:
            os.remove(root / name)
        else:
            with open(root / name, 'ab' if name in model.get('append_files', ()) else 'wb') as f:
                f.write(data)
    return static


def write_with_converter(root, file_out, file_type, rows):
    csv_fp = root / f'{file_type}.csv'
    pd.DataFrame(rows, columns=list(TOOL_INFO[file_type]['dtype'].names)).to_csv(csv_fp, index=False)
    csvtobin(csv_fp, file_out, file_type)


def write_vulnerability(static, model, vulnerability_format):
    rows = np.array(model['vulnerability'], dtype=vulnerability_dtype)
    if vulnerability_format == 'csv':
        pd.DataFrame(rows).to_csv(static / 'vulnerability.csv', index=False)
        return
    if vulnerability_format == 'bin':
        with open(static / 'vulnerability.bin', 'wb') as f:
            np.array([model['num_damage_bins']], dtype=np.int32).tofile(f)
            rows.tofile(f)
        return
    csv_fp = static.parent / 'vulnerability_src.csv'
    pd.DataFrame(rows).to_csv(csv_fp, index=False)
    idx = static / 'vulnerability.idx' if vulnerability_format in ('idx', 'zip') else None
    csvtobin(csv_fp, static / 'vulnerability.bin', 'vulnerability', idx_file_out=idx, max_damage_bin_idx=model['num_damage_bins'],
             no_validation=True, suppress_int_bin_checks=True, zip_files=vulnerability_format == 'zip')
    if vulnerability_format == 'parquet':
        vulnerability_to_parquet(str(static))
        os.remove(static / 'vulnerability.bin')


def run_check(tmp_path, model=None, analysis_settings=None, model_settings=None, **kwargs):
    static = write_run_dir(tmp_path, model or default_model())
    analysis_settings = ANALYSIS_SETTINGS if analysis_settings is None else analysis_settings
    model_settings = MODEL_SETTINGS if model_settings is None else model_settings
    merged = merge_settings(analysis_settings, model_settings)
    return run_model_check(CheckReport(), str(tmp_path), LocalStorage(str(static)), analysis_settings, model_settings, merged, **kwargs)


def merge_settings(analysis_settings, model_settings):
    settings = Settings()
    settings.add_settings(analysis_settings, 'user')
    settings.add_settings(model_settings)
    return settings.get_settings()


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


def test_check_model_does_not_write_through_to_existing_inputs(tmp_path):
    write_run_dir(tmp_path, default_model())
    np.array([1, 2], dtype=np.int32).tofile(tmp_path / 'input' / 'events.bin')
    (tmp_path / 'input' / 'events.csv').write_text('event_id\n1\n2\n')
    analysis_fp, model_fp = write_settings(tmp_path)
    analysis = json.loads(open(analysis_fp).read())
    open(analysis_fp, 'w').write(json.dumps({**analysis, 'event_ids': [2]}))
    CheckModel(model_data_dir=str(tmp_path / 'static'), check_inputs_dir=str(tmp_path / 'input'),
               analysis_settings_json=analysis_fp, model_settings_json=model_fp).run()
    assert np.fromfile(tmp_path / 'input' / 'events.bin', dtype=np.int32).tolist() == [1, 2]
    assert (tmp_path / 'input' / 'events.csv').read_text() == 'event_id\n1\n2\n'


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


def with_second_vulnerability(model):
    model['vulnerability'] = model['vulnerability'] + [(2, 1, 1, 1.), (2, 2, 3, 1.)]
    return model


@pytest.mark.parametrize('vulnerability_format', ['bin', 'csv', 'idx', 'parquet'])
@pytest.mark.parametrize('full_model', [False, True])
def test_vulnerability_formats(tmp_path, vulnerability_format, full_model):
    model = with_second_vulnerability(default_model())
    model['vulnerability_format'] = vulnerability_format
    report = run_check(tmp_path, model, full_model=full_model)
    assert not report.errors and not report.warnings, report.format()

    vuln = load_vulnerability(CheckReport(), str(tmp_path / 'static'), None if full_model else [1])
    expected = [r for r in model['vulnerability'] if full_model or r[0] == 1]
    np.testing.assert_allclose(np.sort(vuln.rows).tolist(), sorted(expected), rtol=1e-6)
    assert vuln.available_ids.tolist() == [1, 2]
    assert vuln.num_damage_bins == 3


@pytest.mark.parametrize('raw_files, append_files, ids_known', [
    ({'static/vulnerability.bin': b'\x00\x00\x00'}, ('static/vulnerability.bin',), False),
    ({'static/vulnerability.idx': np.array([(1, 4, 64, 0)], dtype=VulnerabilityIndex_dtype).tobytes()}, (), True),
])
def test_vulnerability_file_corrupt(tmp_path, raw_files, append_files, ids_known):
    model = default_model()
    model['raw_files'], model['append_files'] = raw_files, append_files
    report = run_check(tmp_path, model)
    assert 'vulnerability.format' in checks(report, 'ERROR')
    assert ('items.vulnerability_exists' in report.passed) == ids_known
    assert 'vulnerability.probability_sum' not in report.passed


def test_vulnerability_zipped_is_error(tmp_path):
    model = default_model()
    model['vulnerability_format'] = 'zip'
    assert 'vulnerability.format' in checks(run_check(tmp_path, model), 'ERROR')


def test_vulnerability_missing(tmp_path):
    model = default_model()
    model['raw_files'] = {'static/vulnerability.bin': None}
    assert 'vulnerability.exists' in checks(run_check(tmp_path, model), 'ERROR')


def test_vulnerability_parquet_intensity_bins_differ_from_footprint(tmp_path):
    model = default_model()
    model['vulnerability_format'] = 'parquet'
    model['num_intensity_bins'] = 3
    assert 'vulnerability.intensity_bins' in checks(run_check(tmp_path, model), 'ERROR')


def test_vulnerability_row_errors(tmp_path):
    model = default_model()
    model['vulnerability'] = [(1, 1, 1, 1.5), (1, 1, 2, -.5), (1, 1, 0, 0.), (1, 1, 4, 0.),
                              (1, 2, 2, .5), (1, 2, 3, .5), (1, 2, 3, 0.), (1, 0, 1, 1.), (1, 3, 1, 1.)]
    report = run_check(tmp_path, model)
    assert {'vulnerability.damage_bin_min', 'vulnerability.damage_bin_header', 'vulnerability.damage_bin_dict',
            'vulnerability.intensity_bin_min', 'vulnerability.probability_range', 'vulnerability.duplicates'} <= checks(report, 'ERROR')
    assert 'vulnerability.intensity_bin_max' in checks(report, 'WARNING')


@pytest.mark.parametrize('first_bin, level', [((1, 0., 0., 0., 1), 'INFO'), ((1, 0., .1, .05, 1), 'ERROR')])
def test_vulnerability_undefined_intensity(tmp_path, first_bin, level):
    model = default_model()
    model['damage_bins'] = [first_bin, (2, model['damage_bins'][0][2] if level == 'INFO' else .1, .5, .3, 1), (3, .5, 1., .75, 1)]
    model['vulnerability'] = [(1, 1, 1, .8), (1, 1, 2, .2)]
    report = run_check(tmp_path, model)
    assert 'vulnerability.intensity_coverage' in checks(report, level)
    assert ('damage_bin_dict.zero_bin' in checks(report, 'WARNING')) == (level == 'ERROR')


def test_damage_bin_dict_missing(tmp_path):
    model = default_model()
    model['damage_bins'] = None
    assert 'damage_bin_dict.exists' in checks(run_check(tmp_path, model), 'ERROR')


@pytest.mark.parametrize('damage_bins, errors, warnings', [
    ([(1, 0., 0., 0., 1), (2, .6, .5, .55, 1), (3, .6, 1., .8, 1)], {'damage_bin_dict.from_to', 'damage_bin_dict.interpolation'}, set()),
    ([(1, 0., 0., 0., 1), (2, 0., .6, .3, 1), (3, .5, 1., .75, 9)], set(), {'damage_bin_dict.overlap', 'damage_bin_dict.damage_type'}),
])
def test_damage_bin_dict_values(tmp_path, damage_bins, errors, warnings):
    model = default_model()
    model['damage_bins'] = damage_bins
    report = run_check(tmp_path, model)
    assert errors <= checks(report, 'ERROR') and warnings <= checks(report, 'WARNING')


@pytest.mark.parametrize('dynamic', [False, True])
def test_footprint_intensity_zero_allowed_for_dynamic(tmp_path, dynamic):
    model = default_model()
    model['footprint'][1] = [(10, 0, 1.)]
    assert ('footprint.intensity_range' in checks(run_check(tmp_path, model, dynamic_footprint=dynamic), 'ERROR')) != dynamic


def test_footprint_row_problems(tmp_path):
    model = default_model()
    model['footprint'] = {1: [(0, 1, 1.), (10, 1, 1.)], 2: [(10, 2, .5), (10, 2, .5), (20, 2, 1.)]}
    model['footprint_order'] = [1, 1, 2]
    model['events_p'] = [1, 2, 3]
    report = run_check(tmp_path, model)
    assert {'footprint.duplicates', 'footprint.index_unique'} <= checks(report, 'ERROR')
    assert {'footprint.areaperil_zero', 'footprint.events_present', 'occurrence.events_covered'} <= checks(report, 'WARNING')


def test_footprint_cannot_load(tmp_path):
    model = default_model()
    model['raw_files'] = {'static/footprint.bin': None, 'static/footprint.idx': None}
    report = run_check(tmp_path, model)
    assert 'footprint.load' in checks(report, 'ERROR')
    assert 'items.hit_by_events' not in checks(report) | set(report.passed)


def test_no_item_hit_by_any_event(tmp_path):
    model = default_model()
    model['footprint'] = {1: [(30, 1, 1.)], 2: [(30, 2, 1.)]}
    assert 'items.hit_by_events' in checks(run_check(tmp_path, model), 'ERROR')


def test_no_item_hit_by_sampled_events_is_warning(tmp_path):
    model = default_model()
    model['footprint'] = {1: [(30, 1, 1.)], 2: [(10, 1, 1.)]}
    report = run_check(tmp_path, model, max_events=1)
    assert 'items.hit_by_events' in checks(report, 'WARNING')
    assert not report.errors


@pytest.mark.parametrize('events, errors', [
    ([1, 2, 2, -1], {'events.duplicates', 'events.positive'}),
    ([], {'events.empty'}),
])
def test_events_file_problems(tmp_path, events, errors):
    model = default_model()
    model['events_p'] = events
    assert errors <= checks(run_check(tmp_path, model), 'ERROR')


def test_occurrence_header_truncated(tmp_path):
    model = default_model()
    model['raw_files'] = {'static/occurrence_lt.bin': np.array([0], dtype=np.int32).tobytes()}
    assert 'occurrence.format' in checks(run_check(tmp_path, model), 'ERROR')


@pytest.mark.parametrize('periods, level, check', [
    ([(1, .5), (2, .25)], 'WARNING', 'periods.weights_sum'),
    ([(1, .25), (2, .25), (3, .5)], 'ERROR', 'periods.format'),
])
def test_periods(tmp_path, periods, level, check):
    model = default_model()
    model['periods'] = periods
    assert check in checks(run_check(tmp_path, model), level)


@pytest.mark.parametrize('return_periods, valid', [([100, 10], True), ([10, 100], False)])
def test_return_periods(tmp_path, return_periods, valid):
    model = default_model()
    model['raw_files'] = {'static/returnperiods.bin': np.array(return_periods, dtype=np.int32).tobytes()}
    settings = {**ANALYSIS_SETTINGS, 'gul_summaries': [{'id': 1, 'ord_output': {'ept_full_uncertainty_aep': True}}]}
    report = run_check(tmp_path, model, analysis_settings=settings)
    assert ('returnperiods.format' in report.passed) == valid
    assert ('returnperiods.format' in checks(report, 'ERROR')) != valid


@pytest.mark.parametrize('aggregate, weights, errors, warnings', [
    ([(100, 1), (100, 2)], [(10, 1, .5), (10, 2, .5), (20, 1, 1.)], set(), set()),
    ([(100, 1), (100, 5)], [(10, 1, 1.)], {'aggregate_vulnerability.sub_ids'}, set()),
    ([(100, 1), (100, 2)], None, {'aggregate_vulnerability.weights'}, set()),
    ([(1, 2)], [(10, 2, 1.)], set(), {'aggregate_vulnerability.id_collision'}),
])
def test_aggregate_vulnerability(tmp_path, aggregate, weights, errors, warnings):
    model = with_second_vulnerability(default_model())
    model['aggregate_vulnerability'], model['weights'] = aggregate, weights
    agg_id = aggregate[0][0]
    model['items'] = [(1, 1, 10, agg_id, 1), (2, 2, 20, 1, 2)]
    report = run_check(tmp_path, model)
    assert checks(report, 'ERROR') == errors, report.format()
    assert warnings <= checks(report, 'WARNING')
    assert 'items.vulnerability_exists' in report.passed


@pytest.mark.parametrize('lossfactors, amplifications, expected', [
    ([(1, 1, 1.1), (2, 1, 1.2)], [(1, 1), (2, 2)], ('WARNING', 'amplifications.lossfactors')),
    ([(1, 1, 1.1)], None, ('ERROR', 'amplifications.exists')),
    (None, [(1, 1), (2, 1)], ('ERROR', 'amplifications.lossfactors')),
    ([(1, 1, 1.1)], 'non-contiguous', ('ERROR', 'amplifications.format')),
])
def test_amplifications(tmp_path, lossfactors, amplifications, expected):
    model = default_model()
    model['lossfactors'] = lossfactors
    if amplifications == 'non-contiguous':
        model['raw_files'] = {'input/amplifications.bin': np.array([0, 1, 1, 3, 1], dtype=np.int32).tobytes()}
    else:
        model['amplifications'] = amplifications
    level, check = expected
    assert check in checks(run_check(tmp_path, model, analysis_settings={**ANALYSIS_SETTINGS, 'pla': True}), level)


def test_amplifications_ignored_without_pla(tmp_path):
    model = default_model()
    model['amplifications'] = [(1, 1), (2, 2)]
    report = run_check(tmp_path, model)
    assert not report.errors and not report.warnings


def test_portfolio_input_problems(tmp_path):
    model = default_model()
    model['items'] = [(1, 1, 0, 1, 1), (2, 3, 20, 1, 2)]
    model['coverages'] = [0., 200.]
    model['keys'] = [(1, 'WTC', 1, 0, 1), (2, 'ORF', 1, 20, 0)]
    model['keys_errors'] = [(3, 'WTC', 1, 'fail', 'no areaperil'), (4, 'WTC', 1, 'nomatch', 'no areaperil')]
    model_settings = {**MODEL_SETTINGS, 'model_settings': {**MODEL_SETTINGS['model_settings'], 'correlation_settings': [
        {'peril_correlation_group': 1, 'damage_correlation_value': .5, 'hazard_correlation_value': 0.}]}}
    report = run_check(tmp_path, model, model_settings=model_settings)
    assert {'items.coverage_range', 'items.areaperil_zero', 'keys.areaperil_range', 'keys.vulnerability_range',
            'keys.supported_perils'} <= checks(report, 'ERROR')
    assert {'coverages.tiv', 'keys.lookup_failures'} <= checks(report, 'WARNING')
    lookup_failures = next(f for f in report.findings if f.check == 'keys.lookup_failures')
    assert lookup_failures.count == 2 and '"no areaperil": 2' in lookup_failures.examples


def test_items_file_missing(tmp_path):
    model = default_model()
    model['items'] = None
    report = run_check(tmp_path, model)
    assert 'inputs.exists' in checks(report, 'ERROR')
    assert 'vulnerability.probability_sum' in report.passed


def test_coverages_file_missing(tmp_path):
    model = default_model()
    model['raw_files'] = {'input/coverages.bin': None}
    report = run_check(tmp_path, model)
    assert 'inputs.exists' in checks(report, 'ERROR')


def test_complex_model_keys_are_not_range_checked(tmp_path):
    model = default_model()
    model['raw_files'] = {'input/keys.csv': b'LocID,PerilID,CoverageTypeID,ModelData\n1,WTC,1,"{}"\n'}
    report = run_check(tmp_path, model)
    assert not checks(report) & {'keys.areaperil_range', 'keys.vulnerability_range'}
    assert not report.errors


def test_keys_parquet_is_read(tmp_path):
    model = default_model()
    model['keys'][0] = (1, 'ORF', 1, 10, 1)
    model['rename_files'] = {'input/keys.csv': 'keys_src.csv'}
    write_run_dir(tmp_path, model)
    pd.read_csv(tmp_path / 'keys_src.csv').to_parquet(tmp_path / 'input' / 'keys.parquet')
    report = run_model_check(CheckReport(), str(tmp_path), LocalStorage(str(tmp_path / 'static')), ANALYSIS_SETTINGS, MODEL_SETTINGS,
                             merge_settings(ANALYSIS_SETTINGS, MODEL_SETTINGS))
    assert 'keys.supported_perils' in checks(report, 'WARNING')


def test_lookup_dict_problems(tmp_path):
    keys_dir = tmp_path / 'keys_data'
    keys_dir.mkdir()
    pd.DataFrame({'AREA_PERIL_ID': [1, 0], 'LAT': [0., 1.]}).to_csv(keys_dir / 'areaperil_dict.csv', index=False)
    pd.DataFrame({'PERIL_ID': ['WTC'], 'CLASS': [1]}).to_csv(keys_dir / 'no_id.csv', index=False)
    pd.DataFrame({'PERIL_ID': ['WTC', 'WTC'], 'AMPLIFICATION_ID': [1, 7]}).to_parquet(keys_dir / 'amplification_dict.parquet')
    lookup = {'keys_data_path': './', 'step_definition': {
        'peril': {'type': 'rtree', 'parameters': {'file_path': '%%KEYS_DATA_PATH%%/areaperil_dict.csv', 'id_columns': ['area_peril_id']}},
        'vulnerability': {'type': 'merge', 'parameters': {'file_path': '%%KEYS_DATA_PATH%%/no_id.csv', 'id_columns': ['vulnerability_id']}},
        'missing': {'type': 'merge', 'parameters': {'file_path': str(keys_dir / 'missing.csv'), 'id_columns': ['vulnerability_id']}},
        'amplification': {'type': 'merge', 'parameters': {'file_path': 'amplification_dict.parquet', 'file_type': 'parquet',
                                                          'id_columns': ['amplification_id']}},
        'class': {'type': 'merge', 'parameters': {'file_path': 'no_id.csv', 'id_columns': ['class']}},
        'split': {'type': 'split_loc_perils_covered', 'parameters': {}},
    }}
    with open(keys_dir / 'lookup.json', 'w') as f:
        json.dump(lookup, f)

    model = default_model()
    model['lossfactors'] = [(1, 1, 1.1)]
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    report = run_check(run_dir, model, lookup_config_json=str(keys_dir / 'lookup.json'))
    assert {'lookup.peril.areaperil_range', 'lookup.vulnerability.vulnerability_id', 'lookup.missing.exists'} <= checks(report, 'ERROR')
    assert next(f for f in report.warnings if f.check == 'lookup.amplification.amplification_id').examples == [7]
    assert not any(f.check.startswith(('lookup.class', 'lookup.split')) for f in report.findings)


def test_no_analysis_settings(tmp_path):
    report = run_check(tmp_path, analysis_settings={})
    assert 'settings.analysis' in checks(report, 'WARNING')
    assert not report.errors, report.format()


def test_event_occurrence_pair_and_peril_filter(tmp_path):
    model_settings = json.loads(json.dumps(MODEL_SETTINGS))
    model_settings['model_settings']['event_occurrence_id']['options'].append({'id': 'st', 'desc': 'Short term'})
    settings = {**ANALYSIS_SETTINGS, 'model_settings': {'event_set': 'p', 'event_occurrence_id': 'st'}, 'peril_filter': ['ORF']}
    errors = checks(run_check(tmp_path, analysis_settings=settings, model_settings=model_settings), 'ERROR')
    assert {'settings.event_occurrence_pair', 'settings.peril_filter'} <= errors
    assert 'settings.declared_options' not in errors


@pytest.mark.parametrize('rename, valid', [
    ({'static/footprint.bin': 'static/footprint_alt.bin', 'static/footprint.idx': 'static/footprint_alt.idx'}, True),
    ({}, False),
])
def test_footprint_set(tmp_path, rename, valid):
    model = default_model()
    model['rename_files'] = rename
    settings = {**ANALYSIS_SETTINGS, 'model_settings': {**ANALYSIS_SETTINGS['model_settings'], 'footprint_set': 'alt'}}
    report = run_check(tmp_path, model, analysis_settings=settings)
    assert ('settings.footprint_set' in report.passed) == valid
    assert ('settings.footprint_set' in checks(report, 'ERROR')) != valid
    if valid:
        assert not report.errors, report.format()


def test_footprint_set_alongside_default_footprint(tmp_path):
    model = default_model()
    write_run_dir(tmp_path, model)
    for ext in ('bin', 'idx'):
        (tmp_path / 'static' / f'footprint_alt.{ext}').write_bytes((tmp_path / 'static' / f'footprint.{ext}').read_bytes())
    settings = {**ANALYSIS_SETTINGS, 'model_settings': {**ANALYSIS_SETTINGS['model_settings'], 'footprint_set': 'alt'}}
    report = run_model_check(CheckReport(), str(tmp_path), LocalStorage(str(tmp_path / 'static')), settings, MODEL_SETTINGS,
                             merge_settings(settings, MODEL_SETTINGS))
    finding = next(f for f in report.errors if f.check == 'settings.footprint_set')
    assert 'FileExistsError' in finding.message


def check_model_step(tmp_path, monkeypatch, generate, **kwargs):
    fixture_dir = tmp_path / 'fixture'
    fixture_dir.mkdir()
    write_run_dir(fixture_dir, kwargs.pop('model', None) or default_model())
    analysis_fp, model_fp = write_settings(tmp_path)
    loc_fp = tmp_path / 'location.csv'
    loc_fp.write_text(MIN_LOC)

    def fake_generate(self):
        if generate is not True:
            raise generate
        assert self.kwargs['exposure_data'].location.dataframe['LocNumber'].tolist() == ['10002082046']
        for name in os.listdir(fixture_dir / 'input'):
            shutil.copy(fixture_dir / 'input' / name, self.oasis_files_dir)

    monkeypatch.setattr('oasislmf.computation.run.check.GenerateFiles.run', fake_generate)
    return CheckModel(model_data_dir=str(fixture_dir / 'static'), analysis_settings_json=analysis_fp, model_settings_json=model_fp,
                      oed_location_csv=str(loc_fp), check_oed=False, **kwargs)


def test_check_model_generates_inputs(tmp_path, monkeypatch):
    check_dir = tmp_path / 'check'
    report = check_model_step(tmp_path, monkeypatch, True, check_dir=str(check_dir)).run()
    assert not report.errors, report.format()
    assert {'inputs.generate', 'items.vulnerability_exists'} <= set(report.passed)
    assert sorted(os.listdir(check_dir)) == ['input', 'static']
    assert 'items.bin' in os.listdir(check_dir / 'input')


def test_check_model_generation_failure_skips_portfolio_checks(tmp_path, monkeypatch):
    step = check_model_step(tmp_path, monkeypatch, RuntimeError('lookup exploded'), check_report_json=str(tmp_path / 'report.json'))
    with pytest.raises(OasisException, match='1 errors'):
        step.run()
    report = json.loads((tmp_path / 'report.json').read_text())
    [error] = [f for f in report['findings'] if f['level'] == 'ERROR']
    assert error['check'] == 'inputs.generate' and 'lookup exploded' in error['message']
    assert 'vulnerability.probability_sum' in report['passed']
    assert not any(c.startswith(('items.', 'keys.', 'inputs.exists')) for c in report['passed'])


def test_check_model_warnings_as_errors(tmp_path, monkeypatch):
    model = default_model()
    model['coverages'] = [0., 200.]
    step = check_model_step(tmp_path, monkeypatch, True, model=model, check_warnings_as_errors=True)
    with pytest.raises(OasisException, match='0 errors, 1 warnings'):
        step.run()


def test_check_model_rejects_used_check_dir(tmp_path, monkeypatch):
    (tmp_path / 'check' / 'input').mkdir(parents=True)
    step = check_model_step(tmp_path, monkeypatch, True, check_dir=str(tmp_path / 'check'))
    with pytest.raises(OasisException, match='already contains'):
        step.run()
