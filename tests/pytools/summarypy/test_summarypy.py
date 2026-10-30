import filecmp
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.common.data import def_to_type_and_size, gul_summary_xref_dtype
from oasislmf.pytools.common.event_stream import (MAX_LOSS_IDX, MEAN_IDX, NUMBER_OF_AFFECTED_RISK_IDX, TIV_IDX,
                                                  decode_building, decode_local_sidx, mv_read)
from oasislmf.pytools.elt.data import SELT_dtype, SELT_headers
from oasislmf.pytools.summary.cli import parser, manager
from oasislmf.utils.exceptions import OasisStreamException
from tests.pytools.utils import packed_gul_stream, read_gul_stream, read_summary_stream


TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_summarypy")

IDX_DTYPE = np.dtype([('summary_id', '<i4'), ('offset', '<i8')])  # fix dtype to reproduce test setup
EVENT_ID_DTYPE, EVENT_ID_SIZE = def_to_type_and_size('event_id')
SUMMARY_ID_DTYPE, SUMMARY_ID_SIZE = def_to_type_and_size('summary_id')
SIDX_DTYPE, SIDX_SIZE = def_to_type_and_size('sidx')
LOSS_DTYPE, LOSS_SIZE = def_to_type_and_size('loss')


def case_runner(test_name, test_case):

    base_path = Path(TESTS_ASSETS_DIR, test_name)
    with TemporaryDirectory() as tmp_result_dir_str:
        for run_type, summary_set_ids in test_case.items():
            if run_type == manager.RUNTYPE_REINSURANCE_LOSS:
                static_path = base_path.joinpath('RI_1')
                output_zeros = ' -z'
            else:
                static_path = base_path
                output_zeros = ''

            summary_sets_cmd = ' '.join(f" -{summary_set_id} "
                                        f"{Path(tmp_result_dir_str, f'{run_type}_S{summary_set_id}_summary.bin')}"
                                        for summary_set_id in summary_set_ids)
            cmd = (f"-m -t {run_type}{output_zeros} -p {static_path}"
                   f" -i {Path(TESTS_ASSETS_DIR, run_type + '.bin')}{summary_sets_cmd}").split()

            kwargs = vars(parser.parse_args(cmd))
            kwargs.pop('logging_level')
            manager.main(**kwargs)
            for summary_set_id in summary_set_ids:
                base_file_name = f"{run_type}_S{summary_set_id}_summary"
                try:
                    for file_extension in ['.bin', '.idx']:
                        assert filecmp.cmp(Path(tmp_result_dir_str, base_file_name + file_extension),
                                           Path(base_path, base_file_name + file_extension), shallow=True)
                    idx_path = Path(tmp_result_dir_str, base_file_name + '.idx')
                    assert idx_path.stat().st_size % IDX_DTYPE.itemsize == 0, \
                        f"{idx_path} size is not a multiple of IDX_DTYPE.itemsize ({IDX_DTYPE.itemsize})"
                    recs = np.fromfile(idx_path, dtype=IDX_DTYPE)
                    assert len(recs) > 0, f"{idx_path} parsed to zero records"
                    assert (recs['summary_id'] >= 1).all(), f"{idx_path} has summary_id < 1"
                    assert (recs['offset'] >= 0).all(), f"{idx_path} has negative offsets"
                    assert (np.diff(recs['offset']) > 0).all(), \
                        f"{idx_path} offsets are not strictly increasing"
                except Exception as e:
                    error_path = base_path.joinpath('error_files')
                    error_path.mkdir(exist_ok=True)
                    for file_extension in ['.bin', '.idx']:
                        shutil.copyfile(Path(tmp_result_dir_str, base_file_name + file_extension),
                                        Path(error_path, base_file_name + file_extension))
                    raise Exception(f"running 'summarypy {' '.join(cmd)}' led to diff, see files at {error_path}") from e


def test_single_summary_set():
    test_case = {
        manager.RUNTYPE_GROUNDUP_LOSS: [1, ],
        manager.RUNTYPE_INSURED_LOSS: [1, ],
        manager.RUNTYPE_REINSURANCE_LOSS: [1, ],
    }
    case_runner('single_summary_set', test_case)


def test_multiple_summary_set():
    test_case = {
        manager.RUNTYPE_GROUNDUP_LOSS: [1, 2],
        manager.RUNTYPE_INSURED_LOSS: [1, 2, 3],
        manager.RUNTYPE_REINSURANCE_LOSS: [1, 2],
    }
    case_runner('multiple_summary_set', test_case)


@pytest.mark.parametrize('map_format', ['parquet', 'csv'])
@pytest.mark.parametrize('sample_size', [10, 0], ids=['samples', 'mean_only'])
def test_a_packed_items_buildings_add_into_its_summary(tmp_path, sample_size, map_format):
    """A summary is per item, and a packed item stands for all its buildings, so every building's
    samples, mean, maximum and TIV go into the item's summary, and each building of an affected
    location counts as an affected risk, as it would as an item of its own. A mean-only run
    (number_of_samples defaults to 0) still carries a set of specials per building.

    The summary map is written here; that generation puts ``number_of_buildings`` on it is tested
    by the summary-info TIV test in tests/computation/test_generate_losses.py. Generation writes it
    as parquet; a csv is what older versions wrote."""
    n_buildings = 3
    run_dir, gul = packed_gul_stream(tmp_path / 'model', n_building=-n_buildings, sample_size=sample_size)
    # a location per coverage, and a summary per parity of the location
    items = pd.read_csv(run_dir / 'input' / 'items.csv').set_index('item_id')['coverage_id']
    xref = np.zeros(len(items), dtype=gul_summary_xref_dtype)
    xref['item_id'], xref['summary_id'], xref['summaryset_id'] = items.index, 1 + items % 2, 1
    xref.tofile(tmp_path / 'gulsummaryxref.bin')
    summary_map = pd.DataFrame({'loc_id': items, 'item_id': items.index, 'building_id': 1,
                                'number_of_buildings': n_buildings})
    if map_format == 'parquet':
        summary_map.to_parquet(tmp_path / 'gul_summary_map.parquet', index=False)
    else:
        summary_map.to_csv(tmp_path / 'gul_summary_map.csv', index=False)

    out = tmp_path / 'summary.bin'
    manager.run(files_in=[str(gul)], static_path=str(tmp_path), run_type=manager.RUNTYPE_GROUNDUP_LOSS,
                low_memory=False, output_zeros=False, summary_sets_output=['-1', str(out)])
    got = read_summary_stream(out)

    stream = read_gul_stream(gul)[2]
    assert decode_building(stream['sidx'].min(), sample_size) == n_buildings, "the stream is not packed"
    stream['loc_id'] = stream['item_id'].map(items)
    stream['summary_id'] = 1 + stream['loc_id'] % 2
    stream['sidx'] = [decode_local_sidx(s, sample_size) for s in stream['sidx']]
    expected = stream.groupby(['event_id', 'summary_id', 'sidx'])['loss'].sum()

    assert set(got['sidx']) == {MAX_LOSS_IDX, NUMBER_OF_AFFECTED_RISK_IDX, MEAN_IDX, *range(1, sample_size + 1)}
    summed = got[got['sidx'] != NUMBER_OF_AFFECTED_RISK_IDX].set_index(['event_id', 'summary_id', 'sidx'])['loss']
    summed_sidx = expected.index.get_level_values('sidx').isin([MAX_LOSS_IDX, MEAN_IDX, *range(1, sample_size + 1)])
    expected_summed = expected[summed_sidx & (expected != 0)]  # summarypy writes no zero loss
    assert summed.index.sort_values().equals(expected_summed.index.sort_values())
    np.testing.assert_allclose(summed, expected_summed.loc[summed.index], rtol=1e-5)
    risks = got[got['sidx'] == NUMBER_OF_AFFECTED_RISK_IDX].set_index(['event_id', 'summary_id'])['loss']
    affected_locations = stream[stream['loss'] != 0].groupby(['event_id', 'summary_id'])['loc_id'].nunique()
    pd.testing.assert_series_equal(risks.sort_index(), (n_buildings * affected_locations).sort_index(),
                                   check_names=False, check_dtype=False)
    exposure = got.groupby(['event_id', 'summary_id'])['exposure_value'].first()
    np.testing.assert_allclose(exposure, expected.xs(TIV_IDX, level='sidx').loc[exposure.index], rtol=1e-5)


def read_summary_bin_as_rows(file_path):
    """Decode a binary summary stream into (EventId, SummaryId, SampleId, Loss, ImpactedExposure) rows."""
    buf = np.fromfile(file_path, dtype=np.uint8)
    cursor = manager.SUMMARY_META_SIZE
    rows = []
    while cursor < buf.shape[0]:
        event_id, cursor = mv_read(buf, cursor, EVENT_ID_DTYPE, EVENT_ID_SIZE)
        summary_id, cursor = mv_read(buf, cursor, SUMMARY_ID_DTYPE, SUMMARY_ID_SIZE)
        impacted_exposure, cursor = mv_read(buf, cursor, LOSS_DTYPE, LOSS_SIZE)
        while True:
            sidx, cursor = mv_read(buf, cursor, SIDX_DTYPE, SIDX_SIZE)
            loss, cursor = mv_read(buf, cursor, LOSS_DTYPE, LOSS_SIZE)
            if sidx == 0:
                break
            rows.append((event_id, summary_id, sidx, loss, impacted_exposure))
    return np.array(rows, dtype=SELT_dtype)


@pytest.mark.parametrize('test_name, test_case', [
    ('single_summary_set', {manager.RUNTYPE_GROUNDUP_LOSS: [1], manager.RUNTYPE_INSURED_LOSS: [1], manager.RUNTYPE_REINSURANCE_LOSS: [1]}),
    ('multiple_summary_set', {manager.RUNTYPE_GROUNDUP_LOSS: [1, 2], manager.RUNTYPE_INSURED_LOSS: [1, 2, 3],
                              manager.RUNTYPE_REINSURANCE_LOSS: [1, 2]}),
    ('multiple_summary_set', {manager.RUNTYPE_GROUNDUP_LOSS: [2, 1], manager.RUNTYPE_INSURED_LOSS: [3, 1, 2],
                              manager.RUNTYPE_REINSURANCE_LOSS: [2, 1]}),
])
@pytest.mark.parametrize('ext', ['csv', 'parquet'])
@pytest.mark.parametrize('buffer_size', [manager.OUTPUT_ROWS_BUFFER_SIZE, 7])
def test_tabular_output_matches_binary_stream(test_name, test_case, ext, buffer_size, monkeypatch):
    monkeypatch.setattr(manager, 'OUTPUT_ROWS_BUFFER_SIZE', buffer_size)
    base_path = Path(TESTS_ASSETS_DIR, test_name)
    with TemporaryDirectory() as tmp_result_dir_str:
        for run_type, summary_set_ids in test_case.items():
            if run_type == manager.RUNTYPE_REINSURANCE_LOSS:
                static_path = base_path.joinpath('RI_1')
                output_zeros = ' -z'
            else:
                static_path = base_path
                output_zeros = ''
            summary_sets_cmd = ''.join(f" -{summary_set_id} {Path(tmp_result_dir_str, f'{run_type}_S{summary_set_id}_summary.{ext}')}"
                                       for summary_set_id in summary_set_ids)
            cmd = (f"-E {ext} -t {run_type}{output_zeros} -p {static_path}"
                   f" -i {Path(TESTS_ASSETS_DIR, run_type + '.bin')}{summary_sets_cmd}").split()
            kwargs = vars(parser.parse_args(cmd))
            kwargs.pop('logging_level')
            manager.main(**kwargs)

            for summary_set_id in summary_set_ids:
                base_file_name = f"{run_type}_S{summary_set_id}_summary"
                expected = pd.DataFrame(read_summary_bin_as_rows(Path(base_path, base_file_name + '.bin')))
                result_path = Path(tmp_result_dir_str, f"{base_file_name}.{ext}")
                if ext == 'parquet':
                    result = pd.read_parquet(result_path)
                    pd.testing.assert_frame_equal(result, expected)
                else:
                    result = pd.read_csv(result_path)
                    assert list(result.columns) == SELT_headers
                    pd.testing.assert_frame_equal(result[['EventId', 'SummaryId', 'SampleId']],
                                                  expected[['EventId', 'SummaryId', 'SampleId']], check_dtype=False)
                    np.testing.assert_allclose(result[['Loss', 'ImpactedExposure']], expected[['Loss', 'ImpactedExposure']],
                                               rtol=1e-6, atol=0.005)


@pytest.mark.parametrize('extra_args, error', [
    ('-E csv -m -1 {tmp}/S1.csv', 'only available for bin output'),
    ('-E parquet -1 {tmp}/S1.csv', 'Invalid file extension'),
])
def test_tabular_output_invalid_arguments(extra_args, error):
    with TemporaryDirectory() as tmp:
        cmd = (f"-t {manager.RUNTYPE_GROUNDUP_LOSS} -p {Path(TESTS_ASSETS_DIR, 'single_summary_set')}"
               f" -i {Path(TESTS_ASSETS_DIR, 'gul.bin')} " + extra_args.format(tmp=tmp)).split()
        kwargs = vars(parser.parse_args(cmd))
        kwargs.pop('logging_level')
        with pytest.raises(ValueError, match=error):
            manager.main(**kwargs)


def test_summary_sets_out_of_order():
    # each summary set's output must not depend on the order its -summary_set_id argument is given in
    test_case = {
        manager.RUNTYPE_GROUNDUP_LOSS: [2, 1],
        manager.RUNTYPE_INSURED_LOSS: [3, 1, 2],
        manager.RUNTYPE_REINSURANCE_LOSS: [2, 1],
    }
    case_runner('multiple_summary_set', test_case)


def test_subset_of_summary_sets():
    # summary sets requested without all of the ones before them in summaryxref
    test_case = {
        manager.RUNTYPE_GROUNDUP_LOSS: [2],
        manager.RUNTYPE_INSURED_LOSS: [3, 2],
        manager.RUNTYPE_REINSURANCE_LOSS: [2],
    }
    case_runner('multiple_summary_set', test_case)


def run_summarypy(run_type, static_path, summary_sets_path):
    # summary sets are passed to the manager directly, as argparse would reject a negative id ('--1')
    cmd = f"-m -t {run_type} -p {static_path} -i {Path(TESTS_ASSETS_DIR, run_type + '.bin')}".split()
    kwargs = vars(parser.parse_args(cmd))
    kwargs.pop('logging_level')
    kwargs['summary_sets_output'] = [arg for summary_set_id, path in summary_sets_path.items() for arg in (f"-{summary_set_id}", str(path))]
    manager.main(**kwargs)


@pytest.mark.parametrize('summary_set_id', [3, -1, 0])
def test_unknown_summary_set_id(summary_set_id):
    # the ground up summaryxref has sets 1 and 2; -1 must not be read as the last set
    with TemporaryDirectory() as tmp_result_dir_str:
        with pytest.raises(ValueError, match='not found in summaryxref'):
            run_summarypy(manager.RUNTYPE_GROUNDUP_LOSS, Path(TESTS_ASSETS_DIR, 'multiple_summary_set'),
                          {summary_set_id: Path(tmp_result_dir_str, 'S.bin')})


def test_unrequested_summary_set_is_not_read():
    # items missing from summary set 3, which isn't requested, must not change sets 1 and 2
    base_path = Path(TESTS_ASSETS_DIR, 'multiple_summary_set')
    with TemporaryDirectory() as tmp_dir_str:
        static_path = Path(tmp_dir_str, 'static')
        shutil.copytree(base_path, static_path, ignore=shutil.ignore_patterns('*.bin', '*.idx', 'error_files', 'RI_1'))
        xref = pd.read_csv(static_path / 'fmsummaryxref.csv')
        xref = xref[~((xref['summaryset_id'] == 3) & (xref['output'] <= 10))]
        xref.to_csv(static_path / 'fmsummaryxref.csv', index=False)

        run_type = manager.RUNTYPE_INSURED_LOSS
        run_summarypy(run_type, static_path, {i: Path(tmp_dir_str, f'{run_type}_S{i}_summary.bin') for i in (1, 2)})
        for summary_set_id in (1, 2):
            for file_extension in ['.bin', '.idx']:
                file_name = f'{run_type}_S{summary_set_id}_summary{file_extension}'
                assert filecmp.cmp(Path(tmp_dir_str, file_name), Path(base_path, file_name), shallow=False)


def test_requested_summary_set_missing_item(monkeypatch):
    # an item in the loss stream with no summaryxref row for a requested set must fail, not add its loss elsewhere
    base_path = Path(TESTS_ASSETS_DIR, 'multiple_summary_set')
    with TemporaryDirectory() as tmp_dir_str:
        monkeypatch.chdir(tmp_dir_str)  # summarypy writes event_error.json to the working directory on failure
        static_path = Path(tmp_dir_str, 'static')
        shutil.copytree(base_path, static_path, ignore=shutil.ignore_patterns('*.bin', '*.idx', 'error_files', 'RI_1'))
        xref = pd.read_csv(static_path / 'fmsummaryxref.csv')
        xref = xref[~((xref['summaryset_id'] == 2) & (xref['output'] <= 10))]
        xref.to_csv(static_path / 'fmsummaryxref.csv', index=False)

        with pytest.raises(OasisStreamException, match='has no summaryxref row'):
            run_summarypy(manager.RUNTYPE_INSURED_LOSS, static_path,
                          {i: Path(tmp_dir_str, f'S{i}_summary.bin') for i in (1, 2)})
