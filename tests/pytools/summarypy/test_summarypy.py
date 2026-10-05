import filecmp
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.elt.data import SELT_dtype, SELT_headers
from oasislmf.pytools.summary.cli import parser, manager


TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_summarypy")

IDX_DTYPE = np.dtype([('summary_id', '<i4'), ('offset', '<i8')])  # fix dtype to reproduce test setup


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
                    for file_extention in ['.bin', '.idx']:
                        assert filecmp.cmp(Path(tmp_result_dir_str, base_file_name + file_extention),
                                           Path(base_path, base_file_name + file_extention), shallow=True)
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
                    for file_extention in ['.bin', '.idx']:
                        shutil.copyfile(Path(tmp_result_dir_str, base_file_name + file_extention),
                                        Path(error_path, base_file_name + file_extention))
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


def read_summary_bin_as_rows(file_path):
    """Decode a binary summary stream into (EventId, SummaryId, SampleId, Loss, ImpactedExposure) rows."""
    data = np.fromfile(file_path, dtype='<i4')[3:]  # skip stream type, sample size and summary set id
    rows = []
    i = 0
    while i < data.shape[0]:
        event_id, summary_id = data[i], data[i + 1]
        impacted_exposure = data[i + 2:i + 3].view('<f4')[0]
        i += 3
        while data[i] != 0:
            rows.append((event_id, summary_id, data[i], data[i + 1:i + 2].view('<f4')[0], impacted_exposure))
            i += 2
        i += 2
    return np.array(rows, dtype=SELT_dtype)


@pytest.mark.parametrize('test_name, test_case', [
    ('single_summary_set', {manager.RUNTYPE_GROUNDUP_LOSS: [1], manager.RUNTYPE_INSURED_LOSS: [1], manager.RUNTYPE_REINSURANCE_LOSS: [1]}),
    ('multiple_summary_set', {manager.RUNTYPE_GROUNDUP_LOSS: [1, 2], manager.RUNTYPE_INSURED_LOSS: [1, 2, 3],
                              manager.RUNTYPE_REINSURANCE_LOSS: [1, 2]}),
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
