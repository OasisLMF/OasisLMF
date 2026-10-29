import filecmp
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.summary.cli import parser, manager
from oasislmf.utils.exceptions import OasisStreamException


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
            for file_extention in ['.bin', '.idx']:
                file_name = f'{run_type}_S{summary_set_id}_summary{file_extention}'
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
