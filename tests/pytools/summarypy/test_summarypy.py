import filecmp
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

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
