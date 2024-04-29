import filecmp
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from oasislmf.pytools.summary.cli import parser, manager


TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_summarypy")


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
