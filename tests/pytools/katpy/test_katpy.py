import filecmp
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from oasislmf.pytools.kat.manager import main

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_katpy")


def case_runner(dir_in, csv_name, sorted):
    with TemporaryDirectory() as tmp_result_dir_str:
        dir_in = Path(TESTS_ASSETS_DIR, dir_in)
        expected_out = Path(TESTS_ASSETS_DIR, csv_name)
        actual_out = Path(tmp_result_dir_str, csv_name)

        kwargs = {
            "dir_in": dir_in,
            "qplt": True,
            "out": actual_out,
            "unsorted": not sorted,
        }

        main(**kwargs)

        try:
            assert filecmp.cmp(expected_out, actual_out, shallow=False)
        except Exception as e:
            error_path = TESTS_ASSETS_DIR.joinpath('error_files')
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(actual_out),
                            Path(error_path, csv_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'eltpy {arg_str}' led to diff, see files at {error_path}") from e


def test_katpy_csv_sorted():
    """Test katpy with csv inputs (using QPLT) sorted"""
    case_runner("qplt", "katpy_qplt.csv", True)


def test_katpy_bin_sorted():
    """Test katpy with bin inputs (using QPLT) sorted"""
    case_runner("bqplt", "bkatpy_qplt.csv", True)


def test_katpy_csv_unsorted():
    """Test katpy with csv inputs (using QPLT) unsorted"""
    case_runner("qplt", "ukatpy_qplt.csv", False)


def test_katpy_bin_unsorted():
    """Test katpy with bin inputs (using QPLT) unsorted"""
    case_runner("bqplt", "ubkatpy_qplt.csv", False)
