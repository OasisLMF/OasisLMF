from pathlib import Path
import pandas as pd
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from oasislmf.pytools.kat.manager import main

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_katpy")


def case_runner(dir_in, out_name, sorted):
    with TemporaryDirectory() as tmp_result_dir_str:
        dir_in = Path(TESTS_ASSETS_DIR, dir_in)
        expected_out = Path(TESTS_ASSETS_DIR, out_name)
        actual_out = Path(tmp_result_dir_str, out_name)

        kwargs = {
            "dir_in": dir_in,
            "qplt": True,
            "out": actual_out,
            "unsorted": not sorted,
        }

        main(**kwargs)

        suffix = Path(out_name).suffix

        try:
            if suffix == ".csv":
                expected_data = np.genfromtxt(expected_out, delimiter=',', skip_header=1)
                actual_data = np.genfromtxt(actual_out, delimiter=',', skip_header=1)
                if expected_data.shape != actual_data.shape:
                    raise AssertionError(
                        f"Shape mismatch: {expected_out} has shape {expected_data.shape}, {actual_out} has shape {actual_data.shape}")
                np.testing.assert_allclose(expected_data, actual_data, rtol=1e-5, atol=1e-8)
            if suffix == ".parquet":
                expected_data = pd.read_parquet(expected_out)
                actual_data = pd.read_parquet(actual_out)
                pd.testing.assert_frame_equal(expected_data, actual_data)
        except Exception as e:
            error_path = TESTS_ASSETS_DIR.joinpath('error_files')
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(actual_out),
                            Path(error_path, out_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'katpy {arg_str}' led to diff, see files at {error_path}") from e


def test_katpy_csv_sorted():
    """Test katpy with csv inputs (using QPLT) sorted"""
    case_runner("qplt", "katpy_qplt.csv", True)


def test_katpy_bin_sorted():
    """Test katpy with bin inputs (using QPLT) sorted"""
    case_runner("bqplt", "bkatpy_qplt.csv", True)


def test_katpy_parquet_sorted():
    """Test katpy with parquet inputs (using QPLT) sorted"""
    case_runner("pqplt", "pkatpy_qplt.parquet", True)


def test_katpy_csv_unsorted():
    """Test katpy with csv inputs (using QPLT) unsorted"""
    case_runner("qplt", "ukatpy_qplt.csv", False)


def test_katpy_bin_unsorted():
    """Test katpy with bin inputs (using QPLT) unsorted"""
    case_runner("bqplt", "ubkatpy_qplt.csv", False)


def test_katpy_parquet_unsorted():
    """Test katpy with parquet inputs (using QPLT) unsorted"""
    case_runner("pqplt", "upkatpy_qplt.parquet", False)
