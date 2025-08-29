from io import BufferedReader
import shutil
import sys
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from oasislmf.pytools.common.input_files import read_event_rates
from oasislmf.pytools.elt.manager import main
from oasislmf.pytools.common.data import (oasis_int, oasis_float)

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_eltpy")


def case_runner(test_name, out_ext="csv", with_event_rate=False):
    if out_ext not in ["csv", "bin", "parquet"]:
        raise Exception(f"Invalid or unimplemented test case for .{out_ext} output files for eltpy")

    outfile_name = f"py_{test_name}.{out_ext}"
    summary_bin_input = Path(TESTS_ASSETS_DIR, "summarypy.bin")
    if with_event_rate:
        outfile_name = f"py_{test_name}_er.{out_ext}"
    expected_outfile = Path(TESTS_ASSETS_DIR, outfile_name)
    with TemporaryDirectory() as tmp_result_dir_str:
        actual_outfile = Path(tmp_result_dir_str, outfile_name)

        kwargs = {
            "run_dir": TESTS_ASSETS_DIR,
            "files_in": summary_bin_input,
            "ext": out_ext,
        }

        if test_name in ["selt", "melt", "qelt"]:
            kwargs[f"{test_name}"] = actual_outfile
        else:
            raise Exception(f"Invalid or unimplemented test case {test_name} for eltpy")

        if with_event_rate:
            eids, ers = read_event_rates(Path(TESTS_ASSETS_DIR, "input"), filename="er.csv")
            with patch('oasislmf.pytools.elt.manager.read_event_rates', return_value=(eids, ers)):
                main(**kwargs)
        else:
            with patch('oasislmf.pytools.elt.manager.read_event_rates', return_value=(np.array([], dtype=oasis_int), np.array([], dtype=oasis_float))):
                main(**kwargs)

        try:
            if out_ext == "csv":
                expected_outfile_data = np.genfromtxt(expected_outfile, delimiter=',', skip_header=1)
                actual_outfile_data = np.genfromtxt(actual_outfile, delimiter=',', skip_header=1)
                if expected_outfile_data.shape != actual_outfile_data.shape:
                    raise AssertionError(
                        f"Shape mismatch: {expected_outfile} has shape {expected_outfile_data.shape}, {actual_outfile} has shape {actual_outfile_data.shape}")
                np.testing.assert_allclose(expected_outfile_data, actual_outfile_data, rtol=1e-5, atol=1e-8)
            if out_ext == "parquet":
                expected_outfile_data = pd.read_parquet(expected_outfile)
                actual_outfile_data = pd.read_parquet(actual_outfile)
                pd.testing.assert_frame_equal(expected_outfile_data, actual_outfile_data)
            if out_ext == "bin":
                with open(expected_outfile, 'rb') as f1, open(actual_outfile, 'rb') as f2:
                    assert f1.read() == f2.read()
        except Exception as e:
            error_path = TESTS_ASSETS_DIR.joinpath('error_files')
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(actual_outfile),
                            Path(error_path, outfile_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'eltpy {arg_str}' led to diff, see files at {error_path}") from e


def test_selt_output():
    case_runner("selt")


def test_melt_output():
    case_runner("melt")


def test_melt_output_with_event_rate():
    case_runner("melt", with_event_rate=True)


def test_qelt_output():
    case_runner("qelt")


def test_selt_output_bin():
    case_runner("selt", "bin")


def test_melt_output_bin():
    case_runner("melt", "bin")


def test_melt_output_bin_with_event_rate():
    case_runner("melt", "bin", with_event_rate=True)


def test_qelt_output_bin():
    case_runner("qelt", "bin")


def test_selt_output_parquet():
    case_runner("selt", "parquet")


def test_melt_output_parquet():
    case_runner("melt", "parquet")


def test_melt_output_parquet_with_event_rate():
    case_runner("melt", "parquet", with_event_rate=True)


def test_qelt_output_parquet():
    case_runner("qelt", "bin")


def test_selt_stdin(monkeypatch):
    input_file = Path(TESTS_ASSETS_DIR, "summarypy.bin")
    expected_outfile = Path(TESTS_ASSETS_DIR, "py_selt.csv")

    with TemporaryDirectory() as tmp_result_dir_str:
        actual_outfile = Path(tmp_result_dir_str, "py_selt.csv")

        f = open(input_file, "rb")

        mock_stdin = Mock()
        mock_stdin.buffer = BufferedReader(f)
        monkeypatch.setattr(sys, "stdin", mock_stdin)

        kwargs = {
            "run_dir": TESTS_ASSETS_DIR,
            "files_in": ["-"],  # default value to use stdin
            "ext": "csv",
            "selt": actual_outfile,
        }

        with patch('oasislmf.pytools.elt.manager.read_event_rates', return_value=(np.array([], dtype=oasis_int), np.array([], dtype=oasis_float))):
            main(**kwargs)

        f.close()

        try:
            expected_outfile_data = np.genfromtxt(expected_outfile, delimiter=',', skip_header=1)
            actual_outfile_data = np.genfromtxt(actual_outfile, delimiter=',', skip_header=1)
            if expected_outfile_data.shape != actual_outfile_data.shape:
                raise AssertionError(
                    f"Shape mismatch: {expected_outfile} has shape {expected_outfile_data.shape}, {actual_outfile} has shape {actual_outfile_data.shape}")
            np.testing.assert_allclose(expected_outfile_data, actual_outfile_data, rtol=1e-5, atol=1e-8)
        except Exception as e:
            error_path = TESTS_ASSETS_DIR.joinpath('error_files')
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(actual_outfile),
                            Path(error_path, "py_selt.csv"))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'eltpy {arg_str}' led to diff, see files at {error_path}") from e
