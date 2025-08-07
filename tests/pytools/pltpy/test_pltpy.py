from io import BufferedReader
import sys
from unittest.mock import Mock
import numpy as np
import shutil
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from oasislmf.pytools.plt.manager import main

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_pltpy")


def case_runner(sub_folder, test_name, out_ext="csv"):
    """Run output file correctness tests

    Args:
        sub_folder (str | os.PathLike): path to input files root
        test_name (str): test name
    """
    outfile_name = f"py_{test_name}.{out_ext}"
    summary_bin_input = Path(TESTS_ASSETS_DIR, "summarypy.bin")
    expected_outfile = Path(TESTS_ASSETS_DIR, sub_folder, outfile_name)
    with TemporaryDirectory() as tmp_result_dir_str:
        actual_outfile = Path(tmp_result_dir_str, outfile_name)

        kwargs = {
            "run_dir": Path(TESTS_ASSETS_DIR, sub_folder),
            "files_in": summary_bin_input,
            "ext": out_ext,
        }

        if test_name in ["splt", "mplt", "qplt"]:
            kwargs[f"{test_name}"] = actual_outfile
        else:
            raise Exception(f"Invalid or unimplemented test case {test_name} for pltpy")

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
            error_path = Path(TESTS_ASSETS_DIR, sub_folder, 'error_files')
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(actual_outfile),
                            Path(error_path, outfile_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'pltpy {arg_str}' led to diff, see files at {error_path}") from e


def test_splt_output():
    """Tests splt outputs
    """
    case_runner("all_files", "splt")  # All optional input files present
    case_runner("no_files", "splt")  # No optional input files present
    case_runner("occ_gran_files", "splt")  # Granular occurrence input file present


def test_mplt_output():
    """Tests mplt outputs
    """
    case_runner("all_files", "mplt")  # All optional input files present
    case_runner("no_files", "mplt")  # No optional input files present
    case_runner("occ_gran_files", "mplt")


def test_qplt_output():
    """Tests qplt outputs
    """
    case_runner("all_files", "qplt")  # All optional input files present
    case_runner("no_files", "qplt")  # No optional input files present
    case_runner("occ_gran_files", "qplt")  # Granular occurrence input file present


def test_splt_output_bin():
    """Tests splt outputs
    """
    case_runner("all_files", "splt", "bin")  # All optional input files present
    case_runner("no_files", "splt", "bin")  # No optional input files present
    case_runner("occ_gran_files", "splt", "bin")  # Granular occurrence input file present


def test_mplt_output_bin():
    """Tests mplt outputs
    """
    case_runner("all_files", "mplt", "bin")  # All optional input files present
    case_runner("no_files", "mplt", "bin")  # No optional input files present
    case_runner("occ_gran_files", "mplt", "bin")


def test_qplt_output_bin():
    """Tests qplt outputs
    """
    case_runner("all_files", "qplt", "bin")  # All optional input files present
    case_runner("no_files", "qplt", "bin")  # No optional input files present
    case_runner("occ_gran_files", "qplt", "bin")  # Granular occurrence input file present


def test_splt_output_parquet():
    """Tests splt outputs
    """
    case_runner("all_files", "splt", "parquet")  # All optional input files present
    case_runner("no_files", "splt", "parquet")  # No optional input files present
    case_runner("occ_gran_files", "splt", "parquet")  # Granular occurrence input file present


def test_mplt_output_parquet():
    """Tests mplt outputs
    """
    case_runner("all_files", "mplt", "parquet")  # All optional input files present
    case_runner("no_files", "mplt", "parquet")  # No optional input files present
    case_runner("occ_gran_files", "mplt", "parquet")


def test_qplt_output_parquet():
    """Tests qplt outputs
    """
    case_runner("all_files", "qplt", "parquet")  # All optional input files present
    case_runner("no_files", "qplt", "parquet")  # No optional input files present
    case_runner("occ_gran_files", "qplt", "parquet")  # Granular occurrence input file present


def test_splt_stdin(monkeypatch):
    test_asset_subdir = Path(TESTS_ASSETS_DIR, "all_files")
    input_file = Path(TESTS_ASSETS_DIR, "summarypy.bin")
    expected_outfile = Path(test_asset_subdir, "py_splt.csv")

    with TemporaryDirectory() as tmp_result_dir_str:
        actual_outfile = Path(tmp_result_dir_str, "py_splt.csv")

        f = open(input_file, "rb")

        mock_stdin = Mock()
        mock_stdin.buffer = BufferedReader(f)
        monkeypatch.setattr(sys, "stdin", mock_stdin)

        kwargs = {
            "run_dir": test_asset_subdir,
            "files_in": ["-"],  # default value to use stdin
            "ext": "csv",
            "splt": actual_outfile,
        }

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
            error_path = test_asset_subdir.joinpath('error_files')
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(actual_outfile),
                            Path(error_path, "py_splt.csv"))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'pltpy {arg_str}' led to diff, see files at {error_path}") from e
