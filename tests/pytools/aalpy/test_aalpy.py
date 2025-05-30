import numpy as np
import os
import shutil
from pathlib import Path
import pandas as pd
from tempfile import TemporaryDirectory

from oasislmf.pytools.aal.manager import main

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_aalpy")


def case_runner(sub_folder, test_name, out_ext="csv", meanonly=False):
    """Run output file correctness tests
    Args:
        sub_folder (str | os.PathLike): path to workspace sub folder
        test_name (str): test name
    """
    print("HERE")
    outfile_name = f"py_{test_name}{sub_folder}.{out_ext}"
    if meanonly:
        outfile_name = f"py_{test_name}meanonly{sub_folder}.{out_ext}"
    expected_outfile = Path(TESTS_ASSETS_DIR, "all_files", outfile_name)
    with TemporaryDirectory() as tmp_result_dir_str:
        tmp_workspace_dir = Path(tmp_result_dir_str) / "workspace"
        shutil.copytree(Path(TESTS_ASSETS_DIR, "all_files"), tmp_workspace_dir)

        out_dir = tmp_workspace_dir / "out"
        out_dir.mkdir()

        actual_outfile = out_dir / outfile_name
        print("Workspace directory structure:")
        for root, dirs, files in os.walk(tmp_workspace_dir):
            print(f"Root: {root}")
            for d in dirs:
                print(f"  Dir: {d}")
            for f in files:
                print(f"  File: {f}")

        kwargs = {
            "run_dir": tmp_workspace_dir,
            "subfolder": sub_folder,
            "meanonly": meanonly,
            "ext": out_ext,
        }

        if test_name in ["aal", "alct"]:
            kwargs[f"{test_name}"] = actual_outfile
        else:
            raise Exception(f"Invalid or unimplemented test case {test_name} for aalpy")

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
            error_path = Path(TESTS_ASSETS_DIR, "all_files", "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_outfile, Path(error_path, outfile_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'aalpy {arg_str}' led to diff, see files at {error_path}") from e


def test_aal_output():
    """Tests AAL output
    """
    case_runner("gul", "aal", "csv")


def test_aalmeanonly_output():
    """Tests AAL meanonly output
    """
    case_runner("gul", "aal", "csv", meanonly=True)


def test_alct_output():
    """Tests ALCT output
    """
    case_runner("gul", "alct", "csv")


def test_aal_output_bin():
    """Tests AAL output
    """
    case_runner("gul", "aal", "bin")


def test_aalmeanonly_output_bin():
    """Tests AAL meanonly output
    """
    case_runner("gul", "aal", "bin", meanonly=True)


def test_alct_output_bin():
    """Tests ALCT output
    """
    case_runner("gul", "alct", "bin")


def test_aal_output_parquet():
    """Tests AAL output
    """
    case_runner("gul", "aal", "parquet")


def test_aalmeanonly_output_parquet():
    """Tests AAL meanonly output
    """
    case_runner("gul", "aal", "parquet", meanonly=True)


def test_alct_output_parquet():
    """Tests ALCT output
    """
    case_runner("gul", "alct", "parquet")
