"""Shared helpers for the converter tests (csvtobin / bintocsv / bintoparquet / parquettobin).

Test layout, so new tests have an obvious home:
- test_converters_<file type>.py: file types with their own logic and several tests
  (footprint, vulnerability). Give a type its own file once it outgrows its group (~3-4 tests).
- test_converters_model_files.py / _fm_files.py / _outputs.py: simple round trips, grouped.
- test_converters_io.py: stdin/stdout, pipe and seekability behaviour across types.
- test_converters_env.py: OASIS_FLOAT / OASIS_INT / areaperil build variants across types.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.converters.bintoparquet.manager import bintoparquet
from oasislmf.pytools.converters.parquettobin.manager import parquettobin
from oasislmf.pytools.converters.data import TOOL_INFO


TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_converters")


def case_runner(converter, file_type, sub_dir, filename=None, abnormal_dtype=False, **kwargs):
    if converter == "bintocsv":
        in_ext = ".bin"
        out_ext = ".csv"
        converter = bintocsv
    elif converter == "csvtobin":
        in_ext = ".csv"
        out_ext = ".bin"
        converter = csvtobin
    elif converter == "bintoparquet":
        in_ext = ".bin"
        out_ext = ".parquet"
        converter = bintoparquet
    elif converter == "parquettobin":
        in_ext = ".parquet"
        out_ext = ".bin"
        converter = parquettobin
    else:
        raise RuntimeError(f"Unknown test type {file_type}")

    if filename == None:
        filename = file_type
    with TemporaryDirectory() as tmp_result_dir_str:
        infile_name = f"{filename}{in_ext}"
        outfile_name = f"{filename}{out_ext}"
        infile = Path(TESTS_ASSETS_DIR, sub_dir, infile_name)
        expected_outfile = Path(TESTS_ASSETS_DIR, sub_dir, outfile_name)
        actual_outfile = Path(tmp_result_dir_str, outfile_name)

        converter_args = {
            "file_in": infile,
            "file_out": actual_outfile,
            "file_type": file_type,
            **kwargs,
        }
        converter(**converter_args)

        try:
            compare_conversion_outputs(expected_outfile, actual_outfile, file_type, out_ext, abnormal_dtype)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, sub_dir, "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_outfile, Path(error_path, outfile_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in converter_args.items()])
            raise Exception(f"running '{converter} {arg_str}' led to diff, see files at {error_path}") from e


def compare_conversion_outputs(expected_outfile, actual_outfile, file_type, out_ext, abnormal_dtype=False,
                               dtype=None):
    if out_ext == ".csv":
        expected_outfile_data = np.genfromtxt(expected_outfile, delimiter=',', skip_header=1)
        actual_outfile_data = np.genfromtxt(actual_outfile, delimiter=',', skip_header=1)
        if expected_outfile_data.shape != actual_outfile_data.shape:
            raise AssertionError(
                f"Shape mismatch: {expected_outfile} has shape {expected_outfile_data.shape}, {actual_outfile} has shape {actual_outfile_data.shape}"
            )
        np.testing.assert_allclose(expected_outfile_data, actual_outfile_data, rtol=1e-5, atol=1e-8)
    if out_ext == ".bin":
        if abnormal_dtype:  # This is if the binary file has headers or does not have a simple dtype, then compare raw bytes
            custom_dtype = "u1"
        elif dtype:
            custom_dtype = dtype
        else:  # Default dtype
            custom_dtype = TOOL_INFO[file_type]["dtype"]
        expected_outfile_data = pd.DataFrame(np.fromfile(expected_outfile, dtype=custom_dtype))
        actual_outfile_data = pd.DataFrame(np.fromfile(actual_outfile, dtype=custom_dtype))
        pd.testing.assert_frame_equal(expected_outfile_data, actual_outfile_data, check_exact=False, rtol=1e-3, atol=1e-4)
    if out_ext == ".parquet":
        expected_outfile_data = pd.read_parquet(expected_outfile)
        actual_outfile_data = pd.read_parquet(actual_outfile)
        pd.testing.assert_frame_equal(expected_outfile_data, actual_outfile_data, check_exact=False, rtol=1e-3, atol=1e-4)
    return


def case_runner_tocsv_with_zip_and_idx(file_type, sub_dir, filename, **kwargs):
    in_ext = ".bin"
    out_ext = ".csv"
    if kwargs["zip_files"]:
        in_ext = ".bin.z"

    with TemporaryDirectory() as tmp_result_dir_str:
        infile_name = f"{filename}{in_ext}"
        outfile_name = f"{filename}{out_ext}"
        infile = Path(TESTS_ASSETS_DIR, sub_dir, infile_name)
        expected_outfile = Path(TESTS_ASSETS_DIR, sub_dir, outfile_name)
        actual_outfile = Path(tmp_result_dir_str, outfile_name)

        converter_args = {
            "file_in": infile,
            "file_out": actual_outfile,
            "file_type": file_type,
            **kwargs,
        }
        bintocsv(**converter_args)

        try:
            expected_outfile_data = np.genfromtxt(expected_outfile, delimiter=',', skip_header=1)
            actual_outfile_data = np.genfromtxt(actual_outfile, delimiter=',', skip_header=1)
            if expected_outfile_data.shape != actual_outfile_data.shape:
                raise AssertionError(
                    f"Shape mismatch: {expected_outfile} has shape {expected_outfile_data.shape}, {actual_outfile} has shape {actual_outfile_data.shape}"
                )
            np.testing.assert_allclose(expected_outfile_data, actual_outfile_data, rtol=1e-5, atol=1e-8)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, sub_dir, "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_outfile, Path(error_path, outfile_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in converter_args.items()])
            raise Exception(f"running 'bintocsv {arg_str}' led to diff, see files at {error_path}") from e


def case_runner_tobin_with_zip_and_idx(file_type, sub_dir, filename, **kwargs):
    in_ext = ".csv"
    out_ext = ".bin"
    if kwargs["zip_files"]:
        out_ext = ".bin.z"

    with TemporaryDirectory() as tmp_result_dir_str:
        infile_name = f"{filename}{in_ext}"
        infile = Path(TESTS_ASSETS_DIR, sub_dir, infile_name)

        expected_idx_outfile = kwargs["idx_file_out"]
        idx_outfile_name = expected_idx_outfile.name
        actual_idx_outfile = Path(tmp_result_dir_str, idx_outfile_name)
        kwargs["idx_file_out"] = actual_idx_outfile

        outfile_name = f"{filename}{out_ext}"
        expected_outfile = Path(TESTS_ASSETS_DIR, sub_dir, outfile_name)
        actual_outfile = Path(tmp_result_dir_str, outfile_name)

        converter_args = {
            "file_in": infile,
            "file_out": actual_outfile,
            "file_type": file_type,
            **kwargs,
        }
        csvtobin(**converter_args)

        try:
            expected_outfile_data = pd.DataFrame(np.fromfile(expected_outfile, dtype="u1"))
            actual_outfile_data = pd.DataFrame(np.fromfile(actual_outfile, dtype="u1"))
            pd.testing.assert_frame_equal(expected_outfile_data, actual_outfile_data, check_exact=False, rtol=1e-3, atol=1e-4)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, sub_dir, "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_outfile, Path(error_path, outfile_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in converter_args.items()])
            raise Exception(f"running 'bintocsv {arg_str}' led to diff, see files at {error_path}") from e

        try:
            expected_idx_outfile_data = pd.DataFrame(np.fromfile(expected_idx_outfile, dtype="u1"))
            actual_idx_outfile_data = pd.DataFrame(np.fromfile(actual_idx_outfile, dtype="u1"))
            pd.testing.assert_frame_equal(expected_idx_outfile_data, actual_idx_outfile_data, check_exact=False, rtol=1e-3, atol=1e-4)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, sub_dir, "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_outfile, Path(error_path, outfile_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in converter_args.items()])
            raise Exception(f"running 'bintocsv {arg_str}' led to diff, see files at {error_path}") from e
