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
import zlib

from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.converters.bintoparquet.manager import bintoparquet
from oasislmf.pytools.converters.parquettobin.manager import parquettobin
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.common.data import vulnerability_bin_header_type
from oasislmf.pytools.getmodel.common import FootprintHeader, EventIndexBin_dtype, EventIndexBinZ_dtype
from oasislmf.pytools.getmodel.manager import VulnerabilityIndex_dtype


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
            if kwargs["zip_files"]:
                compare_zipped_bin_and_idx(expected_outfile, actual_outfile, expected_idx_outfile, actual_idx_outfile,
                                           file_type, kwargs.get("decompressed_size", False))
            else:
                for expected, actual in [(expected_outfile, actual_outfile), (expected_idx_outfile, actual_idx_outfile)]:
                    expected_data = pd.DataFrame(np.fromfile(expected, dtype="u1"))
                    actual_data = pd.DataFrame(np.fromfile(actual, dtype="u1"))
                    pd.testing.assert_frame_equal(expected_data, actual_data, check_exact=False, rtol=1e-3, atol=1e-4)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, sub_dir, "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_outfile, Path(error_path, outfile_name))
            shutil.copyfile(actual_idx_outfile, Path(error_path, idx_outfile_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in converter_args.items()])
            raise Exception(f"running 'csvtobin {arg_str}' led to diff, see files at {error_path}") from e


def compare_zipped_bin_and_idx(expected_bin, actual_bin, expected_idx, actual_idx, file_type, decompressed_size=False):
    """Compare zipped .bin/.idx outputs by their decompressed blocks rather than raw bytes.
    Deflate output isn't fixed across zlib implementations (e.g. zlib-ng, #2236), so the compressed
    bytes, and the idx offsets/sizes that follow from them, can differ while the data is identical.
    """
    if file_type == "footprint":
        header_size = FootprintHeader.itemsize
        idx_dtype = EventIndexBinZ_dtype if decompressed_size else EventIndexBin_dtype
    elif file_type == "vulnerability":
        header_size = vulnerability_bin_header_type.itemsize
        idx_dtype = VulnerabilityIndex_dtype
    else:
        raise RuntimeError(f"No zipped comparison for file type {file_type}")

    expected_bin_bytes = Path(expected_bin).read_bytes()
    actual_bin_bytes = Path(actual_bin).read_bytes()
    assert expected_bin_bytes[:header_size] == actual_bin_bytes[:header_size], "bin header mismatch"

    expected_idx_data = np.fromfile(expected_idx, dtype=idx_dtype)
    actual_idx_data = np.fromfile(actual_idx, dtype=idx_dtype)
    id_field = idx_dtype.names[0]
    np.testing.assert_array_equal(expected_idx_data[id_field], actual_idx_data[id_field])
    if "d_size" in idx_dtype.names:
        np.testing.assert_array_equal(expected_idx_data["d_size"], actual_idx_data["d_size"])

    # The actual idx must describe the actual bin: contiguous blocks from the header to EOF
    sizes = actual_idx_data["size"]
    np.testing.assert_array_equal(actual_idx_data["offset"], header_size + np.cumsum(sizes) - sizes)
    assert header_size + int(sizes.sum()) == len(actual_bin_bytes), "idx sizes don't cover the bin file"

    for expected_entry, actual_entry in zip(expected_idx_data, actual_idx_data):
        expected_block = zlib.decompress(
            expected_bin_bytes[expected_entry["offset"]:expected_entry["offset"] + expected_entry["size"]])
        actual_block = zlib.decompress(
            actual_bin_bytes[actual_entry["offset"]:actual_entry["offset"] + actual_entry["size"]])
        assert expected_block == actual_block, f"decompressed block mismatch for {id_field} {actual_entry[id_field]}"
