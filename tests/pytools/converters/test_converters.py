import struct
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.converters.bintoparquet.manager import bintoparquet
from oasislmf.pytools.converters.parquettobin.manager import parquettobin
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.getmodel.common import Event_dtype, EventIndexBin_dtype
from oasislmf.pytools.getmodel.manager import VulnerabilityIndex_dtype
from oasislmf.utils.exceptions import OasisException

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
                else:
                    custom_dtype = TOOL_INFO[file_type]["dtype"]
                expected_outfile_data = pd.DataFrame(np.fromfile(expected_outfile, dtype=custom_dtype))
                actual_outfile_data = pd.DataFrame(np.fromfile(actual_outfile, dtype=custom_dtype))
                pd.testing.assert_frame_equal(expected_outfile_data, actual_outfile_data, check_exact=False, rtol=1e-3, atol=1e-4)
            if out_ext == ".parquet":
                expected_outfile_data = pd.read_parquet(expected_outfile)
                actual_outfile_data = pd.read_parquet(actual_outfile)
                pd.testing.assert_frame_equal(expected_outfile_data, actual_outfile_data, check_exact=False, rtol=1e-3, atol=1e-4)
        except Exception as e:
            error_path = Path(TESTS_ASSETS_DIR, sub_dir, "error_files")
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(actual_outfile, Path(error_path, outfile_name))
            arg_str = ' '.join([f"{k}={v}" for k, v in converter_args.items()])
            raise Exception(f"running '{converter} {arg_str}' led to diff, see files at {error_path}") from e


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


def test_aggregatevulnerability():
    case_runner("bintocsv", "aggregatevulnerability", "static")
    case_runner("csvtobin", "aggregatevulnerability", "static")


def test_damagebin():
    case_runner("bintocsv", "damagebin", "static")
    case_runner("csvtobin", "damagebin", "static", no_validation=False)


def test_footprint():
    # zip_input = False
    case_runner_tocsv_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint",
        idx_file_in=Path(TESTS_ASSETS_DIR, "static", "footprint.idx"),
        event_from_to="1-3",
        zip_files=False
    )
    case_runner_tobin_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "footprint.idx"),
        max_intensity_bin_idx=3,
        no_intensity_uncertainty=True,
        decompressed_size=False,
        no_validation=False,
        zip_files=False
    )
    # no_validation=True must produce identical output to validated path on clean input
    case_runner_tobin_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "footprint.idx"),
        max_intensity_bin_idx=3,
        no_intensity_uncertainty=True,
        decompressed_size=False,
        no_validation=True,
        zip_files=False
    )

    # zip_input = True
    case_runner_tocsv_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint_zip",
        idx_file_in=Path(TESTS_ASSETS_DIR, "static", "footprint_zip.idx.z"),
        event_from_to="1-3",
        zip_files=True
    )
    case_runner_tobin_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint_zip",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "footprint_zip.idx.z"),
        max_intensity_bin_idx=58,
        no_intensity_uncertainty=False,
        decompressed_size=False,
        no_validation=False,
        zip_files=True
    )
    # no_validation=True must produce identical output to validated path on clean input
    case_runner_tobin_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint_zip",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "footprint_zip.idx.z"),
        max_intensity_bin_idx=58,
        no_intensity_uncertainty=False,
        decompressed_size=False,
        no_validation=True,
        zip_files=True
    )
    # decompressed_size=True: index includes uncompressed size field alongside compressed
    case_runner_tobin_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint_zip_dsize",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "footprint_zip_dsize.idx.z"),
        max_intensity_bin_idx=58,
        no_intensity_uncertainty=False,
        decompressed_size=True,
        no_validation=False,
        zip_files=True
    )
    # decompressed_size=True tocsv: exercises pre-allocated buffer path (d_size in index)
    case_runner_tocsv_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint_zip_dsize",
        idx_file_in=Path(TESTS_ASSETS_DIR, "static", "footprint_zip_dsize.idx.z"),
        event_from_to="1-3",
        zip_files=True
    )


def test_footprint_no_validation_unsorted():
    """no_validation=False rejects unsorted input; no_validation=True accepts it and preserves order."""
    unsorted_csv = (
        b"event_id,areaperil_id,intensity_bin_id,probability\n"
        b"2,1,1,1.0\n"
        b"1,1,1,1.0\n"
    )
    with TemporaryDirectory() as tmp:
        csv_in = Path(tmp) / "footprint.csv"
        csv_in.write_bytes(unsorted_csv)

        # validated path must reject unsorted input
        with pytest.raises(OasisException, match="not in ascending order"):
            csvtobin(
                file_in=csv_in,
                file_out=Path(tmp) / "v.bin",
                file_type="footprint",
                idx_file_out=Path(tmp) / "v.idx",
                max_intensity_bin_idx=3,
                no_intensity_uncertainty=True,
                decompressed_size=False,
                no_validation=False,
                zip_files=False,
            )

        # no_validation path must accept unsorted input and preserve input order
        csvtobin(
            file_in=csv_in,
            file_out=Path(tmp) / "n.bin",
            file_type="footprint",
            idx_file_out=Path(tmp) / "n.idx",
            max_intensity_bin_idx=3,
            no_intensity_uncertainty=True,
            decompressed_size=False,
            no_validation=True,
            zip_files=False,
        )
        idx = np.fromfile(Path(tmp) / "n.idx", dtype=EventIndexBin_dtype)
        assert idx["event_id"].tolist() == [2, 1], (
            f"Expected events written in input order [2, 1], got {idx['event_id'].tolist()}"
        )


def test_footprint_unsorted_index():
    """bintocsv must sort events by event_id even when the index is out of order."""
    # Build a binary footprint: header + event 2 data (offset 8) + event 1 data (offset 20)
    header = struct.pack("<ii", 3, 1)  # num_intensity_bins=3, has_intensity_uncertainty=1
    ev2 = np.array([(1, 1, 1.0)], dtype=Event_dtype).tobytes()
    ev1 = np.array([(1, 1, 1.0)], dtype=Event_dtype).tobytes()
    bin_bytes = header + ev2 + ev1

    item_size = Event_dtype.itemsize  # 12
    header_size = 8
    # Index with event 2 first (unsorted order)
    idx = np.array(
        [(2, header_size, item_size), (1, header_size + item_size, item_size)],
        dtype=EventIndexBin_dtype,
    )

    with TemporaryDirectory() as tmp:
        bin_path = Path(tmp) / "footprint.bin"
        idx_path = Path(tmp) / "footprint.idx"
        out_path = Path(tmp) / "footprint.csv"
        bin_path.write_bytes(bin_bytes)
        idx.tofile(idx_path)

        bintocsv(
            file_in=bin_path,
            file_out=out_path,
            file_type="footprint",
            idx_file_in=idx_path,
            zip_files=False,
            event_from_to=None,
        )

        data = np.genfromtxt(out_path, delimiter=",", skip_header=1,
                             dtype=[("event_id", int), ("areaperil_id", int),
                                    ("intensity_bin_id", int), ("probability", float)])
        assert data["event_id"].tolist() == [1, 2], (
            f"Expected sorted output [1, 2], got {data['event_id'].tolist()}"
        )


def test_lossfactors():
    case_runner("bintocsv", "lossfactors", "static")
    case_runner("csvtobin", "lossfactors", "static")


def test_random():
    case_runner("bintocsv", "random", "static")
    case_runner("csvtobin", "random", "static")


def test_vulnerability():
    # zip_input = False
    # bintocsv
    case_runner(
        converter="bintocsv",
        file_type="vulnerability",
        sub_dir="static",
        filename="vulnerability_noidx",
        abnormal_dtype=True,
        idx_file_in=None,
        zip_files=False
    )
    case_runner_tocsv_with_zip_and_idx(
        file_type="vulnerability",
        sub_dir="static",
        filename="vulnerability_idx",
        idx_file_in=Path(TESTS_ASSETS_DIR, "static", "vulnerability_idx.idx"),
        zip_files=False
    )
    # csvtobin
    case_runner(
        converter="csvtobin",
        file_type="vulnerability",
        sub_dir="static",
        filename="vulnerability_noidx",
        abnormal_dtype=True,
        idx_file_out=None,
        max_damage_bin_idx=2,
        no_validation=False,
        suppress_int_bin_checks=False,
        zip_files=False
    )
    case_runner_tobin_with_zip_and_idx(
        file_type="vulnerability",
        sub_dir="static",
        filename="vulnerability_idx",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "vulnerability_idx.idx"),
        max_damage_bin_idx=2,
        no_validation=False,
        suppress_int_bin_checks=False,
        zip_files=False
    )

    # no_validation=True must produce identical output to validated path on clean input
    case_runner(
        converter="csvtobin",
        file_type="vulnerability",
        sub_dir="static",
        filename="vulnerability_noidx",
        abnormal_dtype=True,
        idx_file_out=None,
        max_damage_bin_idx=2,
        no_validation=True,
        suppress_int_bin_checks=False,
        zip_files=False
    )
    case_runner_tobin_with_zip_and_idx(
        file_type="vulnerability",
        sub_dir="static",
        filename="vulnerability_idx",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "vulnerability_idx.idx"),
        max_damage_bin_idx=2,
        no_validation=True,
        suppress_int_bin_checks=False,
        zip_files=False
    )

    # zip_input = True
    # bintocsv
    case_runner_tocsv_with_zip_and_idx(
        file_type="vulnerability",
        sub_dir="static",
        filename="vulnerability_idx",
        idx_file_in=Path(TESTS_ASSETS_DIR, "static", "vulnerability_idx.idx.z"),
        zip_files=True
    )
    # csvtobin
    case_runner_tobin_with_zip_and_idx(
        file_type="vulnerability",
        sub_dir="static",
        filename="vulnerability_idx",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "vulnerability_idx.idx.z"),
        max_damage_bin_idx=2,
        no_validation=False,
        suppress_int_bin_checks=False,
        zip_files=True
    )
    # no_validation=True must produce identical output to validated path on clean input
    case_runner_tobin_with_zip_and_idx(
        file_type="vulnerability",
        sub_dir="static",
        filename="vulnerability_idx",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "vulnerability_idx.idx.z"),
        max_damage_bin_idx=2,
        no_validation=True,
        suppress_int_bin_checks=False,
        zip_files=True
    )


def test_vulnerability_no_validation_unsorted():
    """no_validation=False rejects unsorted input; no_validation=True accepts it and preserves order."""
    unsorted_csv = (
        b"vulnerability_id,intensity_bin_id,damage_bin_id,probability\n"
        b"2,1,1,1.0\n"
        b"1,1,1,1.0\n"
    )
    with TemporaryDirectory() as tmp:
        csv_in = Path(tmp) / "vulnerability.csv"
        csv_in.write_bytes(unsorted_csv)

        with pytest.raises(OasisException, match="not in ascending order"):
            csvtobin(
                file_in=csv_in,
                file_out=Path(tmp) / "v.bin",
                file_type="vulnerability",
                idx_file_out=Path(tmp) / "v.idx",
                max_damage_bin_idx=2,
                no_validation=False,
                suppress_int_bin_checks=True,
                zip_files=False,
            )

        csvtobin(
            file_in=csv_in,
            file_out=Path(tmp) / "n.bin",
            file_type="vulnerability",
            idx_file_out=Path(tmp) / "n.idx",
            max_damage_bin_idx=2,
            no_validation=True,
            suppress_int_bin_checks=True,
            zip_files=False,
        )
        idx = np.fromfile(Path(tmp) / "n.idx", dtype=VulnerabilityIndex_dtype)
        assert idx["vulnerability_id"].tolist() == [2, 1], (
            f"Expected vulns written in input order [2, 1], got {idx['vulnerability_id'].tolist()}"
        )


def test_weights():
    case_runner("bintocsv", "weights", "static")
    case_runner("csvtobin", "weights", "static")


def test_amplifications():
    case_runner("bintocsv", "amplifications", "input")
    case_runner("csvtobin", "amplifications", "input")


def test_correlations_items():
    case_runner("bintocsv", "correlations", "input")
    case_runner("csvtobin", "correlations", "input")


def test_complex_items():
    case_runner("bintocsv", "complex_items", "input")
    case_runner("csvtobin", "complex_items", "input")


def test_coverages():
    case_runner("bintocsv", "coverages", "input")
    case_runner("csvtobin", "coverages", "input")


def test_eve():
    case_runner("bintocsv", "eve", "input")
    case_runner("csvtobin", "eve", "input")


def test_fm_policytc():
    case_runner("bintocsv", "fm_policytc", "input")
    case_runner("csvtobin", "fm_policytc", "input")


def test_fm_profile():
    case_runner("bintocsv", "fm_profile", "input")
    case_runner("csvtobin", "fm_profile", "input")


def test_fm_profile_step():
    case_runner("bintocsv", "fm_profile_step", "input")
    case_runner("csvtobin", "fm_profile_step", "input")


def test_fm_programme():
    case_runner("bintocsv", "fm_programme", "input")
    case_runner("csvtobin", "fm_programme", "input")


def test_fm_summary_xref():
    case_runner("bintocsv", "fm_summary_xref", "input")
    case_runner("csvtobin", "fm_summary_xref", "input")


def test_gul_summary_xref():
    case_runner("bintocsv", "gul_summary_xref", "input")
    case_runner("csvtobin", "gul_summary_xref", "input")


def test_fm_xref():
    case_runner("bintocsv", "fm_xref", "input")
    case_runner("csvtobin", "fm_xref", "input")


def test_items():
    case_runner("bintocsv", "items", "input")
    case_runner("csvtobin", "items", "input")


def test_occurrence():
    case_runner("bintocsv", "occurrence", "input", "occurrence")
    case_runner("bintocsv", "occurrence", "input", "occurrence_gran")
    case_runner("bintocsv", "occurrence", "input", "occurrence_noalg")
    case_runner("csvtobin", "occurrence", "input", "occurrence", no_of_periods=9)
    case_runner("csvtobin", "occurrence", "input", "occurrence_gran", no_of_periods=9, granular=True)
    case_runner("csvtobin", "occurrence", "input", "occurrence_noalg", no_of_periods=9, no_date_alg=True)


def test_periods():
    case_runner("bintocsv", "periods", "input")
    case_runner("csvtobin", "periods", "input")


def test_quantile():
    case_runner("bintocsv", "quantile", "input")
    case_runner("csvtobin", "quantile", "input")


def test_returnperiods():
    case_runner("bintocsv", "returnperiods", "input")
    case_runner("csvtobin", "returnperiods", "input")


def test_aal():
    # Test bin/csv
    case_runner("bintocsv", "aal", "output")
    case_runner("csvtobin", "aal", "output")
    case_runner("bintocsv", "aalmeanonly", "output")
    case_runner("csvtobin", "aalmeanonly", "output")
    case_runner("bintocsv", "alct", "output")
    case_runner("csvtobin", "alct", "output")

    # Test bin/parquet
    case_runner("bintoparquet", "aal", "output")
    case_runner("parquettobin", "aal", "output")
    case_runner("bintoparquet", "aalmeanonly", "output")
    case_runner("parquettobin", "aalmeanonly", "output")
    case_runner("bintoparquet", "alct", "output")
    case_runner("parquettobin", "alct", "output")


def test_elt():
    # Test bin/csv
    case_runner("bintocsv", "selt", "output")
    case_runner("csvtobin", "selt", "output")
    case_runner("bintocsv", "melt", "output")
    case_runner("csvtobin", "melt", "output")
    case_runner("bintocsv", "qelt", "output")
    case_runner("csvtobin", "qelt", "output")

    # Test bin/parquet
    case_runner("bintoparquet", "selt", "output")
    case_runner("parquettobin", "selt", "output")
    case_runner("bintoparquet", "melt", "output")
    case_runner("parquettobin", "melt", "output")
    case_runner("bintoparquet", "qelt", "output")
    case_runner("parquettobin", "qelt", "output")


def test_lec():
    # Test bin/csv
    case_runner("bintocsv", "ept", "output")
    case_runner("csvtobin", "ept", "output")
    case_runner("bintocsv", "psept", "output")
    case_runner("csvtobin", "psept", "output")

    # Test bin/parquet
    case_runner("bintoparquet", "ept", "output")
    case_runner("parquettobin", "ept", "output")
    case_runner("bintoparquet", "psept", "output")
    case_runner("parquettobin", "psept", "output")


def test_plt():
    # Test bin/csv
    case_runner("bintocsv", "splt", "output")
    case_runner("csvtobin", "splt", "output")
    case_runner("bintocsv", "mplt", "output")
    case_runner("csvtobin", "mplt", "output")
    case_runner("bintocsv", "qplt", "output")
    case_runner("csvtobin", "qplt", "output")

    # Test bin/parquet
    case_runner("bintoparquet", "splt", "output")
    case_runner("parquettobin", "splt", "output")
    case_runner("bintoparquet", "mplt", "output")
    case_runner("parquettobin", "mplt", "output")
    case_runner("bintoparquet", "qplt", "output")
    case_runner("parquettobin", "qplt", "output")


def test_fm():
    case_runner("bintocsv", "fm", "misc", "raw_ils")
    case_runner("csvtobin", "fm", "misc", "raw_ils", abnormal_dtype=True, stream_type=2, max_sample_index=1)


def test_gul():
    case_runner("bintocsv", "gul", "misc", "raw_guls")
    case_runner("csvtobin", "gul", "misc", "raw_guls", abnormal_dtype=True, stream_type=2, max_sample_index=1)


def test_summarycalc():
    case_runner("csvtobin", "summarycalc", "misc", "summary", abnormal_dtype=True, summary_set_id=1, max_sample_index=100)


def test_cdf():
    case_runner("bintocsv", "cdf", "cdftocsv", "getmodel", run_dir=Path(TESTS_ASSETS_DIR, "cdftocsv"))
