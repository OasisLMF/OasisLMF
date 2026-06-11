import numpy as np
import os
import shutil
from pathlib import Path
import pandas as pd
import pytest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from oasislmf.pytools.common.data import occurrence_dtype, summary_stream_index_dtype
from oasislmf.pytools.common.event_stream import SUMMARY_STREAM_ID, stream_info_to_bytes
from oasislmf.pytools.aal.manager import _SUMMARIES_DTYPE_size, main

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_aalpy")


def make_idx_from_bin(bin_path: Path, idx_path: Path) -> None:
    """Scan a summary .bin and write a paired .idx recording each event block's byte offset.

    If the bin contains no data blocks (header only), creates a 0-byte idx to match
    summarypy --low-memory behaviour for partitions that received no events.
    """
    raw = np.fromfile(str(bin_path), dtype=np.int32)
    pos = 3  # skip 3-int stream header (stream_type, sample_size, summary_set_id)
    entries = []
    while pos < len(raw):
        byte_offset = pos * 4
        summary_id = int(raw[pos + 1])
        pos += 3  # event_id, summary_id, expval
        while pos < len(raw) and raw[pos] != 0:
            pos += 2  # sidx + loss pair (any non-zero sidx, including special negatives)
        pos += 2  # terminating (sidx=0, loss=0.0)
        entries.append((summary_id, byte_offset))
    if entries:
        np.array(entries, dtype=summary_stream_index_dtype).tofile(str(idx_path))
    else:
        idx_path.touch()  # empty partition → 0-byte idx


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


def test_empty_input():
    """Test AAL does not crash and produces no output when summary binary has no loss records"""
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        input_dir = tmp_dir / "input"
        work_dir = tmp_dir / "work" / "gul"
        input_dir.mkdir(parents=True)
        work_dir.mkdir(parents=True)

        # Minimal occurrence.bin: date_opts=1, no_of_periods=1000, no event records
        np.array([1, 1000], dtype=np.int32).tofile(input_dir / "occurrence.bin")

        # 12-byte header-only summary binary (no loss records)
        stream_header_int32 = np.frombuffer(stream_info_to_bytes(SUMMARY_STREAM_ID, 1), dtype=np.int32)[0]
        np.array([stream_header_int32, 10, 1], dtype=np.int32).tofile(work_dir / "summarypy1.bin")

        out_dir = tmp_dir / "out"
        out_dir.mkdir()
        outfile = out_dir / "aal.csv"

        kwargs = {
            "run_dir": tmp_dir,
            "subfolder": "gul",
            "ext": "csv",
            "aal": outfile,
        }
        main(**kwargs)

        # Early return when no loss records: output file should not be created
        assert not outfile.exists(), "aal.csv should not be created when there are no loss records"


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


# ---------------------------------------------------------------------------
# .idx path tests
# ---------------------------------------------------------------------------

def test_aal_idx_output_matches_sequential():
    """AAL .idx path produces identical output to the sequential path."""
    outfile_name = "py_aalgul.csv"
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str) / "workspace"
        shutil.copytree(Path(TESTS_ASSETS_DIR, "all_files"), tmp_dir)

        work_dir = tmp_dir / "work" / "gul"
        for bin_path in sorted(work_dir.glob("*.bin")):
            make_idx_from_bin(bin_path, bin_path.with_suffix(".idx"))

        out_dir = tmp_dir / "out"
        out_dir.mkdir()
        actual_outfile = out_dir / outfile_name

        main(run_dir=tmp_dir, subfolder="gul", ext="csv", aal=actual_outfile)

        expected = np.genfromtxt(tmp_dir / outfile_name, delimiter=',', skip_header=1)
        actual = np.genfromtxt(actual_outfile, delimiter=',', skip_header=1)
        assert expected.shape == actual.shape, f"Shape mismatch: {expected.shape} vs {actual.shape}"
        np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-8,
                                   err_msg=".idx path output differs from sequential for AAL")


def test_aal_idx_empty_file_no_crash():
    """A 0-byte .idx for an empty partition does not raise ValueError (crash regression).

    summarypy --low-memory creates 0-byte .idx files for partitions that received no
    events. Before the fix np.memmap raised 'ValueError: cannot mmap an empty file'.
    """
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str) / "workspace"
        shutil.copytree(Path(TESTS_ASSETS_DIR, "all_files"), tmp_dir)

        work_dir = tmp_dir / "work" / "gul"
        for bin_path in sorted(work_dir.glob("*.bin")):
            make_idx_from_bin(bin_path, bin_path.with_suffix(".idx"))

        # Add an empty partition: header-only .bin + 0-byte .idx
        stream_hdr = np.frombuffer(stream_info_to_bytes(SUMMARY_STREAM_ID, 1), dtype=np.int32)[0]
        np.array([stream_hdr, 100, 1], dtype=np.int32).tofile(work_dir / "summarypy9.bin")
        (work_dir / "summarypy9.idx").touch()

        out_dir = tmp_dir / "out"
        out_dir.mkdir()

        main(run_dir=tmp_dir, subfolder="gul", ext="csv", aal=out_dir / "py_aalgul.csv")
        assert (out_dir / "py_aalgul.csv").exists(), "AAL output should be created when populated bins are present"


# ---------------------------------------------------------------------------
# Buffer overflow guard tests
# ---------------------------------------------------------------------------

def _make_occurrence_with_event(input_dir, event_id, n_periods):
    """Write an occurrence.bin where event_id appears once in each of n_periods periods."""
    header = np.array([1, n_periods], dtype=np.int32)  # date_opts=1, no_of_periods
    records = np.zeros(n_periods, dtype=occurrence_dtype)
    records["event_id"] = event_id
    records["period_no"] = np.arange(1, n_periods + 1)
    with open(input_dir / "occurrence.bin", "wb") as f:
        f.write(header.tobytes())
        f.write(records.tobytes())


def _make_summary_bin_with_event(work_dir, filename, event_id, summary_id, sample_size=1):
    """Write a summary .bin containing one event block with one sample loss."""
    stream_hdr = np.frombuffer(stream_info_to_bytes(SUMMARY_STREAM_ID, sample_size), dtype=np.int32)[0]
    header = np.array([stream_hdr, sample_size, 1], dtype=np.int32)

    # Pack event_id(i32), summary_id(i32), expval(f32), sidx=1(i32), loss(f32), sidx=0(i32), loss=0(f32)
    # Build as int32 array, encoding floats via view to avoid struct dependency.
    block = np.zeros(7, dtype=np.int32)
    block[0] = event_id
    block[1] = summary_id
    block[2] = np.float32(0.0).view(np.int32)   # expval
    block[3] = 1                                  # sidx
    block[4] = np.float32(500.0).view(np.int32)  # loss
    block[5] = 0                                  # terminating sidx=0
    block[6] = np.float32(0.0).view(np.int32)    # terminating loss=0

    path = work_dir / filename
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(block.tobytes())
    return path


@pytest.mark.parametrize("use_idx", [False, True], ids=["sequential", "idx"])
def test_aal_oversize_event_raises(use_idx):
    """ValueError is raised when a single event's occurrence count exceeds the buffer size.

    Sets OASIS_AAL_MEMORY to exactly one buffer slot (buffer_size=1) and creates an
    event that maps to two periods (n_rows=2), so n_rows > buffer_size triggers the guard
    in both process_bin_file (sequential path) and process_idx_file (idx path).
    """
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        input_dir = tmp_dir / "input"
        work_dir = tmp_dir / "work" / "gul"
        input_dir.mkdir(parents=True)
        work_dir.mkdir(parents=True)

        # event_id=1 in 2 periods → n_rows=2
        _make_occurrence_with_event(input_dir, event_id=1, n_periods=2)
        bin_path = _make_summary_bin_with_event(work_dir, "summarypy1.bin", event_id=1, summary_id=1)

        if use_idx:
            make_idx_from_bin(bin_path, bin_path.with_suffix(".idx"))

        out_dir = tmp_dir / "out"
        out_dir.mkdir()

        # buffer_size = int(tiny_memory * 1024**3 / _SUMMARIES_DTYPE_size) = 1
        tiny_memory = _SUMMARIES_DTYPE_size / 1024 ** 3
        with patch("oasislmf.pytools.aal.manager.OASIS_AAL_MEMORY", tiny_memory):
            with pytest.raises(ValueError, match="OASIS_AAL_MEMORY is too small"):
                main(run_dir=tmp_dir, subfolder="gul", ext="csv", aal=out_dir / "aal.csv")
