from io import BufferedReader
import struct
import sys
from unittest.mock import Mock, patch
import numpy as np
import shutil
import pandas as pd
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from oasislmf.pytools.common.event_stream import SUMMARY_STREAM_ID, stream_info_to_bytes
from oasislmf.pytools.common.id_index import build as id_index_build
from oasislmf.pytools.common.input_files import OccurrenceCSR
import oasislmf.pytools.plt.manager as plt_manager
from oasislmf.pytools.plt.manager import main
from oasislmf.utils.exceptions import OasisStreamException

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_pltpy")


def _build_summary_stream(sample_size, summaries):
    """Build a minimal summary-stream binary, same layout as eltpy's test helper.

    summaries: list of (event_id, summary_id, impacted_exposure, [(sidx, loss), ...])
    """
    buf = bytearray()
    buf += struct.pack("<i", np.frombuffer(stream_info_to_bytes(SUMMARY_STREAM_ID, sample_size), dtype=np.int32)[0])
    buf += struct.pack("<i", sample_size)
    buf += struct.pack("<i", 1)  # summaryset_id, read once
    for event_id, summary_id, impacted_exposure, samples in summaries:
        buf += struct.pack("<i", event_id)
        buf += struct.pack("<i", summary_id)
        buf += struct.pack("<f", impacted_exposure)
        for sidx, loss in samples:
            buf += struct.pack("<i", sidx)
            buf += struct.pack("<f", loss)
        buf += struct.pack("<i", 0)
        buf += struct.pack("<f", 0.0)
    return bytes(buf)


def _make_occ_csr(event_to_periods):
    """event_to_periods: dict event_id -> list of period_no (occ_date_id fixed at 1)"""
    event_ids = sorted(event_to_periods.keys())
    valtype = np.dtype([("period_no", np.int32), ("occ_date_id", np.int32)])
    occ_flat = np.empty(sum(len(v) for v in event_to_periods.values()), dtype=valtype)
    occ_offsets = np.zeros(len(event_ids) + 1, dtype=np.int64)
    pos = 0
    for i, eid in enumerate(event_ids):
        for p in event_to_periods[eid]:
            occ_flat[pos] = (p, 1)
            pos += 1
        occ_offsets[i + 1] = pos
    return OccurrenceCSR(id_index_build(np.array(event_ids, dtype=np.int64)), occ_offsets, occ_flat)


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


def test_empty_input():
    """Test PLT does not crash and produces header-only output when summary binary has no loss records"""
    from oasislmf.pytools.common.event_stream import SUMMARY_STREAM_ID, stream_info_to_bytes

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        input_dir = tmp_dir / "input"
        input_dir.mkdir()

        # Minimal occurrence.bin: date_opts=1, no_of_periods=1000, no event records
        np.array([1, 1000], dtype=np.int32).tofile(input_dir / "occurrence.bin")

        # 12-byte header-only summary binary
        stream_header_int32 = np.frombuffer(stream_info_to_bytes(SUMMARY_STREAM_ID, 1), dtype=np.int32)[0]
        empty_bin = tmp_dir / "empty_summary.bin"
        np.array([stream_header_int32, 10, 1], dtype=np.int32).tofile(empty_bin)

        outfile = tmp_dir / "splt.csv"
        kwargs = {
            "run_dir": tmp_dir,
            "files_in": empty_bin,
            "ext": "csv",
            "splt": outfile,
        }
        main(**kwargs)

        assert outfile.exists(), "splt.csv was not created"
        lines = outfile.read_text().strip().splitlines()
        assert len(lines) == 1, f"splt.csv should contain only a header line, got {len(lines)} lines"


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


def test_mplt_qplt_buffer_full_across_summaries():
    """An MPLT/QPLT output buffer filling up mid-run must not skip, duplicate, or
    misattribute rows for any summary, including ones that don't trigger the overflow
    themselves. Uses a tiny DEFAULT_BUFFER_SIZE to force this deterministically.
    """
    sample_size = 3
    occ_csr = _make_occ_csr({999: [1]})
    period_weights = np.array([(1, 1.0)], dtype=np.dtype([("period_no", np.int32), ("weighting", "f4")]))
    intervals = np.array([(0.5, 2, 0.0)], dtype=np.dtype(
        [("quantile", "f4"), ("integer_part", "i4"), ("fractional_part", "f4")]))
    summaries = [
        (999, 101, 1000.0, [(1, 10.0), (2, 20.0), (3, 30.0)]),
        (999, 102, 2000.0, [(1, 40.0), (2, 50.0), (3, 60.0)]),
        (999, 103, 3000.0, [(1, 70.0), (2, 80.0), (3, 90.0)]),
    ]
    stream_bytes = _build_summary_stream(sample_size, summaries)

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        stream_file = tmp_dir / "summary.bin"
        stream_file.write_bytes(stream_bytes)
        mplt_out = tmp_dir / "mplt.csv"
        qplt_out = tmp_dir / "qplt.csv"

        with patch('oasislmf.pytools.plt.manager.DEFAULT_BUFFER_SIZE', 2), \
                patch('oasislmf.pytools.plt.manager.read_occurrence', return_value=(occ_csr, 1, False, 1)), \
                patch('oasislmf.pytools.plt.manager.read_periods', return_value=period_weights), \
                patch('oasislmf.pytools.plt.manager.read_quantile', return_value=intervals):
            main(run_dir=tmp_dir, files_in=stream_file, mplt=mplt_out, qplt=qplt_out, ext="csv")

        mplt = pd.read_csv(mplt_out)
        qplt = pd.read_csv(qplt_out)

        assert len(mplt) == 3, f"expected 3 MPLT rows (3 summaries x 1 period sample-mean), got {len(mplt)}"
        assert list(mplt["SummaryId"]) == [101, 102, 103]
        assert (mplt["EventId"] == 999).all()

        assert len(qplt) == 3, f"expected 3 QPLT rows (3 summaries x 1 period x 1 interval), got {len(qplt)} - MPLT's overflow must not skip QPLT"
        assert list(qplt["SummaryId"]) == [101, 102, 103]
        assert (qplt["EventId"] == 999).all()


def test_splt_buffer_full_across_summaries():
    """An SPLT output buffer filling up mid-run must not skip, duplicate, or
    misattribute rows for any summary. Uses a tiny DEFAULT_BUFFER_SIZE to force
    this deterministically.
    """
    sample_size = 2
    occ_csr = _make_occ_csr({999: [1]})
    period_weights = np.array([(1, 1.0)], dtype=np.dtype([("period_no", np.int32), ("weighting", "f4")]))
    summaries = [
        (999, 101, 1000.0, [(1, 10.0), (2, 20.0)]),
        (999, 102, 2000.0, [(1, 40.0), (2, 50.0)]),
        (999, 103, 3000.0, [(1, 70.0), (2, 80.0)]),
    ]
    stream_bytes = _build_summary_stream(sample_size, summaries)

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        stream_file = tmp_dir / "summary.bin"
        stream_file.write_bytes(stream_bytes)
        splt_out = tmp_dir / "splt.csv"

        with patch('oasislmf.pytools.plt.manager.DEFAULT_BUFFER_SIZE', 4), \
                patch('oasislmf.pytools.plt.manager.read_occurrence', return_value=(occ_csr, 1, False, 1)), \
                patch('oasislmf.pytools.plt.manager.read_periods', return_value=period_weights):
            main(run_dir=tmp_dir, files_in=stream_file, splt=splt_out, ext="csv")

        splt = pd.read_csv(splt_out)
        assert len(splt) == 6, f"expected 6 SPLT rows (3 summaries x 1 period x 2 samples), got {len(splt)}"
        assert list(splt["SummaryId"]) == [101, 101, 102, 102, 103, 103]
        assert (splt["EventId"] == 999).all()


def test_mplt_buffer_full_immediately_before_new_event():
    """An MPLT buffer-full flush that happens to land right before a genuine new event
    must still detect that event boundary correctly (not merge or lose it).
    """
    sample_size = 2
    occ_csr = _make_occ_csr({1000719084: [1], 1100028063: [1]})
    period_weights = np.array([(1, 1.0)], dtype=np.dtype([("period_no", np.int32), ("weighting", "f4")]))
    summaries = [
        (1000719084, 3820941, 100.0, [(1, 1.0), (2, 2.0)]),
        (1000719084, 3820948, 200.0, [(1, 3.0), (2, 4.0)]),
        (1100028063, 111, 300.0, [(1, 5.0), (2, 6.0)]),
    ]
    stream_bytes = _build_summary_stream(sample_size, summaries)

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        stream_file = tmp_dir / "summary.bin"
        stream_file.write_bytes(stream_bytes)
        mplt_out = tmp_dir / "mplt.csv"

        with patch('oasislmf.pytools.plt.manager.DEFAULT_BUFFER_SIZE', 2), \
                patch('oasislmf.pytools.plt.manager.read_occurrence', return_value=(occ_csr, 1, False, 1)), \
                patch('oasislmf.pytools.plt.manager.read_periods', return_value=period_weights):
            main(run_dir=tmp_dir, files_in=stream_file, mplt=mplt_out, ext="csv")

        mplt = pd.read_csv(mplt_out)
        assert len(mplt) == 3
        assert list(mplt["EventId"]) == [1000719084, 1000719084, 1100028063]
        assert list(mplt["SummaryId"]) == [3820941, 3820948, 111]


def test_qplt_buffer_full_across_summaries():
    """A QPLT output buffer filling up mid-run (without MPLT also overflowing) must
    not skip, duplicate, or misattribute rows for any summary.
    """
    sample_size = 3
    occ_csr = _make_occ_csr({999: [1]})
    period_weights = np.array([(1, 1.0)], dtype=np.dtype([("period_no", np.int32), ("weighting", "f4")]))
    intervals = np.array([(0.25, 3, 0.0), (0.75, 3, 0.0)], dtype=np.dtype(
        [("quantile", "f4"), ("integer_part", "i4"), ("fractional_part", "f4")]))
    summaries = [
        (999, 101, 1000.0, [(1, 10.0), (2, 20.0), (3, 30.0)]),
        (999, 102, 2000.0, [(1, 40.0), (2, 50.0), (3, 60.0)]),
        (999, 103, 3000.0, [(1, 70.0), (2, 80.0), (3, 90.0)]),
    ]
    stream_bytes = _build_summary_stream(sample_size, summaries)

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        stream_file = tmp_dir / "summary.bin"
        stream_file.write_bytes(stream_bytes)
        qplt_out = tmp_dir / "qplt.csv"

        with patch('oasislmf.pytools.plt.manager.DEFAULT_BUFFER_SIZE', 3), \
                patch('oasislmf.pytools.plt.manager.read_occurrence', return_value=(occ_csr, 1, False, 1)), \
                patch('oasislmf.pytools.plt.manager.read_periods', return_value=period_weights), \
                patch('oasislmf.pytools.plt.manager.read_quantile', return_value=intervals):
            main(run_dir=tmp_dir, files_in=stream_file, qplt=qplt_out, ext="csv")

        qplt = pd.read_csv(qplt_out)
        assert len(qplt) == 6, f"expected 6 QPLT rows (3 summaries x 1 period x 2 intervals), got {len(qplt)}"
        assert list(qplt["SummaryId"]) == [101, 101, 102, 102, 103, 103]
        assert (qplt["EventId"] == 999).all()


def test_splt_reservation_impossible_raises_instead_of_hanging():
    """If a single summary's worst-case SPLT output can never fit in the buffer
    (e.g. max_records_per_event x sample size too large for DEFAULT_BUFFER_SIZE),
    read_buffer must raise rather than repeatedly yield a "buffer full" signal with
    zero progress, which would otherwise hang run() in an infinite loop.
    """
    sample_size = 5
    occ_csr = _make_occ_csr({999: [1]})
    period_weights = np.array([(1, 1.0)], dtype=np.dtype([("period_no", np.int32), ("weighting", "f4")]))
    summaries = [(999, 101, 1000.0, [(i, float(i)) for i in range(1, sample_size + 1)])]
    stream_bytes = _build_summary_stream(sample_size, summaries)

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        stream_file = tmp_dir / "summary.bin"
        stream_file.write_bytes(stream_bytes)
        splt_out = tmp_dir / "splt.csv"

        # Buffer smaller than a single summary's worst case (1 period x (5 + 1) = 6).
        with patch('oasislmf.pytools.plt.manager.DEFAULT_BUFFER_SIZE', 3), \
                patch('oasislmf.pytools.plt.manager.read_occurrence', return_value=(occ_csr, 1, False, 1)), \
                patch('oasislmf.pytools.plt.manager.read_periods', return_value=period_weights):
            with pytest.raises(OasisStreamException, match="SPLT reservation"):
                main(run_dir=tmp_dir, files_in=stream_file, splt=splt_out, ext="csv")


def test_multifile_current_event_id_not_leaked_across_files():
    """current_event_id (and the rest of read_buffer's per-record state) is tracked
    per input file, not reader-wide. Without that, switching from file A to file B
    could compare file B's first header against file A's last event id, spuriously
    re-yielding file A's already-reported event with zero new rows whenever the two
    files' event ids happen to differ. Verify each file's first read_buffer call
    starts with current_event_id == 0 (no leakage), and output content is unaffected.
    """
    sample_size = 2
    occ_csr = _make_occ_csr({999: [1], 1000: [1]})
    period_weights = np.array([(1, 1.0)], dtype=np.dtype([("period_no", np.int32), ("weighting", "f4")]))
    stream_a = _build_summary_stream(sample_size, [
        (999, 101, 1000.0, [(1, 10.0), (2, 20.0)]),
        (999, 102, 2000.0, [(1, 30.0), (2, 40.0)]),
    ])
    stream_b = _build_summary_stream(sample_size, [
        (1000, 201, 3000.0, [(1, 50.0), (2, 60.0)]),
        (1000, 202, 4000.0, [(1, 70.0), (2, 80.0)]),
    ])

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        file_a = tmp_dir / "a.bin"
        file_b = tmp_dir / "b.bin"
        file_a.write_bytes(stream_a)
        file_b.write_bytes(stream_b)
        mplt_out = tmp_dir / "mplt.csv"

        calls = []
        orig_read_buffer = plt_manager.PLTReader.read_buffer

        def spy_read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, file_idx):
            current_event_id_before = int(self.state[file_idx]["current_event_id"])
            result = orig_read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, file_idx)
            calls.append((file_idx, current_event_id_before, result[3]))
            return result

        with patch.object(plt_manager.PLTReader, "read_buffer", spy_read_buffer), \
                patch('oasislmf.pytools.plt.manager.read_occurrence', return_value=(occ_csr, 1, False, 1)), \
                patch('oasislmf.pytools.plt.manager.read_periods', return_value=period_weights):
            main(run_dir=tmp_dir, files_in=[file_a, file_b], mplt=mplt_out, ext="csv")

        # Each file's very first read_buffer call must start with current_event_id == 0 -
        # not just the reader's first-ever call across all files.
        first_call_per_file = {}
        for file_idx, current_event_id_before, _ret in calls:
            first_call_per_file.setdefault(file_idx, current_event_id_before)
        assert all(v == 0 for v in first_call_per_file.values()), first_call_per_file

        # No spurious extra yield: exactly one read_buffer call per file (each file
        # is small enough to finish in one call with no event boundary inside it).
        assert len(calls) == 2, f"expected 1 read_buffer call per file (2 total), got {len(calls)}: {calls}"

        # read_streams interleaves files via a selector, so file processing order isn't
        # guaranteed - compare content order-independently rather than assuming file A's
        # rows come first.
        mplt = pd.read_csv(mplt_out)
        assert len(mplt) == 4, f"expected 4 MPLT rows (4 summaries x 1 period), got {len(mplt)}"
        expected = sorted([(999, 101), (999, 102), (1000, 201), (1000, 202)])
        assert sorted(zip(mplt["EventId"], mplt["SummaryId"])) == expected
