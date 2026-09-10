from io import BufferedReader
import shutil
import struct
import sys
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from oasislmf.pytools.common.event_stream import SUMMARY_STREAM_ID, stream_info_to_bytes, MEAN_IDX, NUMBER_OF_AFFECTED_RISK_IDX
from oasislmf.pytools.common.input_files import read_event_rates
import oasislmf.pytools.elt.manager as elt_manager
from oasislmf.pytools.elt.manager import main
from oasislmf.pytools.common.data import (oasis_int, oasis_float, quantile_interval_dtype)
from oasislmf.utils.exceptions import OasisStreamException

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_eltpy")


def _build_summary_stream(sample_size, summaries):
    """Build a minimal summary-stream binary: header, one summaryset_id, then per summary
    (event_id, summary_id, impacted_exposure, samples..., terminator).

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


def _make_intervals(quantiles, sample_size):
    rows = []
    for q in quantiles:
        pos = (sample_size - 1) * q + 1
        integer_part = int(pos)
        fractional_part = pos - integer_part
        rows.append((q, integer_part, fractional_part))
    return np.array(rows, dtype=quantile_interval_dtype)


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


def test_empty_input():
    """Test ELT does not crash and produces header-only output when summary binary has no loss records"""
    from oasislmf.pytools.common.event_stream import SUMMARY_STREAM_ID, stream_info_to_bytes

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # 12-byte header-only summary binary: stream_type (4B) + sample_size (4B) + reserved (4B)
        stream_header_int32 = np.frombuffer(stream_info_to_bytes(SUMMARY_STREAM_ID, 1), dtype=np.int32)[0]
        empty_bin = tmp_dir / "empty_summary.bin"
        np.array([stream_header_int32, 10, 1], dtype=np.int32).tofile(empty_bin)

        for elt_type in ["selt", "melt"]:
            outfile = tmp_dir / f"{elt_type}.csv"
            kwargs = {
                "run_dir": tmp_dir,
                "files_in": empty_bin,
                "ext": "csv",
                elt_type: outfile,
            }
            main(**kwargs)

            assert outfile.exists(), f"{elt_type}.csv was not created"
            lines = outfile.read_text().strip().splitlines()
            assert len(lines) == 1, f"{elt_type}.csv should contain only a header line, got {len(lines)} lines"


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


def test_melt_qelt_buffer_full_across_summaries():
    """A MELT/QELT output buffer filling up mid-run must not skip, duplicate, or
    misattribute rows for any summary, including ones that don't trigger the overflow
    themselves. Uses a tiny DEFAULT_BUFFER_SIZE to force this deterministically.
    """
    sample_size = 3
    summaries = [
        (999, 101, 1000.0, [(1, 10.0), (2, 20.0), (3, 30.0)]),
        (999, 102, 2000.0, [(1, 40.0), (2, 50.0), (3, 60.0)]),
        (999, 103, 3000.0, [(1, 70.0), (2, 80.0), (3, 90.0)]),
    ]
    stream_bytes = _build_summary_stream(sample_size, summaries)
    intervals = _make_intervals([0.5], sample_size)

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        stream_file = tmp_dir / "summary.bin"
        stream_file.write_bytes(stream_bytes)
        melt_out = tmp_dir / "melt.csv"
        qelt_out = tmp_dir / "qelt.csv"

        with patch('oasislmf.pytools.elt.manager.DEFAULT_BUFFER_SIZE', 4), \
                patch('oasislmf.pytools.elt.manager.read_event_rates', return_value=(np.array([], dtype=oasis_int), np.array([], dtype=oasis_float))), \
                patch('oasislmf.pytools.elt.manager.read_quantile', return_value=intervals):
            main(run_dir=tmp_dir, files_in=stream_file, melt=melt_out, qelt=qelt_out, ext="csv")

        melt = pd.read_csv(melt_out)
        qelt = pd.read_csv(qelt_out)

        assert len(melt) == 6, f"expected 6 MELT rows (3 summaries x 2), got {len(melt)}"
        assert list(melt["SummaryId"]) == [101, 101, 102, 102, 103, 103]
        assert (melt["EventId"] == 999).all()

        assert len(qelt) == 3, f"expected 3 QELT rows (3 summaries x 1 interval), got {len(qelt)} - MELT's overflow must not skip QELT"
        assert list(qelt["SummaryId"]) == [101, 102, 103]
        assert (qelt["EventId"] == 999).all()


def test_melt_buffer_full_immediately_before_new_event():
    """A MELT buffer-full flush that happens to land right before a genuine new event
    must still detect that event boundary correctly (not merge or lose it).
    """
    sample_size = 2
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
        melt_out = tmp_dir / "melt.csv"

        with patch('oasislmf.pytools.elt.manager.DEFAULT_BUFFER_SIZE', 4), \
                patch('oasislmf.pytools.elt.manager.read_event_rates', return_value=(np.array([], dtype=oasis_int), np.array([], dtype=oasis_float))):
            main(run_dir=tmp_dir, files_in=stream_file, melt=melt_out, ext="csv")

        melt = pd.read_csv(melt_out)
        assert len(melt) == 6
        assert list(melt["EventId"]) == [1000719084] * 4 + [1100028063] * 2
        assert list(melt["SummaryId"]) == [3820941, 3820941, 3820948, 3820948, 111, 111]


def test_selt_reservation_holds_with_mean_and_affected_risk_idx():
    """The SELT buffer-capacity reservation must account for every sidx that currently
    reaches SELT's write path: the len_sample real samples, plus MEAN_IDX, plus
    NUMBER_OF_AFFECTED_RISK_IDX. Undersizing the reservation is a silent out-of-bounds
    write under numba, not a catchable Python exception.
    """
    sample_size = 2
    summaries = [
        (999, 101, 1000.0, [(NUMBER_OF_AFFECTED_RISK_IDX, 2.0), (MEAN_IDX, 15.0), (1, 10.0), (2, 20.0)]),
    ]
    stream_bytes = _build_summary_stream(sample_size, summaries)

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        stream_file = tmp_dir / "summary.bin"
        stream_file.write_bytes(stream_bytes)
        selt_out = tmp_dir / "selt.csv"

        # Buffer sized to exactly what the reservation should reserve (len_sample + 2,
        # for the 2 samples + MEAN_IDX + NUMBER_OF_AFFECTED_RISK_IDX).
        with patch('oasislmf.pytools.elt.manager.DEFAULT_BUFFER_SIZE', sample_size + 2), \
                patch('oasislmf.pytools.elt.manager.read_event_rates', return_value=(np.array([], dtype=oasis_int), np.array([], dtype=oasis_float))):
            main(run_dir=tmp_dir, files_in=stream_file, selt=selt_out, ext="csv")

        selt = pd.read_csv(selt_out)
        expected_rows = sample_size + 2
        assert len(selt) == expected_rows, (
            f"expected {expected_rows} rows (MEAN_IDX + NUMBER_OF_AFFECTED_RISK_IDX + {sample_size} samples), got {len(selt)}"
        )
        assert sorted(selt["SampleId"]) == [-4, -1, 1, 2]


def test_selt_buffer_full_across_summaries():
    """A SELT output buffer filling up mid-run must not skip, duplicate, or
    misattribute rows for any summary. Uses a tiny DEFAULT_BUFFER_SIZE to force
    this deterministically.
    """
    sample_size = 2
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
        selt_out = tmp_dir / "selt.csv"

        with patch('oasislmf.pytools.elt.manager.DEFAULT_BUFFER_SIZE', 5), \
                patch('oasislmf.pytools.elt.manager.read_event_rates', return_value=(np.array([], dtype=oasis_int), np.array([], dtype=oasis_float))):
            main(run_dir=tmp_dir, files_in=stream_file, selt=selt_out, ext="csv")

        selt = pd.read_csv(selt_out)
        assert len(selt) == 6, f"expected 6 SELT rows (3 summaries x 2 samples), got {len(selt)}"
        assert list(selt["SummaryId"]) == [101, 101, 102, 102, 103, 103]
        assert (selt["EventId"] == 999).all()


def test_qelt_buffer_full_across_summaries():
    """A QELT output buffer filling up mid-run (without MELT also overflowing) must
    not skip, duplicate, or misattribute rows for any summary.
    """
    sample_size = 3
    summaries = [
        (999, 101, 1000.0, [(1, 10.0), (2, 20.0), (3, 30.0)]),
        (999, 102, 2000.0, [(1, 40.0), (2, 50.0), (3, 60.0)]),
        (999, 103, 3000.0, [(1, 70.0), (2, 80.0), (3, 90.0)]),
    ]
    stream_bytes = _build_summary_stream(sample_size, summaries)
    intervals = _make_intervals([0.25, 0.75], sample_size)

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        stream_file = tmp_dir / "summary.bin"
        stream_file.write_bytes(stream_bytes)
        qelt_out = tmp_dir / "qelt.csv"

        with patch('oasislmf.pytools.elt.manager.DEFAULT_BUFFER_SIZE', 3), \
                patch('oasislmf.pytools.elt.manager.read_quantile', return_value=intervals):
            main(run_dir=tmp_dir, files_in=stream_file, qelt=qelt_out, ext="csv")

        qelt = pd.read_csv(qelt_out)
        assert len(qelt) == 6, f"expected 6 QELT rows (3 summaries x 2 intervals), got {len(qelt)}"
        assert list(qelt["SummaryId"]) == [101, 101, 102, 102, 103, 103]
        assert (qelt["EventId"] == 999).all()


def test_selt_reservation_impossible_raises_instead_of_hanging():
    """If a single summary's worst-case SELT output can never fit in the buffer
    (e.g. sample size too large for DEFAULT_BUFFER_SIZE), read_buffer must raise
    rather than repeatedly yield a "buffer full" signal with zero progress, which
    would otherwise hang run() in an infinite loop.
    """
    sample_size = 5
    summaries = [(999, 101, 1000.0, [(i, float(i)) for i in range(1, sample_size + 1)])]
    stream_bytes = _build_summary_stream(sample_size, summaries)

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        stream_file = tmp_dir / "summary.bin"
        stream_file.write_bytes(stream_bytes)
        selt_out = tmp_dir / "selt.csv"

        # Buffer smaller than a single summary's worst case (len_sample + 2 = 7).
        with patch('oasislmf.pytools.elt.manager.DEFAULT_BUFFER_SIZE', 3), \
                patch('oasislmf.pytools.elt.manager.read_event_rates', return_value=(np.array([], dtype=oasis_int), np.array([], dtype=oasis_float))):
            with pytest.raises(OasisStreamException, match="SELT reservation"):
                main(run_dir=tmp_dir, files_in=stream_file, selt=selt_out, ext="csv")


def test_multifile_current_event_id_not_leaked_across_files():
    """current_event_id (and the rest of read_buffer's per-record state) is tracked
    per input file, not reader-wide. Without that, switching from file A to file B
    could compare file B's first header against file A's last event id, spuriously
    re-yielding file A's already-reported event with zero new rows whenever the two
    files' event ids happen to differ. Verify each file's first read_buffer call
    starts with current_event_id == 0 (no leakage), and output content is unaffected.
    """
    sample_size = 2
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
        melt_out = tmp_dir / "melt.csv"

        calls = []
        orig_read_buffer = elt_manager.ELTReader.read_buffer

        def spy_read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, file_idx):
            current_event_id_before = int(self.state[file_idx]["current_event_id"])
            result = orig_read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, file_idx)
            calls.append((file_idx, current_event_id_before, result[3]))
            return result

        with patch.object(elt_manager.ELTReader, "read_buffer", spy_read_buffer), \
                patch('oasislmf.pytools.elt.manager.read_event_rates', return_value=(np.array([], dtype=oasis_int), np.array([], dtype=oasis_float))):
            main(run_dir=tmp_dir, files_in=[file_a, file_b], melt=melt_out, ext="csv")

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
        # guaranteed (confirmed: CI processed file B before file A here) - compare content
        # order-independently rather than assuming file A's rows come first.
        melt = pd.read_csv(melt_out)
        assert len(melt) == 8, f"expected 8 MELT rows (4 summaries x 2), got {len(melt)}"
        expected = sorted([(999, 101), (999, 101), (999, 102), (999, 102),
                           (1000, 201), (1000, 201), (1000, 202), (1000, 202)])
        assert sorted(zip(melt["EventId"], melt["SummaryId"])) == expected
