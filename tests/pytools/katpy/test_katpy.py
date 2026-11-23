from pathlib import Path
import pandas as pd
import shutil
import tempfile
from tempfile import TemporaryDirectory

import numpy as np
import psutil
import pytest
from oasislmf.pytools.kat import manager
from oasislmf.pytools.kat.data import KAT_MAP, KAT_SELT, KAT_SPLT
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


def test_empty_input():
    """Test katpy does not crash and produces header-only output when CSV inputs have no data rows"""
    from oasislmf.pytools.kat.manager import KAT_MAP, KAT_QPLT

    qplt_headers = KAT_MAP[KAT_QPLT]["headers"]

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        dir_in = tmp_dir / "qplt_empty"
        dir_in.mkdir()

        # Create a header-only QPLT CSV file
        (dir_in / "py_qplt1.csv").write_text(",".join(qplt_headers) + "\n")

        outfile = tmp_dir / "katpy_qplt_empty.csv"
        kwargs = {
            "dir_in": dir_in,
            "qplt": True,
            "out": outfile,
            "unsorted": False,
        }
        main(**kwargs)

        assert outfile.exists(), "Output CSV was not created"
        lines = outfile.read_text().strip().splitlines()
        assert len(lines) == 1, f"Output CSV should contain only a header line, got {len(lines)} lines"


def test_empty_input_bin_to_tabular():
    """Test katpy does not crash and produces empty output when binary inputs have no data"""
    from oasislmf.pytools.kat.manager import KAT_MAP, KAT_QPLT

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        dir_in = tmp_dir / "qplt_empty_bin"
        dir_in.mkdir()

        # 0-byte binary triggers the empty-input path in bin_concat_sort_by_headers
        (dir_in / "py_qplt1.bin").write_bytes(b"")

        for ext in ["csv", "parquet"]:
            outfile = tmp_dir / f"katpy_qplt_empty.{ext}"
            main(dir_in=dir_in, qplt=True, out=outfile, unsorted=False)

            assert outfile.exists(), f"Output {ext} was not created"
            if ext == "csv":
                lines = outfile.read_text().strip().splitlines()
                assert len(lines) == 1, f"Output CSV should contain only a header line, got {len(lines)} lines"
                assert lines[0] == ",".join(KAT_MAP[KAT_QPLT]["headers"])
            else:
                df = pd.read_parquet(outfile)
                assert len(df) == 0, f"Output parquet should have no rows, got {len(df)}"
                assert list(df.columns) == list(KAT_MAP[KAT_QPLT]["dtype"].names)


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


def merge_inputs(file_type):
    """Rows of three sorted binary inputs, and the same rows in merged order.

    Events go to the inputs in turn and have one to four rows each. Event 4 is in every input,
    with three rows in two periods. SummaryId numbers the rows in input order.

    Args:
        file_type (int): katpy table type (index into KAT_MAP)

    Returns:
        arrays (List[np.ndarray]): rows of each input, sorted by EventId and, for PLT, Period
        expected (np.ndarray): all rows sorted by the same columns, ties in input order
    """
    dtype = KAT_MAP[file_type]["dtype"]
    events = [[], [], []]
    periods = [[], [], []]
    for event_id in range(1, 13):
        event_rows = 3 if event_id == 4 else 1 + event_id % 4
        for file_idx in range(3) if event_id == 4 else [(event_id - 1) % 3]:
            events[file_idx] += [event_id] * event_rows
            periods[file_idx] += [1 + row // 2 for row in range(event_rows)]

    arrays = []
    n_rows = 0
    for file_events, file_periods in zip(events, periods):
        data = np.zeros(len(file_events), dtype=dtype)
        data["EventId"] = file_events
        data["SummaryId"] = np.arange(n_rows, n_rows + len(data))
        if "Period" in dtype.names:
            data["Period"] = file_periods
        n_rows += len(data)
        arrays.append(data)

    rows = np.concatenate(arrays)
    keys = [rows["Period"], rows["EventId"]] if "Period" in dtype.names else [rows["EventId"]]
    return arrays, rows[np.lexsort(keys)]


@pytest.mark.parametrize("flag, file_type", [("selt", KAT_SELT), ("splt", KAT_SPLT)])
def test_katpy_bin_sorted_merge_order(tmp_path, flag, file_type):
    """Sorted bin inputs are merged by EventId and, for PLT, Period. Rows with equal keys keep the input order."""
    arrays, expected = merge_inputs(file_type)
    files_in = [tmp_path / f"{flag}_P{file_idx + 1}.bin" for file_idx in range(len(arrays))]
    for file_path, data in zip(files_in, arrays):
        data.tofile(file_path)
    out_file = tmp_path / f"{flag}.bin"

    main(files_in=files_in, out=out_file, unsorted=False, **{flag: True})

    np.testing.assert_array_equal(np.fromfile(out_file, dtype=expected.dtype), expected)


@pytest.mark.parametrize("merge_name, file_type", [("merge_elt_data", KAT_SELT), ("merge_plt_data", KAT_SPLT)])
def test_merge_yields_full_buffers(monkeypatch, merge_name, file_type):
    """The merged rows are yielded in buffers of DEFAULT_BUFFER_SIZE rows, followed by the remaining rows.

    The buffer size is a compile time constant of the jitted functions, so their Python implementations run here.
    """
    buffer_size = 4
    monkeypatch.setattr(manager, "DEFAULT_BUFFER_SIZE", buffer_size)
    merge = getattr(manager, merge_name)
    merge = getattr(merge, "py_func", merge)
    arrays, expected = merge_inputs(file_type)

    chunks = [chunk.copy() for chunk in merge(arrays)]

    assert [len(chunk) for chunk in chunks] == [buffer_size] * (len(expected) // buffer_size) + [len(expected) % buffer_size]
    np.testing.assert_array_equal(np.concatenate(chunks), expected)


REAL_UNLINK = Path.unlink


def windows_unlink(self, *args, **kwargs):
    """``Path.unlink`` failing as on Windows when the file is still open or memory mapped (WinError 32)."""
    process = psutil.Process()
    in_use = {Path(f.path) for f in process.open_files()}
    if hasattr(process, "memory_maps"):
        in_use |= {Path(m.path) for m in process.memory_maps() if m.path}
    if Path(self).resolve() in in_use:
        raise PermissionError(f"[WinError 32] The process cannot access the file because it is being used by another process: '{self}'")
    return REAL_UNLINK(self, *args, **kwargs)


@pytest.fixture
def temp_dir(monkeypatch, tmp_path):
    """Directory where katpy creates its temporary .bin, apart from the output."""
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    return temp_dir


@pytest.mark.parametrize("out_name", ["katpy_qplt.csv", "katpy_qplt.parquet"])
def test_bin_to_tabular_releases_temp_file(monkeypatch, tmp_path, temp_dir, out_name):
    """The temporary .bin of a bin to csv/parquet run is closed and unmapped before it is deleted.

    Windows refuses to delete a file that is still open or memory mapped (WinError 32); deleting
    such a file is made to fail the same way here.
    """
    monkeypatch.setattr(Path, "unlink", windows_unlink)

    out_file = tmp_path / out_name
    main(dir_in=Path(TESTS_ASSETS_DIR, "bqplt"), qplt=True, out=out_file, unsorted=False)

    assert out_file.stat().st_size > 0
    assert list(temp_dir.iterdir()) == []


@pytest.mark.parametrize("out_name", ["katpy_qplt.csv", "katpy_qplt.parquet"])
def test_bin_to_tabular_removes_temp_file_on_error(monkeypatch, tmp_path, temp_dir, out_name):
    """The temporary .bin is deleted when the concatenation fails before the conversion starts."""
    def failing_concat(*args, **kwargs):
        raise RuntimeError("concatenation failed")

    monkeypatch.setattr("oasislmf.pytools.kat.manager.bin_concat_sort_by_headers", failing_concat)

    with pytest.raises(RuntimeError, match="concatenation failed"):
        main(dir_in=Path(TESTS_ASSETS_DIR, "bqplt"), qplt=True, out=tmp_path / out_name, unsorted=False)

    assert list(temp_dir.iterdir()) == []


@pytest.mark.parametrize("out_name, failing", [("katpy_qplt.csv", "write_ndarray_to_fmt_csv"), ("katpy_qplt.parquet", "pa.array")])
def test_bin_to_tabular_releases_temp_file_on_conversion_error(monkeypatch, tmp_path, temp_dir, out_name, failing):
    """A conversion that fails part way still lets the temporary .bin be deleted, and its error is the one raised.

    The failing callee keeps its arguments, as a real one would: its frame stays in the traceback with the chunk read.
    """
    def fail(*args, **kwargs):
        raise ValueError("conversion failed")

    monkeypatch.setattr(f"oasislmf.pytools.kat.manager.{failing}", fail)
    monkeypatch.setattr(Path, "unlink", windows_unlink)

    with pytest.raises(ValueError, match="conversion failed"):
        main(dir_in=Path(TESTS_ASSETS_DIR, "bqplt"), qplt=True, out=tmp_path / out_name, unsorted=False)

    assert list(temp_dir.iterdir()) == []
