import os
import threading
import numpy as np
import pytest
from pathlib import Path
from contextlib import ExitStack
from tempfile import TemporaryDirectory

from oasislmf.pytools.converters.csvtobin.manager import default_tobin, TOBIN_FUNC_MAP
from oasislmf.pytools.converters.csvtobin.utils import footprint_tobin, vulnerability_tobin
from oasislmf.pytools.converters.data import TOOL_INFO
from tests.pytools.converters.helpers import TESTS_ASSETS_DIR


def test_csvtobin_non_seekable_stream():
    """default_tobin() must work when file_out is a pipe, not just a seekable file —
    this is what execution/runner.py's rerun() pipes csvtobin's output into, and a
    plain ndarray.tofile() call fails on a non-seekable destination."""
    infile = Path(TESTS_ASSETS_DIR, "input", "eve.csv")
    expected_outfile = Path(TESTS_ASSETS_DIR, "input", "eve.bin")
    dtype = TOOL_INFO["eve"]["dtype"]

    read_fd, write_fd = os.pipe()
    with ExitStack() as stack, os.fdopen(write_fd, "wb") as file_out, os.fdopen(read_fd, "rb") as file_in:
        default_tobin(stack, infile, file_out, "eve")
        file_out.close()
        actual_bytes = file_in.read()

    assert actual_bytes == expected_outfile.read_bytes()
    assert np.array_equal(np.frombuffer(actual_bytes, dtype=dtype), np.fromfile(expected_outfile, dtype=dtype))


def _run_writer_via_pipe(write_fn):
    """Run write_fn(file_out) with file_out as the write end of an OS pipe, draining the
    read end concurrently on a thread. A synchronous write larger than the OS pipe buffer
    (64KB on Linux) would otherwise deadlock: the writer blocks once the buffer fills,
    and nothing is reading it back out yet.
    """
    read_fd, write_fd = os.pipe()
    read_chunks = []

    def _drain():
        with os.fdopen(read_fd, "rb") as file_in:
            read_chunks.append(file_in.read())

    reader = threading.Thread(target=_drain)
    reader.start()
    with os.fdopen(write_fd, "wb") as file_out:
        write_fn(file_out)
    reader.join()
    return read_chunks[0]


@pytest.mark.parametrize(
    "file_type, sub_dir, filename, kwargs",
    [
        ("gul", "misc", "raw_guls", dict(stream_type=2, max_sample_index=1)),
        ("fm", "misc", "raw_ils", dict(stream_type=2, max_sample_index=1)),
        ("summarycalc", "misc", "summary", dict(summary_set_id=1, max_sample_index=100)),
        ("occurrence", "input", "occurrence", dict(no_of_periods=9)),
        ("damagebin", "static", "damagebin", dict(no_validation=False)),
        ("lossfactors", "static", "lossfactors", dict()),
        ("amplifications", "input", "amplifications", dict()),
        ("coverages", "input", "coverages", dict()),
        ("complex_items", "input", "complex_items", dict()),
        ("returnperiods", "input", "returnperiods", dict()),
    ],
)
def test_csvtobin_pipe_output_matches_file_output(file_type, sub_dir, filename, kwargs):
    """Every TOBIN_FUNC_MAP converter (not just the default_tobin fallback) must write
    identical bytes whether file_out is a regular seekable file or a non-seekable pipe —
    see test_csvtobin_non_seekable_stream."""
    infile = Path(TESTS_ASSETS_DIR, sub_dir, f"{filename}.csv")
    tobin_func = TOBIN_FUNC_MAP[file_type]

    with TemporaryDirectory() as tmp_dir, ExitStack() as stack:
        seekable_out = Path(tmp_dir, "seekable.bin")
        with open(seekable_out, "wb") as file_out:
            tobin_func(stack, infile, file_out, file_type, **kwargs)
        expected_bytes = seekable_out.read_bytes()

    with ExitStack() as stack:
        actual_bytes = _run_writer_via_pipe(lambda file_out: tobin_func(stack, infile, file_out, file_type, **kwargs))

    assert actual_bytes == expected_bytes


def test_footprint_pipe_output_matches_file_output():
    """footprint_tobin's main file_out must tolerate a non-seekable pipe (idx_file_out
    is written separately and is not exercised as a pipe here)."""
    infile = Path(TESTS_ASSETS_DIR, "static", "footprint.csv")
    kwargs = dict(
        max_intensity_bin_idx=3,
        no_intensity_uncertainty=True,
        decompressed_size=False,
        no_validation=False,
        zip_files=False,
    )

    with TemporaryDirectory() as tmp_dir, ExitStack() as stack:
        seekable_out = Path(tmp_dir, "seekable.bin")
        with open(seekable_out, "wb") as file_out:
            footprint_tobin(stack, infile, file_out, "footprint", idx_file_out=Path(tmp_dir, "seekable.idx"), **kwargs)
        expected_bytes = seekable_out.read_bytes()

    with TemporaryDirectory() as tmp_dir, ExitStack() as stack:
        actual_bytes = _run_writer_via_pipe(
            lambda file_out: footprint_tobin(stack, infile, file_out, "footprint", idx_file_out=Path(tmp_dir, "pipe.idx"), **kwargs)
        )

    assert actual_bytes == expected_bytes


def test_vulnerability_pipe_output_matches_file_output():
    """vulnerability_tobin's no-idx path must tolerate a non-seekable pipe for file_out."""
    infile = Path(TESTS_ASSETS_DIR, "static", "vulnerability_noidx.csv")
    kwargs = dict(
        idx_file_out=None,
        max_damage_bin_idx=2,
        no_validation=False,
        suppress_int_bin_checks=False,
        zip_files=False,
    )

    with TemporaryDirectory() as tmp_dir, ExitStack() as stack:
        seekable_out = Path(tmp_dir, "seekable.bin")
        with open(seekable_out, "wb") as file_out:
            vulnerability_tobin(stack, infile, file_out, "vulnerability", **kwargs)
        expected_bytes = seekable_out.read_bytes()

    with ExitStack() as stack:
        actual_bytes = _run_writer_via_pipe(lambda file_out: vulnerability_tobin(stack, infile, file_out, "vulnerability", **kwargs))

    assert actual_bytes == expected_bytes
