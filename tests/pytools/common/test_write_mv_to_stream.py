import io
import os
import sys

import numpy as np
import pytest

from oasislmf.pytools.common import event_stream
from oasislmf.pytools.common.event_stream import stream_is_selectable, write_mv_to_stream

PAYLOAD = np.arange(64, dtype=np.uint8)


def macos_select(rlist, wlist, xlist):
    """``select`` as macOS answers it for a regular file: writable, but also exceptional."""
    return [], list(wlist), list(xlist)


def test_write_to_regular_file_does_not_select(monkeypatch, tmp_path):
    """Writing to a regular file must not depend on ``select``, which reports one as exceptional on macOS."""
    monkeypatch.setattr(event_stream, "select", macos_select)
    out_path = tmp_path / "out.bin"
    with open(out_path, "wb") as stream:
        assert not stream_is_selectable(stream)
        write_mv_to_stream(stream, memoryview(PAYLOAD), PAYLOAD.shape[0])
    assert out_path.read_bytes() == PAYLOAD.tobytes()


def test_write_to_buffer_without_fileno():
    stream = io.BytesIO()
    assert not stream_is_selectable(stream)
    write_mv_to_stream(stream, memoryview(PAYLOAD), 10)
    assert stream.getvalue() == PAYLOAD[:10].tobytes()


@pytest.mark.skipif(sys.platform == "win32", reason="select does not accept pipes on Windows")
def test_write_to_pipe_selects(monkeypatch):
    """A pipe still waits on ``select`` for back pressure."""
    calls = []
    real_select = event_stream.select

    def spy_select(rlist, wlist, xlist):
        calls.append(wlist)
        return real_select(rlist, wlist, xlist)

    monkeypatch.setattr(event_stream, "select", spy_select)
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "rb") as reader, os.fdopen(write_fd, "wb", buffering=0) as writer:
        assert stream_is_selectable(writer)
        write_mv_to_stream(writer, memoryview(PAYLOAD), PAYLOAD.shape[0])
        writer.close()
        assert reader.read() == PAYLOAD.tobytes()
    assert calls
