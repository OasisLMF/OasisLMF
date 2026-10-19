import logging
import os
import select
import sys
import threading

import numpy as np
import pytest

from oasislmf.pytools.common import event_stream
from oasislmf.pytools.common.event_stream import EventReader, write_mv_to_stream

# A reader that hangs on EOF would block the whole test session, so every read
# runs in a thread and a test fails instead once this deadline passes.
READ_DEADLINE = 30  # seconds

needs_fifo = pytest.mark.skipif(sys.platform == 'win32', reason="named pipes (os.mkfifo) are POSIX-only")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ByteCountingReader(EventReader):
    """Consume every byte without parsing events, so only EOF detection is exercised."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nb_bytes = 0

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, **kwargs):
        self.nb_bytes += valid_buff - cursor
        return valid_buff, 0, 0, 0


def _read_to_eof(stream):
    """Run read_streams on stream in a thread; return (finished, nb_bytes_read)."""
    reader = _ByteCountingReader()
    thread = threading.Thread(target=lambda: list(reader.read_streams([stream])), daemon=True)
    thread.start()
    thread.join(READ_DEADLINE)
    return not thread.is_alive(), reader.nb_bytes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_write_mv_to_stream_regular_file(tmp_path):
    # macOS reports regular files in select()'s exceptfds; that must not be taken as a stream error
    payload = np.arange(1000, dtype=np.int32).view('b')
    path = tmp_path / "out.bin"
    with open(path, 'wb') as stream:
        write_mv_to_stream(stream, payload, payload.shape[0])
    assert path.read_bytes() == payload.tobytes()


def test_read_streams_regular_file_reaches_eof(tmp_path):
    # kqueue (macOS DefaultSelector) accepts regular files and never reports EOF on them
    path = tmp_path / "in.bin"
    path.write_bytes(b"x" * 100_000)
    with open(path, 'rb') as stream:
        finished, nb_bytes = _read_to_eof(stream)
    assert finished, "read_streams did not return at end of a regular file"
    assert nb_bytes == 100_000


@needs_fifo
def test_read_streams_fifo_reaches_eof_after_writer_closed(tmp_path):
    # the writer has written everything and closed before the reader starts selecting
    path = str(tmp_path / "fifo")
    os.mkfifo(path)

    def write_all():
        with open(path, 'wb') as writer:
            writer.write(b"x" * 3000)

    writer = threading.Thread(target=write_all)
    writer.start()
    with open(path, 'rb') as stream:
        writer.join()
        finished, nb_bytes = _read_to_eof(stream)
    assert finished, "read_streams did not return at EOF of a FIFO whose writer closed"
    assert nb_bytes == 3000


@needs_fifo
def test_nudge_fifo_eof_keeps_live_writer(tmp_path):
    # re-arming EOF must never end a stream whose writer is still attached
    path = str(tmp_path / "fifo")
    os.mkfifo(path)
    fd_r = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    fd_w = os.open(path, os.O_WRONLY)
    try:
        with os.fdopen(fd_r, 'rb', buffering=0, closefd=False) as stream:
            event_stream._nudge_fifo_eof(stream)
            assert select.select([fd_r], [], [], 0)[0] == [], "nudge delivered EOF with a writer attached"
            os.write(fd_w, b"data")
            assert os.read(fd_r, 16) == b"data"
    finally:
        os.close(fd_w)
        os.close(fd_r)
