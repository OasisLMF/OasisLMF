import io
import os
import socket
import threading
import time

import numpy as np
import pytest

from oasislmf.pytools.common.event_stream import PIPE_CAPACITY, write_mv_to_stream

PAYLOAD = np.arange(64, dtype=np.uint8)


def test_write_to_regular_file(tmp_path):
    out_path = tmp_path / "out.bin"
    with open(out_path, "wb") as stream:
        write_mv_to_stream(stream, PAYLOAD, PAYLOAD.shape[0])
    assert out_path.read_bytes() == PAYLOAD.tobytes()


def test_write_to_buffer_without_fileno():
    stream = io.BytesIO()
    write_mv_to_stream(stream, PAYLOAD, 10)
    assert stream.getvalue() == PAYLOAD[:10].tobytes()


def slow_read(stream, received):
    """Read in small chunks with pauses, so the writer fills the kernel buffer and has to wait."""
    while chunk := stream.read(PIPE_CAPACITY // 4):
        received.append(chunk)
        time.sleep(0.001)


def pipe_pair():
    read_fd, write_fd = os.pipe()
    return os.fdopen(read_fd, "rb"), os.fdopen(write_fd, "wb")


def socket_pair():
    a, b = socket.socketpair()
    reader, writer = b.makefile("rb"), a.makefile("wb")
    a.close()  # the file objects keep the sockets open
    b.close()
    return reader, writer


@pytest.mark.parametrize("make_pair", [pipe_pair, socket_pair], ids=["pipe", "socket"])
def test_write_waits_for_a_slow_reader(make_pair):
    """A payload several times the kernel buffer arrives whole and in order: the blocking write is the back pressure."""
    payload = np.random.default_rng(0).integers(0, 256, 16 * PIPE_CAPACITY, dtype=np.uint8)
    reader, writer = make_pair()
    received = []
    thread = threading.Thread(target=slow_read, args=(reader, received))
    thread.start()
    try:
        write_mv_to_stream(writer, payload, payload.shape[0])
    finally:
        writer.close()
        thread.join()
        reader.close()
    assert b"".join(received) == payload.tobytes()
