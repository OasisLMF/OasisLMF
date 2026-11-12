import io
import os
import subprocess
import sys

import pytest

from oasislmf.pytools.common import stream_ref
from oasislmf.pytools.common.stream_ref import (FD_KIND, HANDLE_KIND, open_stream_out, parse_stream_reference,
                                                stream_reference)


def test_reference_round_trip():
    assert stream_reference(FD_KIND, 7) == '<oasis-fd>/7'
    assert stream_reference(HANDLE_KIND, 1234) == '<oasis-handle>/1234'
    assert parse_stream_reference('<oasis-fd>/7') == (FD_KIND, 7)
    assert parse_stream_reference('<oasis-handle>/1234') == (HANDLE_KIND, 1234)
    with pytest.raises(ValueError):
        stream_reference('pipe', 1)


@pytest.mark.parametrize('path', ['out.bin', '/dev/fd/3', '<oasis-fd>/', '<oasis-fd>/3x', 'x<oasis-fd>/3', '<oasis-sock>/3',
                                  '<oasis-fd>/3/', b'<oasis-fd>/3', 3])
def test_not_a_reference(path):
    assert parse_stream_reference(path) is None


@pytest.mark.parametrize('buffering', [-1, 0])
def test_open_path_matches_open(tmp_path, buffering):
    path = tmp_path / 'out.bin'
    with open_stream_out(str(path), buffering=buffering) as f, open(tmp_path / 'ref.bin', 'wb', buffering=buffering) as ref:
        assert type(f) is type(ref)
        assert f.mode == ref.mode == 'wb'
        f.write(b'abc')
    assert path.read_bytes() == b'abc'
    with open_stream_out(path) as f:
        assert isinstance(f, io.BufferedWriter)


def test_open_fd_reference_closes_write_end():
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, 'rb') as reader:
        f = open_stream_out(stream_reference(FD_KIND, write_fd))
        assert isinstance(f, io.BufferedWriter)
        f.write(b'payload')
        f.close()
        assert reader.read() == b'payload'
    with pytest.raises(OSError):
        os.fstat(write_fd)


def test_open_fd_reference_unbuffered():
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, 'rb') as reader:
        with open_stream_out(stream_reference(FD_KIND, write_fd), buffering=0) as f:
            assert isinstance(f, io.FileIO)
            f.write(b'x')
        assert reader.read() == b'x'


@pytest.mark.skipif(os.name == 'nt', reason='pass_fds is POSIX only')
def test_open_fd_reference_in_child():
    """A child given ``<oasis-fd>/n`` with ``pass_fds`` writes to the parent's pipe."""
    read_fd, write_fd = os.pipe()
    script = ('import sys; from oasislmf.pytools.common.stream_ref import open_stream_out\n'
              'with open_stream_out(sys.argv[1]) as f: f.write(b"from child")')
    proc = subprocess.Popen([sys.executable, '-c', script, stream_reference(FD_KIND, write_fd)], pass_fds=(write_fd,))
    os.close(write_fd)
    with os.fdopen(read_fd, 'rb') as reader:
        assert reader.read() == b'from child'
    assert proc.wait(timeout=30) == 0


def test_handle_reference_needs_windows(monkeypatch):
    monkeypatch.setattr(stream_ref, 'msvcrt', None)
    with pytest.raises(ValueError, match='Windows handle'):
        open_stream_out(stream_reference(HANDLE_KIND, 42))


def test_handle_reference_windows_branch(monkeypatch):
    """The Windows branch with a fake ``msvcrt``: the handle becomes a C runtime descriptor, opened binary."""
    read_fd, write_fd = os.pipe()
    calls = []

    class FakeMsvcrt:
        @staticmethod
        def open_osfhandle(handle, flags):
            calls.append((handle, flags))
            return write_fd

    monkeypatch.setattr(stream_ref, 'msvcrt', FakeMsvcrt)
    with os.fdopen(read_fd, 'rb') as reader:
        with open_stream_out(stream_reference(HANDLE_KIND, 9999)) as f:
            f.write(b'handle')
        assert reader.read() == b'handle'
    assert calls == [(9999, 0)]
