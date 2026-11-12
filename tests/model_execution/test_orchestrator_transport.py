import os
import subprocess
import sys

import pytest

from oasislmf.execution.orchestrator.model import Process, Stage, Stream, StreamPath
from oasislmf.execution.orchestrator.transport import PosixTransport, get_transport

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='the Windows transport is not implemented')

PY = sys.executable
TEEPY = [PY, '-m', 'oasislmf.execution.orchestrator.tee']
TIMEOUT = 30

WRITE_STDOUT = 'import sys; sys.stdout.buffer.write(bytes.fromhex(sys.argv[1]) * int(sys.argv[2]))'
WRITE_PATH = 'import sys; f = open(sys.argv[1], "wb"); f.write(bytes.fromhex(sys.argv[2]) * int(sys.argv[3])); f.close()'
SLOW_WRITE_PATH = 'import sys, time; time.sleep(1.5); f = open(sys.argv[1], "wb"); f.write(b"late data"); f.close()'
COPY = 'import sys, shutil; shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)'
UPPER = 'import sys; sys.stdout.buffer.write(sys.stdin.buffer.read().upper())'
WRITE_FOREVER_PATH = 'import sys\nf = open(sys.argv[1], "wb")\nwhile True: f.write(b"x" * 65536)'
WRITE_FOREVER_STDOUT = 'import sys\nwhile True: sys.stdout.buffer.write(b"x" * 65536)'


def open_fds():
    return sorted(os.listdir('/proc/self/fd'))


def run_stage(stage, order=None):
    """Spawn the stage's processes in ``order`` through a transport, wait, and return return codes by name."""
    procs = {}
    with get_transport() as transport:
        transport.open(stage)
        try:
            for process in order or stage.processes:
                procs[process.name] = subprocess.Popen(transport.resolve_argv(process), **transport.popen_kwargs(process))
                transport.after_spawn(process)
        except BaseException:
            for proc in procs.values():
                proc.kill()
            raise
    try:
        return {name: proc.wait(timeout=TIMEOUT) for name, proc in procs.items()}
    except subprocess.TimeoutExpired:
        for proc in procs.values():
            proc.kill()
            proc.wait()
        pytest.fail('stage did not finish: deadlock or missing EOF')


def test_stdout_stdin_chain(tmp_path):
    s1, s2 = Stream('s1'), Stream('s2')
    out = tmp_path / 'out.txt'
    stage = Stage('chain', [
        Process('src', [PY, '-c', WRITE_STDOUT, b'abc'.hex(), '100000'], stdout=s1),
        Process('upper', [PY, '-c', UPPER], stdin=s1, stdout=s2),
        Process('sink', [PY, '-c', COPY], stdin=s2, stdout=str(out)),
    ])
    assert run_stage(stage) == {'src': 0, 'upper': 0, 'sink': 0}
    assert out.read_bytes() == b'ABC' * 100000


@pytest.mark.parametrize('reverse', [False, True])
def test_writer_opens_stream_path(tmp_path, reverse):
    s = Stream('s')
    out = tmp_path / 'out.bin'
    stage = Stage('path', [
        Process('reader', [PY, '-c', COPY], stdin=s, stdout=str(out)),
        Process('writer', [PY, '-c', WRITE_PATH, StreamPath(s), b'\x00\x01\xff'.hex(), '500000']),
    ])
    order = stage.processes[::-1] if reverse else None
    assert run_stage(stage, order) == {'reader': 0, 'writer': 0}
    assert out.read_bytes() == b'\x00\x01\xff' * 500000


@pytest.mark.parametrize('reverse', [False, True])
def test_teepy_fan_out(tmp_path, reverse):
    src, a, b = Stream('src'), Stream('a'), Stream('b')
    out_a, out_b, out_file = tmp_path / 'a.bin', tmp_path / 'b.bin', tmp_path / 'file.bin'
    payload = bytes(range(256)) * 20000
    stage = Stage('fan', [
        Process('reader_a', [PY, '-c', COPY], stdin=a, stdout=str(out_a)),
        Process('reader_b', [PY, '-c', COPY], stdin=b, stdout=str(out_b)),
        Process('tee', TEEPY + [StreamPath(a), StreamPath(b), str(out_file)], stdin=src),
        Process('src', [PY, '-c', WRITE_STDOUT, bytes(range(256)).hex(), '20000'], stdout=src),
    ])
    order = stage.processes[::-1] if reverse else None
    assert set(run_stage(stage, order).values()) == {0}
    for path in (out_a, out_b, out_file):
        assert path.read_bytes() == payload


def test_slow_writer_no_early_eof(tmp_path):
    s = Stream('s')
    out = tmp_path / 'out.bin'
    stage = Stage('slow', [
        Process('reader', [PY, '-c', COPY], stdin=s, stdout=str(out)),
        Process('writer', [PY, '-c', SLOW_WRITE_PATH, StreamPath(s)]),
    ])
    assert run_stage(stage) == {'reader': 0, 'writer': 0}
    assert out.read_bytes() == b'late data'


def test_writer_never_opens_path(tmp_path):
    s = Stream('s')
    out = tmp_path / 'out.bin'
    stage = Stage('noopen', [
        Process('reader', [PY, '-c', COPY], stdin=s, stdout=str(out)),
        Process('writer', [PY, '-c', 'import sys; sys.exit(3)', StreamPath(s)]),
    ])
    assert run_stage(stage) == {'reader': 0, 'writer': 3}
    assert out.read_bytes() == b''


@pytest.mark.parametrize('via_path', [True, False])
def test_reader_exits_immediately_fails_writer(via_path):
    s = Stream('s')
    writer = (Process('writer', [PY, '-c', WRITE_FOREVER_PATH, StreamPath(s)]) if via_path
              else Process('writer', [PY, '-c', WRITE_FOREVER_STDOUT], stdout=s))
    writer.stderr_log = None
    stage = Stage('earlyexit', [Process('reader', [PY, '-c', 'pass'], stdin=s), writer])
    codes = run_stage(stage)
    assert codes['reader'] == 0
    assert codes['writer'] != 0


def test_popen_kwargs_and_argv():
    s = Stream('s')
    writer = Process('writer', ['w', '-o', StreamPath(s)])
    reader = Process('reader', ['r'], stdin=s)
    with PosixTransport() as transport:
        transport.open(Stage('x', [reader, writer]))
        argv = transport.resolve_argv(writer)
        kwargs = transport.popen_kwargs(writer)
        assert argv[:2] == ['w', '-o'] and argv[2] == f'/dev/fd/{kwargs["pass_fds"][0]}'
        assert kwargs['pass_fds'][0] >= 3
        assert kwargs['stdin'] == subprocess.DEVNULL and kwargs['stdout'] == subprocess.DEVNULL
        assert transport.popen_kwargs(reader)['stdin'] >= 0
        assert transport.resolve_argv(reader) == ['r']


def test_open_rejects_invalid_stage():
    with PosixTransport() as transport, pytest.raises(ValueError):
        transport.open(Stage('bad', [Process('w', ['w'], stdout=Stream('s'))]))


def test_close_after_partial_spawn(tmp_path, monkeypatch):
    monkeypatch.setenv('OASIS_TMPDIR', str(tmp_path))
    linux = os.path.isdir('/proc/self/fd')
    before = open_fds() if linux else None
    s1, s2 = Stream('s1'), Stream('s2')
    stage = Stage('partial', [
        Process('reader', [PY, '-c', COPY], stdin=s2, stdout=str(tmp_path / 'out.bin')),
        Process('missing', [str(tmp_path / 'no-such-exe')], stdin=s1, stdout=s2),
        Process('writer', [PY, '-c', WRITE_PATH, StreamPath(s1), '00', '10']),
    ])
    transport = get_transport()
    transport.open(stage)
    reader = subprocess.Popen(transport.resolve_argv(stage.processes[0]), **transport.popen_kwargs(stage.processes[0]))
    transport.after_spawn(stage.processes[0])
    with pytest.raises(FileNotFoundError):
        subprocess.Popen(transport.resolve_argv(stage.processes[1]), **transport.popen_kwargs(stage.processes[1]))
    transport.close()
    transport.close()
    assert reader.wait(timeout=TIMEOUT) == 0
    assert sorted(os.listdir(tmp_path)) == ['out.bin']
    if linux:
        assert open_fds() == before
