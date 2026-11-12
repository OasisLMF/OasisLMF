import io
import os
import subprocess
import sys

import pytest

from oasislmf.execution.orchestrator.tee import TeeError, run

TEEPY = [sys.executable, '-m', 'oasislmf.execution.orchestrator.tee']
TIMEOUT = 60


def test_run_fans_out_to_files_and_file_objects(tmp_path):
    payload = os.urandom(300_000)
    paths = [tmp_path / f'out{i}.bin' for i in range(3)]
    sink = io.BytesIO()
    run([str(p) for p in paths] + [sink], stream_in=io.BytesIO(payload), buffer_size=4096)
    for p in paths:
        assert p.read_bytes() == payload
    assert sink.getvalue() == payload
    assert not sink.closed


def test_run_reports_output_that_cannot_be_opened(tmp_path):
    with pytest.raises(TeeError, match='missing'):
        run([str(tmp_path / 'missing' / 'out.bin')], stream_in=io.BytesIO(b'x'))


def test_cli_fans_out_to_files_and_stdout(tmp_path):
    payload = b'abc' * 1000
    paths = [tmp_path / 'a.bin', tmp_path / 'b.bin']
    result = subprocess.run(TEEPY + [str(paths[0]), '-', str(paths[1])], input=payload, capture_output=True, timeout=TIMEOUT)
    assert result.returncode == 0, result.stderr
    assert result.stdout == payload
    assert all(p.read_bytes() == payload for p in paths)


def test_cli_large_payload_through_pipes(tmp_path):
    payload = os.urandom(40 * 1024 * 1024 + 123)
    out = tmp_path / 'big.bin'
    result = subprocess.run(TEEPY + ['-', str(out)], input=payload, capture_output=True, timeout=TIMEOUT)
    assert result.returncode == 0, result.stderr
    assert result.stdout == payload
    assert out.read_bytes() == payload


def test_cli_empty_input(tmp_path):
    out = tmp_path / 'empty.bin'
    result = subprocess.run(TEEPY + [str(out), '-'], input=b'', capture_output=True, timeout=TIMEOUT)
    assert result.returncode == 0, result.stderr
    assert result.stdout == b''
    assert out.exists() and out.read_bytes() == b''


def test_cli_stdout_reader_exits_early(tmp_path):
    reader = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.read(10)'], stdin=subprocess.PIPE)
    tee = subprocess.Popen(TEEPY + [str(tmp_path / 'out.bin'), '-'], stdin=subprocess.PIPE, stdout=reader.stdin, stderr=subprocess.PIPE)
    reader.stdin.close()
    try:
        tee.stdin.write(b'x' * (20 * 1024 * 1024))
    except BrokenPipeError:
        pass
    tee.stdin.close()
    tee.wait(timeout=TIMEOUT)
    reader.wait(timeout=TIMEOUT)
    stderr = tee.stderr.read().decode()
    tee.stderr.close()
    assert tee.returncode == 1
    assert "teepy: error writing to '<stdout>'" in stderr
    assert 'Traceback' not in stderr


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='needs mkfifo')
def test_cli_path_reader_exits_early(tmp_path):
    fifo = tmp_path / 'fifo'
    os.mkfifo(fifo)
    tee = subprocess.Popen(TEEPY + [str(tmp_path / 'out.bin'), str(fifo)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    reader = subprocess.Popen([sys.executable, '-c', 'import sys; open(sys.argv[1], "rb").read(10)', str(fifo)])
    try:
        tee.stdin.write(b'x' * (20 * 1024 * 1024))
    except BrokenPipeError:
        pass
    tee.stdin.close()
    tee.wait(timeout=TIMEOUT)
    reader.wait(timeout=TIMEOUT)
    stderr = tee.stderr.read().decode()
    tee.stderr.close()
    assert tee.returncode == 1
    assert f"teepy: error writing to '{fifo}'" in stderr
    assert 'Traceback' not in stderr
