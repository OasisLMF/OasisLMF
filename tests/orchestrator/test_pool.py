import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import warnings

import numpy as np
import psutil
import pytest

from oasislmf.execution.orchestrator import cli, inline, pool
from oasislmf.execution.orchestrator.chain import Chain
from oasislmf.execution.orchestrator.chunks import make_chunks
from oasislmf.execution.orchestrator.worker import stage_name

from .test_chain import FakeFm, FakeGul, FakeSummary, expected_streams, expected_tree_files, make_spec, make_tree_spec, read_tree_files
from .tools.compare_streams import compare_dirs_bytes
from .tools.reference_streams import DEFAULT_GUL_ARGS

TAPS = ['gul', 'il', 'ri']
THIS = 'tests.orchestrator.test_pool'
FAKE_FACTORIES = {'gul': 'tests.orchestrator.test_chain:FakeGul', 'fm': 'tests.orchestrator.test_chain:FakeFm'}
RAISE_EVENT = 13
CRASH_EVENT = 17
HANG_EVENT = 2
FM_RAISE_EVENT = 22
TIMEOUT = 120
STDERR_EVENT = 5


class RaisingGul(FakeGul):
    """Raises on RAISE_EVENT."""

    def process_event(self, event_id, out):
        if event_id == RAISE_EVENT:
            raise ValueError(f'bad event {event_id}')
        super().process_event(event_id, out)


class CrashingGul(FakeGul):
    """Exits the worker process abruptly on CRASH_EVENT, as a segfault would."""

    def process_event(self, event_id, out):
        if event_id == CRASH_EVENT:
            sys.stderr.write('about to crash\n')
            sys.stderr.flush()
            os._exit(3)
        super().process_event(event_id, out)


class StderrGul(FakeGul):
    """Writes an error line to stderr on STDERR_EVENT and carries on."""

    def process_event(self, event_id, out):
        if event_id == STDERR_EVENT:
            sys.stderr.write('ERROR: something went wrong\n')
            sys.stderr.flush()
        super().process_event(event_id, out)


class WarningGul(FakeGul):
    """Issues a Python warning on STDERR_EVENT, which the stderr guard ignores."""

    def process_event(self, event_id, out):
        if event_id == STDERR_EVENT:
            warnings.warn('this is only a warning', UserWarning)
            sys.stderr.flush()
        super().process_event(event_id, out)


class FailingSetupGul(FakeGul):
    def __init__(self, run_dir, **kwargs):
        raise OSError('cannot open footprint')


class RaisingFm(FakeFm):
    """Raises in the IL step for FM_RAISE_EVENT."""

    def process_block(self, block, gross_out=None, net_out=None):
        values = np.frombuffer(block.tobytes(), dtype=np.int32)
        if self.static_path == 'input' and values[0] == FM_RAISE_EVENT:
            raise RuntimeError('fm kernel failed')
        super().process_block(block, gross_out, net_out)


class HangOrRaiseGul(FakeGul):
    """On HANG_EVENT starts a grandchild process and hangs; on RAISE_EVENT waits for that grandchild, then raises."""

    def process_event(self, event_id, out):
        pid_file = os.path.join(self.run_dir, 'grandchild.pid')
        if event_id == HANG_EVENT:
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
            with open(pid_file + '.tmp', 'w') as f:
                f.write(str(child.pid))
            os.replace(pid_file + '.tmp', pid_file)
            time.sleep(600)
        if event_id == RAISE_EVENT:
            deadline = time.monotonic() + 60
            while not os.path.exists(pid_file) and time.monotonic() < deadline:
                time.sleep(0.05)
            raise ValueError('raised while another worker hangs')
        super().process_event(event_id, out)


@contextlib.contextmanager
def deadline(seconds=TIMEOUT):
    """Fail with TimeoutError instead of hanging, where SIGALRM exists (a no-op elsewhere)."""
    if not hasattr(signal, 'SIGALRM'):
        yield
        return

    def expire(signum, frame):
        raise TimeoutError(f'no result after {seconds} s')

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def run_pool(*args, timeout=TIMEOUT, **kwargs):
    """pool.run_chunks under a deadline."""
    with deadline(timeout):
        return pool.run_chunks(*args, **kwargs)


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / 'run'
    (path / 'input').mkdir(parents=True)
    return path


def fake_headers():
    return Chain(make_spec(), factories={'gul': FakeGul, 'fm': FakeFm}).headers


def check_fake_outputs(out_dir, chunks, taps=TAPS):
    headers = fake_headers()
    assert sorted(os.listdir(out_dir)) == sorted(f'{tap}_P{k}.bin' for tap in taps for k in range(1, len(chunks) + 1))
    for k, chunk in enumerate(chunks, 1):
        expected = expected_streams([int(e) for e in chunk])
        for tap in taps:
            assert (out_dir / f'{tap}_P{k}.bin').read_bytes() == headers[tap] + expected[tap], (tap, k)


def descendants():
    """Live descendants of the test process."""
    return [p for p in psutil.Process().children(recursive=True) if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]


def check_stats_shape(stats, chunks, n_workers):
    assert stats['executor'] == 'pool'
    assert stats['n_workers'] == n_workers
    assert stats['events'] == sum(len(c) for c in chunks)
    assert stats['wall_seconds'] >= stats['setup_seconds'] > 0
    assert [c['chunk'] for c in stats['chunks']] == list(range(1, len(chunks) + 1))
    assert [c['events'] for c in stats['chunks']] == [len(c) for c in chunks]
    assert all(set(c) == {'chunk', 'events', 'seconds', 'worker'} for c in stats['chunks'])
    assert all(0 <= c['worker'] < n_workers and c['seconds'] >= 0 for c in stats['chunks'])
    assert [w['worker'] for w in stats['workers']] == list(range(n_workers))
    assert all({'worker', 'chunks', 'setup_seconds', 'busy_seconds'} <= set(w) for w in stats['workers'])
    assert sum(w['chunks'] for w in stats['workers']) == len(chunks)
    assert stats['setup_seconds'] == max(w['setup_seconds'] for w in stats['workers'])
    for w in stats['workers']:
        mine = [c for c in stats['chunks'] if c['worker'] == w['worker']]
        assert w['chunks'] == len(mine)
        assert w['busy_seconds'] == pytest.approx(sum(c['seconds'] for c in mine))
    json.dumps(stats)


@pytest.mark.parametrize('n_chunks, n_workers', [(5, 2), (2, 4), (4, 1)])
def test_pool_fake_stages_outputs_and_stats(tmp_path, run_dir, n_chunks, n_workers):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), n_chunks)
    out_dir = tmp_path / 'out'
    cwd = os.getcwd()
    stats = run_pool(make_spec(str(run_dir)), chunks, out_dir, TAPS, n_workers, factories=FAKE_FACTORIES)
    assert os.getcwd() == cwd
    check_fake_outputs(out_dir, chunks)
    check_stats_shape(stats, chunks, min(n_workers, n_chunks))
    assert len({w['pid'] for w in stats['workers']}) == len(stats['workers'])
    assert descendants() == []


def test_pool_empty_chunks_relative_paths_and_subset_of_taps(tmp_path, run_dir, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chunks = [np.array([3, 1], dtype=np.int32), np.array([], dtype=np.int32), np.array([5], dtype=np.int32)]
    stats = run_pool(make_spec('run'), chunks, 'out', ['il'], 2, factories=FAKE_FACTORIES)
    check_fake_outputs(tmp_path / 'out', chunks, taps=['il'])
    assert (tmp_path / 'out' / 'il_P2.bin').read_bytes() == fake_headers()['il']
    assert stats['events'] == 3


def test_pool_no_chunks_starts_no_worker(tmp_path, run_dir):
    stats = run_pool(make_spec(str(run_dir)), [], tmp_path / 'out', TAPS, 4, factories=FAKE_FACTORIES)
    assert stats['n_workers'] == 0
    assert stats['chunks'] == [] and stats['workers'] == []


def test_pool_rejects_live_factories_and_bad_worker_count(tmp_path, run_dir):
    with pytest.raises(TypeError, match='strings'):
        run_pool(make_spec(str(run_dir)), [[1]], tmp_path / 'out', TAPS, 1, factories={'gul': FakeGul, 'fm': FakeFm})
    with pytest.raises(ValueError):
        run_pool(make_spec(str(run_dir)), [[1]], tmp_path / 'out', TAPS, 0, factories=FAKE_FACTORIES)


def test_pool_workers_pull_and_policy_can_spawn_and_retire(tmp_path, run_dir):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 10)
    seen = {'spawned': False, 'retired': None}
    progress = []

    def policy(p):
        if not seen['spawned'] and p.chunk_stats:
            p.spawn()
            seen['spawned'] = True
        elif seen['spawned'] and seen['retired'] is None and len(p.chunk_stats) >= 3:
            seen['retired'] = p.retire(p.workers[0]).number

    stats = run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, 1, factories=FAKE_FACTORIES,
                            policy=policy, progress=lambda done, total: progress.append((done, total)))
    check_fake_outputs(tmp_path / 'out', chunks)
    assert stats['n_workers'] == 2
    assert seen['retired'] == 0
    assert sum(w['chunks'] for w in stats['workers']) == 10
    assert progress[-1] == (30, 30)
    assert [d for d, _ in progress] == sorted(d for d, _ in progress)
    assert descendants() == []


def test_pool_retiring_every_worker_with_chunks_left_is_an_error(tmp_path, run_dir):
    def policy(p):
        for w in p.workers:
            p.retire(w)

    with pytest.raises(pool.WorkerError, match='chunks left'):
        run_pool(make_spec(str(run_dir)), make_chunks(np.arange(1, 31, dtype=np.int32), 5), tmp_path / 'out', TAPS, 1,
                        factories=FAKE_FACTORIES, policy=policy)
    assert descendants() == []


@pytest.mark.parametrize('n_workers', [1, 3])
def test_pool_stage_exception_stops_run_with_stage_chunk_event(tmp_path, run_dir, n_workers):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 4)
    k = next(i for i, c in enumerate(chunks, 1) if RAISE_EVENT in c)
    with pytest.raises(pool.WorkerError) as info:
        run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, n_workers,
                        factories={'gul': f'{THIS}:RaisingGul', 'fm': FAKE_FACTORIES['fm']})
    error = info.value
    assert (error.stage, error.chunk, error.event) == ('gul', k, RAISE_EVENT)
    assert error.returncode is not None and error.returncode != 0
    text = str(error)
    assert f'stage gul, chunk {k}, event {RAISE_EVENT}' in text
    assert f'ValueError: bad event {RAISE_EVENT}' in text
    assert 'RaisingGul' in text or 'process_event' in text
    assert descendants() == []


def test_pool_fm_exception_names_fm_step(tmp_path, run_dir):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 3)
    with pytest.raises(pool.WorkerError) as info:
        run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, 2,
                        factories={'gul': FAKE_FACTORIES['gul'], 'fm': f'{THIS}:RaisingFm'})
    assert info.value.stage == 'il (fm step 1)'
    assert info.value.event == FM_RAISE_EVENT
    assert 'fm kernel failed' in str(info.value)
    assert descendants() == []


def test_pool_setup_failure(tmp_path, run_dir):
    with pytest.raises(pool.WorkerError) as info:
        run_pool(make_spec(str(run_dir)), [[1], [2]], tmp_path / 'out', TAPS, 2,
                        factories={'gul': f'{THIS}:FailingSetupGul', 'fm': FAKE_FACTORIES['fm']})
    assert info.value.stage == 'setup'
    assert 'cannot open footprint' in str(info.value)
    assert descendants() == []


def test_pool_worker_crash_reports_chunk_exit_code_and_stderr(tmp_path, run_dir):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 3)
    k = next(i for i, c in enumerate(chunks, 1) if CRASH_EVENT in c)
    with pytest.raises(pool.WorkerError) as info:
        run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, 2,
                        factories={'gul': f'{THIS}:CrashingGul', 'fm': FAKE_FACTORIES['fm']})
    assert info.value.chunk == k
    assert info.value.returncode == 3
    assert 'about to crash' in info.value.stderr_tail
    assert 'about to crash' in str(info.value)
    assert descendants() == []


def test_stderr_errors_ignores_blank_lines_and_python_warnings():
    text = ('\n   \n/x/y.py:12: UserWarning: careful\n  warnings.warn("careful")\n'
            '/x/z.py:3: NumbaPerformanceWarning: slow\nTraceback (most recent call last):\nERROR here\n')
    assert pool.stderr_errors(text) == ['Traceback (most recent call last):', 'ERROR here']
    assert pool.stderr_errors('') == []


def test_pool_stderr_guard_fails_on_an_error_line(tmp_path, run_dir):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 3)
    factories = {'gul': f'{THIS}:StderrGul', 'fm': FAKE_FACTORIES['fm']}
    with pytest.raises(pool.WorkerError) as info:
        run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, 2, factories=factories, stderr_guard=True,
                 log_dir=str(tmp_path / 'log'))
    assert 'wrote to stderr: ERROR: something went wrong' in str(info.value)
    assert 'ERROR: something went wrong' in info.value.stderr_tail
    assert descendants() == []
    logs = sorted(os.listdir(tmp_path / 'log'))
    assert 'orchestrator_worker_0.err' in logs and 'orchestrator_worker_0.log' in logs

    stats = run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out2', TAPS, 2, factories=factories, stderr_guard=False)
    check_fake_outputs(tmp_path / 'out2', chunks)
    assert stats['events'] == 30


def test_pool_stderr_guard_ignores_warnings(tmp_path, run_dir):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 3)
    log_dir = tmp_path / 'log'
    run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, 2, stderr_guard=True, log_dir=str(log_dir),
             factories={'gul': f'{THIS}:WarningGul', 'fm': FAKE_FACTORIES['fm']})
    check_fake_outputs(tmp_path / 'out', chunks)
    stderr = ''.join(p.read_text() for p in log_dir.glob('orchestrator_worker_*.err'))
    assert 'UserWarning: this is only a warning' in stderr


def test_pool_failure_terminates_hanging_workers_and_their_children(tmp_path, run_dir):
    chunks = [np.array([1, HANG_EVENT, 3], dtype=np.int32), np.array([4, RAISE_EVENT], dtype=np.int32)]
    t_start = time.monotonic()
    with pytest.raises(pool.WorkerError) as info:
        run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, 2,
                        factories={'gul': f'{THIS}:HangOrRaiseGul', 'fm': FAKE_FACTORIES['fm']}, kill_grace=2)
    assert time.monotonic() - t_start < 60
    assert (info.value.chunk, info.value.event) == (2, RAISE_EVENT)
    grandchild = int((run_dir / 'grandchild.pid').read_text())
    deadline = time.monotonic() + 10
    while psutil.pid_exists(grandchild) and psutil.Process(grandchild).status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not psutil.pid_exists(grandchild) or psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE
    assert descendants() == []


def test_stage_name():
    chain = Chain(make_spec(n_ri=2), factories={'gul': FakeGul, 'fm': FakeFm})
    assert [stage_name(chain, i) for i in (None, 0, 1, 2)] == ['gul', 'il (fm step 1)', 'ri (fm step 2)', 'ri (fm step 3)']


def test_cli_get_executor_pool():
    assert cli.get_executor('pool') is pool


def kernel_spec(kernel_run):
    args = cli.build_parser().parse_args([
        '--run-dir', str(kernel_run.run_dir), '--out-dir', 'unused', f'-S{kernel_run.samples}', *DEFAULT_GUL_ARGS,
        '--il-alloc-rule', str(kernel_run.il_alloc_rule), '--ri-layers', '1', '--ri-alloc-rule', str(kernel_run.ri_alloc_rule),
    ])
    return cli.spec_from_args(args)


def assert_same_streams(dir_a, dir_b, n_chunks):
    results = compare_dirs_bytes(dir_a, dir_b, taps=TAPS)
    assert len(results) == 3 * n_chunks
    assert all(r.ok for r in results), [str(r) for r in results if not r.ok]


@pytest.mark.parametrize('n_chunks, n_workers', [(1, 1), (3, 1), (3, 2), (2, 4), (5, 3)])
def test_pool_kernel_byte_identical_to_reference_and_inline(kernel_run, kernel_reference, tmp_path, n_chunks, n_workers):
    spec = kernel_spec(kernel_run)
    chunks = make_chunks(kernel_run.event_ids, n_chunks)
    stats = run_pool(spec, chunks, tmp_path / 'pool', TAPS, n_workers, timeout=600)
    assert_same_streams(tmp_path / 'pool', kernel_reference(n_chunks), n_chunks)
    inline.run_chunks(spec, chunks, tmp_path / 'inline', TAPS)
    assert_same_streams(tmp_path / 'pool', tmp_path / 'inline', n_chunks)
    check_stats_shape(stats, chunks, min(n_workers, n_chunks))
    assert descendants() == []


TREE_FACTORY_STRINGS = {**FAKE_FACTORIES, 'summary': 'tests.orchestrator.test_chain:FakeSummary',
                        'elt': 'tests.orchestrator.test_chain:FakeElt', 'plt': 'tests.orchestrator.test_chain:FakePlt'}


class RaisingSummary(FakeSummary):
    def process_block(self, block, outs, idx_outs=None):
        if self.run_type == 'il' and np.frombuffer(block.tobytes(), dtype=np.int32)[0] == 2 * FM_RAISE_EVENT + 2:
            raise ValueError('summary failed')
        super().process_block(block, outs, idx_outs)


@pytest.mark.parametrize('n_chunks, n_workers', [(5, 2), (3, 3)])
def test_pool_tree_writes_chunk_files_and_taps(tmp_path, run_dir, n_chunks, n_workers):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), n_chunks)
    out_dir = tmp_path / 'out'
    run_pool(make_tree_spec(str(run_dir), n_ri=1), chunks, out_dir, TAPS, n_workers, factories=TREE_FACTORY_STRINGS)
    check_fake_outputs(out_dir, chunks)
    for k, chunk in enumerate(chunks, 1):
        assert read_tree_files(run_dir, k) == expected_tree_files([int(e) for e in chunk], n_ri=1, k=k), k


def test_pool_summary_exception_names_summary_stage(tmp_path, run_dir):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 3)
    with pytest.raises(pool.WorkerError) as info:
        run_pool(make_tree_spec(str(run_dir)), chunks, tmp_path / 'out', [], 2,
                 factories={**TREE_FACTORY_STRINGS, 'summary': f'{THIS}:RaisingSummary'})
    assert info.value.stage == 'summary il'
    assert info.value.event == FM_RAISE_EVENT
    assert 'summary failed' in str(info.value)
    assert descendants() == []


@pytest.mark.parametrize('executor', ['inline', 'pool'])
def test_chunk_numbers_run_only_those_chunks(tmp_path, run_dir, executor):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 4)
    out_dir = tmp_path / 'out'
    if executor == 'pool':
        stats = run_pool(make_spec(str(run_dir)), chunks, out_dir, TAPS, 4, factories=FAKE_FACTORIES, chunk_numbers=[3, 1])
        assert stats['n_workers'] == 2
        assert descendants() == []
    else:
        stats = inline.run_chunks(make_spec(str(run_dir)), chunks, out_dir, TAPS, factories={'gul': FakeGul, 'fm': FakeFm},
                                  chunk_numbers=[3, 1])
    assert sorted(c['chunk'] for c in stats['chunks']) == [1, 3]
    assert stats['events'] == len(chunks[0]) + len(chunks[2])
    assert sorted(os.listdir(out_dir)) == sorted(f'{tap}_P{k}.bin' for tap in TAPS for k in (1, 3))
    selected = tmp_path / 'selected'
    selected.mkdir()
    for name in os.listdir(out_dir):
        shutil.copy(out_dir / name, selected / name.replace('_P3', '_P2'))
    check_fake_outputs(selected, [chunks[0], chunks[2]])


@pytest.mark.parametrize('chunk_numbers', [[0], [5], [1, 1]])
def test_chunk_numbers_are_checked(tmp_path, run_dir, chunk_numbers):
    chunks = make_chunks(np.arange(1, 9, dtype=np.int32), 4)
    with pytest.raises(ValueError, match='chunk_numbers'):
        run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, 1, factories=FAKE_FACTORIES, chunk_numbers=chunk_numbers)
    with pytest.raises(ValueError, match='chunk_numbers'):
        inline.run_chunks(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, factories={'gul': FakeGul, 'fm': FakeFm},
                          chunk_numbers=chunk_numbers)


def test_pool_progress_messages_count_every_event_once(tmp_path, run_dir):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 3)
    calls = []
    stats = run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, 2, factories=FAKE_FACTORIES, progress_interval=0,
                     progress=lambda done, total: calls.append((done, total)), log_dir=str(tmp_path / 'logs'))
    assert calls[-1] == (30, 30)
    assert [d for d, _ in calls] == sorted(d for d, _ in calls)
    assert len(calls) > len(chunks)
    assert sorted(os.listdir(tmp_path / 'logs')) == sorted(f'orchestrator_worker_{w["worker"]}.{ext}'
                                                           for w in stats['workers'] for ext in ('err', 'log'))
