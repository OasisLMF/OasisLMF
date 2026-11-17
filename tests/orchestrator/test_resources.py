"""Tests of the pool's resource manager (``orchestrator/resources.py``).

The policy is tested against a fake pool and a fake probe (memory, CPUs and worker sizes set by the test), then on a
real pool with fake stages, and its cgroup reading on fake cgroup trees.
"""
import collections
import dataclasses
import os
import types

import numpy as np
import pytest

from oasislmf.execution.orchestrator import pool
from oasislmf.execution.orchestrator.chunks import make_chunks
from oasislmf.execution.orchestrator.resources import (GB, MB, ResourceManager, SystemProbe, cgroup_memory, default_margin)

from .test_chain import make_spec
from .test_pool import FAKE_FACTORIES, TAPS, check_fake_outputs, descendants, run_pool

WORKER = GB // 2


class FakeProbe:
    """Memory, CPUs and worker sizes as the test sets them."""

    def __init__(self, available=30 * GB, total=32 * GB, cpus=12, worker_memory=WORKER, busy=0.0):
        self.available = available
        self.total = total
        self.cpus = cpus
        self.worker_memory = worker_memory
        self.sizes = {}
        self.busy = busy

    def memory(self):
        return self.available, self.total

    def cpu_count(self):
        return self.cpus

    def processes_memory(self, pids):
        return {pid: self.sizes.get(pid, self.worker_memory) for pid in pids}

    def busy_cpus(self, exclude_pids=()):
        return self.busy


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@dataclasses.dataclass
class FakeWorker:
    number: int
    popen: object
    jobs_only: bool = False
    setup_seconds: float = None
    chunks: int = 0
    jobs: int = 0
    stopped: bool = False
    retire: bool = False


class FakePool:
    """The parts of ``WorkerPool`` the manager uses."""

    def __init__(self, n_chunks=40, n_jobs=0):
        self.workers = []
        self.pending = collections.deque(range(1, n_chunks + 1))
        self.pending_jobs = collections.deque(range(n_jobs))
        self.n_chunks = n_chunks
        self.done = 0

    @property
    def chunks_done(self):
        return self.done == self.n_chunks

    @property
    def active(self):
        return [w for w in self.workers if not w.stopped and not w.retire]

    @property
    def live(self):
        return [w for w in self.workers if not w.stopped]

    def spawn(self, jobs_only=False):
        worker = FakeWorker(len(self.workers), types.SimpleNamespace(pid=10000 + len(self.workers)), jobs_only=jobs_only)
        self.workers.append(worker)
        return worker

    def retire(self, worker=None):
        worker = worker or self.active[-1]
        worker.retire = True
        return worker

    def ready_all(self, chunks=0):
        """Every live worker has built its chain; with ``chunks``, each has finished that many chunks."""
        for w in self.live:
            w.setup_seconds = 1.0
            w.chunks = max(w.chunks, chunks)

    def stop_retiring(self):
        for w in self.workers:
            if w.retire:
                w.stopped = True


def manager(probe, max_workers=8, clock=None, **options):
    options.setdefault('margin', default_margin(probe.total))
    return ResourceManager(max_workers, probe=probe, clock=clock or FakeClock(), **options)


def tick(m, p, clock, seconds=1.0):
    clock.t += seconds
    m(p)


def test_default_margin():
    assert default_margin(4 * GB) == GB
    assert default_margin(32 * GB) == int(3.2 * GB)


def test_starts_initial_workers_and_waits_for_a_measurement():
    probe, clock, p = FakeProbe(), FakeClock(), FakePool()
    m = manager(probe, clock=clock)
    m.start(p)
    assert len(p.workers) == 2
    tick(m, p, clock)
    assert len(p.workers) == 2, 'nothing grows before a worker has built its chain'
    assert m.decisions[0].action == 'start'


def test_grows_to_the_hard_maximum():
    probe, clock, p = FakeProbe(), FakeClock(), FakePool()
    m = manager(probe, max_workers=8, clock=clock)
    m.start(p)
    p.ready_all()
    tick(m, p, clock)
    assert len(p.workers) == 8
    assert m.estimate == 2 * WORKER, 'unmeasured factor until a chunk is done'
    p.ready_all(chunks=1)
    for _ in range(5):
        tick(m, p, clock)
    assert len(p.live) == 8
    assert m.estimate == WORKER
    assert [d.action for d in m.decisions] == ['start', 'spawn']
    assert m.summary()['max_live_workers'] == 8


def test_never_more_workers_than_cpus_or_chunks():
    probe, clock = FakeProbe(cpus=4), FakeClock()
    p = FakePool()
    m = manager(probe, max_workers=16, clock=clock)
    m.start(p)
    p.ready_all(chunks=1)
    tick(m, p, clock)
    assert len(p.workers) == 4

    p = FakePool(n_chunks=3)
    m = manager(FakeProbe(), max_workers=16, clock=clock)
    m.start(p)
    p.pending.popleft()
    p.pending.popleft()
    p.ready_all()
    tick(m, p, clock)
    assert len(p.workers) == 3, 'one chunk left, so one more worker'


def test_honours_the_hard_maximum_from_the_start():
    probe, clock, p = FakeProbe(), FakeClock(), FakePool()
    m = manager(probe, max_workers=1, clock=clock)
    m.start(p)
    p.ready_all(chunks=1)
    for _ in range(3):
        tick(m, p, clock)
    assert len(p.workers) == 1


def test_stops_growing_at_the_memory_limit():
    probe, clock, p = FakeProbe(available=5 * GB), FakeClock(), FakePool()
    m = manager(probe, clock=clock)
    m.start(p)
    p.ready_all(chunks=1)
    tick(m, p, clock)
    assert len(p.workers) == 5, '5 GB available, 3.2 GB margin: room for 3 more at 0.5 GB'
    for w in p.workers[2:]:
        probe.sizes[w.popen.pid] = 0
    for _ in range(5):
        tick(m, p, clock)
    assert len(p.workers) == 5, 'the new workers reserve their expected growth'


def test_reserves_growth_of_workers_still_setting_up():
    probe, clock, p = FakeProbe(available=6 * GB), FakeClock(), FakePool()
    m = manager(probe, clock=clock)
    m.start(p)
    p.workers[0].setup_seconds = 1.0
    p.workers[0].chunks = 1
    probe.sizes[p.workers[1].popen.pid] = 0
    tick(m, p, clock)
    # 6 - 3.2 margin - 0.5 reserved for worker 1 = 2.3 GB: 4 more
    assert len(p.workers) == 6


def test_retires_under_pressure_at_chunk_boundaries_and_cools_down():
    probe, clock, p = FakeProbe(), FakeClock(), FakePool()
    m = manager(probe, clock=clock, grow_cooldown=10.0)
    m.start(p)
    p.ready_all(chunks=1)
    tick(m, p, clock)
    assert len(p.active) == 8
    probe.available = int(2.2 * GB)
    tick(m, p, clock)
    assert len(p.active) == 6, '1 GB below the margin at 0.5 GB per worker'
    assert all(not w.stopped for w in p.workers), 'retired workers are not killed'
    tick(m, p, clock)
    assert len(p.active) == 6, 'workers already retiring count against the deficit'
    p.stop_retiring()
    probe.available = 30 * GB
    tick(m, p, clock)
    assert len(p.live) == 6, 'no growth during the cool-down'
    tick(m, p, clock, seconds=10.0)
    assert len(p.live) == 8
    assert [d.action for d in m.decisions] == ['start', 'spawn', 'retire', 'spawn']


def test_never_below_one_worker_and_warns():
    probe, clock, p = FakeProbe(), FakeClock(), FakePool()
    m = manager(probe, max_workers=2, clock=clock)
    m.start(p)
    p.ready_all(chunks=1)
    probe.available = 0
    tick(m, p, clock)
    assert len(p.active) == 1
    p.stop_retiring()
    for _ in range(3):
        tick(m, p, clock)
    assert len(p.active) == 1 and len(p.live) == 1
    assert len(m.warnings) == 1 and 'one worker' in m.warnings[0]


def test_warns_when_one_worker_does_not_fit():
    probe, clock, p = FakeProbe(available=int(3.5 * GB), worker_memory=GB), FakeClock(), FakePool()
    m = manager(probe, clock=clock, initial_workers=1)
    m.start(p)
    p.ready_all(chunks=1)
    for _ in range(3):
        tick(m, p, clock)
    assert len(p.workers) == 1
    assert len(m.warnings) == 1 and 'running with one worker' in m.warnings[0]


def test_starts_one_worker_when_memory_is_already_short():
    probe, clock, p = FakeProbe(available=GB), FakeClock(), FakePool()
    m = manager(probe, clock=clock)
    m.start(p)
    assert len(p.workers) == 1
    assert m.warnings and 'starting one worker' in m.warnings[0]


def test_counts_cpus_busy_elsewhere_when_asked():
    probe, clock, p = FakeProbe(cpus=8, busy=5.0), FakeClock(), FakePool()
    m = manager(probe, max_workers=8, clock=clock, cpu_load=True)
    m.start(p)
    p.ready_all(chunks=1)
    tick(m, p, clock)
    assert len(p.workers) == 3


def test_job_phase_adds_job_only_workers():
    probe, clock, p = FakeProbe(), FakeClock(), FakePool(n_chunks=0, n_jobs=6)
    m = manager(probe, max_workers=4, clock=clock)
    m.start(p)
    assert [w.jobs_only for w in p.workers] == [True, True]
    p.ready_all()
    p.workers[0].jobs = 1
    tick(m, p, clock)
    assert len(p.workers) == 4 and all(w.jobs_only for w in p.workers)


def test_spawns_one_when_no_worker_is_left_with_work():
    probe, clock, p = FakeProbe(), FakeClock(), FakePool()
    m = manager(probe, clock=clock)
    m.start(p)
    for w in p.workers:
        w.stopped = True
    m(p)
    assert len(p.live) == 1


def test_samples_at_most_every_interval_unless_a_worker_changes():
    probe, clock, p = FakeProbe(), FakeClock(), FakePool()
    m = manager(probe, max_workers=2, clock=clock, sample_interval=1.0)
    m.start(p)
    m(p)
    m(p)
    assert m.samples == 1
    p.ready_all()
    m(p)
    assert m.samples == 1, 'a change within MIN_SAMPLE_GAP waits'
    clock.t += 0.3
    m(p)
    assert m.samples == 2
    p.workers[0].jobs = 5
    clock.t += 0.3
    m(p)
    assert m.samples == 2, 'output jobs do not trigger a measurement'
    clock.t += 1.0
    m(p)
    assert m.samples == 3


def test_rejects_bad_settings():
    with pytest.raises(ValueError):
        ResourceManager(0, probe=FakeProbe())
    with pytest.raises(ValueError):
        ResourceManager(2, probe=FakeProbe(), initial_workers=0)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cgroup_memory_v2_takes_the_tightest_ancestor(tmp_path):
    root = tmp_path / 'cg'
    write(tmp_path / 'proc_cgroup', '0::/a/b\n')
    write(root / 'a' / 'memory.max', str(8 * GB))
    write(root / 'a' / 'memory.current', str(7 * GB))
    write(root / 'a' / 'memory.stat', f'anon 1\ninactive_file {GB}\n')
    write(root / 'a' / 'b' / 'memory.max', 'max')
    write(root / 'a' / 'b' / 'memory.current', str(GB))
    assert cgroup_memory(str(tmp_path / 'proc_cgroup'), str(root)) == (8 * GB, 2 * GB)


def test_cgroup_memory_none_without_a_limit(tmp_path):
    write(tmp_path / 'proc_cgroup', '0::/a\n')
    write(tmp_path / 'cg' / 'a' / 'memory.max', 'max')
    assert cgroup_memory(str(tmp_path / 'proc_cgroup'), str(tmp_path / 'cg')) is None
    assert cgroup_memory(str(tmp_path / 'missing'), str(tmp_path / 'cg')) is None


def test_system_probe_reads_this_machine():
    probe = SystemProbe()
    available, total = probe.memory()
    assert 0 < available <= total
    assert probe.cpu_count() >= 1
    gone = 2 ** 22 + 12345
    sizes = probe.processes_memory([os.getpid(), gone])
    assert sizes[os.getpid()] > MB and gone not in sizes
    assert probe.busy_cpus([os.getpid()]) >= 0.0


# --- on a real pool, fake stages ------------------------------------------------------------------------------------


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / 'run'
    (path / 'input').mkdir(parents=True)
    return path


class PressureProbe(FakeProbe):
    """Plenty of memory until ``after`` samples, then below the margin."""

    def __init__(self, after, **kwargs):
        super().__init__(**kwargs)
        self.after = after
        self.calls = 0

    def memory(self):
        self.calls += 1
        if self.calls > self.after:
            return GB // 2, self.total
        return self.available, self.total


def run_managed(tmp_path, run_dir, chunks, m):
    return run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, m.max_workers, factories=FAKE_FACTORIES,
                    resource_manager=m)


def test_pool_with_the_manager_grows_and_writes_every_chunk(tmp_path, run_dir):
    chunks = make_chunks(np.arange(1, 61, dtype=np.int32), 12)
    m = ResourceManager(3, probe=FakeProbe(), sample_interval=0.05)
    stats = run_managed(tmp_path, run_dir, chunks, m)
    check_fake_outputs(tmp_path / 'out', chunks)
    assert sum(w['chunks'] for w in stats['workers']) == 12
    summary = stats['resource_manager']
    assert summary['max_live_workers'] <= 3
    assert summary['decisions'][0]['action'] == 'start'
    assert descendants() == []


def test_pool_with_the_manager_retires_under_pressure_and_finishes(tmp_path, run_dir):
    chunks = make_chunks(np.arange(1, 61, dtype=np.int32), 12)
    m = ResourceManager(2, probe=PressureProbe(after=1), sample_interval=0.0, initial_workers=2)
    stats = run_managed(tmp_path, run_dir, chunks, m)
    check_fake_outputs(tmp_path / 'out', chunks)
    actions = [d['action'] for d in stats['resource_manager']['decisions']]
    assert 'retire' in actions
    assert sum(w['chunks'] for w in stats['workers']) == 12
    assert descendants() == []


def test_pool_rejects_manager_and_policy_together(tmp_path, run_dir):
    with pytest.raises(ValueError, match='exclusive'):
        pool.run_chunks(make_spec(str(run_dir)), [[1]], tmp_path / 'out', TAPS, 1, factories=FAKE_FACTORIES,
                        resource_manager=True, policy=lambda p: None)
