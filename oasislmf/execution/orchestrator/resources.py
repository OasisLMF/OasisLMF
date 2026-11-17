"""Resource manager of the fused pool: sizes the pool from measured memory and CPU so a run does not exhaust memory.

``ResourceManager`` is a ``WorkerPool`` policy. It starts a few workers, measures what each one uses (the private
memory of the worker's process tree, ``private_memory``, sampled every ``sample_interval`` seconds, its peak kept) and grows the pool while
the available memory, less a safety margin and less what the workers still setting up are expected to add, holds one
more worker of the largest size seen. Under memory pressure it retires workers at their next chunk boundary (their
next ``ready`` is answered with ``stop``), never below one; a worker is never killed for memory. When even one worker
does not fit, the run carries on with one and a warning is logged.

The machine is read through a probe (``SystemProbe``: psutil, and the cgroup memory limit when the process runs under
one, as in a container or a ``systemd-run`` scope), so tests can pass a fake one.

Every decision is logged (``logger``) and kept, with the measurements it was made from, in ``summary()``, which the
pool adds to its statistics as ``resource_manager``.
"""
import dataclasses
import logging
import math
import os
import time

import psutil

logger = logging.getLogger(__name__)

GB = 1024 ** 3
MB = 1024 ** 2
DEFAULT_INITIAL_WORKERS = 2
DEFAULT_MARGIN_MIN = 1 * GB
DEFAULT_MARGIN_FRACTION = 0.10
DEFAULT_UNMEASURED_FACTOR = 2.0
DEFAULT_SAMPLE_INTERVAL = 1.0
MIN_SAMPLE_GAP = 0.2
DEFAULT_GROW_COOLDOWN = 10.0
CGROUP_V2_ROOT = '/sys/fs/cgroup'
CGROUP_V1_MEMORY = '/sys/fs/cgroup/memory'


def _read_int(path):
    """The integer in a cgroup file, None for ``max``, a missing or unreadable file."""
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_stat(path, key):
    """The value of ``key`` in a cgroup ``memory.stat`` file, 0 if absent."""
    try:
        with open(path) as f:
            for line in f:
                name, _, value = line.partition(' ')
                if name == key:
                    return int(value)
    except (OSError, ValueError):
        pass
    return 0


def cgroup_memory(proc_cgroup='/proc/self/cgroup', v2_root=CGROUP_V2_ROOT, v1_memory=CGROUP_V1_MEMORY):
    """The memory limit of this process's cgroups and the room left under it.

    With cgroup v2 every cgroup from this process's up to the root is looked at, and the tightest one counts. Used
    memory is ``memory.current`` less the reclaimable ``inactive_file`` page cache. With cgroup v1 the memory
    controller's ``memory.limit_in_bytes`` and ``memory.usage_in_bytes`` are read.

    Returns:
        tuple[int, int] | None: ``(limit, room)`` in bytes, None when no limit applies or none can be read.
    """
    try:
        with open(proc_cgroup) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    best = None
    for line in lines:
        hierarchy, _, rest = line.partition(':')
        controllers, _, path = rest.partition(':')
        if hierarchy == '0' and controllers == '':
            directory = os.path.normpath(os.path.join(v2_root, path.lstrip('/')))
            while True:
                limit = _read_int(os.path.join(directory, 'memory.max'))
                current = _read_int(os.path.join(directory, 'memory.current'))
                if limit is not None and current is not None:
                    used = current - _read_stat(os.path.join(directory, 'memory.stat'), 'inactive_file')
                    room = max(limit - used, 0)
                    if best is None or room < best[1]:
                        best = (limit, room)
                if os.path.normpath(directory) == os.path.normpath(v2_root) or directory == os.path.dirname(directory):
                    break
                directory = os.path.dirname(directory)
        elif 'memory' in controllers.split(','):
            limit = _read_int(os.path.join(v1_memory, 'memory.limit_in_bytes'))
            usage = _read_int(os.path.join(v1_memory, 'memory.usage_in_bytes'))
            if limit is not None and usage is not None and limit < psutil.virtual_memory().total:
                used = usage - _read_stat(os.path.join(v1_memory, 'memory.stat'), 'total_inactive_file')
                room = max(limit - used, 0)
                if best is None or room < best[1]:
                    best = (limit, room)
    return best


def available_cpus():
    """CPUs this process may run on: its affinity where the platform has one, else ``os.cpu_count()``."""
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


class SystemProbe:
    """Reads the machine for ``ResourceManager``: memory, CPUs and the memory of worker processes."""

    def __init__(self):
        self._procs = {}

    def memory(self):
        """Available and total memory in bytes, the tighter of the machine and this process's cgroup limit.

        Returns:
            tuple[int, int]: ``(available, total)``.
        """
        vm = psutil.virtual_memory()
        available, total = vm.available, vm.total
        cgroup = cgroup_memory()
        if cgroup is not None:
            limit, room = cgroup
            total = min(total, limit)
            available = min(available, room)
        return available, total

    def cpu_count(self):
        """CPUs the workers may use."""
        return available_cpus()

    def processes_memory(self, pids):
        """Memory of processes, each with its descendants (found from one snapshot of the process table).

        Args:
            pids (Iterable[int]): process ids.

        Returns:
            dict[int, int]: ``private_memory`` of each process still running and its descendants, by pid.
        """
        pids = list(pids)
        children = {}
        for proc in psutil.process_iter(['ppid']):
            children.setdefault(proc.info['ppid'], []).append(proc.pid)
        result = {}
        for pid in pids:
            tree, pending = [], [pid]
            while pending:
                current = pending.pop()
                tree.append(current)
                pending.extend(children.get(current, []))
            total, alive = 0, False
            for member in tree:
                try:
                    total += private_memory(psutil.Process(member))
                    alive = alive or member == pid
                except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                    continue
            if alive:
                result[pid] = total
        return result

    def busy_cpus(self, exclude_pids=()):
        """CPUs kept busy by other processes since the previous call: system-wide use less that of ``exclude_pids``.

        The first call measures from the import of psutil (or the previous call of ``psutil.cpu_percent``).
        """
        system = psutil.cpu_percent(interval=None) / 100.0 * (os.cpu_count() or 1)
        own = 0.0
        for pid in exclude_pids:
            proc = self._procs.get(pid)
            try:
                if proc is None:
                    proc = self._procs[pid] = psutil.Process(pid)
                own += proc.cpu_percent(interval=None) / 100.0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._procs.pop(pid, None)
        return max(system - own, 0.0)


def private_memory(proc):
    """Memory a process holds for itself, what one more such process costs.

    Resident memory less its file-backed shared part on Linux (libraries and memory-mapped model files stay in the
    page cache, shared by every worker and reclaimable), the private bytes on Windows, the resident size elsewhere.
    Read from ``memory_info``, which is cheap; the unique set size would need a walk of every mapping.
    """
    info = proc.memory_info()
    if hasattr(info, 'shared'):
        return max(info.rss - info.shared, 0)
    if hasattr(info, 'private'):
        return info.private
    return info.rss


def default_margin(total, margin_min=DEFAULT_MARGIN_MIN, fraction=DEFAULT_MARGIN_FRACTION):
    """The memory kept free by default: ``max(margin_min, fraction * total)``."""
    return int(max(margin_min, fraction * total))


def _gb(n_bytes):
    return f'{n_bytes / GB:.2f} GB'


@dataclasses.dataclass
class Decision:
    """One decision of the manager, with what it was based on."""
    seconds: float
    action: str
    workers_before: int
    workers_after: int
    available: int
    margin: int
    estimate: int
    reason: str

    def text(self):
        """One line for the run log."""
        return (f'{self.seconds:.1f} s: {self.action} {self.workers_before} -> {self.workers_after} workers ({self.reason}; '
                f'available {_gb(self.available)}, margin {_gb(self.margin)}, estimate {_gb(self.estimate)} per worker)')


class ResourceManager:
    """Pool policy sizing the pool from measured memory and CPU (see the module docstring).

    ``start(pool)`` spawns the first workers; the pool then calls the manager (``manager(pool)``) after every message
    and on every pass of its loop. Decisions are taken on fresh measurements only: every ``sample_interval``
    seconds, and as soon as a worker becomes ready or finishes a chunk (but not within ``MIN_SAMPLE_GAP`` seconds of
    the previous measurement).

    Growth: one more worker is spawned when chunks (or, once every chunk is done, output jobs) are left for it, the
    hard maximum and the CPUs allow it, and ``available - margin - reserve >= estimate``, where ``estimate`` is the
    largest peak memory of a worker seen so far (times ``unmeasured_factor`` until a worker has finished a chunk or
    job, as a worker's memory grows while it runs its first chunk) and ``reserve`` what the live workers may still
    grow to that estimate. Nothing grows until a worker has finished setting up (its chain is built), nor for
    ``grow_cooldown`` seconds after a retirement.

    Back-off: when ``available < margin``, enough of the newest active workers are retired to cover the deficit at
    ``estimate`` each (less the workers already retiring), keeping one; each stops at its next chunk boundary.
    """

    def __init__(self, max_workers, probe=None, initial_workers=DEFAULT_INITIAL_WORKERS, margin=None,
                 margin_min=DEFAULT_MARGIN_MIN, margin_fraction=DEFAULT_MARGIN_FRACTION, unmeasured_factor=DEFAULT_UNMEASURED_FACTOR,
                 sample_interval=DEFAULT_SAMPLE_INTERVAL, grow_cooldown=DEFAULT_GROW_COOLDOWN, cpu_load=False, clock=time.monotonic):
        """Set the policy up; nothing is measured or started before ``start``.

        Args:
            max_workers (int): hard maximum of live workers (``--workers``, ``ktools_num_processes``).
            probe (SystemProbe, optional): what the machine is read through; ``SystemProbe()`` by default.
            initial_workers (int): workers started before anything is measured.
            margin (int, optional): bytes of memory kept free; ``max(margin_min, margin_fraction * total memory)`` by
                default.
            margin_min (int): see ``margin``.
            margin_fraction (float): see ``margin``.
            unmeasured_factor (float): the estimate is the largest worker peak seen times this until a worker has
                finished a chunk or job.
            sample_interval (float): seconds between two measurements.
            grow_cooldown (float): seconds after a retirement during which the pool does not grow.
            cpu_load (bool): also count the CPUs other processes keep busy (``probe.busy_cpus``) as taken.
            clock (callable): monotonic clock in seconds.
        """
        if max_workers < 1:
            raise ValueError(f'max_workers must be positive, got {max_workers}')
        if initial_workers < 1:
            raise ValueError(f'initial_workers must be positive, got {initial_workers}')
        self.max_workers = int(max_workers)
        self.probe = probe if probe is not None else SystemProbe()
        self.initial_workers = int(initial_workers)
        self.margin = margin
        self.margin_min = margin_min
        self.margin_fraction = margin_fraction
        self.unmeasured_factor = unmeasured_factor
        self.sample_interval = sample_interval
        self.grow_cooldown = grow_cooldown
        self.cpu_load = cpu_load
        self.clock = clock
        self.decisions = []
        self.warnings = []
        self.peaks = {}
        self.current = {}
        self.estimate = 0
        self.available = None
        self.total = None
        self.max_live = 0
        self.samples = 0
        self._t0 = None
        self._t_sample = None
        self._t_retire = None
        self._signature = None
        self._warned_one = False

    def _now(self):
        return self.clock() - self._t0

    @staticmethod
    def _live(pool):
        return [w for w in pool.workers if not w.stopped]

    def _record(self, action, before, after, reason):
        decision = Decision(self._now(), action, before, after, self.available or 0, self.margin or 0, self.estimate, reason)
        self.decisions.append(decision)
        logger.info('resource manager: %s', decision.text())
        return decision

    def _warn(self, message):
        text = f'{self._now():.1f} s: {message}'
        self.warnings.append(text)
        logger.warning('resource manager: %s', message)

    def _useful(self, pool):
        """How many more workers would find work.

        Chunks not yet handed out, or output jobs once every chunk is done, less the workers still setting up, which
        will take them.
        """
        setting_up = sum(1 for w in self._live(pool) if w.setup_seconds is None)
        if pool.pending:
            return max(len(pool.pending) - setting_up, 0)
        if pool.chunks_done and pool.pending_jobs:
            return max(len(pool.pending_jobs) - setting_up, 0)
        return 0

    def _spawn(self, pool, n):
        jobs_only = not pool.pending and pool.chunks_done
        for _ in range(n):
            pool.spawn(jobs_only=jobs_only)

    def _measure(self):
        self.available, total = self.probe.memory()
        if self.total is None:
            self.total = total
            if self.margin is None:
                self.margin = default_margin(total, self.margin_min, self.margin_fraction)

    def start(self, pool):
        """Spawn the first workers: ``initial_workers``, within the hard maximum, the CPUs and the work available."""
        self._t0 = self.clock()
        self._measure()
        useful = len(pool.pending) if pool.pending else (len(pool.pending_jobs) if pool.chunks_done else 0)
        n = min(self.initial_workers, self.max_workers, max(self.probe.cpu_count(), 1), useful)
        if n <= 0:
            return
        if self.available < self.margin:
            n = 1
            self._warn(f'available memory {_gb(self.available)} is below the margin {_gb(self.margin)}: starting one worker')
        self._spawn(pool, n)
        self.max_live = max(self.max_live, len(self._live(pool)))
        self._record('start', 0, n, f'initial workers, maximum {self.max_workers}')

    def _sample(self, pool):
        """Measure the memory and every live worker; update the peaks and the estimate."""
        self._measure()
        self.samples += 1
        self.current = {}
        live = self._live(pool)
        sizes = self.probe.processes_memory([w.popen.pid for w in live])
        for w in live:
            used = sizes.get(w.popen.pid)
            if used is None:
                continue
            self.current[w.number] = used
            if w.setup_seconds is not None or w.chunks or w.jobs:
                self.peaks[w.number] = max(self.peaks.get(w.number, 0), used)
        peak = max(self.peaks.values(), default=0)
        proven = any(w.chunks or w.jobs for w in pool.workers)
        self.estimate = int(peak if proven else peak * self.unmeasured_factor)

    def _due(self, pool):
        signature = (sum(1 for w in pool.workers if w.setup_seconds is not None), sum(w.chunks for w in pool.workers),
                     len(pool.workers))
        now = self.clock()
        since = None if self._t_sample is None else now - self._t_sample
        changed = signature != self._signature and (since is None or since >= min(self.sample_interval, MIN_SAMPLE_GAP))
        if changed or since is None or since >= self.sample_interval:
            self._signature = signature
            self._t_sample = now
            return True
        return False

    def __call__(self, pool):
        """Measure when due and spawn or retire workers (see the class docstring)."""
        if self._t0 is None:
            self._t0 = self.clock()
        live = self._live(pool)
        work_left = bool(pool.pending or not pool.chunks_done or pool.pending_jobs)
        if not live and work_left:
            self._measure()
            self._spawn(pool, 1)
            self._record('spawn', 0, 1, 'no worker left with work remaining')
            return
        if not self._due(pool):
            return
        self._sample(pool)
        if self.available < self.margin:
            self._back_off(pool)
        else:
            self._grow(pool)
        self.max_live = max(self.max_live, len(self._live(pool)))

    def _back_off(self, pool):
        active = pool.active
        retiring = sum(1 for w in pool.workers if w.retire and not w.stopped)
        deficit = self.margin - self.available
        wanted = math.ceil(deficit / max(self.estimate, MB)) - retiring
        n = min(max(wanted, 0), len(active) - 1)
        if n > 0:
            for _ in range(n):
                pool.retire()
            self._t_retire = self.clock()
            self._record('retire', len(active), len(active) - n,
                         f'available memory {_gb(deficit)} below the margin; stopping at their next chunk boundary')
        if len(pool.active) == 1 and retiring == 0 and n <= 0 and not self._warned_one:
            self._warned_one = True
            self._warn(f'available memory {_gb(self.available)} is below the margin {_gb(self.margin)} with one worker left: '
                       f'carrying on with one worker')

    def _grow(self, pool):
        if self.estimate <= 0 or not any(w.setup_seconds is not None for w in pool.workers):
            return
        if self._t_retire is not None and self.clock() - self._t_retire < self.grow_cooldown:
            return
        live = self._live(pool)
        reserve = sum(max(self.estimate - self.current.get(w.number, 0), 0) for w in live)
        room = self.available - self.margin - reserve
        by_memory = int(room // self.estimate) if room > 0 else 0
        cpus = self.probe.cpu_count()
        if self.cpu_load:
            cpus -= int(round(self.probe.busy_cpus([w.popen.pid for w in live])))
        n = min(by_memory, self.max_workers - len(live), cpus - len(live), self._useful(pool))
        if n > 0:
            self._spawn(pool, n)
            self._record('spawn', len(live), len(live) + n, f'room for {by_memory} more after reserving {_gb(reserve)} for '
                                                            f'the live workers')
        elif by_memory <= 0 and len(live) == 1 and self._useful(pool) > 0 and not self._warned_one:
            self._warned_one = True
            self._warn(f'one worker needs about {_gb(self.estimate)} and leaves no room for a second '
                       f'(available {_gb(self.available)}, margin {_gb(self.margin)}): running with one worker')

    def summary(self):
        """The manager's settings, measurements and decisions, for the pool statistics."""
        return {
            'max_workers': self.max_workers,
            'initial_workers': self.initial_workers,
            'margin_bytes': self.margin,
            'estimate_bytes': self.estimate,
            'worker_peak_bytes': {str(k): v for k, v in sorted(self.peaks.items())},
            'max_live_workers': self.max_live,
            'samples': self.samples,
            'decisions': [dataclasses.asdict(d) | {'text': d.text()} for d in self.decisions],
            'warnings': list(self.warnings),
        }
