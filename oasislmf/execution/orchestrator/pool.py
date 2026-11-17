"""Fused pool executor: W worker processes, each running the whole chain in-process, pulling chunks.

Workers are started with ``subprocess`` (never ``fork``), so the executor runs the same way on Windows. Each one
connects back over ``multiprocessing.connection`` (TCP on the loopback interface, with an authentication key),
builds one ``Chain`` and asks for chunks; see ``worker.py`` for the protocol. Because workers pull, the fastest
ones take the most chunks.

Elasticity is a mechanism of ``WorkerPool``: ``spawn()`` adds a worker at any time, it connects and starts
pulling; ``retire()`` answers the next ``ready`` of a worker with ``stop``. ``run_chunks`` uses the simplest policy,
W = min(n_workers, K) from the start; a ``policy`` callable may be passed to change the pool during the run.

Output jobs (``output_jobs``: the katpy/aalpy/lecpy runs of the plan, see ``outputs.py``) are handed out on the same
pool once every chunk is done: a worker whose ``ready`` arrives while the last chunks are still running waits for
them, then gets a ``job``. The jobs are independent (``outputs.check_independent``), so any free worker takes the
next one.

Any failure stops the run: a worker that reports an exception, exits or closes its connection before it was told
to stop. The remaining workers are terminated with their process trees and ``WorkerError`` is raised, naming the
stage, chunk and event when known, or the output job (tool and output files), with the tail of the worker's stderr.

With ``stderr_guard`` a worker that writes anything but Python warnings to its stderr also fails the run, as bash's
stderr guard (``oasis_exec_monitor.sh`` and the chunk script's last check) fails a script whose ``stderror.err`` has
such lines: the workers' stderr files are checked every ``STDERR_POLL_INTERVAL`` seconds and once more at the end.
"""
import collections
import dataclasses
import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Client, Listener, wait

import psutil

import oasislmf

from .chain import spec_for_taps
from .outputs import as_job, check_independent, job_label, job_outputs, prepare_output_dirs

logger = logging.getLogger(__name__)

WORKER_MODULE = 'oasislmf.execution.orchestrator.worker'
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(oasislmf.__file__)))
POLL_INTERVAL = 0.1
KILL_GRACE = 5.0
STDERR_TAIL_LINES = 40
STOP_TIMEOUT = 30.0
STDERR_POLL_INTERVAL = 2.0
STDERR_IGNORED = re.compile(r'(^\s*$|.+:[0-9]+: [A-Za-z]+Warning:|^\s+warnings\.warn)')


def stderr_errors(text):
    """The lines of ``text`` bash's stderr guard counts as errors.

    Every line but blank ones, Python warning headers (``<file>:<line>: <Category>Warning:``) and indented
    ``warnings.warn`` source lines, the pattern ``oasis_exec_monitor.sh`` and the chunk script give ``grep -v``.
    """
    return [line for line in text.splitlines() if not STDERR_IGNORED.search(line)]


class WorkerError(RuntimeError):
    """A worker failed; the run was stopped.

    Attributes:
        worker (int): worker number.
        stage (str | None): stage that raised (``'gul'``, ``'il (fm step 1)'``, ``'setup'``...), when reported.
        chunk (int | None): chunk the worker was running.
        event (int | None): event id the worker was processing, when reported.
        returncode (int | None): exit status of the worker, None if it was still running.
        stderr_tail (str): last lines of the worker's stderr.
        job (int | None): index of the output job the worker was running.
        job_label (str | None): that job's tool and output files (``outputs.job_label``).
    """

    def __init__(self, message, worker=None, stage=None, chunk=None, event=None, returncode=None, stderr_tail='', job=None,
                 job_label=None):
        super().__init__(message)
        self.worker = worker
        self.stage = stage
        self.chunk = chunk
        self.event = event
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        self.job = job
        self.job_label = job_label


def _descendants(pids):
    """Every process descending from ``pids``, found from one snapshot of the process table."""
    children = {}
    for proc in psutil.process_iter(['ppid']):
        children.setdefault(proc.info['ppid'], []).append(proc)
    found, pending = [], list(pids)
    while pending:
        for child in children.get(pending.pop(), []):
            found.append(child)
            pending.append(child.pid)
    return found


def _terminate(popens, grace=KILL_GRACE):
    """Terminate every process of ``popens`` still running, with its descendants.

    Politely first; whatever is still alive after ``grace`` seconds is killed. Direct children are reaped through
    their ``Popen`` so their return code stays consistent.
    """
    direct = [popen for popen in popens if popen.poll() is None]
    if not direct:
        return
    descendants = _descendants([popen.pid for popen in direct])
    for popen in direct:
        try:
            popen.terminate()
        except OSError:
            pass
    for proc in descendants:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass

    deadline = time.monotonic() + grace
    _gone, alive = psutil.wait_procs(descendants, timeout=grace)
    for popen in direct:
        try:
            popen.wait(timeout=max(deadline - time.monotonic(), 0))
        except subprocess.TimeoutExpired:
            popen.kill()
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=grace)
    for popen in direct:
        try:
            popen.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            logger.error('worker process %s did not exit after kill', popen.pid)


def _tail(path, n_lines=STDERR_TAIL_LINES):
    """Last ``n_lines`` lines of a text file, '' if it cannot be read."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(f.tell() - 64 * 1024, 0))
            text = f.read().decode(errors='replace')
    except OSError:
        return ''
    return '\n'.join(text.splitlines()[-n_lines:])


@dataclasses.dataclass
class Worker:
    """Orchestrator-side state of one worker process."""
    number: int
    popen: subprocess.Popen
    log_path: str
    err_path: str
    t_spawn: float
    conn: object = None
    pid: int = None
    setup_seconds: float = None
    chain_seconds: float = None
    chunk: int = None
    chunks: int = 0
    busy_seconds: float = 0.0
    job: int = None
    jobs: int = 0
    job_seconds: float = 0.0
    waiting: bool = False
    retire: bool = False
    stopped: bool = False
    t_stop: float = None
    killed: bool = False
    log_file: object = None
    err_file: object = None
    jobs_only: bool = False


class WorkerPool:
    """The worker processes of one run and the chunk queue they pull from.

    Usage::

        with WorkerPool(spec, chunks, out_dir, taps) as pool:
            for _ in range(n_workers):
                pool.spawn()
            stats = pool.run()
    """

    def __init__(self, spec, chunks, out_dir, taps, factories=None, policy=None, progress=None, python=None, env=None,
                 kill_grace=KILL_GRACE, output_jobs=None, job_workers=0, chunk_numbers=None, log_dir=None, progress_interval=None,
                 stderr_guard=False):
        """Open the listener; no worker is started yet.

        Args:
            spec (ChainSpec): the chain, with an absolute ``run_dir``.
            chunks (Sequence[array-like]): event ids of each chunk; chunk k (1-based) is ``chunks[k - 1]``.
            out_dir (str): absolute output directory.
            taps (list[str]): taps to write.
            factories (dict, optional): ``{'gul'|'fm'|'summary'|'elt'|'plt': 'package.module:attribute'}`` replacing the real stages in
                the workers. Only import strings can be sent to a spawned process.
            policy (callable, optional): ``policy(pool)`` called after every message handled, to ``spawn`` or
                ``retire`` workers.
            progress (callable, optional): ``progress(events_done, events_total)`` called after every chunk, and on every
                ``progress`` message of a worker when ``progress_interval`` is set.
            python (str, optional): interpreter for the workers, ``sys.executable`` by default.
            env (dict, optional): environment of the workers, ``os.environ`` by default.
            kill_grace (float): seconds a terminated worker has before it is killed.
            output_jobs (Sequence[OutputJob | dict], optional): run once every chunk is done; they must be
                independent (``outputs.check_independent``).
            job_workers (int): when the output jobs start, job-only workers (``spawn(jobs_only=True)``) are added
                until this many workers are active, but never more than there are jobs.
            chunk_numbers (Sequence[int], optional): the 1-based numbers of the chunks to run, all by default. The others
                are not run and write nothing (a distributed run hands each chunk to a different machine).
            log_dir (str, optional): directory the workers' stdout and stderr are written to, as
                ``orchestrator_worker_<n>.log`` and ``orchestrator_worker_<n>.err``, and kept; by default a temporary
                directory removed by ``close``.
            progress_interval (float, optional): workers report the events done in their current chunk at most every
                this many seconds, and ``progress`` is called with them; by default they report whole chunks only.
            stderr_guard (bool): fail the run when a worker writes to its stderr anything ``stderr_errors`` keeps.
        """
        if factories is not None:
            not_strings = sorted(key for key, value in factories.items() if not isinstance(value, str))
            if not_strings:
                raise TypeError(f"pool factories must be 'package.module:attribute' strings, got objects for {not_strings}")
        self.spec = spec
        self.chunks = list(chunks)
        self.out_dir = out_dir
        self.taps = list(taps)
        self.factories = dict(factories) if factories is not None else None
        self.policy = policy
        self.progress = progress
        self.python = python or sys.executable
        self.env = env
        self.kill_grace = kill_grace
        self.job_workers = job_workers
        self._job_phase_started = False

        self.output_jobs = [as_job(job) for job in output_jobs or []]
        check_independent(self.output_jobs)
        self.chunk_numbers = list(range(1, len(self.chunks) + 1)) if chunk_numbers is None else [int(k) for k in chunk_numbers]
        unknown = [k for k in self.chunk_numbers if not 1 <= k <= len(self.chunks)]
        if unknown or len(set(self.chunk_numbers)) != len(self.chunk_numbers):
            raise ValueError(f'chunk_numbers must be distinct chunk numbers in 1..{len(self.chunks)}, got {self.chunk_numbers}')
        self.progress_interval = progress_interval
        self.stderr_guard = stderr_guard
        self._t_stderr_check = time.perf_counter()
        self.pending = collections.deque(self.chunk_numbers)
        self.chunk_stats = {}
        self.pending_jobs = collections.deque(range(len(self.output_jobs)))
        self.job_stats = {}
        self.t_jobs_start = None
        self.t_jobs_end = None
        self.events_total = sum(len(self.chunks[k - 1]) for k in self.chunk_numbers)
        self.events_done = 0
        self._events_reported = {}
        self.workers = []
        self._authkey = os.urandom(32)
        self._listener = Listener(('127.0.0.1', 0), family='AF_INET', authkey=self._authkey)
        self._accepted = queue.Queue()
        self._closing = False
        self._accept_thread = threading.Thread(target=self._accept_loop, name='pool-accept', daemon=True)
        self._accept_thread.start()
        self._tmp_log_dir = None
        if log_dir is None:
            self._tmp_log_dir = tempfile.TemporaryDirectory(prefix='oasis-pool-')
            log_dir = self._tmp_log_dir.name
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

    def _accept_loop(self):
        """Accept worker connections until the pool closes; a failed handshake is ignored."""
        while not self._closing:
            try:
                conn = self._listener.accept()
            except OSError:
                if self._closing:
                    return
                continue
            except Exception:
                continue
            if self._closing:
                conn.close()
                return
            self._accepted.put(conn)

    @property
    def chunks_done(self):
        """True when every chunk has been reported done."""
        return len(self.chunk_stats) == len(self.chunk_numbers)

    @property
    def remaining(self):
        """Number of chunks not yet handed out."""
        return len(self.pending)

    @property
    def active(self):
        """Workers neither stopped nor retiring."""
        return [w for w in self.workers if not w.stopped and not w.retire]

    def _worker_env(self):
        """Environment of the workers: ``env`` or ``os.environ``, with this oasislmf package first on PYTHONPATH."""
        env = dict(os.environ if self.env is None else self.env)
        env['PYTHONPATH'] = os.pathsep.join(p for p in (PACKAGE_ROOT, env.get('PYTHONPATH')) if p)
        return env

    def spawn(self, jobs_only=False):
        """Start one more worker; it connects and starts pulling chunks when it is ready.

        Args:
            jobs_only (bool): the worker builds no chain and only takes output jobs.

        Returns:
            Worker: the new worker.
        """
        number = len(self.workers)
        log_path = os.path.join(self.log_dir, f'orchestrator_worker_{number}.log')
        err_path = os.path.join(self.log_dir, f'orchestrator_worker_{number}.err')
        log_file = open(log_path, 'wb')
        try:
            err_file = open(err_path, 'wb')
        except BaseException:
            log_file.close()
            raise
        config = {
            'address': list(self._listener.address), 'authkey': self._authkey.hex(), 'worker': number,
            'spec': self.spec.to_dict(), 'factories': self.factories, 'out_dir': self.out_dir, 'taps': self.taps,
            'sys_path': [p for p in sys.path if p], 'jobs_only': jobs_only, 'progress_interval': self.progress_interval,
        }
        try:
            popen = subprocess.Popen([self.python, '-m', WORKER_MODULE], stdin=subprocess.PIPE, stdout=log_file, stderr=err_file,
                                     cwd=self.spec.run_dir, env=self._worker_env())
        except BaseException:
            log_file.close()
            err_file.close()
            raise
        worker = Worker(number=number, popen=popen, log_path=log_path, err_path=err_path, t_spawn=time.perf_counter(),
                        log_file=log_file, err_file=err_file, jobs_only=jobs_only)
        self.workers.append(worker)
        try:
            popen.stdin.write(json.dumps(config).encode())
            popen.stdin.close()
        except OSError:
            pass
        return worker

    def retire(self, worker=None):
        """Retire a worker: its next ``ready`` is answered with ``stop``.

        Args:
            worker (Worker, optional): the worker; the most recently started active one by default.

        Returns:
            Worker | None: the worker retired, None if none is active.
        """
        if worker is None:
            active = self.active
            if not active:
                return None
            worker = active[-1]
        worker.retire = True
        return worker

    def _fail(self, worker, reason, message=None):
        """Build the ``WorkerError`` for a failed worker, after terminating every worker."""
        self.terminate()
        message = message or {}
        stage, chunk, event = message.get('stage'), message.get('chunk', worker.chunk), message.get('event')
        if chunk is None:
            chunk = worker.chunk
        job = message.get('job', worker.job)
        if job is None:
            job = worker.job
        label = job_label(self.output_jobs[job]) if job is not None else None
        returncode = worker.popen.poll()
        stderr_tail = _tail(worker.err_path)
        where = [f'stage {stage}' if stage else None, f'chunk {chunk}' if chunk is not None else None,
                 f'event {event}' if event is not None else None, f'output job {label}' if label else None]
        where = ', '.join(w for w in where if w)
        text = f'pool worker {worker.number} (pid {worker.popen.pid}) {reason}'
        if where:
            text += f' [{where}]'
        if message.get('error'):
            text += f": {message['error']}"
        if returncode is not None:
            text += f' (exit code {returncode})'
        if message.get('traceback'):
            text += '\nworker traceback:\n' + message['traceback'].rstrip()
        if stderr_tail:
            text += '\nworker stderr (last lines):\n' + stderr_tail
        return WorkerError(text, worker=worker.number, stage=stage, chunk=chunk, event=event, returncode=returncode,
                           stderr_tail=stderr_tail, job=job, job_label=label)

    def _register(self, conn):
        """Attach a new connection to its worker from the ``hello`` it sends first."""
        if not conn.poll(self.kill_grace):
            conn.close()
            return
        try:
            hello = conn.recv()
        except (EOFError, OSError):
            conn.close()
            return
        if not isinstance(hello, dict) or hello.get('type') != 'hello':
            conn.close()
            return
        number = hello.get('worker')
        if not isinstance(number, int) or not 0 <= number < len(self.workers) or self.workers[number].conn:
            conn.close()
            return
        worker = self.workers[number]
        worker.conn = conn
        worker.pid = hello.get('pid')

    def _dispatch(self, worker):
        """Answer a ``ready``: the next chunk; once every chunk is done the next output job; else ``stop``.

        A worker asking while chunks are still running and jobs are left waits (``waiting``) and is answered by
        ``_dispatch_waiting`` when the last chunk is done.
        """
        worker.waiting = False
        if not worker.retire and self.pending and not worker.jobs_only:
            k = self.pending.popleft()
            worker.chunk = k
            worker.conn.send({'type': 'chunk', 'chunk': k, 'events': self.chunks[k - 1]})
            return
        if not worker.retire and self.pending_jobs:
            if not self.chunks_done:
                worker.waiting = True
                return
            i = self.pending_jobs.popleft()
            if self.t_jobs_start is None:
                self.t_jobs_start = time.perf_counter()
            worker.job = i
            worker.conn.send({'type': 'job', 'job': i, **self.output_jobs[i].to_dict()})
            return
        worker.conn.send({'type': 'stop'})
        worker.stopped = True
        worker.t_stop = time.perf_counter()

    def _start_job_phase(self):
        """Once every chunk is done, add job-only workers up to ``job_workers`` (at most one per job); once only."""
        if self._job_phase_started or not self.chunks_done or not self.pending_jobs:
            return
        self._job_phase_started = True
        target = min(self.job_workers, len(self.output_jobs))
        for _ in range(target - len(self.active)):
            self.spawn(jobs_only=True)

    def _dispatch_waiting(self):
        """Answer the workers waiting for the last chunks, once those are done."""
        if not self.chunks_done:
            return
        self._start_job_phase()
        for worker in self.workers:
            if worker.waiting and not worker.stopped:
                self._dispatch(worker)

    def _handle(self, worker, message):
        """Handle one message of a worker; raise ``WorkerError`` when it reports a failure."""
        kind = message.get('type')
        if kind == 'ready':
            if worker.setup_seconds is None:
                worker.setup_seconds = time.perf_counter() - worker.t_spawn
                worker.chain_seconds = message.get('chain_seconds')
            self._dispatch(worker)
        elif kind == 'done' and 'job' in message:
            i = message['job']
            self.job_stats[i] = {'job': i, 'tool': self.output_jobs[i].tool, 'outputs': job_outputs(self.output_jobs[i]),
                                 'seconds': message['seconds'], 'worker': worker.number}
            worker.jobs += 1
            worker.job_seconds += message['seconds']
            worker.job = None
            if len(self.job_stats) == len(self.output_jobs):
                self.t_jobs_end = time.perf_counter()
        elif kind == 'done':
            k = message['chunk']
            self.chunk_stats[k] = {'chunk': k, 'events': message['events'], 'seconds': message['seconds'], 'worker': worker.number}
            worker.chunks += 1
            worker.busy_seconds += message['seconds']
            worker.chunk = None
            self.events_done += message['events'] - self._events_reported.pop(k, 0)
            if self.progress is not None:
                self.progress(self.events_done, self.events_total)
            self._dispatch_waiting()
        elif kind == 'progress':
            k = message['chunk']
            self._events_reported[k] = self._events_reported.get(k, 0) + message['events']
            self.events_done += message['events']
            if self.progress is not None:
                self.progress(self.events_done, self.events_total)
        elif kind == 'error':
            raise self._fail(worker, 'failed', message)
        else:
            raise self._fail(worker, f'sent an unexpected message {message!r}')

    def _check_exits(self):
        """Raise ``WorkerError`` for a worker that exited before it was told to stop, or with a non-zero status."""
        for worker in self.workers:
            returncode = worker.popen.poll()
            if returncode is None:
                if worker.stopped and time.perf_counter() - worker.t_stop > STOP_TIMEOUT:
                    logger.warning('pool worker %s did not exit %.0f s after stop, terminating it', worker.number, STOP_TIMEOUT)
                    worker.killed = True
                    _terminate([worker.popen], self.kill_grace)
                continue
            if worker.killed:
                continue
            if returncode != 0:
                self._drain(worker)
                raise self._fail(worker, 'exited')
            if not worker.stopped:
                self._drain(worker)
                raise self._fail(worker, 'exited before it was told to stop')

    def _check_stderr(self, force=False):
        """With ``stderr_guard``, raise ``WorkerError`` for the first worker whose stderr has an error line.

        Checked at most every ``STDERR_POLL_INTERVAL`` seconds unless ``force``.
        """
        if not self.stderr_guard:
            return
        now = time.perf_counter()
        if not force and now - self._t_stderr_check < STDERR_POLL_INTERVAL:
            return
        self._t_stderr_check = now
        for worker in self.workers:
            try:
                if os.path.getsize(worker.err_path) == 0:
                    continue
                with open(worker.err_path, 'rb') as f:
                    errors = stderr_errors(f.read().decode(errors='replace'))
            except OSError:
                continue
            if errors:
                raise self._fail(worker, 'wrote to stderr', {'error': errors[0]})

    def _drain(self, worker):
        """Handle what a worker sent before it exited, so that a reported error is raised with its details."""
        if worker.conn is None:
            while True:
                try:
                    self._register(self._accepted.get_nowait())
                except queue.Empty:
                    break
        if worker.conn is None:
            return
        try:
            while worker.conn.poll():
                message = worker.conn.recv()
                if message.get('type') == 'error':
                    self._handle(worker, message)
        except (EOFError, OSError):
            pass

    def _finished(self):
        """True when every chunk is done and every worker has been stopped and has exited."""
        return (not self.pending and self.chunks_done and not self.pending_jobs and len(self.job_stats) == len(self.output_jobs)
                and all(w.stopped and w.popen.poll() is not None for w in self.workers))

    def run(self):
        """Hand out every chunk and wait for the workers to finish.

        Returns:
            dict: statistics, see ``run_chunks``.

        Raises:
            WorkerError: a worker failed; every worker has been terminated.
        """
        try:
            while not self._finished():
                self._start_job_phase()
                while True:
                    try:
                        self._register(self._accepted.get_nowait())
                    except queue.Empty:
                        break
                conns = {w.conn: w for w in self.workers if w.conn is not None and not w.conn.closed}
                for conn in wait(list(conns), timeout=POLL_INTERVAL) if conns else []:
                    worker = conns[conn]
                    try:
                        message = conn.recv()
                    except (EOFError, OSError):
                        conn.close()
                        if worker.stopped:
                            continue
                        try:
                            worker.popen.wait(timeout=self.kill_grace)
                        except subprocess.TimeoutExpired:
                            pass
                        raise self._fail(worker, 'closed its connection')
                    self._handle(worker, message)
                    if self.policy is not None:
                        self.policy(self)
                if not conns:
                    time.sleep(POLL_INTERVAL)
                self._check_exits()
                self._check_stderr()
                work_left = (self.pending or not self.chunks_done or self.pending_jobs
                             or len(self.job_stats) < len(self.output_jobs))
                if not any(not w.stopped for w in self.workers) and work_left:
                    if self.policy is not None:
                        self.policy(self)
                    if not any(not w.stopped for w in self.workers):
                        self.terminate()
                        raise WorkerError(f'every worker was stopped with {len(self.pending)} chunks left and '
                                          f'{len(self.pending_jobs)} output jobs left')
            self._check_stderr(force=True)
        except BaseException:
            self.terminate()
            raise
        return self.stats()

    def stats(self):
        """Statistics of the run so far, see ``run_chunks``."""
        setups = [w.setup_seconds for w in self.workers if w.setup_seconds is not None and not w.jobs_only]
        return {
            'executor': 'pool',
            'n_workers': len(self.workers),
            'setup_seconds': max(setups) if setups else 0.0,
            'events': sum(c['events'] for c in self.chunk_stats.values()),
            'chunks': [self.chunk_stats[k] for k in sorted(self.chunk_stats)],
            'jobs': [self.job_stats[i] for i in sorted(self.job_stats)],
            'output_seconds': (self.t_jobs_end - self.t_jobs_start) if self.t_jobs_end is not None else 0.0,
            'workers': [{'worker': w.number, 'pid': w.popen.pid, 'chunks': w.chunks, 'setup_seconds': w.setup_seconds,
                         'chain_seconds': w.chain_seconds, 'busy_seconds': w.busy_seconds, 'jobs': w.jobs,
                         'job_seconds': w.job_seconds, 'jobs_only': w.jobs_only} for w in self.workers],
        }

    def terminate(self):
        """Terminate every worker still running, with its descendants."""
        _terminate([w.popen for w in self.workers], self.kill_grace)

    def close(self):
        """Stop the accept thread, close the connections, terminate the workers left and remove their logs."""
        self.terminate()
        self._closing = True
        try:
            Client(self._listener.address, family='AF_INET', authkey=self._authkey).close()
        except Exception:
            pass
        self._accept_thread.join(timeout=self.kill_grace)
        self._listener.close()
        while True:
            try:
                self._accepted.get_nowait().close()
            except queue.Empty:
                break
        for worker in self.workers:
            if worker.conn is not None:
                worker.conn.close()
            if worker.popen.stdin is not None and not worker.popen.stdin.closed:
                worker.popen.stdin.close()
            worker.log_file.close()
            worker.err_file.close()
        if self._tmp_log_dir is not None:
            self._tmp_log_dir.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def run_chunks(spec, chunks, out_dir, taps, n_workers=1, **options):
    """Run the chunks on a pool of worker processes and write ``<out_dir>/<tap>_P<k>.bin`` for each.

    Starts W = min(n_workers, K) workers, K being the number of chunks to run (``chunk_numbers``, all by default).
    Each builds one ``Chain`` with the run directory as working directory and runs every chunk between
    ``Chain.start_chunk(k)`` and ``Chain.finish_chunk()``. Once every chunk is done the
    workers close their chain and run the ``output_jobs``, each tool's ``main`` called in-process with the run
    directory as working directory (``outputs.run_job``). When there are more jobs than chunks, job-only workers,
    which build no chain, are added then, up to min(n_workers, number of jobs) workers in all.

    Args:
        spec (ChainSpec): the chain to run.
        chunks (Sequence[array-like]): event ids of each chunk; chunk k (1-based) is ``chunks[k - 1]``.
        out_dir (str | os.PathLike): output directory, relative to the current working directory.
        taps (Iterable[str]): taps to write.
        n_workers (int): maximum number of workers.
        **options: ``factories`` (``{'gul'|'fm'|'summary'|'elt'|'plt': 'package.module:attribute'}``), ``policy``, ``progress``,
            ``python``, ``env``, ``kill_grace``, ``output_jobs`` (Sequence[OutputJob | dict]), ``chunk_numbers``, ``log_dir``,
            ``progress_interval`` and ``stderr_guard``, see ``WorkerPool``; other options are ignored.

    Returns:
        dict: statistics: ``executor`` ('pool'), ``n_workers``, ``wall_seconds``, ``setup_seconds`` (the longest time
        from starting a chain-building worker to its first ``ready``), ``events``, ``chunks`` (one ``{'chunk', 'events',
        'seconds', 'worker'}`` per chunk), ``jobs`` (one ``{'job', 'tool', 'outputs', 'seconds', 'worker'}`` per output
        job, by index), ``output_seconds`` (from handing out the first job to the last one done) and ``workers`` (one
        ``{'worker', 'pid', 'chunks', 'setup_seconds', 'chain_seconds', 'busy_seconds', 'jobs', 'job_seconds',
        'jobs_only'}`` per worker; ``busy_seconds`` counts chunks only).

    Raises:
        WorkerError: a worker failed; every worker has been terminated.
    """
    t_start = time.perf_counter()
    taps = list(taps)
    out_dir = os.path.abspath(out_dir)
    spec = dataclasses.replace(spec_for_taps(spec, taps), run_dir=os.path.abspath(spec.run_dir))
    chunks = list(chunks)
    if n_workers <= 0:
        raise ValueError(f'n_workers must be positive, got {n_workers}')
    os.makedirs(out_dir, exist_ok=True)
    output_jobs = [as_job(job) for job in options.get('output_jobs') or []]
    chunk_numbers = options.get('chunk_numbers')
    n = min(n_workers, len(chunks) if chunk_numbers is None else len(chunk_numbers))
    prepare_output_dirs(output_jobs, spec.run_dir)
    pool_options = {key: options[key] for key in ('factories', 'policy', 'progress', 'python', 'env', 'kill_grace', 'chunk_numbers',
                                                  'log_dir', 'progress_interval', 'stderr_guard') if key in options}
    pool_options['output_jobs'] = output_jobs
    pool_options['job_workers'] = min(n_workers, max(n, len(output_jobs)))
    with WorkerPool(spec, chunks, out_dir, taps, **pool_options) as pool:
        for _ in range(n):
            pool.spawn()
        stats = pool.run()
    stats['wall_seconds'] = time.perf_counter() - t_start
    return stats
