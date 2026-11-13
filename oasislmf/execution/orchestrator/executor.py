"""Run a :class:`~.build.Run` in the current working directory, without a shell.

This replaces what the generated bash script does around the kernel processes:

* ``bash_wrapper``'s header: the log directory is (re)created, every child gets
  ``OASIS_PYTOOLS_LOG_DIR`` and an empty ``stderror.err`` is created;
* ``set -e``/``pipefail``, the ``exit_handler`` trap and ``oasis_exec_monitor.sh``: the processes
  of each stage are supervised while they run, and the first failure (a non-zero exit, or with the
  stderr guard an error line in ``stderror.err``) terminates every process of the run, including
  descendants, before a :class:`subprocess.CalledProcessError` is raised;
* ``bash_wrapper``'s footer for a chunk: :class:`~.build.ChunkCheck`.

A JSON manifest of what is about to run is written first; it replaces the ``.sh`` file as the
record of the run. The relative ``log_dir`` is passed to the children as given, so the pytools
resolve it against the same working directory, exactly as with the bash script.

Nothing here is POSIX-only: process trees are handled with ``psutil`` and streams by the
transport returned by :func:`~.transport.get_transport`.
"""

import contextlib
import glob
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time

import psutil

from .build import IGNORABLE_STDERR_LINE
from .model import STDERR_LOG, Stream, StreamPath
from .transport import get_transport

logger = logging.getLogger(__name__)

STDERR_POLL_INTERVAL = 0.5
"""Seconds between two reads of ``stderror.err`` (``oasis_exec_monitor.sh`` polls every 2 s)."""

EXIT_POLL_INTERVAL = 0.05
"""Seconds between two checks of the processes' exit status."""

KILL_GRACE = 3.0
"""Seconds a process gets to exit after SIGTERM (``TerminateProcess`` on Windows) before it is killed."""

EXECUTABLE_NOT_FOUND = 127
ORCHESTRATOR_CMD = 'oasislmf-orchestrator'


class _RunFailed(Exception):
    """Internal: the run must stop; carries what the CalledProcessError will report."""

    def __init__(self, message, returncode, cmd):
        super().__init__(message)
        self.message = message
        self.returncode = returncode
        self.cmd = cmd


def render_endpoint(endpoint):
    """Readable form of a stdin/stdout endpoint for the manifest and messages."""
    if isinstance(endpoint, Stream):
        return f'<stream:{endpoint.name}>'
    return endpoint


def render_argv(argv):
    """``argv`` with each :class:`~.model.StreamPath` shown as ``<stream:NAME>``."""
    return [f'<stream:{arg.stream.name}>' if isinstance(arg, StreamPath) else arg for arg in argv]


def _process_manifest(process):
    return {
        'name': process.name,
        'argv': render_argv(process.argv),
        'stdin': render_endpoint(process.stdin),
        'stdout': render_endpoint(process.stdout),
        'stderr_log': process.stderr_log,
    }


def manifest(run, log_dir):
    """Return the JSON-serialisable description of ``run`` that :func:`execute` writes.

    Args:
        run: the :class:`~.build.Run`.
        log_dir (str): log directory of the run, as passed to :func:`execute`.
    """
    chunk_check = run.chunk_check
    return {
        'log_dir': log_dir,
        'env': dict(run.env),
        'background': [_process_manifest(process) for process in run.background],
        'stages': [{
            'name': stage.name,
            'remove_paths': list(stage.remove_paths),
            'clean_output_dir': stage.clean_output_dir,
            'make_dirs': list(stage.make_dirs),
            'cleanup_paths': list(stage.cleanup_paths),
            'processes': [_process_manifest(process) for process in stage.processes],
        } for stage in run.stages],
        'chunk_check': None if chunk_check is None else {
            'stderr_log': chunk_check.stderr_log,
            'ignorable_stderr_line': chunk_check.ignorable_stderr_line,
            'non_empty_glob': chunk_check.non_empty_glob,
        },
    }


def format_manifest(description):
    """Return ``description`` (from :func:`manifest`) as JSON text with one process per line.

    The structure is indented, but each process is kept on a single line so the file reads like
    a list of commands; with every argument on its own line a run of a few hundred processes
    takes thousands of lines.
    """
    lines = {}

    def one_line(process):
        token = f'@process-{len(lines)}@'
        lines[json.dumps(token)] = json.dumps(process)
        return token

    outline = dict(description)
    outline['background'] = [one_line(process) for process in description['background']]
    outline['stages'] = [{**stage, 'processes': [one_line(process) for process in stage['processes']]}
                         for stage in description['stages']]
    text = json.dumps(outline, indent=2)
    return re.sub(r'"@process-\d+@"', lambda match: lines[match.group(0)], text) + '\n'


def write_manifest(run, log_dir, path):
    """Write :func:`manifest` of ``run`` to ``path``, formatted by :func:`format_manifest`."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(format_manifest(manifest(run, log_dir)))


def connected_components(stage):
    """Group the processes of ``stage`` that are connected, directly or not, by streams.

    The transport holds a descriptor for every stream that has one end spawned and the other not
    yet. Spawning all readers of a large stage before any writer would hold one per stream at
    once (hundreds with many partitions), so the executor spawns one group at a time instead:
    each kernel partition is its own group, and at most one group's streams are half connected.

    Returns:
        list of lists of processes: groups ordered by their first process in ``stage.processes``,
        each keeping the listed order.
    """
    parent = list(range(len(stage.processes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_user = {}
    for i, process in enumerate(stage.processes):
        for stream in process.streams_read() + process.streams_written():
            if stream.name in first_user:
                a, b = find(i), find(first_user[stream.name])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first_user[stream.name] = i

    groups = {}
    for i, process in enumerate(stage.processes):
        groups.setdefault(find(i), []).append(process)
    return [groups[root] for root in sorted(groups)]


def resolve_executable(argv, env):
    """Return ``argv`` with ``argv[0]`` resolved against ``env['PATH']`` when it is found there.

    On POSIX ``Popen`` already searches the child's ``PATH``. On Windows ``CreateProcess`` searches
    the current directory first and then the *parent's* ``PATH``, ignoring ``env``; resolving
    here with :func:`shutil.which` (which applies ``PATHEXT``, so ``fmpy`` finds ``fmpy.exe``)
    gives the same lookup on every OS. An executable that is not found is left as given so that
    ``Popen`` raises the usual :class:`FileNotFoundError`.
    """
    resolved = shutil.which(argv[0], path=env.get('PATH', os.defpath))
    if resolved is None:
        return argv
    return [resolved, *argv[1:]]


def _remove_glob(pattern):
    """``rm -R -f <pattern>``."""
    for path in glob.glob(pattern):
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def _clean_output_dir(directory):
    """``find <dir> -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -f {} +``."""
    for root, _dirs, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            if 'summary-info' in name or name.endswith('.json'):
                continue
            if os.path.isfile(path) and not os.path.islink(path):
                os.remove(path)


def prepare_log_dir(log_dir):
    """Create ``log_dir``, empty it and create an empty ``stderror.err`` (``bash_wrapper`` header)."""
    os.makedirs(log_dir, exist_ok=True)
    for entry in os.listdir(log_dir):
        path = os.path.join(log_dir, entry)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    open(os.path.join(log_dir, STDERR_LOG), 'ab').close()


WARNING_LINE = re.compile(r'.+:[0-9]+: [A-Za-z]+Warning:')


class _ErrorLines:
    """Decides, line by line, which lines of ``stderror.err`` are errors.

    A line is an error unless it matches ``ignorable`` (blank lines and Python warnings, the
    ``oasis_exec_monitor.sh`` grep). Python prints the source line that raised a warning,
    indented, on the line after it; that line belongs to the warning and is not an error either,
    whatever its text.
    """

    def __init__(self, ignorable):
        self.ignorable = re.compile(ignorable)
        self.after_warning = False

    def is_error(self, line):
        if self.after_warning and line[:1] in (' ', '\t'):
            self.after_warning = False
            return False
        self.after_warning = bool(WARNING_LINE.search(line))
        return not self.ignorable.search(line)


class _StderrWatch:
    """Incremental reader of ``stderror.err`` reporting its first error line (see :class:`_ErrorLines`)."""

    def __init__(self, path, ignorable=IGNORABLE_STDERR_LINE):
        self.path = path
        self.lines = _ErrorLines(ignorable)
        self.offset = 0
        self.partial = b''

    def first_error(self, final=False):
        """Return the first error line written since the last call, or None.

        Only complete lines are judged, unless ``final`` is set (all writers have exited).
        """
        try:
            with open(self.path, 'rb') as f:
                f.seek(self.offset)
                data = f.read()
        except FileNotFoundError:
            return None
        self.offset += len(data)
        data = self.partial + data
        lines = data.split(b'\n')
        self.partial = lines.pop()
        if final and self.partial:
            lines.append(self.partial)
            self.partial = b''
        for raw in lines:
            line = raw.decode('utf-8', errors='replace').rstrip('\r')
            if self.lines.is_error(line):
                return line
        return None


def _stderr_errors(path, ignorable):
    """Every error line of ``path`` (see :class:`_ErrorLines`), or [] if it does not exist."""
    if not os.path.isfile(path):
        return []
    with open(path, 'rb') as f:
        lines = [line.rstrip('\r') for line in f.read().decode('utf-8', errors='replace').split('\n')]
    if lines and lines[-1] == '':
        lines.pop()
    error_lines = _ErrorLines(ignorable)
    return [line for line in lines if error_lines.is_error(line)]


def _describe(process, returncode=None):
    text = f"'{process.name}' ({subprocess.list2cmdline(render_argv(process.argv))})"
    if returncode is not None:
        text += f' exited with code {returncode}'
    return text


def _descendants(pids):
    """Every process descending from ``pids``, found from one snapshot of the process table.

    Asking each process for its children scans the whole table once per process, which delays the
    first terminate by seconds when a stage has hundreds of them.
    """
    children = {}
    for proc in psutil.process_iter(['ppid']):
        children.setdefault(proc.info['ppid'], []).append(proc)
    found, pending = [], list(pids)
    while pending:
        for child in children.get(pending.pop(), []):
            found.append(child)
            pending.append(child.pid)
    return found


def _terminate(popens, grace):
    """Terminate every process of ``popens`` still running, with its descendants.

    Politely first; whatever is still alive after ``grace`` seconds is killed. Direct children are
    reaped through their ``Popen`` so their return code stays consistent.
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
            logger.error('Process %s did not exit after SIGKILL', popen.pid)


class _Report:
    """Collects the text returned by :func:`execute`."""

    def __init__(self):
        self.lines = []
        self.start = time.monotonic()

    def add(self, line, level=logging.INFO):
        logger.log(level, line)
        self.lines.append(f'[{time.monotonic() - self.start:9.2f}s] {line}')

    def text(self):
        return '\n'.join(self.lines) + '\n'


class _Executor:
    def __init__(self, log_dir, env, stderr_guard, poll_interval, kill_grace, report):
        self.log_dir = log_dir
        self.env = env
        self.stderr_guard = stderr_guard
        self.poll_interval = poll_interval
        self.kill_grace = kill_grace
        self.report = report
        self.background = []
        self.running = []
        self.stderr_watch = _StderrWatch(os.path.join(log_dir, STDERR_LOG)) if stderr_guard else None
        self.next_stderr_check = 0.0

    def _popen(self, process, argv, **kwargs):
        argv = resolve_executable(argv, self.env)
        try:
            return subprocess.Popen(argv, env=self.env, **kwargs)
        except FileNotFoundError as e:
            raise _RunFailed(f"executable not found for process {_describe(process)}: '{argv[0]}' ({e.strerror or e})",
                             EXECUTABLE_NOT_FOUND, argv) from e

    def start_background(self, processes):
        for process in processes:
            argv = list(process.argv)
            popen = self._popen(process, argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.background.append((process, popen))
            self.report.add(f'background process {_describe(process)} started (pid {popen.pid})')

    def stop_background(self):
        if not self.background:
            return
        popens = [popen for _process, popen in self.background]
        for process, popen in self.background:
            if popen.poll() is not None:
                self.report.add(f'background process {_describe(process, popen.returncode)} before the end of the run',
                                logging.WARNING)
        _terminate(popens, self.kill_grace)
        self.report.add(f'background processes stopped ({len(popens)})')
        self.background = []

    def terminate_running(self):
        _terminate([popen for _process, popen in self.running], self.kill_grace)
        self.running = []

    def _check(self, stage, final=False):
        """Raise :class:`_RunFailed` if a process of ``stage`` failed or stderror.err has an error line.

        Returns True when every process has exited.
        """
        failed = []
        done = True
        for process, popen in self.running:
            returncode = popen.poll()
            if returncode is None:
                done = False
            elif returncode != 0:
                failed.append((process, popen))
        if failed:
            process, popen = failed[0]
            message = f"stage '{stage.name}': process {_describe(process, popen.returncode)}"
            others = [f"'{p.name}' ({o.returncode})" for p, o in failed[1:]]
            if others:
                message += f"; also exited non-zero by the same check (likely consequences): {', '.join(others)}"
            raise _RunFailed(message, popen.returncode, render_argv(process.argv))

        now = time.monotonic()
        if self.stderr_watch is not None and (final or done or now >= self.next_stderr_check):
            self.next_stderr_check = now + self.poll_interval
            line = self.stderr_watch.first_error(final=final)
            if line is not None:
                message = f"stage '{stage.name}': error detected in {self.stderr_watch.path}: {line}"
                raise _RunFailed(message, 1, [ORCHESTRATOR_CMD, stage.name])
        return done

    def _prepare_filesystem(self, stage):
        for pattern in stage.remove_paths:
            _remove_glob(pattern)
        if stage.clean_output_dir and os.path.isdir(stage.clean_output_dir):
            _clean_output_dir(stage.clean_output_dir)
        for directory in stage.make_dirs:
            os.makedirs(directory, exist_ok=True)

    def run_stage(self, stage):
        started = time.monotonic()
        self._prepare_filesystem(stage)
        if stage.processes:
            groups = connected_components(stage)
            self.report.add(f"stage '{stage.name}': starting {len(stage.processes)} processes in {len(groups)} groups")
            self._spawn(stage, groups)
            while not self._check(stage):
                time.sleep(EXIT_POLL_INTERVAL)
            self._check(stage, final=True)
            self.running = []
        for pattern in stage.cleanup_paths:
            _remove_glob(pattern)
        self.report.add(f"stage '{stage.name}': finished in {time.monotonic() - started:.2f}s")

    def _spawn(self, stage, groups):
        stderr_files = {}
        try:
            with get_transport() as transport:
                transport.open(stage)
                for group in groups:
                    for process in group:
                        stderr = None
                        if process.stderr_log:
                            if process.stderr_log not in stderr_files:
                                stderr_files[process.stderr_log] = open(os.path.join(self.log_dir, process.stderr_log), 'ab')
                            stderr = stderr_files[process.stderr_log]
                        argv = transport.resolve_argv(process)
                        popen = self._popen(process, argv, stderr=stderr, **transport.popen_kwargs(process))
                        self.running.append((process, popen))
                        transport.after_spawn(process)
                        logger.debug("stage '%s': started %s (pid %s)", stage.name, _describe(process), popen.pid)
                    self._check(stage)
        finally:
            for f in stderr_files.values():
                f.close()

    def chunk_check(self, check):
        errors = _stderr_errors(os.path.join(self.log_dir, check.stderr_log), check.ignorable_stderr_line)
        if errors:
            path = os.path.join(self.log_dir, check.stderr_log)
            raise _RunFailed(f'Error detected in {path}: {errors[0]}', 1, [ORCHESTRATOR_CMD, 'chunk-check'])
        self.report.add(' === Checking analysis output chunks === ')
        files = sorted(glob.glob(check.non_empty_glob, recursive=True), reverse=True)
        sizes = [(os.path.getsize(path), path) for path in files]
        for size, path in sizes:
            self.report.add(f'{size} {path}')
        for size, path in sizes:
            if size == 0:
                raise _RunFailed(f"Chunk output error: File '{path}' is empty", 1, [ORCHESTRATOR_CMD, 'chunk-check'])
        self.report.add('Chunk output check [OK]')


@contextlib.contextmanager
def _terminate_signal_raises():
    """Make SIGTERM raise ``SystemExit`` for the duration of a run, so that :func:`execute` cleans up.

    By default SIGTERM ends the Python process at once and its children carry on without a
    supervisor. Nothing is changed when a handler is already installed (a Celery pool worker has
    one that raises ``SystemExit`` itself) or outside the main thread, where handlers cannot be set.
    """
    if threading.current_thread() is not threading.main_thread() or signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
        yield
        return

    def handler(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def execute(run, log_dir, manifest_path=None, stderr_guard=True, poll_interval=STDERR_POLL_INTERVAL, kill_grace=KILL_GRACE):
    """Run ``run`` in the current working directory and return a text log of what happened.

    Steps: prepare ``log_dir`` (see :func:`prepare_log_dir`); write the manifest; start the
    background processes; run each stage (filesystem preparation, processes started group by group
    as returned by :func:`connected_components`, supervision until all have exited, then
    ``cleanup_paths``); run the chunk check; stop the background processes.

    A process with ``stderr_log`` set appends its stderr to ``<log_dir>/<stderr_log>``; one with
    ``stderr_log=None`` inherits this process's stderr. Every child gets ``os.environ``,
    ``run.env`` and ``OASIS_PYTOOLS_LOG_DIR=<log_dir>``.

    The run fails when a process exits non-zero, when an executable is not found, when (with
    ``stderr_guard``) ``<log_dir>/stderror.err`` contains a line that is neither blank nor a Python
    warning, or when the chunk check fails. Every process still running, background ones included,
    is then terminated with its descendants, and the same happens on any other exception
    (``KeyboardInterrupt`` included), which is re-raised. SIGTERM is turned into ``SystemExit`` while
    the run lasts, so a terminated run cleans up too (see :func:`_terminate_signal_raises`).

    Args:
        run: the :class:`~.build.Run` to execute.
        log_dir (str): log directory, normally relative (``log``, ``log/<n>``, ``log/out``).
        manifest_path (str or None): where to write the JSON manifest; None for none.
        stderr_guard (bool): watch ``stderror.err`` as ``oasis_exec_monitor.sh`` does.
        poll_interval (float): seconds between two reads of ``stderror.err``.
        kill_grace (float): seconds between terminate and kill when cleaning up.

    Returns:
        str: the run log (stages started and finished, process counts, durations, chunk check).

    Raises:
        subprocess.CalledProcessError: the run failed. ``returncode`` and ``cmd`` are those of the
            first process that failed (127 for an executable not found, 1 for a stderr or chunk
            check failure); ``output`` is the run log as bytes, ending with the reason, which names
            the stage and the process.
    """
    report = _Report()
    prepare_log_dir(log_dir)
    if manifest_path:
        write_manifest(run, log_dir, manifest_path)
        report.add(f'manifest written to {manifest_path}')

    env = os.environ.copy()
    env.update(run.env)
    env['OASIS_PYTOOLS_LOG_DIR'] = log_dir
    executor = _Executor(log_dir, env, stderr_guard, poll_interval, kill_grace, report)
    try:
        with _terminate_signal_raises():
            executor.start_background(run.background)
            for stage in run.stages:
                executor.run_stage(stage)
            if run.chunk_check is not None:
                executor.chunk_check(run.chunk_check)
    except _RunFailed as failure:
        executor.terminate_running()
        executor.stop_background()
        report.add(f'Kernel execution error: {failure.message}', logging.ERROR)
        raise subprocess.CalledProcessError(failure.returncode, failure.cmd, output=report.text().encode('utf-8')) from None
    except BaseException:
        executor.terminate_running()
        executor.stop_background()
        raise
    executor.stop_background()
    report.add(f'run completed in {time.monotonic() - report.start:.2f}s')
    return report.text()
