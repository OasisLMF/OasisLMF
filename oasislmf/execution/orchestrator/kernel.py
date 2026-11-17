"""Run a kernel step from ``bash_params`` with the orchestrator, in place of the script ``genbash`` would run.

``execution/runner.py`` calls ``run_plan`` for the three kernel steps when the ``orchestrator`` option is on and the
planner supports the run (``plan.plan_from_bash_params`` raises ``Unsupported`` otherwise):

* ``all`` (``runner.run``, the whole ``generate-losses``): every chunk, then the output jobs;
* ``analysis`` (``runner.run_analysis``, ``generate-losses-chunk``): every chunk, or with ``process_number`` only that
  one, into ``<process_number>.work/``; no output jobs;
* ``outputs`` (``runner.run_outputs``, ``generate-losses-output``): the output jobs only, over the work files the
  analysis left in ``work/``.

Each step does what its bash script does around the pipelines: it empties its log directory (``bash_wrapper``),
the analysis removes the previous output files and work files as ``create_bash_analysis`` does, and the work files
are removed after the outputs when ``remove_working_files`` is set, as ``create_bash_outputs`` does. The work files
have the names bash gives them, so a chunk run by bash and one run by the orchestrator are interchangeable.

Logs go where the script's would: the workers' stdout and stderr to ``<log_dir>/orchestrator_worker_<n>.log``, the
pytools logs of the output jobs (``OASIS_PYTOOLS_LOG_DIR``) to ``<log_dir>``, and a run log, the counterpart of the
script's stdout, to ``<log_dir>/orchestrator.log``. A failure is appended to ``<log_dir>/stderror.err``, where the
script's error trap writes, and raised as ``subprocess.CalledProcessError`` as a failed script is.

bash's stderr guard (``stderr_guard`` in bash_params, off with ``kernel_disable_guard``) is mirrored with the pool
executor: the workers' stderr (``orchestrator_worker_<n>.err``) is appended to ``stderror.err`` as the guarded
commands' ``2>> $LOG_DIR/stderror.err`` appends theirs, and a line that is not a Python warning fails the run
(``pool.stderr_errors``). A distributed chunk also fails, as its script does, when one of its ``P<k>.bin`` work files
is empty. The inline executor runs in this process and has no stderr of its own to check.

Progress is reported while the chunks run as ``gulmc --socket-server`` reports it in the script: pings to the platform
when bash_params has an ``analysis_pk``, else the local progress bar (``progress.ProgressReporter``).
"""
import fnmatch
import glob
import os
import shutil
import subprocess
import time

from oasislmf.utils.defaults import SERVER_UPDATE_TIME

from . import inline, pool
from .chunks import read_events
from .progress import ProgressReporter

EXECUTORS = {'pool': pool, 'inline': inline}
MODES = ('all', 'analysis', 'outputs')
EVENTS_FILE = os.path.join('input', 'events.bin')
RUN_LOG = 'orchestrator.log'
STDERR_LOG = 'stderror.err'
KEPT_OUTPUTS = ('*summary-info*', '*.json')


def command(mode, script):
    """The command a ``CalledProcessError`` of an orchestrated step names, in place of ``['bash', script]``."""
    return ['oasislmf-orchestrator', mode, script]


def _resolve(run_dir, path):
    return path if os.path.isabs(path) else os.path.join(run_dir, path)


def empty_dir(path, create=True):
    """Remove everything inside ``path`` (``rm -R -f path/*``); create ``path`` if it does not exist and ``create``."""
    if os.path.isdir(path):
        for name in os.listdir(path):
            entry = os.path.join(path, name)
            if os.path.isdir(entry) and not os.path.islink(entry):
                shutil.rmtree(entry)
            else:
                os.remove(entry)
    elif create:
        os.makedirs(path, exist_ok=True)


def remove_previous_outputs(output_dir):
    """Remove the files under ``output_dir`` but the summary-info and JSON ones, as the analysis script does.

    That is ``find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +``.
    """
    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            if not any(fnmatch.fnmatch(name, pattern) for pattern in KEPT_OUTPUTS):
                os.remove(os.path.join(root, name))


def _worker_env(log_dir, log_level):
    env = dict(os.environ)
    env['OASIS_PYTOOLS_LOG_DIR'] = log_dir
    if log_level is not None:
        env['OASIS_PYTOOLS_LOG_LEVEL'] = str(log_level)
    return env


def append_worker_stderr(log_dir):
    """Append the workers' stderr files to ``<log_dir>/stderror.err``, in worker order."""
    paths = sorted(glob.glob(os.path.join(log_dir, 'orchestrator_worker_*.err')),
                   key=lambda path: int(os.path.basename(path).split('_')[-1].split('.')[0]))
    with open(os.path.join(log_dir, STDERR_LOG), 'ab') as out:
        for path in paths:
            with open(path, 'rb') as f:
                shutil.copyfileobj(f, out)


def empty_chunk_outputs(run_dir, process_number):
    """The empty ``P<k>.bin`` files under ``<k>.work``, the files the chunk script requires to be non-empty."""
    name = f'P{process_number}.bin'
    empty = []
    for root, _dirs, files in os.walk(os.path.join(run_dir, f'{process_number}.work')):
        if name in files and os.path.getsize(os.path.join(root, name)) == 0:
            empty.append(os.path.relpath(os.path.join(root, name), run_dir))
    return sorted(empty)


def _run_log(plan, mode, stats, chunk_numbers, n_workers, error=None):
    """The lines of the run log."""
    lines = [f'orchestrator: {mode}, executor {stats.get("executor", "pool") if stats else "pool"}, {n_workers} workers, '
             f'chunks {chunk_numbers} of {plan.n_chunks}']
    if stats:
        for chunk in stats.get('chunks', []):
            lines.append(f"chunk {chunk['chunk']}: {chunk['events']} events, {chunk['seconds']:.2f} s, worker {chunk.get('worker', 0)}")
        for job in stats.get('jobs', []):
            lines.append(f"output job {job['tool']} {' '.join(job['outputs'])}: {job['seconds']:.2f} s, worker {job.get('worker', 0)}")
        lines.append(f"done: {stats.get('events', 0)} events, {len(stats.get('jobs', []))} output jobs, "
                     f"{stats.get('wall_seconds', 0.0):.2f} s (setup {stats.get('setup_seconds', 0.0):.2f} s, "
                     f"output jobs {stats.get('output_seconds', 0.0):.2f} s)")
    if error is not None:
        lines.append(f'FAILED: {error}')
    return '\n'.join(lines) + '\n'


def _failure(plan, mode, stats, chunk_numbers, n_workers, message, log_dir, script, t_start):
    """Write a failed run's run log and ``stderror.err`` line; the ``CalledProcessError`` to raise."""
    text = _run_log(plan, mode, stats, chunk_numbers, n_workers, error=message)
    with open(os.path.join(log_dir, RUN_LOG), 'w') as f:
        f.write(text)
    with open(os.path.join(log_dir, STDERR_LOG), 'a') as f:
        f.write(f'orchestrator {mode} failed after {time.perf_counter() - t_start:.2f} s\n{message}\n')
    return subprocess.CalledProcessError(1, command(mode, script), output=text.encode('utf-8'))


def run_plan(plan, bash_params, mode, log_dir, script, executor='pool', n_workers=None, progress_interval=SERVER_UPDATE_TIME,
             monitor=None):
    """Run one kernel step of a plan, with the side effects its bash script has.

    Args:
        plan (Plan): from ``plan_from_bash_params(bash_params)``; its ``spec.run_dir`` is the directory the script
            would run in, and relative paths are taken in it.
        bash_params (dict): the dict the plan was made from; ``work_dir``, ``work_kat_dir``, ``output_dir``,
            ``remove_working_files``, ``analysis_pk``, ``socket_server_size``, ``log_level``, ``stderr_guard`` and
            ``process_number`` are read.
        mode (str): ``all``, ``analysis`` or ``outputs`` (see the module docstring).
        log_dir (str): the step's log directory (``log``, ``log/<process_number>`` or ``log/out``), relative to the
            run directory or absolute.
        script (str): the script bash would have run, named in errors.
        executor (str): ``pool`` (default) or ``inline``.
        n_workers (int, optional): worker processes; by default one per chunk computed (``max_process_id``, or 1 for
            a single distributed chunk), and for the outputs step one per partition.
        progress_interval (float): seconds between two progress reports of a worker.
        monitor (ResourceMonitor, optional): started with this process's id once the log directory is emptied, and
            stopped when the run ends.

    Returns:
        str: the run log, also written to ``<log_dir>/orchestrator.log``.

    Raises:
        subprocess.CalledProcessError: the run failed; the reason is in its output and in ``<log_dir>/stderror.err``.
    """
    if mode not in MODES:
        raise ValueError(f'unknown mode {mode!r}, expected one of {MODES}')
    if executor not in EXECUTORS:
        raise ValueError(f'unknown executor {executor!r}, expected one of {sorted(EXECUTORS)}')
    run_dir = os.path.abspath(plan.spec.run_dir)
    log_dir = os.path.abspath(_resolve(run_dir, log_dir))
    empty_dir(log_dir)
    open(os.path.join(log_dir, STDERR_LOG), 'a').close()

    run_chunks = mode in ('all', 'analysis')
    run_outputs = mode in ('all', 'outputs')
    chunk_numbers = plan.chunk_numbers if run_chunks else []
    if n_workers is None:
        n_workers = len(chunk_numbers) if run_chunks else plan.n_chunks
    work_dir = _resolve(run_dir, bash_params.get('work_dir', 'work/'))
    if run_chunks:
        remove_previous_outputs(_resolve(run_dir, bash_params.get('output_dir', 'output/')))
        empty_dir(work_dir)
        os.makedirs(_resolve(run_dir, bash_params.get('work_kat_dir', 'work/kat/')), exist_ok=True)

    stderr_guard = bool(bash_params.get('stderr_guard', True))
    chunks = plan.make_chunks(read_events(os.path.join(run_dir, EVENTS_FILE))) if run_chunks else []
    options = {
        'output_jobs': plan.output_jobs if run_outputs else [],
        'chunk_numbers': chunk_numbers,
        'env': _worker_env(log_dir, bash_params.get('log_level')),
        'log_dir': log_dir,
        'progress_interval': progress_interval,
        'stderr_guard': stderr_guard,
    }
    run_log_path = os.path.join(log_dir, RUN_LOG)
    analysis_pk = bash_params.get('analysis_pk')
    bar_total = bash_params.get('socket_server_size') if run_chunks else None
    t_start = time.perf_counter()
    if monitor is not None:
        monitor.start(os.getpid())
    try:
        with ProgressReporter(analysis_pk=analysis_pk if run_chunks else None, bar_total=bar_total) as reporter:
            options['progress'] = lambda done, _total: reporter.update(done)
            stats = EXECUTORS[executor].run_chunks(plan.spec, chunks, log_dir, [], max(n_workers, 1), **options)
    except Exception as exc:
        if stderr_guard:
            append_worker_stderr(log_dir)
        raise _failure(plan, mode, None, chunk_numbers, n_workers, f'{type(exc).__name__}: {exc}', log_dir, script,
                       t_start) from exc
    finally:
        if monitor is not None:
            monitor.stop()

    if stderr_guard:
        append_worker_stderr(log_dir)
        process_number = bash_params.get('process_number')
        if mode == 'analysis' and process_number:
            empty = empty_chunk_outputs(run_dir, process_number)
            if empty:
                raise _failure(plan, mode, stats, chunk_numbers, n_workers, f"Chunk output error: File '{empty[0]}' is empty",
                               log_dir, script, t_start)

    if run_outputs and bash_params.get('remove_working_files'):
        empty_dir(work_dir, create=False)
    text = _run_log(plan, mode, stats, chunk_numbers, n_workers)
    with open(run_log_path, 'w') as f:
        f.write(text)
    return text
