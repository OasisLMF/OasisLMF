"""The output stage: the katpy, aalpy and lecpy runs bash makes once every partition is done, as in-process jobs.

A job is an ``OutputJob`` of the plan (``plan.py``): a tool and the keyword arguments of its ``main``. ``run_job``
calls that ``main`` in the calling process with the run directory as working directory, which is what the pool's
workers and the inline executor do instead of starting one Python interpreter per output file.

Ordering. The jobs of a plan are independent: a katpy job reads only ``work/kat/*_P<k>`` files, an aalpy or lecpy
job only its own ``work/<subfolder>/`` (where it also keeps its scratch files, ``aal_files/`` and ``lec_files/``),
and none of them reads anything under ``output/`` but the summary-info files the analysis step writes. ALCT is
written by the same aalpy run as its ALT. With join_summary_info, a job joins the summary info onto its own files
once its tool is done, as bash does once every tool is done; no other job reads them. They can therefore run in
any order and in parallel. ``check_independent`` verifies this for a
list of jobs, so that a planner change that introduces a dependency fails loudly instead of racing.

In-process safety. The tools keep no state between calls in module globals (aalpy reads ``OASIS_AAL_MEMORY`` once
at import, which a fresh interpreter does too), and their Numba kernels are pure functions, compiled once per
process and cached on disk. What does leak is process-wide state:

* logging: ``redirect_logging`` re-levels every handler of every logger, sets the ``oasislmf`` logger to INFO and
  calls ``logging.shutdown()``; ``isolated_logging`` puts back the levels, handlers, propagation and
  ``warnings.showwarning`` it found;
* the working directory: set to the run directory for the call and restored afterwards, also on error.
"""
import importlib
import logging
import os
import time
import warnings
from contextlib import contextmanager

from oasislmf.pytools.join_summary_info.manager import main as join_summary_info

from .plan import OutputJob

__all__ = ['JobError', 'as_job', 'job_label', 'job_outputs', 'job_inputs', 'check_independent', 'jobs_for_chunks',
           'prepare_output_dirs', 'isolated_logging', 'run_job', 'run_jobs']


class JobError(RuntimeError):
    """An output job failed.

    Attributes:
        job (OutputJob): the job.
        index (int | None): its position in the list of jobs run.
        label (str): the tool and the files it writes, as in the message.
    """

    def __init__(self, message, job=None, index=None):
        super().__init__(message)
        self.job = job
        self.index = index
        self.label = job_label(job) if job is not None else None


def as_job(job):
    """Return ``job`` as an ``OutputJob``; it may be one already or its ``to_dict`` form."""
    return job if isinstance(job, OutputJob) else OutputJob.from_dict(job)


def job_outputs(job):
    """The files a job writes, relative to the run directory as the plan gives them.

    Args:
        job (OutputJob | dict): the job.

    Returns:
        list[str]: the paths, in the order of ``plan.OUTPUT_KWARGS``.
    """
    return as_job(job).outputs()


def job_inputs(job, work_dir='work'):
    """What a job reads: its input files (katpy) or its work sub-directory (aalpy, lecpy, which also write scratch files there).

    Args:
        job (OutputJob | dict): the job.
        work_dir (str): the work directory aalpy and lecpy resolve ``subfolder`` in.

    Returns:
        list[str]: paths relative to the run directory.
    """
    job = as_job(job)
    if job.tool == 'katpy':
        return list(job.kwargs.get('files_in') or []) + ([job.kwargs['dir_in']] if job.kwargs.get('dir_in') else [])
    return [os.path.join(work_dir, job.kwargs['subfolder'])] if job.kwargs.get('subfolder') else []


def job_label(job):
    """``'<tool> <output file>[, <output file>]'``: how a job is named in statistics and errors."""
    job = as_job(job)
    outputs = job_outputs(job)
    return f"{job.tool} {', '.join(outputs)}" if outputs else job.tool


def _overlaps(path, other):
    """True when two relative paths are the same or one is inside the other."""
    path, other = os.path.normpath(path), os.path.normpath(other)
    return path == other or path.startswith(other + os.sep) or other.startswith(path + os.sep)


def check_independent(jobs):
    """Check that jobs can run in any order and in parallel.

    Two jobs conflict when they write the same file, when one reads what the other writes, or when two aalpy/lecpy
    jobs share a work sub-directory (they keep scratch files there).

    Args:
        jobs (Iterable[OutputJob | dict]): the jobs.

    Raises:
        ValueError: naming the first two conflicting jobs.
    """
    jobs = [as_job(j) for j in jobs]
    for i, a in enumerate(jobs):
        for b in jobs[i + 1:]:
            reason = None
            if any(_overlaps(x, y) for x in job_outputs(a) for y in job_outputs(b)):
                reason = 'write the same file'
            elif any(_overlaps(x, y) for x in job_inputs(a) for y in job_outputs(b)) or \
                    any(_overlaps(x, y) for x in job_inputs(b) for y in job_outputs(a)):
                reason = "read each other's output"
            elif a.tool != 'katpy' and b.tool != 'katpy' and any(_overlaps(x, y) for x in job_inputs(a) for y in job_inputs(b)):
                reason = 'share a work sub-directory'
            if reason:
                raise ValueError(f'output jobs [{job_label(a)}] and [{job_label(b)}] {reason}; they cannot run in any order')


def jobs_for_chunks(jobs, n_chunks):
    """The jobs of a plan for a run with ``n_chunks`` chunks instead of the plan's partition count.

    Only katpy depends on the count: it reads one ``..._P<k>`` file per chunk.

    Args:
        jobs (Iterable[OutputJob | dict]): the plan's jobs.
        n_chunks (int): chunk count of the run.

    Returns:
        list[OutputJob]: new jobs; katpy ``files_in`` lists ``_P1`` to ``_P<n_chunks>``.
    """
    result = []
    for job in map(as_job, jobs):
        kwargs = dict(job.kwargs)
        files_in = kwargs.get('files_in')
        if job.tool == 'katpy' and files_in:
            stem, sep, k = files_in[0].rpartition('_P')
            if not sep or not k.isdigit():
                raise ValueError(f'katpy input {files_in[0]!r} is not a per-chunk file name ending in _P<k>')
            kwargs['files_in'] = [f'{stem}_P{k}' for k in range(1, n_chunks + 1)]
        result.append(OutputJob(job.tool, kwargs, job.summary_info))
    return result


def prepare_output_dirs(jobs, run_dir):
    """Create the directories the jobs write into, as bash finds ``output/`` already there.

    Args:
        jobs (Iterable[OutputJob | dict]): the jobs.
        run_dir (str): the run directory relative paths are resolved in.
    """
    for job in jobs:
        for path in job_outputs(job):
            directory = os.path.dirname(os.path.join(run_dir, path))
            if directory:
                os.makedirs(directory, exist_ok=True)


def _logging_state():
    """Levels, handlers, propagation and handler levels of the root logger and every logger created so far."""
    loggers = [logging.getLogger()] + [lg for lg in logging.Logger.manager.loggerDict.values() if isinstance(lg, logging.Logger)]
    state = [(lg, lg.level, lg.propagate, lg.disabled, list(lg.handlers)) for lg in loggers]
    handlers = {h: h.level for _lg, _level, _prop, _dis, hs in state for h in hs}
    return state, handlers, warnings.showwarning


@contextmanager
def isolated_logging():
    """Undo, on exit, whatever a ``redirect_logging``-decorated tool did to the logging configuration of the process."""
    state, handler_levels, showwarning = _logging_state()
    try:
        yield
    finally:
        for lg, level, propagate, disabled, handlers in state:
            for handler in [h for h in lg.handlers if h not in handlers]:
                lg.removeHandler(handler)
                handler.close()
            for handler in handlers:
                if handler not in lg.handlers:
                    lg.addHandler(handler)
            lg.setLevel(level)
            lg.propagate = propagate
            lg.disabled = disabled
        for handler, level in handler_levels.items():
            handler.setLevel(level)
        warnings.showwarning = showwarning


def _entry_point(job):
    module_name, _, attr = job.entry_point.partition(':')
    return getattr(importlib.import_module(module_name), attr)


def run_job(job, run_dir, index=None):
    """Run one output job in the calling process.

    The tool's ``main`` is called with the job's kwargs and the run directory as working directory, then, when the
    job has a ``summary_info``, ``join-summary-info`` on each file it wrote. The previous working directory and the
    logging configuration are restored afterwards, whatever happens.

    Args:
        job (OutputJob | dict): the job.
        run_dir (str): the run directory.
        index (int, optional): the job's position, reported in the statistics and in a ``JobError``.

    Returns:
        dict: ``{'job', 'tool', 'outputs', 'seconds'}``.

    Raises:
        JobError: the tool raised; the original exception is its ``__cause__``.
    """
    job = as_job(job)
    t_start = time.perf_counter()
    cwd = os.getcwd()
    try:
        main = _entry_point(job)
        os.chdir(run_dir)
        with isolated_logging():
            main(**job.kwargs)
        for path in job.outputs() if job.summary_info else ():
            with isolated_logging():
                join_summary_info(summaryinfo=job.summary_info, data=path, output=path)
    except Exception as exc:
        raise JobError(f'output job {job_label(job)} failed: {type(exc).__name__}: {exc}', job=job, index=index) from exc
    finally:
        os.chdir(cwd)
    return {'job': index, 'tool': job.tool, 'outputs': job_outputs(job), 'seconds': time.perf_counter() - t_start}


def run_jobs(jobs, run_dir):
    """Run output jobs one after the other in the calling process; the first failure stops the run.

    Args:
        jobs (Iterable[OutputJob | dict]): the jobs.
        run_dir (str): the run directory.

    Returns:
        list[dict]: one ``run_job`` result per job, in order, with ``worker`` 0.

    Raises:
        JobError: a job failed; the jobs after it were not run.
        ValueError: the jobs are not independent (``check_independent``).
    """
    jobs = [as_job(j) for j in jobs]
    check_independent(jobs)
    prepare_output_dirs(jobs, run_dir)
    return [{**run_job(job, run_dir, index=i), 'worker': 0} for i, job in enumerate(jobs)]
