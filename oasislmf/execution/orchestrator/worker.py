"""Worker process of the fused pool executor: the whole chain in one process, pulling chunks from the orchestrator.

Started by ``pool.py`` as ``python -m oasislmf.execution.orchestrator.worker`` with the run directory as working
directory. Its configuration arrives as one JSON document on stdin (so the authentication key never shows in the
process list):

* ``address`` (``[host, port]``) and ``authkey`` (hex) of the orchestrator's ``multiprocessing.connection.Listener``;
* ``worker``: the worker number the orchestrator gave it;
* ``spec``: the ``ChainSpec`` as a dict, ``factories``: ``{'gul'|'fm'|...: 'package.module:attribute'}`` or None;
* ``out_dir`` (absolute) and ``taps``;
* ``jobs_only``: build no chain and take only output jobs (the pool adds such workers for the job phase);
* ``progress_interval``: when set, report the events done in the current chunk at most every this many seconds;
* ``sys_path``: the orchestrator's ``sys.path``, whose entries the worker appends to its own so that factories
  importable in the orchestrator are importable here;
* ``log_dir``: the directory of the worker's log file when ``OASIS_PYTOOLS_LOG_DIR`` does not set one.

Logging. For its whole life the worker has the logging setup a pytool's ``main`` has under ``redirect_logging``
(``pytools.utils.pytool_logging``): log records of the stages and Python warnings go to
``orchestrator_worker_<n>_<pid>_<uuid>.log`` in ``OASIS_PYTOOLS_LOG_DIR`` (else ``log_dir``) at
``OASIS_PYTOOLS_LOG_LEVEL``, not to its stderr, so that a warning does not trip the stderr guard where bash's would
not. The stage modules are imported first, so that their loggers are levelled as a pytool's are. The file ends with
``finishing process`` when the worker exits normally; an error the worker reports is also logged there.

Protocol, every message a dict with a ``type``:

* worker -> orchestrator: ``hello`` (``worker``, ``pid``) once connected; ``ready`` (``chain_seconds``) when it
  can take a chunk; ``progress`` (``chunk``, ``events``: done since the last report) while running a chunk, when
  ``progress_interval`` is set; ``done`` (``chunk``, ``events``, ``seconds``) after each chunk and ``done`` (``job``,
  ``seconds``) after each output job, followed by ``ready``; ``error`` (``stage``, ``chunk``, ``event``, ``job``,
  ``error``, ``traceback``) when setting up the chain, running a chunk or running a job fails, after which the
  worker exits with status 1.
* orchestrator -> worker: ``chunk`` (``chunk``, ``events``) or ``job`` (``job``: its index, ``tool``, ``kwargs``)
  in reply to ``ready``, or ``stop``, after which the worker closes its chain and exits with status 0. Jobs come
  only after every chunk of the run is done, so the worker closes its chain before the first one to give the
  memory back.
"""
import contextlib
import gc
import json
import logging
import os
import sys
import time
import traceback
from multiprocessing.connection import Client

from oasislmf.pytools.utils import pytool_logging
from oasislmf.warmup import lock_cache_saves

from .chain import Chain, ChainSpec, import_factories
from .chunks import ChunkWriter
from .outputs import JobError, run_job

# Not __name__: run as 'python -m', this module is __main__, which pytool_logging would not route to the log file.
logger = logging.getLogger('oasislmf.execution.orchestrator.worker')

EXIT_ERROR = 1
LOG_NAME = 'orchestrator_worker_{worker}'


def stage_name(chain, index):
    """Name of a stage of a chain as used in error messages.

    Args:
        chain (Chain): the chain.
        index (int | None): None for the gul stage, else the index of the fm step.

    Returns:
        str: ``'gul'``, or the fm step's name with its position, e.g. ``'il (fm step 1)'``.
    """
    if index is None:
        return 'gul'
    return f"{chain.spec.fm[index].get('name')} (fm step {index + 1})"


def failing_stage(chain, tb):
    """Find which stage of a chain raised an exception, from its traceback.

    The outermost frame whose ``self`` is one of the chain's stages names it.

    Args:
        chain (Chain): the chain that was running.
        tb (types.TracebackType): traceback of the exception.

    Returns:
        str | None: the stage's name (see ``Chain.stages``), None when no stage is on the traceback.
    """
    stages = {id(stage): name for name, stage in chain.stages()}
    while tb is not None:
        owner = tb.tb_frame.f_locals.get('self')
        if owner is not None and id(owner) in stages:
            return stages[id(owner)]
        tb = tb.tb_next
    return None


def error_message(exc, stage, chunk=None, event=None, job=None):
    """Build the ``error`` message describing an exception, and log it."""
    message = {
        'type': 'error', 'stage': stage, 'chunk': chunk, 'event': event, 'job': job,
        'error': f'{type(exc).__name__}: {exc}',
        'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    logger.error('stage %s, chunk %s, event %s, job %s failed: %s\n%s', stage, chunk, event, job, message['error'],
                 message['traceback'].rstrip())
    return message


def run_chunk(chain, out_dir, taps, chunk_number, events, report=None, report_interval=None):
    """Run one chunk through the chain and write its tap streams and the chain's files.

    Args:
        chain (Chain): the chain; the chunk runs between its ``start_chunk`` and ``finish_chunk``.
        out_dir (str): output directory.
        taps (list[str]): taps to write.
        chunk_number (int): 1-based chunk number.
        events (array-like): event ids of the chunk.
        report (callable, optional): ``report(n)`` called with the number of events done since the previous call, at
            most every ``report_interval`` seconds; the events not yet reported when the chunk ends are not.
        report_interval (float, optional): seconds between two calls of ``report``.

    Returns:
        dict: the ``done`` message.

    Raises:
        ChunkError: wrapping whatever a stage raised, with the event id it was processing.
    """
    t_start = time.perf_counter()
    t_report, unreported = t_start, 0
    event_id = None
    try:
        chain.start_chunk(chunk_number)
        with ChunkWriter(out_dir, chunk_number, chain.headers, taps) as writer:
            sinks = writer.sinks
            for event_id in events:
                chain.run_event(event_id, sinks)
                if report is not None:
                    unreported += 1
                    now = time.perf_counter()
                    if now - t_report >= report_interval:
                        report(unreported)
                        t_report, unreported = now, 0
        event_id = None
        chain.finish_chunk()
    except Exception as exc:
        raise ChunkError(event_id) from exc
    return {'type': 'done', 'chunk': chunk_number, 'events': len(events), 'seconds': time.perf_counter() - t_start}


class ChunkError(Exception):
    """Raised by ``run_chunk`` with the id of the event being processed (None before the first event)."""

    def __init__(self, event_id):
        super().__init__(event_id)
        self.event_id = None if event_id is None else int(event_id)


def progress_reporter(conn, chunk_number):
    """The ``report`` callable of ``run_chunk``: sends a ``progress`` message for chunk ``chunk_number``."""
    def report(n_events):
        conn.send({'type': 'progress', 'chunk': chunk_number, 'events': n_events})
    return report


def run_output_job(message, run_dir):
    """Run the output job of a ``job`` message.

    Args:
        message (dict): the message: ``job`` (index), ``tool`` and ``kwargs``.
        run_dir (str): the run directory, the job's working directory.

    Returns:
        dict: the ``done`` message.

    Raises:
        JobError: the tool failed.
    """
    result = run_job({'tool': message['tool'], 'kwargs': message['kwargs'],
                     'summary_info': message.get('summary_info')}, run_dir, index=message['job'])
    return {'type': 'done', 'job': message['job'], 'seconds': result['seconds']}


def serve(conn, config):
    """Build the chain, then answer chunks and output jobs until told to stop.

    Args:
        conn (multiprocessing.connection.Connection): connection to the orchestrator.
        config (dict): the worker configuration (see the module docstring).

    Returns:
        int: exit status, 0 after ``stop``, ``EXIT_ERROR`` after an error was reported.
    """
    conn.send({'type': 'hello', 'worker': config['worker'], 'pid': os.getpid()})
    t_setup = time.perf_counter()
    spec = ChainSpec.from_dict(config['spec'])
    chain = None
    if not config.get('jobs_only'):
        try:
            chain = Chain(spec, factories=config.get('factories'))
        except Exception as exc:
            conn.send(error_message(exc, 'setup'))
            return EXIT_ERROR
    taps = list(config['taps'])
    out_dir = config['out_dir']
    run_dir = spec.run_dir
    progress_interval = config.get('progress_interval')
    try:
        conn.send({'type': 'ready', 'chain_seconds': time.perf_counter() - t_setup})
        while True:
            message = conn.recv()
            if message['type'] == 'stop':
                return 0
            if message['type'] == 'job':
                if chain is not None:
                    chain.close()
                    chain = None
                    gc.collect()
                try:
                    done = run_output_job(message, run_dir)
                except JobError as exc:
                    conn.send(error_message(exc.__cause__, exc.job.tool, job=message['job']))
                    return EXIT_ERROR
                conn.send(done)
                conn.send({'type': 'ready'})
                continue
            chunk_number = message['chunk']
            if chain is None:
                conn.send(error_message(RuntimeError('chunk sent to a worker without a chain'), 'chain', chunk_number))
                return EXIT_ERROR
            report = progress_reporter(conn, chunk_number) if progress_interval is not None else None
            try:
                done = run_chunk(chain, out_dir, taps, chunk_number, message['events'], report=report, report_interval=progress_interval)
            except ChunkError as exc:
                cause = exc.__cause__
                stage = failing_stage(chain, cause.__traceback__) or 'chain'
                conn.send(error_message(cause, stage, chunk_number, exc.event_id))
                return EXIT_ERROR
            conn.send(done)
            conn.send({'type': 'ready'})
    finally:
        if chain is not None:
            chain.close()


def main():
    """Read the configuration from stdin, set up logging, connect to the orchestrator and serve it."""
    lock_cache_saves()
    config = json.loads(sys.stdin.read())
    for path in config.get('sys_path', []):
        if path not in sys.path:
            sys.path.append(path)
    if not config.get('jobs_only'):
        with contextlib.suppress(Exception):
            # A factory that fails to import fails again in Chain(), which reports it as a setup error.
            import_factories(ChainSpec.from_dict(config['spec']), config.get('factories'))
    address = tuple(config['address'])
    with pytool_logging(LOG_NAME.format(worker=config['worker']), config.get('log_dir') or './log') as log:
        log.info('starting process')
        with Client(address, family='AF_INET', authkey=bytes.fromhex(config['authkey'])) as conn:
            status = serve(conn, config)
        log.info('finishing process')
    return status


if __name__ == '__main__':
    sys.exit(main())
