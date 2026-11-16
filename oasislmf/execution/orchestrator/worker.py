"""Worker process of the fused pool executor: the whole chain in one process, pulling chunks from the orchestrator.

Started by ``pool.py`` as ``python -m oasislmf.execution.orchestrator.worker`` with the run directory as working
directory. Its configuration arrives as one JSON document on stdin (so the authentication key never shows in the
process list):

* ``address`` (``[host, port]``) and ``authkey`` (hex) of the orchestrator's ``multiprocessing.connection.Listener``;
* ``worker``: the worker number the orchestrator gave it;
* ``spec``: the ``ChainSpec`` as a dict, ``factories``: ``{'gul'|'fm': 'package.module:attribute'}`` or None;
* ``out_dir`` (absolute) and ``taps``;
* ``sys_path``: the orchestrator's ``sys.path``, whose entries the worker appends to its own so that factories
  importable in the orchestrator are importable here.

Protocol, every message a dict with a ``type``:

* worker -> orchestrator: ``hello`` (``worker``, ``pid``) once connected; ``ready`` (``chain_seconds``) when it
  can take a chunk; ``done`` (``chunk``, ``events``, ``seconds``) after each chunk, followed by ``ready``; ``error``
  (``stage``, ``chunk``, ``event``, ``error``, ``traceback``) when setting up the chain or running a chunk fails,
  after which the worker exits with status 1.
* orchestrator -> worker: ``chunk`` (``chunk``, ``events``) in reply to ``ready``, or ``stop``, after which the
  worker closes its chain and exits with status 0.
"""
import json
import os
import sys
import time
import traceback
from multiprocessing.connection import Client

from .chain import Chain, ChainSpec
from .chunks import ChunkWriter

EXIT_ERROR = 1


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
        str | None: the stage's name (see ``stage_name``), None when no stage is on the traceback.
    """
    stages = {id(chain.gul): None, **{id(stage): i for i, stage in enumerate(chain.fm_steps)}}
    while tb is not None:
        owner = tb.tb_frame.f_locals.get('self')
        if owner is not None and id(owner) in stages:
            return stage_name(chain, stages[id(owner)])
        tb = tb.tb_next
    return None


def error_message(exc, stage, chunk=None, event=None):
    """Build the ``error`` message describing an exception."""
    return {
        'type': 'error', 'stage': stage, 'chunk': chunk, 'event': event,
        'error': f'{type(exc).__name__}: {exc}',
        'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def run_chunk(chain, out_dir, taps, chunk_number, events):
    """Run one chunk through the chain and write its streams.

    Args:
        chain (Chain): the chain, reset first.
        out_dir (str): output directory.
        taps (list[str]): taps to write.
        chunk_number (int): 1-based chunk number.
        events (array-like): event ids of the chunk.

    Returns:
        dict: the ``done`` message.

    Raises:
        ChunkError: wrapping whatever a stage raised, with the event id it was processing.
    """
    t_start = time.perf_counter()
    event_id = None
    try:
        chain.reset()
        with ChunkWriter(out_dir, chunk_number, chain.headers, taps) as writer:
            sinks = writer.sinks
            for event_id in events:
                chain.run_event(event_id, sinks)
    except Exception as exc:
        raise ChunkError(event_id) from exc
    return {'type': 'done', 'chunk': chunk_number, 'events': len(events), 'seconds': time.perf_counter() - t_start}


class ChunkError(Exception):
    """Raised by ``run_chunk`` with the id of the event being processed (None before the first event)."""

    def __init__(self, event_id):
        super().__init__(event_id)
        self.event_id = None if event_id is None else int(event_id)


def serve(conn, config):
    """Build the chain, then answer chunks until told to stop.

    Args:
        conn (multiprocessing.connection.Connection): connection to the orchestrator.
        config (dict): the worker configuration (see the module docstring).

    Returns:
        int: exit status, 0 after ``stop``, ``EXIT_ERROR`` after an error was reported.
    """
    conn.send({'type': 'hello', 'worker': config['worker'], 'pid': os.getpid()})
    t_setup = time.perf_counter()
    try:
        chain = Chain(ChainSpec.from_dict(config['spec']), factories=config.get('factories'))
    except Exception as exc:
        conn.send(error_message(exc, 'setup'))
        return EXIT_ERROR
    taps = list(config['taps'])
    out_dir = config['out_dir']
    try:
        conn.send({'type': 'ready', 'chain_seconds': time.perf_counter() - t_setup})
        while True:
            message = conn.recv()
            if message['type'] == 'stop':
                return 0
            chunk_number = message['chunk']
            try:
                done = run_chunk(chain, out_dir, taps, chunk_number, message['events'])
            except ChunkError as exc:
                cause = exc.__cause__
                stage = failing_stage(chain, cause.__traceback__) or 'chain'
                conn.send(error_message(cause, stage, chunk_number, exc.event_id))
                return EXIT_ERROR
            conn.send(done)
            conn.send({'type': 'ready'})
    finally:
        chain.close()


def main():
    """Read the configuration from stdin, connect to the orchestrator and serve it."""
    config = json.loads(sys.stdin.read())
    for path in config.get('sys_path', []):
        if path not in sys.path:
            sys.path.append(path)
    address = tuple(config['address'])
    with Client(address, family='AF_INET', authkey=bytes.fromhex(config['authkey'])) as conn:
        return serve(conn, config)


if __name__ == '__main__':
    sys.exit(main())
