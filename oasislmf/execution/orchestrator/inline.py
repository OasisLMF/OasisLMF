"""Inline executor: every chunk runs sequentially in the calling process, through one ``Chain``."""
import dataclasses
import os
import time

from .chain import Chain
from .chunks import ChunkWriter


def run_chunks(spec, chunks, out_dir, taps, n_workers=1, **options):
    """Run every chunk in the calling process and write ``<out_dir>/<tap>_P<k>.bin`` for each.

    The working directory is the run directory while the chain is set up and run, and is restored afterwards.
    ``Chain.reset()`` is called at the start of every chunk.

    Args:
        spec (ChainSpec): the chain to run.
        chunks (Sequence[array-like]): event ids of each chunk; chunk k (1-based) is ``chunks[k - 1]``.
        out_dir (str | os.PathLike): output directory, relative to the current working directory.
        taps (Iterable[str]): taps to write.
        n_workers (int): ignored, there is always one worker.
        **options: ``factories`` is passed to ``Chain``. ``chunk_order`` (Sequence[int], optional) is the order the
            chunks are run in, as 1-based chunk numbers, a permutation of ``1..len(chunks)``; it changes only the
            processing order, never which file a chunk writes. Other options are ignored.

    Returns:
        dict: statistics: ``executor``, ``n_workers``, ``wall_seconds``, ``setup_seconds``, ``events``,
        ``chunks`` (one ``{'chunk', 'events', 'seconds'}`` per chunk, in the order run) and ``workers`` (one entry).

    Raises:
        ValueError: if ``chunk_order`` is not a permutation of ``1..len(chunks)``.
    """
    t_start = time.perf_counter()
    taps = list(taps)
    chunk_order = options.get('chunk_order')
    chunk_order = list(range(1, len(chunks) + 1)) if chunk_order is None else [int(k) for k in chunk_order]
    if sorted(chunk_order) != list(range(1, len(chunks) + 1)):
        raise ValueError(f"chunk_order must be a permutation of 1..{len(chunks)}, got {chunk_order}")
    out_dir = os.path.abspath(out_dir)
    spec = dataclasses.replace(spec, run_dir=os.path.abspath(spec.run_dir))
    chunk_stats = []
    cwd = os.getcwd()
    os.chdir(spec.run_dir)
    try:
        with Chain(spec, factories=options.get('factories')) as chain:
            setup_seconds = time.perf_counter() - t_start
            for k in chunk_order:
                chunk = chunks[k - 1]
                t_chunk = time.perf_counter()
                chain.reset()
                with ChunkWriter(out_dir, k, chain.headers, taps) as writer:
                    sinks = writer.sinks
                    for event_id in chunk:
                        chain.run_event(event_id, sinks)
                chunk_stats.append({'chunk': k, 'events': len(chunk), 'seconds': time.perf_counter() - t_chunk})
    finally:
        os.chdir(cwd)
    wall_seconds = time.perf_counter() - t_start
    busy = sum(c['seconds'] for c in chunk_stats)
    return {
        'executor': 'inline',
        'n_workers': 1,
        'wall_seconds': wall_seconds,
        'setup_seconds': setup_seconds,
        'events': sum(c['events'] for c in chunk_stats),
        'chunks': chunk_stats,
        'workers': [{'worker': 0, 'chunks': len(chunk_stats), 'setup_seconds': setup_seconds, 'busy_seconds': busy}],
    }
