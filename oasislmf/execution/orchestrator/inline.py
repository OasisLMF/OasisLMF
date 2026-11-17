"""Inline executor: every chunk runs sequentially in the calling process, through one ``Chain``, then the output jobs."""
import dataclasses
import os
import time

from .chain import Chain, spec_for_taps
from .chunks import ChunkWriter
from .outputs import as_job, check_independent, run_jobs


def run_chunks(spec, chunks, out_dir, taps, n_workers=1, **options):
    """Run every chunk in the calling process and write ``<out_dir>/<tap>_P<k>.bin`` and the chain's files for each.

    The working directory is the run directory while the chain is set up and run, and is restored afterwards.
    Every chunk is run between ``Chain.start_chunk(k)`` and ``Chain.finish_chunk()``, which reset the stages and
    open and close the chunk's files of the summary tree (paths relative to the run directory). Once every chunk is
    done and the chain closed, the ``output_jobs`` run one after the other in this process (``outputs.run_jobs``).

    Args:
        spec (ChainSpec): the chain to run.
        chunks (Sequence[array-like]): event ids of each chunk; chunk k (1-based) is ``chunks[k - 1]``.
        out_dir (str | os.PathLike): output directory, relative to the current working directory.
        taps (Iterable[str]): taps to write.
        n_workers (int): ignored, there is always one worker.
        **options: ``factories`` is passed to ``Chain``. ``chunk_order`` (Sequence[int], optional) is the order the
            chunks are run in, as 1-based chunk numbers, a permutation of ``1..len(chunks)``; it changes only the
            processing order, never which file a chunk writes. ``output_jobs`` (Sequence[OutputJob | dict]) are run
            after the chunks. Other options are ignored.

    Returns:
        dict: statistics: ``executor``, ``n_workers``, ``wall_seconds``, ``setup_seconds``, ``events``,
        ``chunks`` (one ``{'chunk', 'events', 'seconds'}`` per chunk, in the order run), ``jobs`` (one ``{'job', 'tool',
        'outputs', 'seconds', 'worker'}`` per output job, in order), ``output_seconds`` and ``workers`` (one entry).

    Raises:
        ValueError: if ``chunk_order`` is not a permutation of ``1..len(chunks)``, or the output jobs are not
            independent.
        JobError: an output job failed; the jobs after it were not run.
    """
    t_start = time.perf_counter()
    taps = list(taps)
    chunk_order = options.get('chunk_order')
    chunk_order = list(range(1, len(chunks) + 1)) if chunk_order is None else [int(k) for k in chunk_order]
    if sorted(chunk_order) != list(range(1, len(chunks) + 1)):
        raise ValueError(f"chunk_order must be a permutation of 1..{len(chunks)}, got {chunk_order}")
    output_jobs = [as_job(job) for job in options.get('output_jobs') or []]
    check_independent(output_jobs)
    out_dir = os.path.abspath(out_dir)
    spec = dataclasses.replace(spec_for_taps(spec, taps), run_dir=os.path.abspath(spec.run_dir))
    chunk_stats = []
    cwd = os.getcwd()
    os.chdir(spec.run_dir)
    try:
        with Chain(spec, factories=options.get('factories')) as chain:
            setup_seconds = time.perf_counter() - t_start
            for k in chunk_order:
                chunk = chunks[k - 1]
                t_chunk = time.perf_counter()
                chain.start_chunk(k)
                with ChunkWriter(out_dir, k, chain.headers, taps) as writer:
                    sinks = writer.sinks
                    for event_id in chunk:
                        chain.run_event(event_id, sinks)
                chain.finish_chunk()
                chunk_stats.append({'chunk': k, 'events': len(chunk), 'seconds': time.perf_counter() - t_chunk})
    finally:
        os.chdir(cwd)
    t_jobs = time.perf_counter()
    job_stats = run_jobs(output_jobs, spec.run_dir)
    output_seconds = time.perf_counter() - t_jobs if output_jobs else 0.0
    wall_seconds = time.perf_counter() - t_start
    busy = sum(c['seconds'] for c in chunk_stats)
    return {
        'executor': 'inline',
        'n_workers': 1,
        'wall_seconds': wall_seconds,
        'setup_seconds': setup_seconds,
        'events': sum(c['events'] for c in chunk_stats),
        'chunks': chunk_stats,
        'jobs': job_stats,
        'output_seconds': output_seconds,
        'workers': [{'worker': 0, 'chunks': len(chunk_stats), 'setup_seconds': setup_seconds, 'busy_seconds': busy,
                     'jobs': len(job_stats), 'job_seconds': sum(j['seconds'] for j in job_stats)}],
    }
