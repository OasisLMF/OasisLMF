"""Splitting the events into chunks, and writing each chunk's output streams.

Chunk k (1-based) of K holds exactly the events ``evepy k K`` emits, so a run in K chunks can be compared
byte for byte with a bash run in K partitions.
"""
import os

import numpy as np

from oasislmf.pytools.eve.manager import (event_id_dtype, partition_events__no_shuffle, partition_events__random,
                                          partition_events__random_builtin, partition_events__round_robin, read_events)
from oasislmf.utils.defaults import EVE_DEFAULT_SHUFFLE, EVE_FISHER_YATES, EVE_NO_SHUFFLE, EVE_ROUND_ROBIN, EVE_STD_SHUFFLE

__all__ = ['make_chunks', 'partition_events', 'read_events', 'chunk_path', 'ChunkWriter']


def make_chunks(events, n_chunks, event_shuffle=EVE_DEFAULT_SHUFFLE):
    """Split events into chunks, as ``evepy`` partitions them with a shuffle rule (round robin by default).

    Args:
        events (array-like): event ids, in the order they are held in ``events.bin``.
        n_chunks (int): number of chunks.
        event_shuffle (int): evepy partitioning rule (``EVE_*`` of ``oasislmf.utils.defaults``).

    Returns:
        list[np.ndarray]: ``n_chunks`` arrays of event ids; element ``k - 1`` holds exactly the events
        ``evepy k n_chunks`` emits, in the same order.
    """
    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive, got {n_chunks}")
    events = np.asarray(events, dtype=event_id_dtype)
    if event_shuffle == EVE_FISHER_YATES:
        return partition_events__fisher_yates_all(events, n_chunks)
    return [partition_events(events, k, n_chunks, event_shuffle) for k in range(1, n_chunks + 1)]


def partition_events__fisher_yates_all(events, n_chunks):
    """Every partition ``evepy -r`` emits, from a single shuffle.

    ``partition_events__random`` seeds its generator with a constant and draws the same numbers whatever the
    partition, so every partition sees the same shuffle. Its loop fixes position i of the array at step i and
    emits it, descending from the last position, when ``(i - k) % n_chunks == 0``, then position 0 when
    ``k % n_chunks == 0``. So partition k is the fully shuffled array at the positions congruent to k modulo
    n_chunks, in descending order. The shuffle itself is evepy's generator, run to the end once.

    Args:
        events (np.ndarray): event ids, in the order they are held in ``events.bin``.
        n_chunks (int): number of partitions.

    Returns:
        list[np.ndarray]: element ``k - 1`` holds the events ``evepy -r k n_chunks`` emits, in the same order.
    """
    shuffled = events.copy()
    for _ in partition_events__random(shuffled, 1, 1):
        pass
    positions = np.arange(len(shuffled) - 1, -1, -1)
    return [shuffled[positions[(positions - k) % n_chunks == 0]] for k in range(1, n_chunks + 1)]


def partition_events(events, k, n_chunks, event_shuffle=EVE_DEFAULT_SHUFFLE):
    """The events ``evepy`` emits for partition k of n_chunks with a shuffle rule, as an array."""
    if event_shuffle == EVE_ROUND_ROBIN:
        return partition_events__round_robin(events, k, n_chunks)
    if event_shuffle == EVE_NO_SHUFFLE:
        return partition_events__no_shuffle(events, k, n_chunks)
    if event_shuffle == EVE_FISHER_YATES:
        return np.fromiter(partition_events__random(events.copy(), k, n_chunks), dtype=event_id_dtype)
    if event_shuffle == EVE_STD_SHUFFLE:
        return partition_events__random_builtin(events.copy(), k, n_chunks)
    raise ValueError(f'unknown event shuffle rule {event_shuffle!r}')


def chunk_path(out_dir, tap, chunk_number):
    """Path of the stream file of one tap of one chunk.

    Args:
        out_dir (str | os.PathLike): output directory.
        tap (str): tap name (``gul``, ``il``, ``ri``).
        chunk_number (int): 1-based chunk number.

    Returns:
        str: ``<out_dir>/<tap>_P<chunk_number>.bin``.
    """
    return os.path.join(out_dir, f'{tap}_P{chunk_number}.bin')


class ChunkWriter:
    """Write the output streams of one chunk: ``<out_dir>/<tap>_P<k>.bin`` = stream header + the chunk's blocks.

    Every file is created and given its header on construction, so a chunk that produces no events still gets a
    header-only file, as a bash pipeline would. ``sinks`` maps each tap to its open file and can be passed as
    the ``taps`` argument of ``Chain.run_event``.

    Usage::

        with ChunkWriter(out_dir, k, chain.headers, ['gul', 'il']) as writer:
            for event_id in chunk:
                chain.run_event(event_id, writer.sinks)
    """

    def __init__(self, out_dir, chunk_number, headers, taps):
        """Open the files and write the headers.

        Args:
            out_dir (str | os.PathLike): output directory, created if missing.
            chunk_number (int): 1-based chunk number.
            headers (dict[str, bytes]): stream header of each tap.
            taps (Iterable[str]): taps to write.
        """
        os.makedirs(out_dir, exist_ok=True)
        self.chunk_number = chunk_number
        self.sinks = {}
        try:
            for tap in taps:
                f = open(chunk_path(out_dir, tap, chunk_number), 'wb')
                self.sinks[tap] = f
                f.write(headers[tap])
        except BaseException:
            self.close()
            raise

    def write(self, tap, data):
        """Append a block to the stream of one tap.

        Args:
            tap (str): tap name.
            data: bytes-like block.
        """
        self.sinks[tap].write(data)

    def close(self):
        """Close every file."""
        for f in self.sinks.values():
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
