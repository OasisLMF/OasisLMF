"""Splitting the events into chunks, and writing each chunk's output streams.

Chunk k (1-based) of K holds exactly the events ``evepy k K`` emits, so a run in K chunks can be compared
byte for byte with a bash run in K partitions.
"""
import os

import numpy as np

from oasislmf.pytools.eve.manager import event_id_dtype, partition_events__round_robin, read_events

__all__ = ['make_chunks', 'read_events', 'chunk_path', 'ChunkWriter']


def make_chunks(events, n_chunks):
    """Split events into chunks, as ``evepy`` partitions them by default (round robin).

    Args:
        events (array-like): event ids, in the order they are held in ``events.bin``.
        n_chunks (int): number of chunks.

    Returns:
        list[np.ndarray]: ``n_chunks`` arrays of event ids; element ``k - 1`` holds exactly the events
        ``evepy k n_chunks`` emits, in the same order.
    """
    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive, got {n_chunks}")
    events = np.asarray(events, dtype=event_id_dtype)
    return [partition_events__round_robin(events, k, n_chunks) for k in range(1, n_chunks + 1)]


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
