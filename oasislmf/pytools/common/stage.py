"""In-memory building blocks shared by the per-event stages of the kernel tools.

A *block* is a 1-D numpy ``int8`` array holding one or more whole events of a loss stream, without the
stream header. A *sink* is any object with ``write(data)`` accepting a buffer; ``EventBlock`` is a sink that
keeps what it is given in memory so that the next stage can read it as a block.
"""
import numpy as np

from oasislmf.pytools.common.event_stream import PIPE_CAPACITY, write_mv_to_stream


class EventBlock:
    """Growable byte buffer holding whole events.

    ``write`` appends bytes, growing the buffer by doubling; ``view`` returns an ``int8`` view of the bytes
    written so far, without copying.
    """

    __slots__ = ('_buf', 'size')

    def __init__(self, capacity=2 * PIPE_CAPACITY):
        """Create an empty block.

        Args:
            capacity (int): initial capacity in bytes.
        """
        self._buf = np.empty(max(int(capacity), 1), dtype=np.int8)
        self.size = 0
        """int: number of bytes currently held."""

    @property
    def capacity(self):
        """int: number of bytes the block can hold before it grows."""
        return self._buf.shape[0]

    @staticmethod
    def _as_bytes(data):
        """Return ``data`` as a 1-D C-contiguous ``int8`` array (a copy only when ``data`` is not contiguous)."""
        if not isinstance(data, np.ndarray):
            data = np.asarray(memoryview(data))
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)
        return data.reshape(-1).view(np.int8)

    def _reserve(self, needed):
        """Grow the buffer, by doubling, until it holds at least ``needed`` bytes."""
        capacity = self._buf.shape[0]
        if needed <= capacity:
            return
        while capacity < needed:
            capacity *= 2
        new_buf = np.empty(capacity, dtype=np.int8)
        new_buf[:self.size] = self._buf[:self.size]
        self._buf = new_buf

    def write(self, data):
        """Append bytes to the block.

        Args:
            data: ``bytes``, ``bytearray``, ``memoryview``, numpy array or any object supporting the buffer protocol.
                Its raw bytes are appended in C order; non-contiguous arrays and memoryviews are handled.

        Returns:
            int: the number of bytes appended.
        """
        src = self._as_bytes(data)
        n = src.shape[0]
        if n:
            end = self.size + n
            self._reserve(end)
            self._buf[self.size:end] = src
            self.size = end
        return n

    def view(self):
        """Return the bytes held, as an ``int8`` view (no copy).

        The view is invalid after the next ``write`` or ``clear``.

        Returns:
            np.ndarray: 1-D ``int8`` array of length ``size``.
        """
        return self._buf[:self.size]

    def tobytes(self):
        """Return a copy of the bytes held, as ``bytes``."""
        return self._buf[:self.size].tobytes()

    def clear(self):
        """Empty the block, keeping its capacity."""
        self.size = 0

    def __len__(self):
        return self.size


class StreamSink:
    """Sink writing to a real stream (stdout, a pipe, a fifo or a file) with ``write_mv_to_stream``.

    ``write_mv_to_stream`` blocks until the reader takes the bytes and loops on partial writes, so a slow
    consumer applies back-pressure.
    """

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        """Write a numpy byte view to the stream in full.

        Args:
            data (np.ndarray): 1-D int8 array to write.
        """
        write_mv_to_stream(self.stream, data, data.shape[0])
