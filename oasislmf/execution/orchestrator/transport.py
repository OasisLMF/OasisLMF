"""Turn the :class:`~.model.Stream` objects of a :class:`~.model.Stage` into OS objects.

A transport is used by the executor around ``subprocess.Popen``, once per stage::

    with get_transport() as transport:
        transport.open(stage)
        for process in stage.processes:
            proc = subprocess.Popen(transport.resolve_argv(process), **transport.popen_kwargs(process))
            transport.after_spawn(process)
        ...wait for the processes...

Mechanism on POSIX (Linux and macOS): every stream is an anonymous ``os.pipe()``. The reader
always gets the read end as stdin. A writer that takes the stream on stdout gets the write end
as stdout; a writer that takes it as a path argument inherits the write end through
``pass_fds`` and receives ``/dev/fd/<n>`` as the path, which it opens itself. Because the write
end exists from the moment the writer is spawned until it exits, the reader cannot see EOF
before the writer has opened its path (no FIFO open race), nobody ever blocks in ``open()``
(no FIFO rendezvous), and no file is ever created. This was preferred to ``mkfifo`` because a
FIFO's read end must be opened in the parent before the writer is known to have opened it, so
avoiding an early EOF needs a placeholder write end that someone must hold until the writer
exits - which is exactly what the inherited fd is, without the files.

Trade-off: since the writer holds the inherited fd for its whole life, the reader sees EOF when
the writer process exits (or closes every copy), not when it closes the file it opened by path.
A process only reads its stdin, so the stream graph of a stage has no cycles and this delay
cannot deadlock it. A writer that hands the fd on to its own children (``close_fds=False``)
extends the delay to their lifetime.

Opening ``/dev/fd/<n>`` re-opens the pipe on Linux (``/proc/self/fd`` magic link) and
duplicates fd ``n`` on macOS (fdesc); for a pipe either gives a writable fd on the same
pipe, and the same mechanism underlies bash's process substitution on both systems.
"""

import os
import subprocess
import sys

from .model import Stream, StreamPath


class Transport:
    """Interface for the stream transport of one stage; also a context manager closing it.

    Call order: :meth:`open` once, then for each process :meth:`resolve_argv` and
    :meth:`popen_kwargs`, ``Popen``, :meth:`after_spawn`; finally :meth:`close` (also on error).
    """

    def open(self, stage):
        """Validate ``stage`` and get ready to serve its processes.

        Directories named by file endpoints must exist before a process using them is prepared.

        Args:
            stage: the :class:`~.model.Stage` about to be started.
        """
        raise NotImplementedError

    def popen_kwargs(self, process):
        """Return the ``stdin``/``stdout``/``pass_fds``/``close_fds`` keyword arguments for ``Popen``.

        stderr is left to the caller.

        Args:
            process: a process of the opened stage.
        """
        raise NotImplementedError

    def resolve_argv(self, process):
        """Return ``process.argv`` with every :class:`~.model.StreamPath` replaced by a real path.

        Args:
            process: a process of the opened stage.
        """
        raise NotImplementedError

    def after_spawn(self, process):
        """Close the parent's copies of the endpoints that the spawned ``process`` now owns.

        Args:
            process: the process just started successfully.
        """
        raise NotImplementedError

    def close(self):
        """Release every OS object still held; safe to call more than once and after a partial spawn."""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _move_above_stdio(fd):
    """Return a descriptor for ``fd``'s file numbered 3 or more, so ``pass_fds`` cannot clash with stdio."""
    low = []
    while fd < 3:
        low.append(fd)
        fd = os.dup(fd)
    for old in low:
        os.close(old)
    return fd


class PosixTransport(Transport):
    """Anonymous pipes, with ``/dev/fd/<n>`` for writers that take a path (see module docstring).

    A stream's pipe is created when the first of its two processes is prepared and each end is
    closed in the parent as soon as its process has been spawned, so the parent only holds
    descriptors for streams that are half connected. Spawning the processes of one connected
    group of streams before starting the next keeps that number small however many groups the
    stage has.
    """

    def __init__(self):
        self._read_fd = {}
        self._write_fd = {}
        self._created = set()
        self._owned = {}

    def open(self, stage):
        """See :meth:`Transport.open`."""
        stage.validate()

    def _ensure_pipe(self, stream):
        if stream.name not in self._created:
            self._read_fd[stream.name], self._write_fd[stream.name] = os.pipe()
            self._created.add(stream.name)

    def _prepare(self, process):
        """Create what ``process`` needs and return ``(stdin, stdout, {stream name: path fd})``."""
        key = id(process)
        if key in self._owned:
            return self._owned[key]
        owned = [subprocess.DEVNULL, subprocess.DEVNULL, {}]
        self._owned[key] = owned
        for slot, endpoint in enumerate((process.stdin, process.stdout)):
            if isinstance(endpoint, Stream):
                self._ensure_pipe(endpoint)
                owned[slot] = (self._read_fd if slot == 0 else self._write_fd).pop(endpoint.name)
            elif isinstance(endpoint, str):
                flags = os.O_RDONLY if slot == 0 else os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                owned[slot] = os.open(endpoint, flags, 0o666)
        for arg in process.argv:
            if isinstance(arg, StreamPath):
                self._ensure_pipe(arg.stream)
                owned[2][arg.stream.name] = _move_above_stdio(self._write_fd.pop(arg.stream.name))
        return owned

    def popen_kwargs(self, process):
        """See :meth:`Transport.popen_kwargs`."""
        stdin, stdout, path_fds = self._prepare(process)
        return {'stdin': stdin, 'stdout': stdout, 'pass_fds': tuple(path_fds.values()), 'close_fds': True}

    def resolve_argv(self, process):
        """See :meth:`Transport.resolve_argv`."""
        path_fds = self._prepare(process)[2]
        return [f'/dev/fd/{path_fds[arg.stream.name]}' if isinstance(arg, StreamPath) else arg for arg in process.argv]

    @staticmethod
    def _close_owned(owned):
        stdin, stdout, path_fds = owned
        for fd in [stdin, stdout, *path_fds.values()]:
            if fd != subprocess.DEVNULL:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def after_spawn(self, process):
        """See :meth:`Transport.after_spawn`."""
        owned = self._owned.pop(id(process), None)
        if owned:
            self._close_owned(owned)

    def close(self):
        """See :meth:`Transport.close`."""
        for owned in self._owned.values():
            self._close_owned(owned)
        self._owned.clear()
        for table in (self._read_fd, self._write_fd):
            for fd in table.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            table.clear()


class WindowsTransport(Transport):
    r"""Not implemented yet.

    Intended approach: each stream is a named pipe ``\\.\pipe\oasis-<uuid>-<stream>`` created
    by the parent with ``CreateNamedPipe`` (overlapped connect so the parent never blocks). The
    server end, once connected, is the reader's stdin handle. A writer taking stdout gets a
    client handle opened by the parent; a writer taking a path receives the pipe name and opens
    it itself as a client, which is what ``open(path, 'wb')`` does on Windows. Handles are passed
    with ``STARTUPINFO.lpAttributeList`` handle lists rather than ``pass_fds``.
    """

    def __init__(self):
        raise NotImplementedError("the orchestrator stream transport is not implemented on Windows yet")


def get_transport():
    """Return a new :class:`Transport` for the current operating system."""
    if sys.platform == 'win32':
        return WindowsTransport()
    return PosixTransport()
