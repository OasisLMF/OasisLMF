"""Turn the :class:`~.model.Stream` objects of a :class:`~.model.Stage` into OS objects.

A transport is used by the executor around ``subprocess.Popen``, once per stage::

    with get_transport() as transport:
        transport.open(stage)
        for process in stage.processes:
            proc = subprocess.Popen(transport.resolve_argv(process), **transport.popen_kwargs(process))
            transport.after_spawn(process)
        ...wait for the processes...

Mechanism (every OS): every stream is an anonymous ``os.pipe()``. The reader always gets the
read end as stdin. A writer that takes the stream on stdout gets the write end as stdout; a
writer that takes it as a path argument inherits the write end itself and receives an
*inherited stream reference* in place of the path (``<oasis-fd>/<n>`` on POSIX,
``<oasis-handle>/<n>`` on Windows, see :mod:`oasislmf.pytools.common.stream_ref`), which it
wraps with :func:`~oasislmf.pytools.common.stream_ref.open_stream_out` instead of opening a
file. Because the write end exists from the moment the writer is spawned, the reader cannot see
EOF before the writer has started writing (no FIFO open race), nobody ever blocks in ``open()``
(no FIFO rendezvous), and no file is ever created. This was preferred to ``mkfifo`` because a
FIFO's read end must be opened in the parent before the writer is known to have opened it, so
avoiding an early EOF needs a placeholder write end that someone must hold until the writer
exits - which is exactly what the inherited end is, without the files.

The writer wraps the inherited descriptor rather than re-opening it, so closing the file closes
the only write end and the reader sees EOF then, even if the writer keeps running. A writer that
never wraps the reference holds the end until it exits. A writer that hands it on to its own
children (``close_fds=False``) extends that to their lifetime.

Only how a path-writer inherits its end, and how that end is named, differs between operating
systems: see :class:`PosixTransport` and :class:`WindowsTransport`.
"""

import os
import subprocess
import sys

from oasislmf.pytools.common.stream_ref import FD_KIND, HANDLE_KIND, stream_reference

from .model import Stream, StreamPath

if os.name == 'nt':
    import msvcrt
else:
    msvcrt = None


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
        """Return the ``stdin``/``stdout``/``close_fds`` keyword arguments for ``Popen``, plus how path ends are inherited.

        That is ``pass_fds`` on POSIX and, on Windows, ``startupinfo`` when the process takes a path.

        stderr is left to the caller.

        Args:
            process: a process of the opened stage.
        """
        raise NotImplementedError

    def resolve_argv(self, process):
        """Return ``process.argv`` with every :class:`~.model.StreamPath` replaced by an inherited stream reference.

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


class PipeTransport(Transport):
    """Anonymous pipes created lazily; the part shared by every operating system.

    A stream's pipe is created when the first of its two processes is prepared and each end is
    closed in the parent as soon as its process has been spawned, so the parent only holds
    descriptors for streams that are half connected. Spawning the processes of one connected
    group of streams before starting the next keeps that number small however many groups the
    stage has.

    Subclasses define how a writer inherits the write end of a stream it takes as a path
    (:meth:`_adopt_path_end`, :meth:`_path_kwargs`) and how that end is named (:meth:`_reference`).
    """

    def __init__(self):
        self._read_fd = {}
        self._write_fd = {}
        self._created = set()
        self._owned = {}

    def open(self, stage):
        """See :meth:`Transport.open`."""
        stage.validate()

    def _adopt_path_end(self, fd):
        """Make the pipe write end ``fd`` inheritable by a path-writer; return the descriptor to keep."""
        raise NotImplementedError

    def _reference(self, fd):
        """Return the reference string the path-writer receives for the kept descriptor ``fd``."""
        raise NotImplementedError

    def _path_kwargs(self, fds):
        """Return the extra ``Popen`` keyword arguments that make the child inherit ``fds``."""
        raise NotImplementedError

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
                owned[2][arg.stream.name] = self._adopt_path_end(self._write_fd.pop(arg.stream.name))
        return owned

    def popen_kwargs(self, process):
        """See :meth:`Transport.popen_kwargs`."""
        stdin, stdout, path_fds = self._prepare(process)
        return {'stdin': stdin, 'stdout': stdout, 'close_fds': True, **self._path_kwargs(list(path_fds.values()))}

    def resolve_argv(self, process):
        """See :meth:`Transport.resolve_argv`."""
        path_fds = self._prepare(process)[2]
        return [self._reference(path_fds[arg.stream.name]) if isinstance(arg, StreamPath) else arg for arg in process.argv]

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


class PosixTransport(PipeTransport):
    """Path-writers inherit their write end through ``pass_fds`` and receive ``<oasis-fd>/<n>``.

    The descriptor is moved to 3 or above so it cannot clash with the child's stdio.
    """

    def _adopt_path_end(self, fd):
        return _move_above_stdio(fd)

    def _reference(self, fd):
        return stream_reference(FD_KIND, fd)

    def _path_kwargs(self, fds):
        return {'pass_fds': tuple(fds)}


class WindowsTransport(PipeTransport):
    """Path-writers inherit their write handle through a ``STARTUPINFO`` handle list and receive ``<oasis-handle>/<n>``.

    ``n`` is the parent's handle value, which an inherited handle keeps in the child. The handle
    is made inheritable with ``os.set_handle_inheritable`` (``os.pipe`` creates non-inheritable
    ends) and listed in ``lpAttributeList['handle_list']``, which ``CreateProcess`` requires to
    hold inheritable handles. This relies on CPython's Windows ``Popen._execute_child``
    (``subprocess.py`` of 3.12, lines 1480-1507 and 1539-1554):

    * with ``close_fds=True`` and the std handles in use, ``Popen`` appends the stdin, stdout and
      stderr handles to the caller's ``handle_list``, de-duplicates it, and creates the process
      with ``bInheritHandles`` true, so the child inherits exactly the listed handles and no
      other inheritable handle of the parent (another stream's end prepared but not yet
      spawned, for instance);
    * stdin/stdout given as descriptors are turned into handles by ``_get_handles`` (lines
      1361-1365 and 1381-1385): ``msvcrt.get_osfhandle`` then ``_make_inheritable``, a
      ``DuplicateHandle`` that ``_close_pipe_fds`` closes after ``CreateProcess``. ``Popen``
      never takes over the caller's descriptors, so the parent closes its own copies in
      :meth:`after_spawn` as on POSIX; the child's inherited copies are independent.

    ``startupinfo`` is only passed when the process takes a path, so the other processes are
    created exactly as ``Popen`` does by default.
    """

    def _adopt_path_end(self, fd):
        os.set_handle_inheritable(msvcrt.get_osfhandle(fd), True)
        return fd

    def _reference(self, fd):
        return stream_reference(HANDLE_KIND, msvcrt.get_osfhandle(fd))

    def _path_kwargs(self, fds):
        if not fds:
            return {}
        handles = [msvcrt.get_osfhandle(fd) for fd in fds]
        return {'startupinfo': subprocess.STARTUPINFO(lpAttributeList={'handle_list': handles})}


def get_transport():
    """Return a new :class:`Transport` for the current operating system."""
    if sys.platform == 'win32':
        return WindowsTransport()
    return PosixTransport()
