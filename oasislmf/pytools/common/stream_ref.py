"""Inherited stream references: how a parent hands an already-open pipe end to a tool by "path".

Some tools write to outputs named on their command line (``summarypy -1 <path>``, ``fmpy -o
<path>``, ``teepy <path>...``). The orchestrator's transport gives such a writer the write end of
an anonymous pipe, inherited from the parent, and passes a *reference* to it where the path goes:

* ``<oasis-fd>/<n>``: file descriptor ``n`` inherited through ``pass_fds`` (POSIX);
* ``<oasis-handle>/<n>``: OS handle value ``n`` inherited through a ``STARTUPINFO`` handle list
  (Windows).

Neither spelling can be opened as a file by mistake: ``<`` and ``>`` are invalid in Windows file
names, and on POSIX the directory ``<oasis-fd>`` does not exist, so a tool that forgot to use
:func:`open_stream_out` fails loudly instead of writing to a stray file.

This module is imported by ``teepy`` at start-up, so it must stay cheap to import (no numpy or
numba).
"""

import os
import re

if os.name == 'nt':
    import msvcrt
else:
    msvcrt = None

FD_KIND = 'fd'
HANDLE_KIND = 'handle'

_REFERENCE = re.compile(r'<oasis-(fd|handle)>/(\d+)')


def stream_reference(kind, number):
    """Return the reference string for an inherited descriptor or handle.

    Args:
        kind (str): :data:`FD_KIND` or :data:`HANDLE_KIND`.
        number (int): the file descriptor or handle value the child inherits.
    """
    if kind not in (FD_KIND, HANDLE_KIND):
        raise ValueError(f'unknown stream reference kind {kind!r}')
    return f'<oasis-{kind}>/{int(number)}'


def parse_stream_reference(path):
    """Return ``(kind, number)`` if ``path`` is a stream reference, else None."""
    if not isinstance(path, str):
        return None
    match = _REFERENCE.fullmatch(path)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def open_stream_out(path, buffering=-1):
    """Open an output for binary writing: a real path, or an inherited stream reference.

    For a normal path this is exactly ``open(path, 'wb', buffering=buffering)``. For a reference
    the inherited descriptor (POSIX) or handle (Windows) is wrapped without being re-opened, so
    closing the returned file closes the write end and the reader sees EOF at that point.

    Args:
        path: file path, or a reference made by :func:`stream_reference`.
        buffering (int): as for :func:`open`.

    Returns:
        A binary file object open for writing.

    Raises:
        ValueError: ``path`` is a handle reference and this is not Windows.
    """
    reference = parse_stream_reference(path)
    if reference is None:
        return open(path, 'wb', buffering=buffering)
    kind, number = reference
    if kind == HANDLE_KIND:
        if msvcrt is None:
            raise ValueError(f'stream reference {path!r} names a Windows handle, which cannot be used on this OS')
        # flags 0: a binary, writable C runtime descriptor, as subprocess uses for its own pipes
        number = msvcrt.open_osfhandle(number, 0)
    return os.fdopen(number, 'wb', buffering=buffering)
