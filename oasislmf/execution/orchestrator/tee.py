"""``teepy``: a portable ``tee`` for the kernel's data path.

Copies stdin, byte for byte, to every output given on the command line. An output is a path
opened for binary write, an inherited stream reference (see
``oasislmf.pytools.common.stream_ref``), or ``-`` for teepy's own stdout. All outputs are opened
before the first read, so a FIFO output applies back-pressure until its reader is there.

Unlike GNU ``tee``, which by default dies silently on SIGPIPE, a write error on any output
(typically its reader having exited) stops teepy with a message naming that output and a
non-zero exit status: in the kernel, a consumer disappearing must fail the run.

Only portable calls are used (no ``select``, threads or POSIX-only APIs) so the same code runs
on Linux, macOS and Windows.

Usage::

    teepy OUT [OUT ...]
    python -m oasislmf.execution.orchestrator.tee OUT [OUT ...]
"""

import argparse
import os
import signal
import sys

from oasislmf.pytools.common.stream_ref import open_stream_out

BUFFER_SIZE = 65536
"""Default read size in bytes; matches ``PIPE_CAPACITY`` in ``oasislmf.pytools.common.event_stream``."""

STDOUT = '-'


class TeeError(Exception):
    """An output could not be opened or written, or the input could not be read."""


def _open_output(output):
    """Return ``(name, raw binary file, owned)`` for one output spec."""
    if output == STDOUT:
        return '<stdout>', open(sys.stdout.fileno(), 'wb', buffering=0, closefd=False), True
    if isinstance(output, (str, bytes, os.PathLike)):
        try:
            return os.fsdecode(output), open_stream_out(output, buffering=0), True
        except OSError as e:
            raise TeeError(f"cannot open output '{os.fsdecode(output)}': {e.strerror or e}") from e
    return getattr(output, 'name', repr(output)), output, False


def _write_all(stream, view):
    """Write the whole of ``view`` to ``stream``, looping on short writes."""
    while view:
        written = stream.write(view)
        if written is None:
            written = 0
        view = view[written:]


def run(outputs, stream_in=None, buffer_size=BUFFER_SIZE):
    """Copy ``stream_in`` to every output until EOF.

    Args:
        outputs: output specs; each is a path to open for binary write, ``'-'`` for this
            process's stdout, or an already-open binary file object (written to but not closed).
        stream_in: binary file object supporting ``readinto``; defaults to unbuffered stdin.
        buffer_size: size of the reusable read buffer in bytes.

    Raises:
        TeeError: an output could not be opened or written, or the input could not be read.
            The message names the output concerned.
    """
    if stream_in is None:
        stream_in = open(sys.stdin.fileno(), 'rb', buffering=0, closefd=False)
    opened = []
    try:
        for output in outputs:
            opened.append(_open_output(output))
        buffer = memoryview(bytearray(buffer_size))
        while True:
            try:
                n = stream_in.readinto(buffer)
            except OSError as e:
                raise TeeError(f"error reading input: {e.strerror or e}") from e
            if not n:
                break
            chunk = buffer[:n]
            for name, stream, _ in opened:
                try:
                    _write_all(stream, chunk)
                except OSError as e:
                    raise TeeError(f"error writing to '{name}': {e.strerror or e}") from e
        for name, stream, _ in opened:
            try:
                stream.flush()
            except OSError as e:
                raise TeeError(f"error writing to '{name}': {e.strerror or e}") from e
    finally:
        for name, stream, owned in opened:
            if owned:
                try:
                    stream.close()
                except OSError:
                    pass


def main(argv=None):
    """Console entry point for ``teepy``; exits 0 on clean EOF, 1 on any error."""
    if hasattr(signal, 'SIGPIPE'):
        # a broken output must surface as BrokenPipeError (and a message), not a silent death by signal
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    parser = argparse.ArgumentParser(prog='teepy', description='Copy stdin to every OUT; "-" is stdout.')
    parser.add_argument('outputs', nargs='+', metavar='OUT', help="path to write, or '-' for stdout")
    args = parser.parse_args(argv)
    try:
        run(args.outputs)
    except TeeError as e:
        sys.stderr.write(f"teepy: {e}\n")
        sys.stderr.flush()
        sys.exit(1)


if __name__ == '__main__':
    main()
