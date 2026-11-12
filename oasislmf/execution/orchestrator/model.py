"""Data model for a kernel run described as processes connected by streams.

A run is a list of :class:`Stage`. The processes of a stage are started together and the stage
is complete when all of them have exited; stages run one after the other. Nothing in this module
starts a process or touches the filesystem: the builders in ``build.py`` produce these objects
and the executor in ``executor.py`` runs them.

Every :class:`Stream` has exactly one writer and one reader. A process is attached to a stream
either through its stdin/stdout, or by receiving a path in its argument list (a
:class:`StreamPath` placeholder, resolved to a real path by the transport when the stage
starts). A stream may be referenced through :class:`StreamPath` by its writer only; the reader
of a stream always receives it as stdin.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

STDERR_LOG = 'stderror.err'
GUL_STDERR_LOG = 'gul_stderror.err'


@dataclass(frozen=True)
class Stream:
    """A one-writer, one-reader byte stream between two processes of the same stage.

    Attributes:
        name: unique name within the stage, e.g. ``gul_S1_summary_P1``.
    """
    name: str


@dataclass(frozen=True)
class StreamPath:
    """Placeholder in an argument list for the path the writer of ``stream`` must open."""
    stream: Stream


Arg = Union[str, StreamPath]
Endpoint = Union[Stream, str, None]


@dataclass
class Process:
    """One process of the run.

    Attributes:
        name: tool name used for logging and reporting, e.g. ``gulmc``.
        argv: the command, as a list of strings and :class:`StreamPath` placeholders.
        stdin: a :class:`Stream` to read, a file path to read, or None to inherit nothing.
        stdout: a :class:`Stream` to write, a file path to write, or None to discard.
        stderr_log: file name under the run's log directory that stderr is appended to, or
            None to leave stderr unguarded.
    """
    name: str
    argv: List[Arg]
    stdin: Endpoint = None
    stdout: Endpoint = None
    stderr_log: Optional[str] = STDERR_LOG

    def streams_written(self):
        """Streams this process writes, through stdout or a path argument."""
        streams = [arg.stream for arg in self.argv if isinstance(arg, StreamPath)]
        if isinstance(self.stdout, Stream):
            streams.append(self.stdout)
        return streams

    def streams_read(self):
        """Streams this process reads (stdin only)."""
        return [self.stdin] if isinstance(self.stdin, Stream) else []


@dataclass
class Stage:
    """Processes started together and waited on together.

    Attributes:
        name: label used in logs and in the run manifest.
        processes: the processes, in the order they should be started (readers first).
        make_dirs: directories to create before the processes start, after ``remove_paths``
            and ``clean_output_dir`` have been applied.
        remove_paths: glob patterns to delete before the processes start.
        cleanup_paths: glob patterns to delete after the stage succeeds.
        clean_output_dir: directory whose files are deleted, recursively, before the processes
            start, except files whose name contains ``summary-info`` or ends in ``.json``
            (directories themselves are kept); None to leave it alone.
    """
    name: str
    processes: List[Process] = field(default_factory=list)
    make_dirs: List[str] = field(default_factory=list)
    remove_paths: List[str] = field(default_factory=list)
    cleanup_paths: List[str] = field(default_factory=list)
    clean_output_dir: Optional[str] = None

    def streams(self):
        """Every stream referenced by the stage's processes, in first-use order."""
        seen = {}
        for process in self.processes:
            for stream in process.streams_read() + process.streams_written():
                seen.setdefault(stream.name, stream)
        return list(seen.values())

    def validate(self):
        """Raise ValueError unless every stream has exactly one writer and one reader."""
        writers, readers = {}, {}
        for process in self.processes:
            for stream in process.streams_written():
                writers.setdefault(stream.name, []).append(process.name)
            for stream in process.streams_read():
                readers.setdefault(stream.name, []).append(process.name)
        problems = []
        for name in sorted(set(writers) | set(readers)):
            if len(writers.get(name, [])) != 1 or len(readers.get(name, [])) != 1:
                problems.append(f"{name}: writers={writers.get(name, [])} readers={readers.get(name, [])}")
        if problems:
            raise ValueError(f"stage '{self.name}' has streams without exactly one writer and one reader: " + '; '.join(problems))
