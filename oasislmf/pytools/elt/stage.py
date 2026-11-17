"""Event loss table stage: set up once, then append the ELT rows of one block of events at a time.

``EltStage`` holds everything ``eltpy`` builds before its event loop (event rates, quantile
intervals, the ``ELTReader`` and its row buffers) and writes the rows of the events handed to
it in memory into per-chunk output files. The ``eltpy`` command is built on it (see
``manager.run``), so both share one implementation of the per-event work.

``TableStage`` is the part shared with ``PltStage`` (``oasislmf.pytools.plt.stage``): header
validation, the per-chunk output files and the event loop that drains the reader's row buffers.

The reader, its Numba kernels and ``read_input_files`` stay in ``manager``, where tests and
``oasislmf.warmup`` patch them; this module looks them up through the ``manager`` module at
call time so that those patches still apply.
"""
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from oasislmf.pytools.common.data import write_ndarray_to_fmt_csv
from oasislmf.pytools.common.event_stream import STREAM_HEADER_SIZE, SUMMARY_STREAM_ID, bytes_to_stream_types, stream_info_to_bytes
from oasislmf.pytools.common.parquet import BufferedParquetWriter
from oasislmf.pytools.elt import manager as elt_manager
from oasislmf.pytools.elt.data import MELT_fmt, MELT_headers, QELT_fmt, QELT_headers, SELT_fmt, SELT_headers

SUMMARY_STREAM_HEADER_SIZE = 12


def summary_stream_header(stream_source_type, stream_agg_type, len_sample, summary_set_id=None):
    """Build the header of a summary stream.

    Args:
        stream_source_type (int): stream type (``SUMMARY_STREAM_ID`` for a summary stream).
        stream_agg_type (int): stream aggregation type.
        len_sample (int): sample count.
        summary_set_id (int, optional): summary set id; the 8-byte header ``init_streams_in`` reads
            is returned when None.

    Returns:
        bytes: the 8-byte stream header, or the 12-byte summary stream header summarypy writes.
    """
    header = stream_info_to_bytes(stream_source_type, stream_agg_type) + np.array([len_sample], '<i4').tobytes()
    if summary_set_id is not None:
        header += np.array([summary_set_id], '<i4').tobytes()
    return header


def parse_summary_header(input_header):
    """Validate the header of a summary stream as eltpy and pltpy do.

    Args:
        input_header (bytes): the 12-byte summary stream header summarypy writes (stream type,
            sample count, summary set id), or its first 8 bytes.

    Returns:
        tuple: ``(len_sample, summary_set_id)``; summary_set_id is None for an 8-byte header.

    Raises:
        ValueError: the header is neither 8 nor 12 bytes long.
        Exception: the stream is not a summary stream (same message as the tools).
    """
    input_header = bytes(input_header)
    if len(input_header) not in (STREAM_HEADER_SIZE, SUMMARY_STREAM_HEADER_SIZE):
        raise ValueError(f"summary stream header must be {STREAM_HEADER_SIZE} or {SUMMARY_STREAM_HEADER_SIZE} bytes, "
                         f"got {len(input_header)}")
    stream_source_type, stream_agg_type = bytes_to_stream_types(input_header[:4])
    len_sample = np.frombuffer(input_header[4:8], dtype=np.int32)[0]
    if stream_source_type != SUMMARY_STREAM_ID:
        raise Exception(f"unsupported stream type {stream_source_type}, {stream_agg_type}")
    summary_set_id = None
    if len(input_header) == SUMMARY_STREAM_HEADER_SIZE:
        summary_set_id = int(np.frombuffer(input_header[8:], dtype=np.int32)[0])
    return len_sample, summary_set_id


def check_output_paths(paths, output_format):
    """Refuse output paths whose extension does not match the output format, as the tools do.

    Paths without an extension (pipes) are accepted.

    Args:
        paths: iterable of output paths, None entries are skipped.
        output_format (str): output format without the dot ('csv', 'bin' or 'parquet').

    Raises:
        ValueError: a path has another extension.
    """
    output_format = "." + output_format
    for path in paths:
        if path is None:
            continue
        if Path(path).suffix == "":
            continue
        if Path(path).suffix != output_format:
            raise ValueError(f"Invalid file extension for {output_format}, got {path},")


class TableStage:
    """Shared body of ``EltStage`` and ``PltStage``: a summary stream reader writing table rows to per-chunk files.

    Subclasses set ``out_types``, ``headers`` and ``fmts`` and implement ``_make_reader``.

    The reader used for blocks keeps one state row (blocks hold no summary set id, which is in
    the stream header). That state is clean at every event boundary: every per-summary field is
    cleared when a summary's closing record is read, and the only thing left behind is the id of
    the last event, which makes the next block's first read yield that id once more with no rows.
    ``start_chunk`` nevertheless restores the reader's initial state, so a chunk abandoned in the
    middle of an event cannot leak into the next.

    Attributes:
        len_sample (int): sample count of the input stream.
        summary_set_id (int | None): summary set id from a 12-byte header, else None.
        reader (EventReader): the reader blocks are parsed with.
    """

    out_types = ()
    headers = {}
    fmts = {}

    def __init__(self, run_dir, input_header, computes, output_format='bin', noheader=False):
        """Validate the header and set up the output formats; subclasses then call ``_init_reader``.

        Args:
            run_dir (str | os.PathLike | None): run directory (input files are read from its ``input``
                folder); None means './work', as in the tools.
            input_header (bytes): summary stream header, see ``parse_summary_header``.
            computes (dict): out_type -> bool, which tables are produced.
            output_format (str): 'bin', 'csv' or 'parquet'.
            noheader (bool): do not write the csv header line.
        """
        self.len_sample, self.summary_set_id = parse_summary_header(input_header)
        self.run_dir = './work' if run_dir is None else run_dir
        self.output_format = output_format
        self.output_binary = output_format == "bin"
        self.output_parquet = output_format == "parquet"
        self.noheader = noheader
        self.computes = {out_type: bool(computes[out_type]) for out_type in self.out_types}
        self.reader = None
        self._fresh_state = None
        self._stack = None
        self._files = {}
        self._closed = False

    def _make_reader(self, n_files):
        """Return a new reader with ``n_files`` state rows (implemented by subclasses)."""
        raise NotImplementedError

    def _init_reader(self):
        """Build the block reader once the input files are loaded."""
        self.reader = self._make_reader(1)
        self.reader.state["read_summary_set_id"] = True
        self._fresh_state = self.reader.state.copy()

    @property
    def active_types(self):
        """list: the out_types this stage produces, in output order."""
        return [out_type for out_type in self.out_types if self.computes[out_type]]

    def reset(self):
        """Restore the reader's initial state and empty its row buffers."""
        self.reader.state[:] = self._fresh_state
        for out_type in self.out_types:
            self.reader.get_data_idx(out_type)[0] = 0

    def start_chunk(self, paths):
        """Open this chunk's output files and write their headers as the tool does.

        Args:
            paths (dict): out_type -> output path, one entry for each table the stage produces.
        """
        if self._closed:
            raise ValueError("stage is closed")
        if self._stack is not None:
            raise ValueError("a chunk is already open, call finish_chunk first")
        paths = dict(paths)
        if set(paths) != set(self.active_types):
            raise ValueError(f"paths must be given for exactly {self.active_types}, got {sorted(paths)}")
        check_output_paths(paths.values(), self.output_format)
        self.reset()
        self._open(paths, self.reader)

    def _open(self, paths, reader):
        """Open the output files in ``paths`` (out_type -> path), writing csv headers as the tool does."""
        stack = ExitStack()
        files = {}
        try:
            for out_type in self.out_types:
                if out_type not in paths:
                    continue
                if self.output_binary:
                    files[out_type] = stack.enter_context(open(paths[out_type], 'wb'))
                elif self.output_parquet:
                    dtype = reader.get_data(out_type).dtype
                    files[out_type] = stack.enter_context(BufferedParquetWriter(paths[out_type], dtype))
                else:
                    out_file = stack.enter_context(open(paths[out_type], 'w'))
                    if not self.noheader:
                        out_file.write(','.join(self.headers[out_type]) + '\n')
                    files[out_type] = out_file
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        self._files = files

    def process_block(self, block):
        """Append the rows of each event of an in-memory block to the open files.

        Args:
            block (np.ndarray): 1-D int8 array holding one or more whole events of the summary
                stream, without the stream header. It may be read-only.
        """
        if self._stack is None:
            raise ValueError("no chunk is open, call start_chunk first")
        self._write_events(self.reader, self.reader.read_block(block))

    def process_streams(self, streams_in):
        """Append the rows of every event of open summary streams, as the tool reads them.

        The streams are positioned after the 8 bytes ``init_streams_in`` reads, so each still starts
        with its summary set id; they are read with a reader of their own.

        Args:
            streams_in (list): binary streams, as returned by ``init_streams_in``.
        """
        if self._stack is None:
            raise ValueError("no chunk is open, call start_chunk first")
        reader = self._make_reader(len(streams_in))
        self._write_events(reader, reader.read_streams(streams_in))

    def _write_events(self, reader, events):
        """Drain the reader's row buffers to the open files after each event it yields."""
        files = self._files
        for _ in events:
            for out_type in self.out_types:
                if not self.computes[out_type]:
                    continue
                data_idx = reader.get_data_idx(out_type)
                data = reader.get_data(out_type)[:data_idx[0]]
                out_file = files.get(out_type)
                if out_file is not None and data.size > 0:
                    if self.output_binary:
                        data.tofile(out_file)
                    elif self.output_parquet:
                        out_file.write(data)
                    else:
                        write_ndarray_to_fmt_csv(out_file, data, self.headers[out_type], self.fmts[out_type])
                data_idx[0] = 0

    def finish_chunk(self):
        """Flush and close this chunk's output files."""
        if self._stack is None:
            raise ValueError("no chunk is open, call start_chunk first")
        stack, self._stack, self._files = self._stack, None, {}
        stack.close()

    def close(self):
        """Close any chunk still open. The stage cannot be used afterwards."""
        if self._closed:
            return
        self._closed = True
        if self._stack is not None:
            stack, self._stack, self._files = self._stack, None, {}
            stack.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class EltStage(TableStage):
    """``eltpy`` set up once and fed summary blocks in memory, writing SELT, MELT and QELT files per chunk.

    Paths resolve as in ``eltpy``: ``run_dir`` and the output paths are relative to the working
    directory.

    Args:
        run_dir (str | os.PathLike | None): run directory; event rates and quantiles are read from
            its ``input`` folder. None means './work'.
        input_header (bytes): the 12-byte summary stream header summarypy writes, or its first 8
            bytes. It is validated as eltpy validates its input and gives the sample count.
        selt (bool): produce the sample ELT.
        melt (bool): produce the moment ELT.
        qelt (bool): produce the quantile ELT.
        output_format (str): 'bin', 'csv' or 'parquet'.
        noheader (bool): do not write the csv header line.
    """

    out_types = ("selt", "melt", "qelt")
    headers = {"selt": SELT_headers, "melt": MELT_headers, "qelt": QELT_headers}
    fmts = {"selt": SELT_fmt, "melt": MELT_fmt, "qelt": QELT_fmt}

    def __init__(self, run_dir, input_header, selt=False, melt=False, qelt=False, output_format='bin', noheader=False):
        super().__init__(run_dir, input_header, {"selt": selt, "melt": melt, "qelt": qelt}, output_format, noheader)
        self.file_data = elt_manager.read_input_files(self.run_dir, self.computes["melt"], self.computes["qelt"],
                                                      self.len_sample)
        self._init_reader()

    def _make_reader(self, n_files):
        return elt_manager.ELTReader(
            self.len_sample,
            self.computes["selt"],
            self.computes["melt"],
            self.computes["qelt"],
            self.file_data["unique_event_ids"],
            self.file_data["event_rates"],
            self.file_data["intervals"],
            n_files=n_files,
        )
