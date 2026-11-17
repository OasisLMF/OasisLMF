"""Entry point to the run method of summarypy
read an event loss stream and aggregate it into the relevant summary loss stream

The per-event work (reading the loss stream, aggregating it and writing each summary set) is done by
SummaryStage in stage.py, which also holds the Numba kernels and documents the data structures.
This module reads the command-line options, opens the inputs and outputs and drives the stage over them.
"""
import logging
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oasislmf.pytools.common.data import write_ndarray_to_fmt_csv
from oasislmf.pytools.common.event_stream import init_streams_in, stream_info_to_bytes
from oasislmf.pytools.common.stage import StreamSink
from oasislmf.pytools.common.run_types import RUNTYPE_GROUNDUP_LOSS, RUNTYPE_INSURED_LOSS, RUNTYPE_REINSURANCE_LOSS  # noqa: F401
from oasislmf.pytools.elt.data import SELT_dtype, SELT_fmt, SELT_headers, VALID_EXT
from oasislmf.pytools.summary.stage import (SPECIAL_SIDX_COUNT, SUMMARY_HEADER_SIZE, SUMMARY_META_SIZE, SIDX_LOSS_WRITE_SIZE,  # noqa: F401
                                            SUPPORTED_RUN_TYPE, SummaryReader, SummaryStage, get_summary_object, load_summary_object,
                                            load_summary_map, read_summary_objects, extract_risk_info, read_buffer, mv_write_event,
                                            fill_event_rows, get_summary_set_id_to_summary_set_index, get_summary_xref_info)
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


SUPPORTED_SUMMARY_SET_ID = list(range(1, 10))

OUTPUT_ROWS_BUFFER_SIZE = 100_000


def create_summary_object_file(static_path, run_type):
    """Create and write summary object into static path"""
    summary_objects = get_summary_object(static_path, run_type)
    write_summary_objects(static_path, run_type, *summary_objects)


def write_summary_objects(static_path, run_type, summary_info, summary_set_id_to_summary_set_index,
                          summary_set_index_to_loss_ptr, item_id_to_summary_id, item_id_to_risks_i, risk_i_to_n_buildings):
    os.makedirs(os.path.join(static_path, run_type), exist_ok=True)
    np.save(os.path.join(static_path, run_type, 'summary_info'), summary_info)
    np.save(os.path.join(static_path, run_type, 'summary_set_id_to_summary_set_index'), summary_set_id_to_summary_set_index)
    np.save(os.path.join(static_path, run_type, 'summary_set_index_to_loss_ptr'), summary_set_index_to_loss_ptr)
    np.save(os.path.join(static_path, run_type, 'item_id_to_summary_id'), item_id_to_summary_id)
    np.save(os.path.join(static_path, run_type, 'item_id_to_risks_i'), item_id_to_risks_i)
    np.save(os.path.join(static_path, run_type, 'risk_i_to_n_buildings'), risk_i_to_n_buildings)


class SummaryRowWriter:
    """Buffer the SELT rows of one summary set and write them to a csv or parquet file in large batches."""

    def __init__(self, stack, file_path, output_format):
        if OUTPUT_ROWS_BUFFER_SIZE < 3:  # fill_event_rows needs room for the special sidx rows of one summary
            raise ValueError(f"OUTPUT_ROWS_BUFFER_SIZE must be at least 3, got {OUTPUT_ROWS_BUFFER_SIZE}")
        self.rows = np.empty(OUTPUT_ROWS_BUFFER_SIZE, dtype=SELT_dtype)
        self.row_i = 0
        self.output_format = output_format
        if output_format == 'parquet':
            self.schema = pa.schema([(name, pa.from_numpy_dtype(SELT_dtype[name])) for name in SELT_dtype.names])
            self.file = stack.enter_context(pq.ParquetWriter(file_path, self.schema))
        else:
            self.file = stack.enter_context(open(file_path, 'w'))
            self.file.write(','.join(SELT_headers) + '\n')

    def flush(self):
        """Write the buffered rows and empty the buffer."""
        if not self.row_i:
            return
        data = self.rows[:self.row_i]
        if self.output_format == 'parquet':
            self.file.write_table(pa.Table.from_arrays([pa.array(data[name]) for name in data.dtype.names], schema=self.schema))
        else:
            write_ndarray_to_fmt_csv(self.file, data, SELT_headers, SELT_fmt)
        self.row_i = 0



def run(files_in, static_path, run_type, low_memory, output_zeros, output_format='bin', **kwargs):
    """Run the summary calculation, writing a summary output for each summary set.

    With the default ``bin`` format each summary set is written as a binary summary stream. With ``csv``
    or ``parquet`` it is written as a table in the SELT layout (EventId, SummaryId, SampleId, Loss,
    ImpactedExposure), one row per sidx of the stream, special sidx included.

    Args:
        files_in: list of file path to read event from
        run_type: type of the source that is sending the stream
        static_path: path to the static files
        low_memory: if true output summary index file (bin output only)
        output_zeros: if true output 0 loss
        output_format: one of ``bin``, ``csv`` or ``parquet``
        **kwargs: additional options, including ``summary_sets_output``, a list of
            ``-summary_set_id summary_set_path`` pairs
    """
    summary_sets_path = {}
    error_msg = (f"summary_sets_output expected format is a list of -summary_set_id summary_set_path (ex: -1 S1.bin -2 S2.bin')"
                 f", found '{' '.join(kwargs['summary_sets_output'])}'")
    for summary_set_id, summary_set_path in zip_longest(*[iter(kwargs['summary_sets_output'])] * 2):
        if summary_set_id[0] != '-' or summary_set_path is None:
            raise Exception(error_msg)
        try:
            summary_sets_path[int(summary_set_id[1:])] = summary_set_path
        except ValueError:
            raise Exception(error_msg)

    if output_format not in VALID_EXT:
        raise ValueError(f"unsupported output format {output_format}, expected one of {VALID_EXT}")
    output_binary = output_format == 'bin'
    if low_memory and not output_binary:
        raise ValueError(f"low memory index files are only available for bin output, not {output_format}")
    if not output_binary:
        for summary_set_path in summary_sets_path.values():
            suffix = Path(summary_set_path).suffix
            if suffix and suffix != '.' + output_format:
                raise ValueError(f"Invalid file extension for {output_format}, got {summary_set_path}")

    with ExitStack() as stack:
        summary_sets_pipe = summary_sets_index_pipe = summary_row_writers = None
        if output_binary:
            summary_sets_pipe = {i: stack.enter_context(open(summary_set_path, 'wb')) for i, summary_set_path in summary_sets_path.items()}
            if low_memory:
                summary_sets_index_pipe = {summary_set_id: stack.enter_context(open(setpath.rsplit('.', 1)[0] + '.idx', 'wb'))
                                           for summary_set_id, setpath in summary_sets_path.items()}
        else:
            summary_row_writers = {i: SummaryRowWriter(stack, summary_set_path, output_format)
                                   for i, summary_set_path in summary_sets_path.items()}

        streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)
        input_header = stream_info_to_bytes(stream_source_type, stream_agg_type) + len_sample.tobytes()

        stage = stack.enter_context(SummaryStage(input_header, static_path, run_type, summary_sets_path.keys(),
                                                 low_memory=low_memory, output_zeros=output_zeros))

        summary_sets_out = None
        if output_binary:
            for summary_set_id, summary_pipe in summary_sets_pipe.items():
                summary_pipe.write(stage.headers[summary_set_id])
            summary_sets_out = {summary_set_id: StreamSink(summary_pipe) for summary_set_id, summary_pipe in summary_sets_pipe.items()}

        stage.process_events(stage.reader.read_streams(streams_in), summary_sets_out, summary_sets_index_pipe, summary_row_writers)

        if not output_binary:
            for writer in summary_row_writers.values():
                writer.flush()


@redirect_logging(exec_name='summarypy')
def main(create_summarypy_files, static_path, run_type, ext='bin', **kwargs):
    if create_summarypy_files:
        create_summary_object_file(static_path, run_type)
    else:
        run(static_path=static_path, run_type=run_type, output_format=ext, **kwargs)
