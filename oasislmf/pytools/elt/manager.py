# 12598 is the number of summary_ids


import sys
import argparse
import struct
import logging
import numpy as np
import numba as nb
from contextlib import ExitStack

from oasislmf.pytools.common.data import (oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import (EventReader, init_streams_in, stream_info_to_bytes, write_mv_to_stream,
                                                  mv_read, mv_write_delimiter, SUMMARY_STREAM_ID,
                                                  GUL_STREAM_ID, FM_STREAM_ID, LOSS_STREAM_ID, ITEM_STREAM, PIPE_CAPACITY)

logger = logging.getLogger(__name__)


def read_buffer(byte_mv, cursor, valid_buff, event_id, item_id):
    """Read valid part of byte_mv and load relevant data for one event"""
    last_event_id = event_id

    while True:
        if valid_buff - cursor <= 10 * (4 * oasis_int_size + 2 * oasis_float_size):
            break

        header, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
        event_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
        if event_id != last_event_id and last_event_id != 0:
            return cursor - (2 * oasis_int_size), last_event_id, 0, 1
        summary_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
        impacted_exposure, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
        logger.warning(f"First field: {header}, Read event_id: {event_id}, summary_id: {summary_id}, impacted_exposure: {impacted_exposure}")

        sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
        while sidx != 0:
            loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
            logger.warning(f"Read sidx: {sidx}, loss: {loss}")
            sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
        last_event_id = event_id

    return cursor, event_id, 0, 0


class ELTReader(EventReader):
    def __init__(self, len_sample):
        self.len_sample = len_sample
        # self.loss_array = np.zeros(len_sample + 1, dtype=oasis_float)
        # self.exposure_value = np.zeros(1, dtype=oasis_float)
        self.logger = logger

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id):
        return read_buffer(byte_mv, cursor, valid_buff, event_id, item_id)


# @nb.njit(cache=True)
def write_event(byte_mv, event_id, summary_id, len_sample):
    """Write event loss data to byte_mv"""
    cursor = 0
    cursor = write_mv_to_stream(byte_mv, cursor, event_id)
    cursor = write_mv_to_stream(byte_mv, cursor, summary_id)
    # for sidx in range(-1, len_sample + 1):
        # loss = loss_array[sidx]
        # if loss != 0 or sidx == -1:
            # cursor = write_mv_to_stream(byte_mv, cursor, sidx, loss)
    # impacted_exposure = exposure_value if np.any(loss_array != 0) else 0
    # cursor = write_mv_to_stream(byte_mv, cursor, impacted_exposure)
    cursor = mv_write_delimiter(byte_mv, cursor)
    return cursor


def run(files_in, **kwargs):
    with ExitStack() as stack:
        elt_pipe = stack.enter_context(open(kwargs['output_file'], 'wb'))
        streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)
        if stream_source_type != SUMMARY_STREAM_ID:
            raise Exception(f"unsupported stream type {stream_source_type}, {stream_agg_type}")

        elt_reader = ELTReader(len_sample)
        out_byte_mv = np.frombuffer(buffer=memoryview(bytearray(PIPE_CAPACITY)), dtype='b')

        elt_pipe.write(stream_info_to_bytes(SUMMARY_STREAM_ID, ITEM_STREAM))
        elt_pipe.write(len_sample.tobytes())

        for event_id in elt_reader.read_streams(streams_in):
            # cursor = write_event(out_byte_mv, event_id, 1, len_sample)
            # write_mv_to_stream(elt_pipe, out_byte_mv, cursor)
            # elt_reader.loss_array.fill(0)
            # elt_reader.exposure_value.fill(0)
            logger.warning(f"Event {event_id} processed")


def main():
    parser = argparse.ArgumentParser(description='Process event loss table stream')
    parser.add_argument('--files_in', nargs='+', required=True, help='Input files')
    parser.add_argument('--output_file', required=True, help='Output file')

    args = parser.parse_args()
    run(args.files_in, output_file=args.output_file)


if __name__ == "__main__":
    main()
