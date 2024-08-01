import csv
import sys
import argparse
import logging
import numpy as np
import numba as nb
from contextlib import ExitStack

from oasislmf.pytools.common.data import (oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import (EventReader, init_streams_in, stream_info_to_bytes,
                                                  mv_read, SUMMARY_STREAM_ID, ITEM_STREAM, PIPE_CAPACITY)

logger = logging.getLogger(__name__)


class ELTReader(EventReader):
    def __init__(self, len_sample):
        self.len_sample = len_sample
        self.logger = logger

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id):
        return read_buffer(byte_mv, cursor, valid_buff, event_id, item_id)

@nb.njit(nopython=True, cache=True)
def read_buffer(byte_mv, cursor, valid_buff, event_id, item_id):
    """Read valid part of byte_mv and load relevant data for one event"""
    last_event_id = event_id

    while True:
        if valid_buff - cursor <= 10 * (4 * oasis_int_size + 2 * oasis_float_size):
            break

        summary_set, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
        event_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
        if event_id != last_event_id and last_event_id != 0:
            return cursor - (2 * oasis_int_size), last_event_id, 0, 1
        summary_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
        impacted_exposure, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)

        sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
        # logger.warning(f"Read event_id: {event_id}, summary_id: {summary_id}, impacted_exposure: {impacted_exposure}")
        while sidx != 0:
            loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)

            # logger.warning(f"Read sidx: {sidx}, loss: {loss}")
            sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)

        last_event_id = event_id

    return cursor, event_id, 0, 0


def run(files_in, **kwargs):
    with ExitStack() as stack:
        output_file = kwargs['output_file']
        streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)
        if stream_source_type != SUMMARY_STREAM_ID:
            raise Exception(f"unsupported stream type {stream_source_type}, {stream_agg_type}")

        elt_reader = ELTReader(len_sample)

        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['EventId', 'SummaryId', 'SampleId', 'Loss', 'ImpactedExposure'])

            for event_id in elt_reader.read_streams(streams_in):
                csv_writer.writerow([event_id, 0, 0, 0, 0])
                logger.warning(f"Event {event_id} processed")

def main():
    parser = argparse.ArgumentParser(description='Process event loss table stream')
    parser.add_argument('--files_in', nargs='+', required=True, help='Input files')
    parser.add_argument('--output_file', required=True, help='Output CSV file')

    args = parser.parse_args()
    run(args.files_in, output_file=args.output_file)


if __name__ == "__main__":
    main()