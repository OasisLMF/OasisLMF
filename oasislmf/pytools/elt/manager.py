import csv
import sys
import argparse
import logging
import numpy as np
import numba as nb
from contextlib import ExitStack

from oasislmf.pytools.common.data import (oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import (EventReader, init_streams_in, stream_info_to_bytes, mv_write,
                                                  mv_read, SUMMARY_STREAM_ID, PIPE_CAPACITY)

logger = logging.getLogger(__name__)

SELT_dtype = np.dtype([
    ('EventId', oasis_int),
    ('SummaryId', oasis_int),
    ('SampleId', oasis_int),
    ('Loss', oasis_float),
    ('ImpactedExposure', oasis_float)
])


class ELTReader(EventReader):
    def __init__(self, len_sample):
        self.len_sample = len_sample
        self.logger = logger
        self.event_data = np.zeros(100000, dtype=SELT_dtype)  # Adjust size as needed
        self.event_idx = np.zeros(1, dtype=np.int64)  # Index into event_data

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id):
        return read_buffer(byte_mv, cursor, valid_buff, event_id, item_id, self.event_data, self.event_idx)
        # return read_buffer(byte_mv, cursor, valid_buff, event_id, item_id)


# @nb.jit(nopython=True, cache=True)
# def read_buffer(byte_mv, cursor, valid_buff, event_id, out_byte_mv):
#     """Read valid part of byte_mv and load relevant data for one event"""
#     last_event_id = event_id

#     while True:
#         if valid_buff - cursor <= 10 * (4 * oasis_int_size + 2 * oasis_float_size):
#             break
#         # two sidx 0 0 when event ends
#         summary_set, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
#         event_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)

#         if event_id != last_event_id and last_event_id != 0:
#             return cursor - (2 * oasis_int_size), last_event_id, 0, 1

#         summary_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
#         impacted_exposure, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)

#         sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
#         logger.warning(f"Read event_id: {event_id}, summary_id: {summary_id}, impacted_exposure: {impacted_exposure}")

#         while sidx != 0:
#             loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
#             logger.warning(f"Read sidx: {sidx}, loss: {loss}")
#             sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)

#             # write to out_byte_mv which is a memoryview of the output file

#         last_event_id = event_id

#     return cursor, event_id, 0, 0


@nb.jit(nopython=True, cache=True)
def read_buffer(byte_mv, cursor, valid_buff, event_id, item_id, event_data, event_idx):
    last_event_id = event_id
    idx = event_idx[0]
    summary_id = 0
    impacted_exposure = 0.0
    readme = 0

    while cursor < valid_buff:
        if readme:
            # Read sidx and loss
            if valid_buff - cursor < oasis_int_size + oasis_float_size:
                break
            sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            if sidx != 0:
                loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
                if sidx != -5:
                    # Add record to event_data
                    event_data[idx]['EventId'] = event_id
                    event_data[idx]['SummaryId'] = summary_id
                    event_data[idx]['SampleId'] = sidx
                    event_data[idx]['Loss'] = loss
                    event_data[idx]['ImpactedExposure'] = impacted_exposure
                    idx += 1
                    if idx >= event_data.shape[0]:
                        # Output array is full
                        event_idx[0] = idx
                        return cursor, event_id, item_id, 1 # adjust cursor before
            else:
                # sidx == 0, end of summary_id
                # cursor += oasis_float_size  ???
                readme = 0
        else:
            # Read summary header
            if valid_buff - cursor < 3 * oasis_int_size + oasis_float_size:
                break
            summary_set, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            event_id_new, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            if last_event_id != 0 and event_id_new != last_event_id:
                # New event started, return to process the previous event
                event_idx[0] = idx
                return cursor - (2 * oasis_int_size), last_event_id, item_id, 1
            event_id = event_id_new
            summary_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            impacted_exposure, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
            readme = 1  # Start reading sidx, loss pairs
            last_event_id = event_id

    event_idx[0] = idx # ISSUE HERE WHEN BUFFER ENDS IT DESYNCS SOMEHOW
    return cursor, event_id, item_id, 0


def run(files_in, **kwargs):
    with ExitStack() as stack:
        output_file = kwargs['output_file']
        streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)
        if stream_source_type != SUMMARY_STREAM_ID:
            raise Exception(f"unsupported stream type {stream_source_type}, {stream_agg_type}")

        elt_reader = ELTReader(len_sample)

        with open(output_file, 'w') as output_file:
            # Write header
            output_file.write('EventId,SummaryId,SampleId,Loss,ImpactedExposure\n')

            for event_id in elt_reader.read_streams(streams_in):
                # Extract the data up to the current index
                data = elt_reader.event_data[:elt_reader.event_idx[0]]

                # Convert structured array to a 2D array
                data_2d = np.column_stack((
                    data['EventId'],
                    data['SummaryId'],
                    data['SampleId'],
                    data['Loss'],
                    data['ImpactedExposure']
                ))

                # Define the format for each column
                fmt = ['%d', '%d', '%d', '%.2f', '%.2f']

                # Write the data using np.savetxt
                np.savetxt(output_file, data_2d, delimiter=',', fmt=fmt)

                # Reset event_data and event_idx
                elt_reader.event_idx[0] = 0

                # Log the processed event
                logger.warning(f"Event {event_id} processed")



def main():
    parser = argparse.ArgumentParser(description='Process event loss table stream')
    parser.add_argument('--files_in', nargs='+', required=True, help='Input files')
    parser.add_argument('--output_file', required=True, help='Output CSV file')

    args = parser.parse_args()
    run(args.files_in, output_file=args.output_file)


if __name__ == "__main__":
    main()
