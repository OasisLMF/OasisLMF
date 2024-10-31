# plt/manager.py

import logging
import numpy as np
import numba as nb
import os
import struct
from contextlib import ExitStack

from oasislmf.pytools.common.data import (oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import (MEAN_IDX, EventReader, init_streams_in,
                                                  mv_read, SUMMARY_STREAM_ID)
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)

SPLT_output = [
    ('Period', oasis_int, '%d'),
    ('PeriodWeight', oasis_float, '%.6f'),
    ('EventId', oasis_int, '%d'),
    ('Year', oasis_int, '%d'),
    ('Month', oasis_int, '%d'),
    ('Day', oasis_int, '%d'),
    ('Hour', oasis_int, '%d'),
    ('Minute', oasis_int, '%d'),
    ('SummaryId', oasis_int, '%d'),
    ('SampleId', oasis_int, '%d'),
    ('Loss', oasis_float, '%.2f'),
    ('ImpactedExposure', oasis_float, '%.2f'),
]

MPLT_output = [
    ('Period', oasis_int, '%d'),
    ('PeriodWeight', oasis_float, '%.6f'),
    ('EventId', oasis_int, '%d'),
    ('Year', oasis_int, '%d'),
    ('Month', oasis_int, '%d'),
    ('Day', oasis_int, '%d'),
    ('Hour', oasis_int, '%d'),
    ('Minute', oasis_int, '%d'),
    ('SummaryId', oasis_int, '%d'),
    ('SampleType', oasis_int, '%d'),
    ('ChanceOfLoss', oasis_float, '%.4f'),
    ('MeanLoss', oasis_float, '%.2f'),
    ('SDLoss', oasis_float, '%.2f'),
    ('MaxLoss', oasis_float, '%.2f'),
    ('FootprintExposure', oasis_float, '%.2f'),
    ('MeanImpactedExposure', oasis_float, '%.2f'),
    ('MaxImpactedExposure', oasis_float, '%.2f'),
]

QPLT_output = [
    ('Period', oasis_int, '%d'),
    ('PeriodWeight', oasis_float, '%.6f'),
    ('EventId', oasis_int, '%d'),
    ('Year', oasis_int, '%d'),
    ('Month', oasis_int, '%d'),
    ('Day', oasis_int, '%d'),
    ('Hour', oasis_int, '%d'),
    ('Minute', oasis_int, '%d'),
    ('SummaryId', oasis_int, '%d'),
    ('Quantile', oasis_float, '%.2f'),
    ('Loss', oasis_float, '%.2f'),
]


class PLTReader(EventReader):
    def __init__(self, len_sample, compute_splt, compute_mplt, compute_qplt, occ_map):
        self.logger = logger

        SPLT_dtype = np.dtype([(c[0], c[1]) for c in SPLT_output])
        MPLT_dtype = np.dtype([(c[0], c[1]) for c in MPLT_output])
        QPLT_dtype = np.dtype([(c[0], c[1]) for c in QPLT_output])

        # Buffer for SPLT data
        self.splt_data = np.zeros(100000, dtype=SPLT_dtype)
        self.splt_idx = np.zeros(1, dtype=np.int64)

        # Buffer for MPLT data
        self.mplt_data = np.zeros(100000, dtype=MPLT_dtype)
        self.mplt_idx = np.zeros(1, dtype=np.int64)

        # Buffer for QPLT data
        self.qplt_data = np.zeros(100000, dtype=QPLT_dtype)
        self.qplt_idx = np.zeros(1, dtype=np.int64)

        read_buffer_state_dtype = np.dtype([
            ('len_sample', oasis_int),
            ('reading_losses', np.bool_),
            ('compute_splt', np.bool_),
            ('compute_mplt', np.bool_),
            ('compute_qplt', np.bool_),
            ('summary_id', oasis_int),
            ('impacted_exposure', oasis_float),
        ])

        self.state = np.zeros(1, dtype=read_buffer_state_dtype)[0]
        self.state["reading_losses"] = False  # Set to true after reading header in read_buffer
        self.state["len_sample"] = len_sample
        self.state["compute_splt"] = compute_splt
        self.state["compute_mplt"] = compute_mplt
        self.state["compute_qplt"] = compute_qplt
        self.occ_map = occ_map

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id):
        # Pass state variables to read_buffer
        cursor, event_id, item_id, ret = read_buffer(
            byte_mv, cursor, valid_buff, event_id, item_id,
            self.state,
            self.splt_data, self.splt_idx,
            self.mplt_data, self.mplt_idx,
            self.qplt_data, self.qplt_idx,
            self.occ_map
        )
        return cursor, event_id, item_id, ret


@nb.jit(nopython=True, cache=True)
def read_buffer(
        byte_mv, cursor, valid_buff, event_id, item_id,
        state,
        splt_data, splt_idx,
        mplt_data, mplt_idx,
        qplt_data, qplt_idx,
        occ_map
):
    last_event_id = event_id
    si = splt_idx[0]
    mi = mplt_idx[0]
    qi = qplt_idx[0]

    def _update_idxs():
        splt_idx[0] = si
        mplt_idx[0] = mi
        qplt_idx[0] = qi

    def _get_dates(occ_date_id):
        # NOTE: Currently does not support granular dates
        g = occ_date_id

        # Function void d(long long g, int& y, int& mm, int& dd) taken from pltcalc.cpp
        y = (10000 * g + 14780) // 3652425
        ddd = g - (365 * y + y // 4 - y // 100 + y // 400)
        if ddd < 0:
            y = y - 1
            ddd = g - (365 * y + y // 4 - y // 100 + y // 400)
        mi = (100 * ddd + 52) // 3060
        mm = (mi + 2) % 12 + 1
        y = y + (mi + 2) // 12
        dd = ddd - (mi * 306 + 5) // 10 + 1

        return y, mm, dd, 0, 0

    while cursor < valid_buff:
        if not state["reading_losses"]:
            if valid_buff - cursor >= 3 * oasis_int_size + oasis_float_size:
                # Read summary header
                _, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                event_id_new, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                if last_event_id != 0 and event_id_new != last_event_id:
                    # New event, return to process the previous event
                    _update_idxs()
                    return cursor - (2 * oasis_int_size), last_event_id, item_id, 1
                event_id = event_id_new
                state["summary_id"], cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                state["impacted_exposure"], cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
                state["reading_losses"] = True

            else:
                # Not enough for whole summary header
                break

        if state["reading_losses"]:
            if valid_buff - cursor >= oasis_int_size + oasis_float_size:
                # Read sidx and loss
                sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                if sidx != 0:
                    loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
                    if sidx >= MEAN_IDX:
                        if state["compute_splt"]:
                            filtered_occ_map = occ_map[occ_map["event_id"] == event_id]
                            for record in filtered_occ_map:
                                year, month, day, hour, minute = _get_dates(record["occ_date_id"])
                                splt_data[si]["Period"] = record["period_no"]
                                splt_data[si]["PeriodWeight"] = 0
                                splt_data[si]["EventId"] = event_id
                                splt_data[si]["Year"] = year
                                splt_data[si]["Month"] = month
                                splt_data[si]["Day"] = day
                                splt_data[si]["Hour"] = hour
                                splt_data[si]["Minute"] = minute
                                splt_data[si]["SummaryId"] = state["summary_id"]
                                splt_data[si]["SampleId"] = sidx
                                splt_data[si]["Loss"] = loss
                                splt_data[si]["ImpactedExposure"] = state["impacted_exposure"] * (loss > 0)
                                si += 1
                                if si >= splt_data.shape[0]:
                                    # Output array full
                                    _update_idxs()
                                    return cursor, event_id, item_id, 1
                else:
                    # sidx == 0, end of record
                    state["reading_losses"] = False
            else:
                break
        else:
            pass  # Should never reach here anyways

    # Update the indices
    _update_idxs()
    return cursor, event_id, item_id, 0


def read_occurrence(occurrence_fp):
    """Read the occurrence binary file and returns an occurrence map

    Args:
        occurrence_fp (str): Path to the occurrence binary file

    Returns:
        occ_map_dtype: numpy map of event_id, period_no, occ_date_id from the occurrence file
    """
    try:
        with open(occurrence_fp, "rb") as fin:
            # Extract Date Options
            date_opts = fin.read(4)
            if not date_opts or len(date_opts) < 4:
                logger.error("Occurrence file is empty or currupted")
                raise RuntimeError("Occurrence file is empty or currupted")
            date_opts = int.from_bytes(date_opts, byteorder="little", signed=True)

            data_algorithm = date_opts & 1  # Unused as granular_date not supported
            granular_date = date_opts >> 1

            # Currently does not support granular_date (hour/minute)
            if granular_date:
                logger.error("Granular date currently not supported by pltpy")
                raise RuntimeError("Granular date currently not supported by pltpy")

            # Extract no_of_periods
            no_of_periods = fin.read(4)
            if not no_of_periods or len(no_of_periods) < 4:
                logger.error("Occurrence file is empty or currupted")
                raise RuntimeError("Occurrence file is empty or currupted")
            no_of_periods = int.from_bytes(no_of_periods, byteorder="little", signed=True)

            record_size = 12  # Non granular record size
            data = fin.read()

        num_records = len(data) // record_size
        if num_records % record_size != 0:
            logger.warning("Occurrence File size does not align with expected record size")

        occ_map_dtype = np.dtype([
            ("event_id", np.int32),
            ("period_no", np.int32),
            ("occ_date_id", np.int32),
        ])
        occ_map = np.zeros(num_records, dtype=occ_map_dtype)

        for i in range(num_records):
            offset = i * record_size
            curr_data = data[offset:offset + record_size]
            if len(curr_data) < record_size:
                break
            event_id, period_no, occ_date_id = struct.unpack('<iii', curr_data)
            occ_map[i] = (event_id, period_no, occ_date_id)

        return occ_map
    except FileNotFoundError:
        logger.error(f"FATAL: Error opening file {occurrence_fp}")
        raise FileNotFoundError(f"FATAL: Error opening file {occurrence_fp}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise RuntimeError(f"An error occurred: {str(e)}")


def run(run_dir, files_in, splt_output_file=None, mplt_output_file=None, qplt_output_file=None):
    """Runs PLT calculations

    Args:
        run_dir (str): Path to directory containing required files structure
        files_in (str): Path to summary binary input file
        splt_output_file (str, optional): Path to SPLT output file. Defaults to None.
        mplt_output_file (str, optional): Path to MPLT output file. Defaults to None.
        qplt_output_file (str, optional): Path to QPLT output file. Defaults to None.
    """
    compute_splt = splt_output_file is not None
    compute_mplt = mplt_output_file is not None
    compute_qplt = qplt_output_file is not None

    if not compute_splt and not compute_mplt and not compute_qplt:
        logger.warning("No output files specified")

    with ExitStack() as stack:
        streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)
        if stream_source_type != SUMMARY_STREAM_ID:
            raise Exception(f"unsupported stream type {stream_source_type}, {stream_agg_type}")

        occ_map = read_occurrence(os.path.join(run_dir, "input", "occurrence.bin"))
        plt_reader = PLTReader(len_sample, compute_splt, compute_mplt, compute_qplt, occ_map)

        # Initialise csv column names for PLT files
        output_files = {}
        if compute_splt:
            SPLT_headers = ','.join([c[0] for c in SPLT_output])
            splt_file = stack.enter_context(open(splt_output_file, 'w'))
            splt_file.write(SPLT_headers + '\n')
            output_files['splt'] = splt_file
        else:
            output_files['splt'] = None

        if compute_mplt:
            MPLT_headers = ','.join([c[0] for c in MPLT_output])
            mplt_file = stack.enter_context(open(mplt_output_file, 'w'))
            mplt_file.write(MPLT_headers + '\n')
            output_files['mplt'] = mplt_file
        else:
            output_files['mplt'] = None

        if compute_qplt:
            QPLT_headers = ','.join([c[0] for c in QPLT_output])
            qplt_file = stack.enter_context(open(qplt_output_file, 'w'))
            qplt_file.write(QPLT_headers + '\n')
            output_files['qplt'] = qplt_file
        else:
            output_files['qplt'] = None

        SPLT_fmt = ','.join([c[2] for c in SPLT_output])
        MPLT_fmt = ','.join([c[2] for c in MPLT_output])
        QPLT_fmt = ','.join([c[2] for c in QPLT_output])

        for event_id in plt_reader.read_streams(streams_in):
            if compute_splt:
                # Extract SPLT data
                data = plt_reader.splt_data[:plt_reader.splt_idx[0]]
                if output_files['splt'] is not None and data.size > 0:
                    np.savetxt(output_files['splt'], data, delimiter=',', fmt=SPLT_fmt)
                plt_reader.splt_idx[0] = 0

            if compute_mplt:
                # Extract MPLT data
                mplt_data = plt_reader.mplt_data[:plt_reader.mplt_idx[0]]
                if output_files['mplt'] is not None and mplt_data.size > 0:
                    np.savetxt(output_files['mplt'], mplt_data, delimiter=',', fmt=MPLT_fmt)
                plt_reader.mplt_idx[0] = 0

            if compute_qplt:
                # Extract QPLT data
                qplt_data = plt_reader.qplt_data[:plt_reader.qplt_idx[0]]
                if output_files['qplt'] is not None and qplt_data.size > 0:
                    np.savetxt(output_files['qplt'], qplt_data, delimiter=',', fmt=QPLT_fmt)
                plt_reader.qplt_idx[0] = 0


@redirect_logging(exec_name='pltpy')
def main(run_dir='.', files_in=None, splt=None, mplt=None, qplt=None, **kwargs):
    run(
        run_dir,
        files_in,
        splt_output_file=splt,
        mplt_output_file=mplt,
        qplt_output_file=qplt
    )
