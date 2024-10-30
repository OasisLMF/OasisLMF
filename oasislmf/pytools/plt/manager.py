# plt/manager.py

import logging
import numpy as np
import numba as nb
from contextlib import ExitStack

from oasislmf.pytools.common.data import (oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import (EventReader, init_streams_in,
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
    def __init__(self, len_sample, compute_splt, compute_mplt, compute_qplt):
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

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id):
        # Pass state variables to read_buffer
        cursor, event_id, item_id, ret = read_buffer(
            byte_mv, cursor, valid_buff, event_id, item_id,
            self.state,
            self.splt_data, self.splt_idx,
            self.mplt_data, self.mplt_idx,
            self.qplt_data, self.qplt_idx
        )
        return cursor, event_id, item_id, ret


@nb.jit(nopython=True, cache=True)
def read_buffer(
        byte_mv, cursor, valid_buff, event_id, item_id,
        state,
        splt_data, splt_idx,
        mplt_data, mplt_idx,
        qplt_data, qplt_idx
):
    last_event_id = event_id
    idx = splt_idx[0]
    midx = mplt_idx[0]
    qidx = qplt_idx[0]

    while cursor < valid_buff:
        if not state["reading_losses"]:
            if valid_buff - cursor >= 3 * oasis_int_size + oasis_float_size:
                # Read summary header
                _, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                event_id_new, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                if last_event_id != 0 and event_id_new != last_event_id:
                    # New event, return to process the previous event
                    splt_idx[0] = idx
                    mplt_idx[0] = midx
                    qplt_idx[0] = qidx
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
                # TODO: READ LOSSES
                cursor += oasis_float_size
            else:
                break
        else:
            pass  # Should never reach here anyways

    # Update the indices
    splt_idx[0] = idx
    mplt_idx[0] = midx
    qplt_idx[0] = qidx
    return cursor, event_id, item_id, 0


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

        plt_reader = PLTReader(len_sample, compute_splt, compute_mplt, compute_qplt)

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
