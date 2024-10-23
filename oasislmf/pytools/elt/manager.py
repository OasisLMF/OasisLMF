# elt/manager.py

import logging
import numpy as np
import numba as nb
import os
import struct
from contextlib import ExitStack

from oasislmf.pytools.common.data import (oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import (EventReader, init_streams_in,
                                                  mv_read, SUMMARY_STREAM_ID)
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)

SELT_dtype = np.dtype([
    ('EventId', oasis_int),
    ('SummaryId', oasis_int),
    ('SampleId', oasis_int),
    ('Loss', oasis_float),
    ('ImpactedExposure', oasis_float)
])

MELT_dtype = np.dtype([
    ('EventId', oasis_int),
    ('SummaryId', oasis_int),
    ('SampleType', oasis_int),
    ('EventRate', oasis_float),
    ('ChanceOfLoss', oasis_float),
    ('MeanLoss', oasis_float),
    ('SDLoss', oasis_float),
    ('MaxLoss', oasis_float),
    ('FootprintExposure', oasis_float),
    ('MeanImpactedExposure', oasis_float),
    ('MaxImpactedExposure', oasis_float)
])

QELT_dtype = np.dtype([
    ('EventId', oasis_int),
    ('SummaryId', oasis_int),
    ('Quantile', oasis_float),
    ('Loss', oasis_float)
])

quantile_interval_dtype = np.dtype([
    ('q', oasis_float),
    ('integer_part', oasis_int),
    ('fractional_part', oasis_float),
])


class ELTReader(EventReader):
    def __init__(self, len_sample, compute_selt, compute_melt, compute_qelt, unique_event_ids, event_rates, intervals):
        self.logger = logger
        self.selt_data = np.zeros(100000, dtype=SELT_dtype)  # write buffer for SELT
        self.selt_idx = np.zeros(1, dtype=np.int64)

        read_buffer_state_dtype = np.dtype([
            ('len_sample', oasis_int),
            ('reading_losses', np.bool_),
            ('compute_selt', np.bool_),
            ('compute_melt', np.bool_),
            ('compute_qelt', np.bool_),
            ('summary_id', oasis_int),
            ('impacted_exposure', oasis_float),
            ('sumloss', oasis_float),
            ('sumlosssqr', oasis_float),
            ('non_zero_samples', oasis_int),
            ('max_loss', oasis_float),
            ('mean_impacted_exposure', oasis_float),
            ('max_impacted_exposure', oasis_float),
            ('analytical_mean', oasis_float),
            ('losses_vec', oasis_float, (len_sample,)),
        ])

        self.state = np.zeros(1, dtype=read_buffer_state_dtype)[0]
        self.state["len_sample"] = len_sample
        self.state["reading_losses"] = False
        self.state["compute_selt"] = compute_selt
        self.state["compute_melt"] = compute_melt
        self.state["compute_qelt"] = compute_qelt
        self.unique_event_ids = unique_event_ids.astype(oasis_int)
        self.event_rates = event_rates.astype(oasis_float)
        self.state["losses_vec"] = np.zeros(len_sample)

        # Buffer for MELT data
        self.melt_data = np.zeros(100000, dtype=MELT_dtype)  # write buffer for MELT
        self.melt_idx = np.zeros(1, dtype=np.int64)

        # Buffer for QELT data
        self.qelt_data = np.zeros(100000, dtype=QELT_dtype)
        self.qelt_idx = np.zeros(1, dtype=np.int64)
        self.intervals = intervals

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id):
        # Pass state variables to read_buffer
        cursor, event_id, item_id, ret = read_buffer(
            byte_mv, cursor, valid_buff, event_id, item_id, self.selt_data, self.selt_idx,
            self.state, self.melt_data, self.melt_idx, self.qelt_data, self.qelt_idx, self.intervals,
            self.unique_event_ids, self.event_rates
        )
        return cursor, event_id, item_id, ret


@nb.jit(nopython=True, cache=True)
def read_buffer(
        byte_mv, cursor, valid_buff, event_id, item_id,
        selt_data, selt_idx, state,
        melt_data, melt_idx,
        qelt_data, qelt_idx, intervals,
        unique_event_ids, event_rates
):
    last_event_id = event_id
    idx = selt_idx[0]
    midx = melt_idx[0]
    qidx = qelt_idx[0]

    while cursor < valid_buff:

        if not state["reading_losses"]:
            if valid_buff - cursor >= 3 * oasis_int_size + oasis_float_size:
                # Read summary header
                _, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                event_id_new, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                if last_event_id != 0 and event_id_new != last_event_id:
                    # New event, return to process the previous event
                    selt_idx[0] = idx
                    melt_idx[0] = midx
                    qelt_idx[0] = qidx
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
                    if sidx != -5:
                        # Normal data record
                        if state["compute_selt"]:
                            # Update SELT data
                            selt_data[idx]['EventId'] = event_id
                            selt_data[idx]['SummaryId'] = state["summary_id"]
                            selt_data[idx]['SampleId'] = sidx
                            selt_data[idx]['Loss'] = loss
                            selt_data[idx]['ImpactedExposure'] = state["impacted_exposure"]
                            idx += 1
                            if idx >= selt_data.shape[0]:
                                # Output array is full
                                selt_idx[0] = idx
                                melt_idx[0] = midx
                                qelt_idx[0] = qidx
                                return cursor, event_id, item_id, 1
                        if state["compute_melt"]:
                            # Update MELT variables
                            if sidx > 0:
                                state["sumloss"] += loss
                                state["sumlosssqr"] += loss * loss
                                if loss > 0.0:
                                    state["non_zero_samples"] += 1
                        if state["compute_qelt"]:
                            if sidx > 0:
                                state["losses_vec"][sidx - 1] = loss
                    else:
                        # sidx == -5, MaxLoss
                        if state["compute_melt"]:
                            state["max_loss"] = loss

                    if sidx == -1 and state["compute_melt"]:
                        state["analytical_mean"] = loss

                else:
                    # sidx == 0, end of summary_id
                    state["reading_losses"] = False

                    if state["compute_melt"]:

                        # Compute MELT statistics and store them
                        if state["non_zero_samples"] > 0:
                            sample_mean = state["sumloss"] / state["len_sample"]
                            if state["non_zero_samples"] > 1:
                                variance = (state["sumlosssqr"] - (state["sumloss"] * state["sumloss"]) /
                                            state["len_sample"]) / (state["len_sample"] - 1)
                                if variance / state["sumlosssqr"] < 1e-7:
                                    variance = 0.0
                                std_dev = np.sqrt(variance)
                            else:
                                std_dev = 0.0

                            chance_of_loss = state["non_zero_samples"] / state["len_sample"]
                            mean_imp_exp = state["impacted_exposure"] * chance_of_loss

                            idx_ev = np.searchsorted(unique_event_ids, event_id)
                            if idx_ev < unique_event_ids.shape[0] and unique_event_ids[idx_ev] == event_id:
                                event_rate = event_rates[idx_ev]
                            else:
                                event_rate = 0.0

                            # MELT data
                            melt_data[midx]['EventId'] = event_id
                            melt_data[midx]['SummaryId'] = state["summary_id"]
                            melt_data[midx]['SampleType'] = 1
                            melt_data[midx]['EventRate'] = event_rate
                            melt_data[midx]['ChanceOfLoss'] = 0
                            melt_data[midx]['MeanLoss'] = state["analytical_mean"]
                            melt_data[midx]['SDLoss'] = 0.0
                            melt_data[midx]['MaxLoss'] = state["max_loss"]
                            melt_data[midx]['FootprintExposure'] = state["impacted_exposure"]
                            melt_data[midx]['MeanImpactedExposure'] = state["impacted_exposure"]
                            melt_data[midx]['MaxImpactedExposure'] = state["impacted_exposure"]
                            midx += 1

                            # sample mean
                            melt_data[midx]['EventId'] = event_id
                            melt_data[midx]['SummaryId'] = state["summary_id"]
                            melt_data[midx]['SampleType'] = 2
                            melt_data[midx]['EventRate'] = event_rate
                            melt_data[midx]['ChanceOfLoss'] = chance_of_loss
                            melt_data[midx]['MeanLoss'] = sample_mean
                            melt_data[midx]['SDLoss'] = std_dev
                            melt_data[midx]['MaxLoss'] = state["max_loss"]
                            melt_data[midx]['FootprintExposure'] = state["impacted_exposure"]
                            melt_data[midx]['MeanImpactedExposure'] = mean_imp_exp
                            melt_data[midx]['MaxImpactedExposure'] = state["impacted_exposure"]
                            midx += 1

                            if midx >= melt_data.shape[0]:
                                # Output array is full
                                selt_idx[0] = idx
                                melt_idx[0] = midx
                                qelt_idx[0] = qidx
                                return cursor, event_id, item_id, 1

                    if state["compute_qelt"]:
                        state["losses_vec"].sort()
                        for i in range(len(intervals)):
                            q = intervals[i]["q"]
                            ipart = intervals[i]["integer_part"]
                            fpart = intervals[i]["fractional_part"]
                            if ipart == len(state["losses_vec"]):
                                loss = state["losses_vec"][ipart - 1]
                            else:
                                loss = (state["losses_vec"][ipart] - state["losses_vec"][ipart - 1]) * fpart + state["losses_vec"][ipart - 1]

                            qelt_data[qidx]['EventId'] = event_id
                            qelt_data[qidx]['SummaryId'] = state['summary_id']
                            qelt_data[qidx]['Quantile'] = q
                            qelt_data[qidx]['Loss'] = loss

                            qidx += 1
                            if qidx >= qelt_data.shape[0]:
                                # Output array is full
                                selt_idx[0] = idx
                                melt_idx[0] = midx
                                qelt_idx[0] = qidx
                                return cursor, event_id, item_id, 1
                        state["losses_vec"].fill(0)

                        if qidx >= qelt_data.shape[0]:
                            # Output array is full
                            selt_idx[0] = idx
                            melt_idx[0] = midx
                            qelt_idx[0] = qidx
                            return cursor, event_id, item_id, 1

                    # Reset variables
                    sample_mean = 0.0
                    state["sumloss"] = 0.0
                    state["sumlosssqr"] = 0.0
                    state["non_zero_samples"] = 0
                    state["max_loss"] = 0.0
                    state["mean_impacted_exposure"] = 0.0
                    state["max_impacted_exposure"] = 0.0
                    state["analytical_mean"] = 0.0

            else:
                break
        else:
            pass

    # Update the indices
    selt_idx[0] = idx
    melt_idx[0] = midx
    qelt_idx[0] = qidx
    return cursor, event_id, item_id, 0


def read_event_rates_occurrence(occurrence_file):
    """_summary_

    Args:
        occurrence_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(occurrence_file, 'rb') as f:
        date_opts_bytes = f.read(4)
        if not date_opts_bytes or len(date_opts_bytes) < 4:
            logger.error("Occurrence file is empty or corrupted")
            return {}

        no_of_periods_bytes = f.read(4)
        if not no_of_periods_bytes or len(no_of_periods_bytes) < 4:
            logger.error("Occurrence file is empty or corrupted")
            return {}
        no_of_periods = int.from_bytes(no_of_periods_bytes, byteorder='little', signed=True)

        record_size = 12

        data = f.read()

    num_records = len(data) // record_size
    if num_records * record_size != len(data):
        logger.warning("File size does not align with expected record size.")

    event_ids = np.empty(num_records, dtype=oasis_int)
    period_nos = np.empty(num_records, dtype=oasis_int)

    for i in range(num_records):
        offset = i * record_size
        record_bytes = data[offset:offset + record_size]
        if len(record_bytes) < record_size:
            break

        event_id, period_no, _ = struct.unpack('<iii', record_bytes)
        event_ids[i] = event_id
        period_nos[i] = period_no

    max_period_no = np.max(period_nos)
    if max_period_no > no_of_periods:
        logger.error("Maximum period number in occurrence file exceeds that in header.")

    unique_event_ids, counts = np.unique(event_ids, return_counts=True)
    event_rates_values = counts / no_of_periods

    # make sure the event ids are sorted
    sort_idx = np.argsort(unique_event_ids)
    unique_event_ids = unique_event_ids[sort_idx]
    event_rates_values = event_rates_values[sort_idx]

    return unique_event_ids.astype(oasis_int), event_rates_values.astype(oasis_float)


def read_quantile_get_intervals(sample_size, fp):
    intervals_dict = {}

    try:
        with open(fp, "rb") as fin:
            while True:
                data = fin.read(4)  # Reading 4 bytes for a float (32-bit)
                if not data:
                    break

                q = struct.unpack('f', data)[0]

                # Calculate interval index and fractional part
                pos = (sample_size - 1) * q + 1
                integer_part = int(pos)
                fractional_part = pos - integer_part

                intervals_dict[q] = {"integer_part": integer_part, "fractional_part": fractional_part}

    except FileNotFoundError:
        logger.error(f"FATAL: Error opening file {fp}")
        raise FileNotFoundError(f"FATAL: Error opening file {fp}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise RuntimeError(f"An error occurred: {str(e)}")

    intervals = np.zeros(len(intervals_dict), dtype=quantile_interval_dtype)
    for i, (k, v) in enumerate(intervals_dict.items()):
        intervals[i] = (k, v['integer_part'], v['fractional_part'])

    return intervals


def run(run_dir, files_in, selt_output_file=None, melt_output_file=None, qelt_output_file=None):
    compute_selt = selt_output_file is not None
    compute_melt = melt_output_file is not None
    compute_qelt = qelt_output_file is not None

    if not compute_selt and not compute_melt and not compute_qelt:
        logger.warning("No output files specified")

    with ExitStack() as stack:
        streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)
        if stream_source_type != SUMMARY_STREAM_ID:
            raise Exception(f"unsupported stream type {stream_source_type}, {stream_agg_type}")

        intervals = np.array([], dtype=quantile_interval_dtype)
        if compute_qelt:
            intervals = read_quantile_get_intervals(len_sample, os.path.join(run_dir, "input", "quantile.bin"))

        if compute_melt:
            unique_event_ids, event_rates = read_event_rates_occurrence(os.path.join(run_dir, "input", "occurrence.bin"))
        else:
            unique_event_ids = np.array([], dtype=oasis_int)
            event_rates = np.array([], dtype=oasis_float)

        elt_reader = ELTReader(len_sample, compute_selt, compute_melt, compute_qelt, unique_event_ids, event_rates, intervals)

        output_files = {}
        if compute_selt:
            selt_file = stack.enter_context(open(selt_output_file, 'w'))
            selt_file.write('EventId,SummaryId,SampleId,Loss,ImpactedExposure\n')
            output_files['selt'] = selt_file
        else:
            output_files['selt'] = None

        if compute_melt:
            melt_file = stack.enter_context(open(melt_output_file, 'w'))
            melt_file.write(
                'EventId,SummaryId,SampleType,EventRate,ChanceOfLoss,MeanLoss,SDLoss,MaxLoss,FootprintExposure,MeanImpactedExposure,MaxImpactedExposure\n')
            output_files['melt'] = melt_file
        else:
            output_files['melt'] = None

        if compute_qelt:
            qelt_file = stack.enter_context(open(qelt_output_file, 'w'))
            qelt_file.write('EventId,SummaryId,Quantile,Loss\n')
            output_files['qelt'] = qelt_file
        else:
            output_files['qelt'] = None

        for event_id in elt_reader.read_streams(streams_in):
            if compute_selt:
                # Extract SELT data
                data = elt_reader.selt_data[:elt_reader.selt_idx[0]]
                if output_files['selt'] is not None and data.size > 0:
                    np.savetxt(output_files['selt'], data, delimiter=',', fmt='%d,%d,%d,%.2f,%.2f')
                elt_reader.selt_idx[0] = 0

            if compute_melt:
                # Extract MELT data
                melt_data = elt_reader.melt_data[:elt_reader.melt_idx[0]]
                if output_files['melt'] is not None and melt_data.size > 0:
                    np.savetxt(output_files['melt'], melt_data, delimiter=',', fmt='%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f')
                elt_reader.melt_idx[0] = 0

            if compute_qelt:
                # Extract QELT data
                qelt_data = elt_reader.qelt_data[:elt_reader.qelt_idx[0]]
                if output_files['qelt'] is not None and qelt_data.size > 0:
                    np.savetxt(output_files['qelt'], qelt_data, delimiter=',', fmt='%d,%d,%.6f,%.6f')
                elt_reader.qelt_idx[0] = 0


@redirect_logging(exec_name='eltpy')
def main(run_dir='.', files_in=None, selt_output_file=None, melt_output_file=None, qelt_output_file=None, **kwargs):
    run(
        run_dir,
        files_in,
        selt_output_file=selt_output_file,
        melt_output_file=melt_output_file,
        qelt_output_file=qelt_output_file
    )
