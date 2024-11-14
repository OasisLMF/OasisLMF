# elt/manager.py

import logging
import numpy as np
import numba as nb
import os
import struct
from contextlib import ExitStack

from oasislmf.pytools.common.data import (oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import (MAX_LOSS_IDX, MEAN_IDX, EventReader, init_streams_in,
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
        self.selt_data = np.zeros(1000000, dtype=SELT_dtype)  # write buffer for SELT
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
        self.melt_data = np.zeros(1000000, dtype=MELT_dtype)  # write buffer for MELT
        self.melt_idx = np.zeros(1, dtype=np.int64)

        # Buffer for QELT data
        self.qelt_data = np.zeros(1000000, dtype=QELT_dtype)
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


@nb.njit(cache=True, error_model="numpy")
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
                break  # Not enough for whole summary header

        if state["reading_losses"]:
            if valid_buff - cursor < oasis_int_size + oasis_float_size:
                break  # Not enough for whole record

            # Read sidx
            sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            if sidx == 0:  # sidx == 0, end of record
                if state["compute_melt"]:
                    # Compute MELT statistics and store them
                    if state["non_zero_samples"] > 0:
                        sample_mean = state["sumloss"] / state["len_sample"]
                        if state["non_zero_samples"] > 1:
                            variance = (
                                state["sumlosssqr"] - (
                                    (state["sumloss"] * state["sumloss"]) / state["len_sample"]
                                )
                            ) / (state["len_sample"] - 1)
                            if variance / state["sumlosssqr"] < 1e-7:
                                variance = 0
                            std_dev = np.sqrt(variance)
                        else:
                            std_dev = 0

                        chance_of_loss = state["non_zero_samples"] / state["len_sample"]
                        mean_imp_exp = state["impacted_exposure"] * chance_of_loss

                        if unique_event_ids.size > 0:
                            idx_ev = np.searchsorted(unique_event_ids, event_id)
                            if idx_ev < unique_event_ids.size and unique_event_ids[idx_ev] == event_id:
                                event_rate = event_rates[idx_ev]
                            else:
                                event_rate = np.nan
                        else:
                            event_rate = np.nan

                        # Update MELT data (analytical mean)
                        melt_data[midx]['EventId'] = event_id
                        melt_data[midx]['SummaryId'] = state["summary_id"]
                        melt_data[midx]['SampleType'] = 1
                        melt_data[midx]['EventRate'] = event_rate
                        melt_data[midx]['ChanceOfLoss'] = 0
                        melt_data[midx]['MeanLoss'] = state["analytical_mean"]
                        melt_data[midx]['SDLoss'] = 0
                        melt_data[midx]['MaxLoss'] = state["max_loss"]
                        melt_data[midx]['FootprintExposure'] = state["impacted_exposure"]
                        melt_data[midx]['MeanImpactedExposure'] = state["impacted_exposure"]
                        melt_data[midx]['MaxImpactedExposure'] = state["impacted_exposure"]
                        midx += 1

                        # Update MELT data (sample mean)
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

                # Update QELT data
                if state["compute_qelt"]:
                    state["losses_vec"].sort()

                    # Calculate loss for per quantile interval
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

                    if qidx >= qelt_data.shape[0]:
                        # Output array is full
                        selt_idx[0] = idx
                        melt_idx[0] = midx
                        qelt_idx[0] = qidx
                        return cursor, event_id, item_id, 1

                # Reset variables
                sample_mean = 0
                state["sumloss"] = 0
                state["sumlosssqr"] = 0
                state["non_zero_samples"] = 0
                state["max_loss"] = 0
                state["mean_impacted_exposure"] = 0
                state["max_impacted_exposure"] = 0
                state["analytical_mean"] = 0
                state["losses_vec"].fill(0)
                state["reading_losses"] = False
                continue

            # Read loss
            loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
            if sidx == MAX_LOSS_IDX:
                if state["compute_melt"]:
                    state["max_loss"] = loss
            else:  # Normal data record
                # Update SELT data
                if state["compute_selt"]:
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
                # Update MELT variables
                if state["compute_melt"]:
                    if sidx > 0:
                        state["sumloss"] += loss
                        state["sumlosssqr"] += loss * loss
                        if loss > 0:
                            state["non_zero_samples"] += 1
                # Update QELT variables
                if state["compute_qelt"]:
                    if sidx > 0:
                        state["losses_vec"][sidx - 1] = loss

            if sidx == MEAN_IDX:
                # Update MELT variables
                if state["compute_melt"]:
                    state["analytical_mean"] = loss
        else:
            pass  # Should never reach here anyways

    # Update the indices
    selt_idx[0] = idx
    melt_idx[0] = midx
    qelt_idx[0] = qidx
    return cursor, event_id, item_id, 0


def read_event_rate_csv(event_rate_file):
    """Reads event rates from a CSV file

    Args:
        event_rate_file (str): Path to the event rate CSV file

    Returns:
        (ndarray, ndarray): unique event id and event rates
    """
    try:
        data = np.genfromtxt(event_rate_file, delimiter=',', skip_header=1, dtype=[('event_id', oasis_int), ('rate', oasis_float)])
        if data is None or data.size == 0:
            logger.info(f"Event rate file {event_rate_file} is empty, proceeding without event rates.")
            return np.array([], dtype=oasis_int), np.array([], dtype=oasis_float)
        unique_event_ids = data['event_id']
        event_rates = data['rate']

        # Make sure event_ids are sorted
        sort_idx = np.argsort(unique_event_ids)
        unique_event_ids = unique_event_ids[sort_idx]
        event_rates = event_rates[sort_idx]
        return unique_event_ids, event_rates
    except FileNotFoundError:
        logger.info(f"Event rate file {event_rate_file} not found, proceeding without event rates.")
        return np.array([], dtype=oasis_int), np.array([], dtype=oasis_float)
    except Exception as e:
        logger.warning(f"An error occurred while reading event rate file: {str(e)}")
        return np.array([], dtype=oasis_int), np.array([], dtype=oasis_float)


def read_quantile_get_intervals(sample_size, fp):
    """Generate a quantile interval Dictionary based on sample size and quantile binary file

    Args:
        sample_size (int): Sample size
        fp (str): File path to quantile binary input

    Returns:
        quantile_interval_dtype: Numpy array emulating a dictionary for numba
    """
    intervals_dict = {}

    try:
        with open(fp, "rb") as fin:
            while True:
                data = fin.read(4)  # Reading 4 bytes for a float32
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

    # Convert to numpy array
    intervals = np.zeros(len(intervals_dict), dtype=quantile_interval_dtype)
    for i, (k, v) in enumerate(intervals_dict.items()):
        intervals[i] = (k, v['integer_part'], v['fractional_part'])

    return intervals


def run(run_dir, files_in, selt_output_file=None, melt_output_file=None, qelt_output_file=None, noheader=False):
    compute_selt = selt_output_file is not None
    compute_melt = melt_output_file is not None
    compute_qelt = qelt_output_file is not None
    if run_dir is None:
        run_dir = './work'

    if not compute_selt and not compute_melt and not compute_qelt:
        logger.warning("No output files specified")

    with ExitStack() as stack:
        streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)
        if stream_source_type != SUMMARY_STREAM_ID:
            raise Exception(f"unsupported stream type {stream_source_type}, {stream_agg_type}")

        # Initialise intervals array
        intervals = np.array([], dtype=quantile_interval_dtype)
        if compute_qelt:
            intervals = read_quantile_get_intervals(len_sample, os.path.join(run_dir, "input", "quantile.bin"))

        # Initialise event rate data
        if compute_melt:
            unique_event_ids, event_rates = read_event_rate_csv(os.path.join(run_dir, "input", "event_rates.csv"))
            include_event_rate = unique_event_ids.size > 0
        else:
            unique_event_ids = np.array([], dtype=oasis_int)
            event_rates = np.array([], dtype=oasis_float)
            include_event_rate = False

        elt_reader = ELTReader(len_sample, compute_selt, compute_melt, compute_qelt, unique_event_ids, event_rates, intervals)

        # Initialise csv column names for ELT files
        output_files = {}
        if compute_selt:
            selt_file = stack.enter_context(open(selt_output_file, 'w'))
            if not noheader:
                selt_file.write('EventId,SummaryId,SampleId,Loss,ImpactedExposure\n')
            output_files['selt'] = selt_file
        else:
            output_files['selt'] = None

        if compute_melt:
            melt_file = stack.enter_context(open(melt_output_file, 'w'))
            if not noheader:
                if include_event_rate:
                    melt_file.write(
                        'EventId,SummaryId,SampleType,EventRate,ChanceOfLoss,MeanLoss,SDLoss,MaxLoss,FootprintExposure,MeanImpactedExposure,MaxImpactedExposure\n')
                else:
                    melt_file.write(
                        'EventId,SummaryId,SampleType,ChanceOfLoss,MeanLoss,SDLoss,MaxLoss,FootprintExposure,MeanImpactedExposure,MaxImpactedExposure\n')
            output_files['melt'] = melt_file
        else:
            output_files['melt'] = None

        if compute_qelt:
            qelt_file = stack.enter_context(open(qelt_output_file, 'w'))
            if not noheader:
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
                    if include_event_rate:
                        np.savetxt(output_files['melt'], melt_data, delimiter=',', fmt='%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f')
                    else:
                        melt_data = melt_data[['EventId', 'SummaryId', 'SampleType', 'ChanceOfLoss',
                                              'MeanLoss', 'SDLoss', 'MaxLoss', 'FootprintExposure',
                                               'MeanImpactedExposure', 'MaxImpactedExposure']]
                        np.savetxt(output_files['melt'], melt_data, delimiter=',', fmt='%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f')
                elt_reader.melt_idx[0] = 0

            if compute_qelt:
                # Extract QELT data
                qelt_data = elt_reader.qelt_data[:elt_reader.qelt_idx[0]]
                if output_files['qelt'] is not None and qelt_data.size > 0:
                    np.savetxt(output_files['qelt'], qelt_data, delimiter=',', fmt='%d,%d,%.6f,%.6f')
                elt_reader.qelt_idx[0] = 0


@redirect_logging(exec_name='eltpy')
def main(run_dir='.', files_in=None, selt=None, melt=None, qelt=None, noheader=None, **kwargs):
    run(
        run_dir,
        files_in,
        selt_output_file=selt,
        melt_output_file=melt,
        qelt_output_file=qelt,
        noheader=noheader
    )
