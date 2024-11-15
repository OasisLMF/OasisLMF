# elt/manager.py

import logging
import numpy as np
import numba as nb
import struct
from contextlib import ExitStack
from pathlib import Path

from oasislmf.pytools.common.data import (oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import (MAX_LOSS_IDX, MEAN_IDX, EventReader, init_streams_in,
                                                  mv_read, SUMMARY_STREAM_ID)
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


_MEAN_TYPE_ANALYTICAL = 1
_MEAN_TYPE_SAMPLE = 2

SELT_output = [
    ('EventId', oasis_int, '%d'),
    ('SummaryId', oasis_int, '%d'),
    ('SampleId', oasis_int, '%d'),
    ('Loss', oasis_float, '%.2f'),
    ('ImpactedExposure', oasis_float, '%.2f'),
]

MELT_output = [
    ('EventId', oasis_int, '%d'),
    ('SummaryId', oasis_int, '%d'),
    ('SampleType', oasis_int, '%d'),
    ('EventRate', oasis_float, '%.6f'),
    ('ChanceOfLoss', oasis_float, '%.6f'),
    ('MeanLoss', oasis_float, '%.6f'),
    ('SDLoss', oasis_float, '%.6f'),
    ('MaxLoss', oasis_float, '%.6f'),
    ('FootprintExposure', oasis_float, '%.6f'),
    ('MeanImpactedExposure', oasis_float, '%.6f'),
    ('MaxImpactedExposure', oasis_float, '%.6f'),
]

QELT_output = [
    ('EventId', oasis_int, '%d'),
    ('SummaryId', oasis_int, '%d'),
    ('Quantile', oasis_float, '%.6f'),
    ('Loss', oasis_float, '%.6f'),
]


class ELTReader(EventReader):
    def __init__(self, len_sample, compute_selt, compute_melt, compute_qelt, unique_event_ids, event_rates, intervals):
        self.logger = logger

        SELT_dtype = np.dtype([(c[0], c[1]) for c in SELT_output])
        MELT_dtype = np.dtype([(c[0], c[1]) for c in MELT_output])
        QELT_dtype = np.dtype([(c[0], c[1]) for c in QELT_output])

        # Buffer for SELT data
        self.selt_data = np.zeros(1000000, dtype=SELT_dtype)
        self.selt_idx = np.zeros(1, dtype=np.int64)

        # Buffer for MELT data
        self.melt_data = np.zeros(1000000, dtype=MELT_dtype)
        self.melt_idx = np.zeros(1, dtype=np.int64)

        # Buffer for QELT data
        self.qelt_data = np.zeros(1000000, dtype=QELT_dtype)
        self.qelt_idx = np.zeros(1, dtype=np.int64)

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
        self.intervals = intervals

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id):
        # Pass state variables to read_buffer
        cursor, event_id, item_id, ret = read_buffer(
            byte_mv, cursor, valid_buff, event_id, item_id, self.selt_data, self.selt_idx,
            self.state, self.melt_data, self.melt_idx, self.qelt_data, self.qelt_idx, self.intervals,
            self.unique_event_ids, self.event_rates
        )
        return cursor, event_id, item_id, ret


@nb.njit(cache=True)
def _update_selt_data(
    selt_data, si, event_id, summary_id,
    sample_id,
    loss,
    impacted_exposure,
):
    selt_data[si]['EventId'] = event_id
    selt_data[si]['SummaryId'] = summary_id
    selt_data[si]['SampleId'] = sample_id
    selt_data[si]['Loss'] = loss
    selt_data[si]['ImpactedExposure'] = impacted_exposure


@nb.njit(cache=True)
def _update_melt_data(
    melt_data, mi, event_id, summary_id,
    sample_type,
    event_rate,
    chance_of_loss,
    meanloss,
    sdloss,
    maxloss,
    footprint_exposure,
    mean_impacted_exposure,
    max_impacted_exposure
):
    melt_data[mi]['EventId'] = event_id
    melt_data[mi]['SummaryId'] = summary_id
    melt_data[mi]['SampleType'] = sample_type
    melt_data[mi]['EventRate'] = event_rate
    melt_data[mi]['ChanceOfLoss'] = chance_of_loss
    melt_data[mi]['MeanLoss'] = meanloss
    melt_data[mi]['SDLoss'] = sdloss
    melt_data[mi]['MaxLoss'] = maxloss
    melt_data[mi]["FootprintExposure"] = footprint_exposure
    melt_data[mi]["MeanImpactedExposure"] = mean_impacted_exposure
    melt_data[mi]["MaxImpactedExposure"] = max_impacted_exposure


@nb.njit(cache=True)
def _update_qelt_data(
    qelt_data, qi, event_id, summary_id,
    quantile,
    loss,
):
    qelt_data[qi]["EventId"] = event_id
    qelt_data[qi]["SummaryId"] = summary_id
    qelt_data[qi]["Quantile"] = quantile
    qelt_data[qi]["Loss"] = loss


@nb.njit(cache=True, error_model="numpy")
def read_buffer(
        byte_mv, cursor, valid_buff, event_id, item_id,
        selt_data, selt_idx, state,
        melt_data, melt_idx,
        qelt_data, qelt_idx, intervals,
        unique_event_ids, event_rates
):
    # Initialise idxs
    last_event_id = event_id
    si = selt_idx[0]
    mi = melt_idx[0]
    qi = qelt_idx[0]

    # Helper functions
    def _update_idxs():
        selt_idx[0] = si
        melt_idx[0] = mi
        qelt_idx[0] = qi

    def _reset_state():
        state["reading_losses"] = False
        state["non_zero_samples"] = 0
        state["max_loss"] = 0
        state["analytical_mean"] = 0
        state["losses_vec"].fill(0)
        state["sumloss"] = 0
        state["sumlosssqr"] = 0

    def _get_mean_and_sd_loss():
        meanloss = state["sumloss"] / state["len_sample"]
        if state["non_zero_samples"] > 1:
            variance = (
                state["sumlosssqr"] - (
                    (state["sumloss"] * state["sumloss"]) / state["len_sample"]
                )
            ) / (state["len_sample"] - 1)

            # Tolerance check
            if variance / state["sumlosssqr"] < 1e-7:
                variance = 0
            sdloss = np.sqrt(variance)
        else:
            sdloss = 0
        return meanloss, sdloss

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
                        meanloss, sdloss = _get_mean_and_sd_loss()
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
                        _update_melt_data(
                            melt_data, mi,
                            event_id=event_id,
                            summary_id=state["summary_id"],
                            sample_type=_MEAN_TYPE_ANALYTICAL,
                            event_rate=event_rate,
                            chance_of_loss=0,
                            meanloss=state["analytical_mean"],
                            sdloss=0,
                            maxloss=state["max_loss"],
                            footprint_exposure=state["impacted_exposure"],
                            mean_impacted_exposure=state["impacted_exposure"],
                            max_impacted_exposure=state["impacted_exposure"]
                        )
                        mi += 1

                        # Update MELT data (sample mean)
                        _update_melt_data(
                            melt_data, mi,
                            event_id=event_id,
                            summary_id=state["summary_id"],
                            sample_type=_MEAN_TYPE_SAMPLE,
                            event_rate=event_rate,
                            chance_of_loss=chance_of_loss,
                            meanloss=meanloss,
                            sdloss=sdloss,
                            maxloss=state["max_loss"],
                            footprint_exposure=state["impacted_exposure"],
                            mean_impacted_exposure=mean_imp_exp,
                            max_impacted_exposure=state["impacted_exposure"]
                        )
                        mi += 1

                        if mi >= melt_data.shape[0]:
                            # Output array is full
                            _update_idxs()
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

                        _update_qelt_data(
                            qelt_data, qi,
                            event_id=event_id,
                            summary_id=state['summary_id'],
                            quantile=q,
                            loss=loss
                        )
                        qi += 1
                        if qi >= qelt_data.shape[0]:
                            # Output array is full
                            _update_idxs()
                            return cursor, event_id, item_id, 1

                # Reset variables
                _reset_state()
                continue

            # Read loss
            loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
            if sidx == MAX_LOSS_IDX:
                state["max_loss"] = loss
            else:  # Normal data record
                # Update SELT data
                if state["compute_selt"]:
                    _update_selt_data(
                        selt_data, si,
                        event_id=event_id,
                        summary_id=state["summary_id"],
                        sample_id=sidx,
                        loss=loss,
                        impacted_exposure=state["impacted_exposure"]
                    )
                    si += 1
                    if si >= selt_data.shape[0]:
                        # Output array is full
                        _update_idxs()
                        return cursor, event_id, item_id, 1

                if sidx > 0:
                    # Update MELT variables
                    state["sumloss"] += loss
                    state["sumlosssqr"] += loss * loss
                    if loss > 0:
                        state["non_zero_samples"] += 1
                    # Update QELT variables
                    state["losses_vec"][sidx - 1] = loss
            if sidx == MEAN_IDX:
                # Update MELT variables
                state["analytical_mean"] = loss
        else:
            pass  # Should never reach here anyways

    # Update the indices
    _update_idxs()
    return cursor, event_id, item_id, 0


def read_event_rate(event_rate_file):
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


def read_quantile(quantile_fp, sample_size, compute_qelt):
    """Generate a quantile interval Dictionary based on sample size and quantile binary file
    Args:
        sample_size (int): Sample size
        fp (str): File path to quantile binary input
    Returns:
        intervals (quantile_interval_dtype): Numpy array emulating a dictionary for numba
    """
    intervals_dict = {}
    quantile_interval_dtype = np.dtype([
        ('q', oasis_float),
        ('integer_part', oasis_int),
        ('fractional_part', oasis_float),
    ])

    if not compute_qelt:
        return np.array([], dtype=quantile_interval_dtype)

    try:
        with open(quantile_fp, "rb") as fin:
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
        logger.error(f"FATAL: Error opening file {quantile_fp}")
        raise FileNotFoundError(f"FATAL: Error opening file {quantile_fp}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise RuntimeError(f"An error occurred: {str(e)}")

    # Convert to numpy array
    intervals = np.zeros(len(intervals_dict), dtype=quantile_interval_dtype)
    for i, (k, v) in enumerate(intervals_dict.items()):
        intervals[i] = (k, v['integer_part'], v['fractional_part'])

    return intervals


def read_input_files(run_dir, compute_melt, compute_qelt, sample_size):
    """Reads all input files and returns a dict of relevant data
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
        compute_melt (bool): Compute MELT bool
        compute_qelt (bool): Compute QELT bool
        sample_size (int): Sample size
    Returns:
        file_data (Dict[str, Any]): A dict of relevent data extracted from files
    """
    unique_event_ids = np.array([], dtype=oasis_int)
    event_rates = np.array([], dtype=oasis_float)
    include_event_rate = False
    if compute_melt:
        unique_event_ids, event_rates = read_event_rate(Path(run_dir, "input", "event_rates.csv"))
        include_event_rate = unique_event_ids.size > 0

    intervals = read_quantile(Path(run_dir, "input", "quantile.bin"), sample_size, compute_qelt)

    file_data = {
        "unique_event_ids": unique_event_ids,
        "event_rates": event_rates,
        "include_event_rate": include_event_rate,
        "intervals": intervals,
    }
    return file_data


def run(run_dir, files_in, selt_output_file=None, melt_output_file=None, qelt_output_file=None, noheader=False):
    """Runs ELT calculations
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
        files_in (str | os.PathLike): Path to summary binary input file
        selt_output_file (str, optional): Path to SPLT output file. Defaults to None.
        melt_output_file (str, optional): Path to MPLT output file. Defaults to None.
        qelt_output_file (str, optional): Path to QPLT output file. Defaults to None.
        noheader (bool): Boolean value to skip header in output file
    """
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

        file_data = read_input_files(run_dir, compute_melt, compute_qelt, len_sample)
        include_event_rate = file_data["include_event_rate"]
        elt_reader = ELTReader(
            len_sample,
            compute_selt,
            compute_melt,
            compute_qelt,
            file_data["unique_event_ids"],
            file_data["event_rates"],
            file_data["intervals"]
        )

        # Initialise csv column names for ELT files
        output_files = {}
        if compute_selt:
            selt_file = stack.enter_context(open(selt_output_file, 'w'))
            if not noheader:
                SELT_headers = ','.join([c[0] for c in SELT_output])
                selt_file.write(SELT_headers + '\n')
            output_files['selt'] = selt_file
        else:
            output_files['selt'] = None

        if compute_melt:
            melt_file = stack.enter_context(open(melt_output_file, 'w'))
            if not noheader:
                if include_event_rate:
                    MELT_headers = ','.join([c[0] for c in MELT_output])
                    melt_file.write(MELT_headers + '\n')
                else:
                    MELT_headers = ','.join([c[0] for c in MELT_output if c[0] != "EventRate"])
                    melt_file.write(MELT_headers + '\n')
            output_files['melt'] = melt_file
        else:
            output_files['melt'] = None

        if compute_qelt:
            qelt_file = stack.enter_context(open(qelt_output_file, 'w'))
            if not noheader:
                QELT_headers = ','.join([c[0] for c in QELT_output])
                qelt_file.write(QELT_headers + '\n')
            output_files['qelt'] = qelt_file
        else:
            output_files['qelt'] = None

        SELT_fmt = ','.join([c[2] for c in SELT_output])
        if include_event_rate:
            MELT_fmt = ','.join([c[2] for c in MELT_output])
        else:
            MELT_fmt = ','.join([c[2] for c in MELT_output if c[0] != "EventRate"])
        QELT_fmt = ','.join([c[2] for c in QELT_output])

        for event_id in elt_reader.read_streams(streams_in):
            if compute_selt:
                # Extract SELT data
                data = elt_reader.selt_data[:elt_reader.selt_idx[0]]
                if output_files['selt'] is not None and data.size > 0:
                    np.savetxt(output_files['selt'], data, delimiter=',', fmt=SELT_fmt)
                elt_reader.selt_idx[0] = 0

            if compute_melt:
                # Extract MELT data
                melt_data = elt_reader.melt_data[:elt_reader.melt_idx[0]]
                if output_files['melt'] is not None and melt_data.size > 0:
                    if include_event_rate:
                        MELT_cols = [c[0] for c in MELT_output]
                    else:
                        MELT_cols = [c[0] for c in MELT_output if c[0] != "EventRate"]
                    melt_data = melt_data[MELT_cols]
                    np.savetxt(output_files['melt'], melt_data, delimiter=',', fmt=MELT_fmt)
                elt_reader.melt_idx[0] = 0

            if compute_qelt:
                # Extract QELT data
                qelt_data = elt_reader.qelt_data[:elt_reader.qelt_idx[0]]
                if output_files['qelt'] is not None and qelt_data.size > 0:
                    np.savetxt(output_files['qelt'], qelt_data, delimiter=',', fmt=QELT_fmt)
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
