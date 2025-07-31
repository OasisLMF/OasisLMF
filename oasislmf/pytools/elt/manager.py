# elt/manager.py

import logging
import numpy as np
import numba as nb
from contextlib import ExitStack
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oasislmf.pytools.common.data import (DEFAULT_BUFFER_SIZE, MEAN_TYPE_ANALYTICAL, MEAN_TYPE_SAMPLE, oasis_int, oasis_float,
                                          oasis_int_size, oasis_float_size, write_ndarray_to_fmt_csv)
from oasislmf.pytools.common.event_stream import (MAX_LOSS_IDX, MEAN_IDX, EventReader, init_streams_in,
                                                  mv_read, SUMMARY_STREAM_ID)
from oasislmf.pytools.common.input_files import read_event_rates, read_quantile
from oasislmf.pytools.elt.data import MELT_dtype, MELT_fmt, MELT_headers, QELT_dtype, QELT_fmt, QELT_headers, SELT_dtype, SELT_fmt, SELT_headers
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


class ELTReader(EventReader):
    def __init__(self, len_sample, compute_selt, compute_melt, compute_qelt, unique_event_ids, event_rates, intervals):
        self.logger = logger

        # Buffer for SELT data
        self.selt_data = np.zeros(DEFAULT_BUFFER_SIZE, dtype=SELT_dtype)
        self.selt_idx = np.zeros(1, dtype=np.int64)

        # Buffer for MELT data
        self.melt_data = np.zeros(DEFAULT_BUFFER_SIZE, dtype=MELT_dtype)
        self.melt_idx = np.zeros(1, dtype=np.int64)

        # Buffer for QELT data
        self.qelt_data = np.zeros(DEFAULT_BUFFER_SIZE, dtype=QELT_dtype)
        self.qelt_idx = np.zeros(1, dtype=np.int64)

        read_buffer_state_dtype = np.dtype([
            ('len_sample', oasis_int),
            ('read_summary_set_id', np.bool_),
            ('reading_losses', np.bool_),
            ('compute_selt', np.bool_),
            ('compute_melt', np.bool_),
            ('compute_qelt', np.bool_),
            ('summary_id', oasis_int),
            ('impacted_exposure', oasis_float),
            ('sumloss', np.float64),  # intermediary value need higher precision to not accumulate error
            ('sumlosssqr', np.float64),  # intermediary value need higher precision to not accumulate error
            ('non_zero_samples', oasis_int),
            ('max_loss', oasis_float),
            ('analytical_mean', oasis_float),
            ('losses_vec', oasis_float, (len_sample,)),
        ])

        self.state = np.zeros(1, dtype=read_buffer_state_dtype)[0]
        self.state["len_sample"] = len_sample
        self.state["reading_losses"] = False
        self.state["read_summary_set_id"] = False
        self.state["compute_selt"] = compute_selt
        self.state["compute_melt"] = compute_melt
        self.state["compute_qelt"] = compute_qelt
        self.unique_event_ids = unique_event_ids.astype(oasis_int)
        self.event_rates = event_rates.astype(oasis_float)
        self.intervals = intervals

        self.curr_file_idx = None  # Current summary file idx being read

    def get_data(self, out_type):
        if out_type == "selt":
            return self.selt_data
        elif out_type == "melt":
            return self.melt_data
        elif out_type == "qelt":
            return self.qelt_data
        else:
            raise RuntimeError(f"Unknown out_type {out_type}")

    def get_data_idx(self, out_type):
        if out_type == "selt":
            return self.selt_idx
        elif out_type == "melt":
            return self.melt_idx
        elif out_type == "qelt":
            return self.qelt_idx
        else:
            raise RuntimeError(f"Unknown out_type {out_type}")

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, file_idx):
        # Check for new file idx to read summary_set_id at the start of each summary file stream
        # This is not done by init_streams_in as the summary_set_id is unique to the summary_stream only
        if self.curr_file_idx is not None and self.curr_file_idx != file_idx:
            self.curr_file_idx = file_idx
            self.state["read_summary_set_id"] = False
        else:
            self.curr_file_idx = file_idx

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
            # Read summary header
            if valid_buff - cursor >= 3 * oasis_int_size + oasis_float_size:
                # Need to read summary_set_id from summary info first
                if not state["read_summary_set_id"]:
                    _, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                    state["read_summary_set_id"] = True
                event_id_new, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                if last_event_id != 0 and event_id_new != last_event_id:
                    # New event, return to process the previous event
                    _update_idxs()
                    return cursor - oasis_int_size, last_event_id, item_id, 1
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
                cursor += oasis_float_size  # Read extra 0 for end of record
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
                            sample_type=MEAN_TYPE_ANALYTICAL,
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
                            sample_type=MEAN_TYPE_SAMPLE,
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
                        q = intervals[i]["quantile"]
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
        unique_event_ids, event_rates = read_event_rates(Path(run_dir, "input"))
        include_event_rate = unique_event_ids.size > 0

    intervals = read_quantile(sample_size, Path(run_dir, "input"), return_empty=not compute_qelt)

    file_data = {
        "unique_event_ids": unique_event_ids,
        "event_rates": event_rates,
        "include_event_rate": include_event_rate,
        "intervals": intervals,
    }
    return file_data


def run(
    run_dir,
    files_in,
    selt_output_file=None,
    melt_output_file=None,
    qelt_output_file=None,
    noheader=False,
    output_format="csv",
):
    """Runs ELT calculations
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
        files_in (str | os.PathLike): Path to summary binary input file
        selt_output_file (str, optional): Path to SELT output file. Defaults to None.
        melt_output_file (str, optional): Path to MELT output file. Defaults to None.
        qelt_output_file (str, optional): Path to QELT output file. Defaults to None.
        noheader (bool): Boolean value to skip header in output file. Defaults to False.
        output_format (str): Output format extension. Defaults to "csv".
    """
    outmap = {
        "selt": {
            "compute": selt_output_file is not None,
            "file_path": selt_output_file,
            "fmt": SELT_fmt,
            "headers": SELT_headers,
            "file": None,
        },
        "melt": {
            "compute": melt_output_file is not None,
            "file_path": melt_output_file,
            "fmt": MELT_fmt,
            "headers": MELT_headers,
            "file": None,
        },
        "qelt": {
            "compute": qelt_output_file is not None,
            "file_path": qelt_output_file,
            "fmt": QELT_fmt,
            "headers": QELT_headers,
            "file": None,
        },
    }

    output_format = "." + output_format
    output_binary = output_format == ".bin"
    output_parquet = output_format == ".parquet"
    # Check for correct suffix
    for path in [v["file_path"] for v in outmap.values()]:
        if path is None:
            continue
        if Path(path).suffix == "":  # Ignore suffix for pipes
            continue
        if (Path(path).suffix != output_format):
            raise ValueError(f"Invalid file extension for {output_format}, got {path},")

    if run_dir is None:
        run_dir = './work'

    if not all([v["compute"] for v in outmap.values()]):
        logger.warning("No output files specified")

    with ExitStack() as stack:
        streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)
        if stream_source_type != SUMMARY_STREAM_ID:
            raise Exception(f"unsupported stream type {stream_source_type}, {stream_agg_type}")

        file_data = read_input_files(
            run_dir,
            outmap["melt"]["compute"],
            outmap["qelt"]["compute"],
            len_sample
        )
        elt_reader = ELTReader(
            len_sample,
            outmap["selt"]["compute"],
            outmap["melt"]["compute"],
            outmap["qelt"]["compute"],
            file_data["unique_event_ids"],
            file_data["event_rates"],
            file_data["intervals"]
        )

        # Initialise output files ELT
        if output_binary:
            for out_type in outmap:
                if not outmap[out_type]["compute"]:
                    continue
                out_file = stack.enter_context(open(outmap[out_type]["file_path"], 'wb'))
                outmap[out_type]["file"] = out_file
        elif output_parquet:
            for out_type in outmap:
                if not outmap[out_type]["compute"]:
                    continue
                temp_df = pd.DataFrame(elt_reader.get_data(out_type), columns=outmap[out_type]["headers"])
                temp_table = pa.Table.from_pandas(temp_df)
                out_file = pq.ParquetWriter(outmap[out_type]["file_path"], temp_table.schema)
                outmap[out_type]["file"] = out_file
        else:
            for out_type in outmap:
                if not outmap[out_type]["compute"]:
                    continue
                out_file = stack.enter_context(open(outmap[out_type]["file_path"], 'w'))
                if not noheader:
                    csv_headers = ','.join(outmap[out_type]["headers"])
                    out_file.write(csv_headers + '\n')
                outmap[out_type]["file"] = out_file

        # Process summary files
        for event_id in elt_reader.read_streams(streams_in):
            for out_type in outmap:
                if not outmap[out_type]["compute"]:
                    continue

                data_idx = elt_reader.get_data_idx(out_type)
                data = elt_reader.get_data(out_type)[:data_idx[0]]

                if outmap[out_type]["file"] is not None and data.size > 0:
                    if output_binary:
                        data.tofile(outmap[out_type]["file"])
                    elif output_parquet:
                        data_df = pd.DataFrame(data)
                        data_table = pa.Table.from_pandas(data_df)
                        outmap[out_type]["file"].write_table(data_table)
                    else:
                        write_ndarray_to_fmt_csv(
                            outmap[out_type]["file"],
                            data,
                            outmap[out_type]["headers"],
                            outmap[out_type]["fmt"]
                        )
                data_idx[0] = 0


@redirect_logging(exec_name='eltpy')
def main(run_dir='.', files_in=None, selt=None, melt=None, qelt=None, noheader=False, ext="csv", **kwargs):
    run(
        run_dir,
        files_in,
        selt_output_file=selt,
        melt_output_file=melt,
        qelt_output_file=qelt,
        noheader=noheader,
        output_format=ext,
    )
