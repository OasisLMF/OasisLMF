# plt/manager.py

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
from oasislmf.pytools.common.event_stream import (MAX_LOSS_IDX, MEAN_IDX, NUMBER_OF_AFFECTED_RISK_IDX, EventReader, init_streams_in,
                                                  mv_read, SUMMARY_STREAM_ID)
from oasislmf.pytools.common.input_files import occ_get_date, read_occurrence, read_periods, read_quantile
from oasislmf.pytools.plt.data import MPLT_dtype, MPLT_fmt, MPLT_headers, QPLT_dtype, QPLT_fmt, QPLT_headers, SPLT_dtype, SPLT_fmt, SPLT_headers
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


class PLTReader(EventReader):
    def __init__(
        self,
        len_sample,
        compute_splt,
        compute_mplt,
        compute_qplt,
        occ_map,
        period_weights,
        granular_date,
        intervals,
    ):
        self.logger = logger

        # Buffer for SPLT data
        self.splt_data = np.zeros(DEFAULT_BUFFER_SIZE, dtype=SPLT_dtype)
        self.splt_idx = np.zeros(1, dtype=np.int64)

        # Buffer for MPLT data
        self.mplt_data = np.zeros(DEFAULT_BUFFER_SIZE, dtype=MPLT_dtype)
        self.mplt_idx = np.zeros(1, dtype=np.int64)

        # Buffer for QPLT data
        self.qplt_data = np.zeros(DEFAULT_BUFFER_SIZE, dtype=QPLT_dtype)
        self.qplt_idx = np.zeros(1, dtype=np.int64)

        read_buffer_state_dtype = np.dtype([
            ('len_sample', oasis_int),
            ('reading_losses', np.bool_),
            ('read_summary_set_id', np.bool_),
            ('compute_splt', np.bool_),
            ('compute_mplt', np.bool_),
            ('compute_qplt', np.bool_),
            ('summary_id', oasis_int),
            ('exposure_value', oasis_float),
            ('max_loss', oasis_float),
            ('mean_impacted_exposure', oasis_float),
            ('max_impacted_exposure', oasis_float),
            ('chance_of_loss', oasis_float),
            ('vrec', oasis_float, (len_sample,)),
            ('sumloss', oasis_float),
            ('sumlosssqr', oasis_float),
            ('hasrec', np.bool_),
        ])

        self.state = np.zeros(1, dtype=read_buffer_state_dtype)[0]
        self.state["reading_losses"] = False  # Set to true after reading header in read_buffer
        self.state["read_summary_set_id"] = False
        self.state["len_sample"] = len_sample
        self.state["compute_splt"] = compute_splt
        self.state["compute_mplt"] = compute_mplt
        self.state["compute_qplt"] = compute_qplt
        self.state["hasrec"] = False
        self.occ_map = occ_map
        self.period_weights = period_weights
        self.granular_date = granular_date
        self.intervals = intervals

        self.curr_file_idx = None  # Current summary file idx being read

    def get_data(self, out_type):
        if out_type == "splt":
            return self.splt_data
        elif out_type == "mplt":
            return self.mplt_data
        elif out_type == "qplt":
            return self.qplt_data
        else:
            raise RuntimeError(f"Unknown out_type {out_type}")

    def get_data_idx(self, out_type):
        if out_type == "splt":
            return self.splt_idx
        elif out_type == "mplt":
            return self.mplt_idx
        elif out_type == "qplt":
            return self.qplt_idx
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
            byte_mv, cursor, valid_buff, event_id, item_id,
            self.state,
            self.splt_data, self.splt_idx,
            self.mplt_data, self.mplt_idx,
            self.qplt_data, self.qplt_idx,
            self.occ_map,
            self.period_weights,
            self.granular_date,
            self.intervals,
        )
        return cursor, event_id, item_id, ret


@nb.njit(cache=True)
def _update_splt_data(
    splt_data, si, period_weights, granular_date,
    record,
    event_id,
    summary_id,
    sidx,
    loss,
    impacted_exposure
):
    """updates splt_data to write to output
    """
    year, month, day, hour, minute = occ_get_date(record["occ_date_id"], granular_date)
    splt_data[si]["Period"] = record["period_no"]
    splt_data[si]["PeriodWeight"] = period_weights[record["period_no"] - 1]["weighting"]
    splt_data[si]["EventId"] = event_id
    splt_data[si]["Year"] = year
    splt_data[si]["Month"] = month
    splt_data[si]["Day"] = day
    splt_data[si]["Hour"] = hour
    splt_data[si]["Minute"] = minute
    splt_data[si]["SummaryId"] = summary_id
    splt_data[si]["SampleId"] = sidx
    splt_data[si]["Loss"] = loss
    splt_data[si]["ImpactedExposure"] = impacted_exposure


@nb.njit(cache=True)
def _update_mplt_data(
    mplt_data, mi, period_weights, granular_date,
    record,
    event_id,
    summary_id,
    sample_type,
    chance_of_loss,
    meanloss,
    sdloss,
    maxloss,
    footprint_exposure,
    mean_impacted_exposure,
    max_impacted_exposure
):
    """updates mplt_data to write to output
    """
    year, month, day, hour, minute = occ_get_date(record["occ_date_id"], granular_date)
    mplt_data[mi]["Period"] = record["period_no"]
    mplt_data[mi]["PeriodWeight"] = period_weights[record["period_no"] - 1]["weighting"]
    mplt_data[mi]["EventId"] = event_id
    mplt_data[mi]["Year"] = year
    mplt_data[mi]["Month"] = month
    mplt_data[mi]["Day"] = day
    mplt_data[mi]["Hour"] = hour
    mplt_data[mi]["Minute"] = minute
    mplt_data[mi]["SummaryId"] = summary_id
    mplt_data[mi]["SampleType"] = sample_type
    mplt_data[mi]["ChanceOfLoss"] = chance_of_loss
    mplt_data[mi]["MeanLoss"] = meanloss
    mplt_data[mi]["SDLoss"] = sdloss
    mplt_data[mi]["MaxLoss"] = maxloss
    mplt_data[mi]["FootprintExposure"] = footprint_exposure
    mplt_data[mi]["MeanImpactedExposure"] = mean_impacted_exposure
    mplt_data[mi]["MaxImpactedExposure"] = max_impacted_exposure


@nb.njit(cache=True)
def _update_qplt_data(
    qplt_data, qi, period_weights, granular_date,
    record,
    event_id,
    summary_id,
    quantile,
    loss,
):
    """updates mplt_data to write to output
    """
    year, month, day, hour, minute = occ_get_date(record["occ_date_id"], granular_date)
    qplt_data[qi]["Period"] = record["period_no"]
    qplt_data[qi]["PeriodWeight"] = period_weights[record["period_no"] - 1]["weighting"]
    qplt_data[qi]["EventId"] = event_id
    qplt_data[qi]["Year"] = year
    qplt_data[qi]["Month"] = month
    qplt_data[qi]["Day"] = day
    qplt_data[qi]["Hour"] = hour
    qplt_data[qi]["Minute"] = minute
    qplt_data[qi]["SummaryId"] = summary_id
    qplt_data[qi]["Quantile"] = quantile
    qplt_data[qi]["Loss"] = loss


@nb.njit(cache=True, error_model="numpy")
def read_buffer(
        byte_mv, cursor, valid_buff, event_id, item_id,
        state,
        splt_data, splt_idx,
        mplt_data, mplt_idx,
        qplt_data, qplt_idx,
        occ_map,
        period_weights,
        granular_date,
        intervals,
):
    # Initialise idxs
    last_event_id = event_id
    si = splt_idx[0]
    mi = mplt_idx[0]
    qi = qplt_idx[0]

    # Helper functions
    def _update_idxs():
        splt_idx[0] = si
        mplt_idx[0] = mi
        qplt_idx[0] = qi

    def _reset_state():
        state["reading_losses"] = False
        state["max_loss"] = 0
        state["mean_impacted_exposure"] = 0
        state["max_impacted_exposure"] = 0
        state["chance_of_loss"] = 0
        state["vrec"].fill(0)
        state["sumloss"] = 0
        state["sumlosssqr"] = 0
        state["hasrec"] = False

    def _get_mean_and_sd_loss():
        meanloss = state["sumloss"] / state["len_sample"]
        if state["len_sample"] != 1:
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

    # Read input loop
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
                state["exposure_value"], cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
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

                # Update MPLT data (sample mean)
                if state["compute_mplt"]:
                    firsttime = True
                    if event_id in occ_map:
                        for record in occ_map[event_id]:
                            if firsttime:
                                for l in state["vrec"]:
                                    state["sumloss"] += l
                                    state["sumlosssqr"] += l * l
                                firsttime = False
                            if state["hasrec"]:
                                meanloss, sdloss = _get_mean_and_sd_loss()
                                if meanloss > 0 or sdloss > 0:
                                    _update_mplt_data(
                                        mplt_data, mi, period_weights, granular_date,
                                        record=record,
                                        event_id=event_id,
                                        summary_id=state["summary_id"],
                                        sample_type=MEAN_TYPE_SAMPLE,
                                        chance_of_loss=state["chance_of_loss"],
                                        meanloss=meanloss,
                                        sdloss=sdloss,
                                        maxloss=state["max_loss"],
                                        footprint_exposure=state["exposure_value"],
                                        mean_impacted_exposure=state["mean_impacted_exposure"],
                                        max_impacted_exposure=state["max_impacted_exposure"],
                                    )
                                    mi += 1
                                    if mi >= mplt_data.shape[0]:
                                        # Output array full
                                        _update_idxs()
                                        return cursor, event_id, item_id, 1

                # Update QPLT data
                if state["compute_qplt"]:
                    state["vrec"].sort()
                    if event_id in occ_map:
                        for record in occ_map[event_id]:
                            for i in range(len(intervals)):
                                q = intervals[i]["quantile"]
                                ipart = intervals[i]["integer_part"]
                                fpart = intervals[i]["fractional_part"]
                                if ipart == len(state["vrec"]):
                                    loss = state["vrec"][ipart - 1]
                                else:
                                    loss = (
                                        (state["vrec"][ipart] - state["vrec"][ipart - 1]) *
                                        fpart + state["vrec"][ipart - 1]
                                    )
                                _update_qplt_data(
                                    qplt_data, qi, period_weights, granular_date,
                                    record=record,
                                    event_id=event_id,
                                    summary_id=state["summary_id"],
                                    quantile=q,
                                    loss=loss
                                )
                                qi += 1
                                if qi >= qplt_data.shape[0]:
                                    # Output array full
                                    _update_idxs()
                                    return cursor, event_id, item_id, 1
                _reset_state()
                continue

            # Read loss
            loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)

            impacted_exposure = 0
            if sidx == NUMBER_OF_AFFECTED_RISK_IDX:
                continue
            if sidx >= MEAN_IDX:
                impacted_exposure = state["exposure_value"] * (loss > 0)
                # Update SPLT data
                if state["compute_splt"]:
                    if event_id in occ_map:
                        for record in occ_map[event_id]:
                            _update_splt_data(
                                splt_data, si, period_weights, granular_date,
                                record=record,
                                event_id=event_id,
                                summary_id=state["summary_id"],
                                sidx=sidx,
                                loss=loss,
                                impacted_exposure=impacted_exposure,
                            )
                            si += 1
                            if si >= splt_data.shape[0]:
                                # Output array full
                                _update_idxs()
                                return cursor, event_id, item_id, 1
            if sidx == MAX_LOSS_IDX:
                state["max_loss"] = loss
            elif sidx == MEAN_IDX:
                # Update MPLT data (analytical mean)
                if state["compute_mplt"]:
                    if event_id in occ_map:
                        for record in occ_map[event_id]:
                            _update_mplt_data(
                                mplt_data, mi, period_weights, granular_date,
                                record=record,
                                event_id=event_id,
                                summary_id=state["summary_id"],
                                sample_type=MEAN_TYPE_ANALYTICAL,
                                chance_of_loss=0,
                                meanloss=loss,
                                sdloss=0,
                                maxloss=state["max_loss"],
                                footprint_exposure=state["exposure_value"],
                                mean_impacted_exposure=state["exposure_value"],
                                max_impacted_exposure=state["exposure_value"],
                            )
                            mi += 1
                            if mi >= mplt_data.shape[0]:
                                # Output array full
                                _update_idxs()
                                return cursor, event_id, item_id, 1
            else:
                # Update state variables
                if sidx > 0:
                    state["vrec"][sidx - 1] = loss
                    state["hasrec"] = True
                state["mean_impacted_exposure"] += impacted_exposure / state["len_sample"]
                if impacted_exposure > state["max_impacted_exposure"]:
                    state["max_impacted_exposure"] = impacted_exposure
                state["chance_of_loss"] += (loss > 0) / state["len_sample"]
        else:
            pass  # Should never reach here anyways

    # Update the indices
    _update_idxs()
    return cursor, event_id, item_id, 0


def read_input_files(run_dir, compute_qplt, sample_size):
    """Reads all input files and returns a dict of relevant data

    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
        compute_qplt (bool): Compute QPLT bool
        sample_size (int): Sample size

    Returns:
        file_data (Dict[str, Any]): A dict of relevent data extracted from files
    """
    occ_map, date_algorithm, granular_date, no_of_periods = read_occurrence(Path(run_dir, "input"))
    period_weights = read_periods(no_of_periods, Path(run_dir, "input"))
    intervals = read_quantile(sample_size, Path(run_dir, "input"), return_empty=not compute_qplt)

    file_data = {
        "occ_map": occ_map,
        "date_algorithm": date_algorithm,
        "granular_date": granular_date,
        "no_of_periods": no_of_periods,
        "period_weights": period_weights,
        "intervals": intervals,
    }
    return file_data


def run(
    run_dir,
    files_in,
    splt_output_file=None,
    mplt_output_file=None,
    qplt_output_file=None,
    noheader=False,
    output_format="csv",
):
    """Runs PLT calculations

    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
        files_in (str | os.PathLike): Path to summary binary input file
        splt_output_file (str, optional): Path to SPLT output file. Defaults to None.
        mplt_output_file (str, optional): Path to MPLT output file. Defaults to None.
        qplt_output_file (str, optional): Path to QPLT output file. Defaults to None.
        noheader (bool): Boolean value to skip header in output file. Defaults to False.
        output_format (str): Output format extension. Defaults to "csv".
    """
    outmap = {
        "splt": {
            "compute": splt_output_file is not None,
            "file_path": splt_output_file,
            "fmt": SPLT_fmt,
            "headers": SPLT_headers,
            "file": None,
        },
        "mplt": {
            "compute": mplt_output_file is not None,
            "file_path": mplt_output_file,
            "fmt": MPLT_fmt,
            "headers": MPLT_headers,
            "file": None,
        },
        "qplt": {
            "compute": qplt_output_file is not None,
            "file_path": qplt_output_file,
            "fmt": QPLT_fmt,
            "headers": QPLT_headers,
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
            outmap["qplt"]["compute"],
            len_sample
        )
        plt_reader = PLTReader(
            len_sample,
            outmap["splt"]["compute"],
            outmap["mplt"]["compute"],
            outmap["qplt"]["compute"],
            file_data["occ_map"],
            file_data["period_weights"],
            file_data["granular_date"],
            file_data["intervals"],
        )

        # Initialise output files PLT
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
                temp_df = pd.DataFrame(plt_reader.get_data(out_type), columns=outmap[out_type]["headers"])
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
        for event_id in plt_reader.read_streams(streams_in):
            for out_type in outmap:
                if not outmap[out_type]["compute"]:
                    continue

                data_idx = plt_reader.get_data_idx(out_type)
                data = plt_reader.get_data(out_type)[:data_idx[0]]

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


@redirect_logging(exec_name='pltpy')
def main(run_dir='.', files_in=None, splt=None, mplt=None, qplt=None, noheader=False, ext="csv", **kwargs):
    run(
        run_dir,
        files_in,
        splt_output_file=splt,
        mplt_output_file=mplt,
        qplt_output_file=qplt,
        noheader=noheader,
        output_format=ext,
    )
