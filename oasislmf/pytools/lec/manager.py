# lec/manager.py

import logging
import numpy as np
import numba as nb
from contextlib import ExitStack
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oasislmf.pytools.common.data import (DEFAULT_BUFFER_SIZE, oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import MAX_LOSS_IDX, MEAN_IDX, NUMBER_OF_AFFECTED_RISK_IDX, SUMMARY_STREAM_ID, init_streams_in, mv_read
from oasislmf.pytools.common.input_files import PERIODS_FILE, read_occurrence, read_periods, read_returnperiods
from oasislmf.pytools.lec.data import (AEP, AEPTVAR, AGG_FULL_UNCERTAINTY, AGG_SAMPLE_MEAN, AGG_WHEATSHEAF, AGG_WHEATSHEAF_MEAN,
                                       OCC_FULL_UNCERTAINTY, OCC_SAMPLE_MEAN, OCC_WHEATSHEAF, OCC_WHEATSHEAF_MEAN, OEP, OEPTVAR,
                                       OUTLOSS_DTYPE, EPT_dtype, EPT_fmt, EPT_headers, PSEPT_dtype, PSEPT_fmt, PSEPT_headers)
from oasislmf.pytools.lec.aggreports import AggReports
from oasislmf.pytools.lec.utils import get_outloss_mean_idx, get_outloss_sample_idx
from oasislmf.pytools.utils import redirect_logging


logger = logging.getLogger(__name__)


def read_input_files(
    run_dir,
    use_return_period,
    agg_wheatsheaf_mean,
    occ_wheatsheaf_mean,
    sample_size
):
    """Reads all input files and returns a dict of relevant data
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
        use_return_period (bool): Use Return Period file.
        agg_wheatsheaf_mean (bool): Aggregate Wheatsheaf Mean.
        occ_wheatsheaf_mean (bool): Occurrence Wheatsheaf Mean.
        sample_size (int): Sample Size.
    Returns:
        file_data (Dict[str, Any]): A dict of relevent data extracted from files
        use_return_period (bool): Use Return Period file.
        agg_wheatsheaf_mean (bool): Aggregate Wheatsheaf Mean.
        occ_wheatsheaf_mean (bool): Occurrence Wheatsheaf Mean.
    """
    input_dir = Path(run_dir, "input")
    occ_map, date_algorithm, granular_date, no_of_periods = read_occurrence(input_dir)
    period_weights = read_periods(no_of_periods, input_dir)
    periods_fp = Path(input_dir, PERIODS_FILE)
    if not periods_fp.exists():
        period_weights = np.array([], dtype=period_weights.dtype)
    else:  # Normalise period weights
        period_weights["weighting"] /= sample_size
    returnperiods, use_return_period = read_returnperiods(use_return_period, input_dir)

    # User must define return periods if he/she wishes to use non-uniform period weights for
    # Wheatsheaf/per sample mean output
    if agg_wheatsheaf_mean or occ_wheatsheaf_mean:
        if len(period_weights) > 0 and not use_return_period:
            logger.warning("WARNING: Return periods file must be present if you wish to use non-uniform period weights for Wheatsheaf mean/per sample mean output.")
            logger.info("INFO: Wheatsheaf mean/per sample mean output will not be produced.")
            agg_wheatsheaf_mean = False
            occ_wheatsheaf_mean = False
        elif len(period_weights) > 0:
            logger.info("INFO: Tail Value at Risk for Wheatsheaf mean/per sample mean is not supported if you wish to use non-uniform period weights.")

    file_data = {
        "occ_map": occ_map,
        "date_algorithm": date_algorithm,
        "granular_date": granular_date,
        "no_of_periods": no_of_periods,
        "period_weights": period_weights,
        "returnperiods": returnperiods,
    }
    return file_data, use_return_period, agg_wheatsheaf_mean, occ_wheatsheaf_mean


@nb.njit(cache=True, error_model="numpy")
def get_max_summary_id(file_handles):
    """Get max summary_id from all summary files
    Args:
        file_handles (List[np.memmap]): List of memmaps for summary files data
    Returns:
        max_summary_id (int): Max summary ID
    """
    max_summary_id = -1
    for fin in file_handles:
        cursor = oasis_int_size * 3

        valid_buff = len(fin)
        while cursor < valid_buff:
            _, cursor = mv_read(fin, cursor, oasis_int, oasis_int_size)
            summary_id, cursor = mv_read(fin, cursor, oasis_int, oasis_int_size)
            _, cursor = mv_read(fin, cursor, oasis_float, oasis_float_size)

            max_summary_id = max(max_summary_id, summary_id)

            while cursor < valid_buff:
                sidx, cursor = mv_read(fin, cursor, oasis_int, oasis_int_size)
                _, cursor = mv_read(fin, cursor, oasis_float, oasis_float_size)
                if sidx == 0:
                    break
    return max_summary_id


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def do_lec_output_agg_summary(
    summary_id,
    sidx,
    loss,
    filtered_occ_map,
    outloss_mean,
    outloss_sample,
    num_sidxs,
    max_summary_id,
):
    """Populate outloss_mean and outloss_sample with aggregate and max losses
    Args:
        summary_id (oasis_int): summary_id
        sidx (oasis_int): Sample ID
        loss (oasis_float): Loss value
        filtered_occ_map (ndarray[occ_map_dtype]): Filtered numpy map of event_id, period_no, occ_date_id from the occurrence file_
        outloss_mean (ndarray[OUTLOSS_DTYPE]): ndarray indexed by summary_id, period_no containing aggregate and max losses
        outloss_sample (ndarray[OUTLOSS_DTYPE]): ndarray indexed by summary_id, sidx, period_no containing aggregate and max losses
        num_sidxs (int): Number of sidxs to consider for outloss_sample
        max_summary_id (int): Max summary ID
    """
    for row in filtered_occ_map:
        period_no = row["period_no"]
        if sidx == MEAN_IDX:
            idx = get_outloss_mean_idx(period_no, summary_id, max_summary_id)
            outloss = outloss_mean
        else:
            idx = get_outloss_sample_idx(period_no, sidx, summary_id, num_sidxs, max_summary_id)
            outloss = outloss_sample
        outloss[idx]["row_used"] = True
        outloss[idx]["agg_out_loss"] += loss
        max_out_loss = max(outloss[idx]["max_out_loss"], loss)
        outloss[idx]["max_out_loss"] = max_out_loss


@nb.njit(cache=True, error_model="numpy")
def process_input_file(
    fin,
    outloss_mean,
    outloss_sample,
    summary_ids,
    occ_map,
    use_return_period,
    num_sidxs,
    max_summary_id,
):
    """Process summary file and populate outloss_mean and outloss_sample with losses
    Args:
        fin (np.memmap): summary binary memmap
        outloss_mean (ndarray[OUTLOSS_DTYPE]): ndarray indexed by summary_id, period_no containing aggregate and max losses
        outloss_sample (ndarray[OUTLOSS_DTYPE]): ndarray indexed by summary_id, sidx, period_no containing aggregate and max losses
        summary_ids (ndarray[bool]): bool array marking which summary_ids are used
        occ_map (nb.typed.Dict): numpy map of event_id, period_no, occ_date_id from the occurrence file
        use_return_period (bool): Use Return Period file.
        num_sidxs (int): Number of sidxs to consider for outloss_sample
        max_summary_id (int): Max summary ID
    """
    # Set cursor to end of stream header (stream_type, sample_size, summary_set_id)
    cursor = oasis_int_size * 3

    # Read all samples
    valid_buff = len(fin)
    while cursor < valid_buff:
        event_id, cursor = mv_read(fin, cursor, oasis_int, oasis_int_size)
        summary_id, cursor = mv_read(fin, cursor, oasis_int, oasis_int_size)
        expval, cursor = mv_read(fin, cursor, oasis_float, oasis_float_size)

        # Discard samples if event_id not found
        if event_id not in occ_map:
            while cursor < valid_buff:
                sidx, cursor = mv_read(fin, cursor, oasis_int, oasis_int_size)
                _, cursor = mv_read(fin, cursor, oasis_float, oasis_float_size)
                if sidx == 0:
                    break
            continue

        filtered_occ_map = occ_map[event_id]
        summary_ids[summary_id - 1] = True

        while cursor < valid_buff:
            sidx, cursor = mv_read(fin, cursor, oasis_int, oasis_int_size)
            loss, cursor = mv_read(fin, cursor, oasis_float, oasis_float_size)
            if sidx == 0:
                break
            if sidx == NUMBER_OF_AFFECTED_RISK_IDX or sidx == MAX_LOSS_IDX:
                continue
            if loss > 0 or use_return_period:
                do_lec_output_agg_summary(
                    summary_id,
                    sidx,
                    loss,
                    filtered_occ_map,
                    outloss_mean,
                    outloss_sample,
                    num_sidxs,
                    max_summary_id,
                )


@nb.njit(cache=True, error_model="numpy")
def run_lec(
    file_handles,
    outloss_mean,
    outloss_sample,
    summary_ids,
    occ_map,
    use_return_period,
    num_sidxs,
    max_summary_id,
):
    """Process each summary file and populate outloss_mean and outloss_sample
    Args:
        file_handles (List[np.memmap]): List of memmaps for summary files data
        outloss_mean (ndarray[OUTLOSS_DTYPE]): ndarray indexed by summary_id, period_no containing aggregate and max losses
        outloss_sample (ndarray[OUTLOSS_DTYPE]): ndarray indexed by summary_id, sidx, period_no containing aggregate and max losses
        summary_ids (ndarray[bool]): bool array marking which summary_ids are used
        occ_map (nb.typed.Dict): numpy map of event_id, period_no, occ_date_id from the occurrence file
        use_return_period (bool): Use Return Period file.
        num_sidxs (int): Number of sidxs to consider for outloss_sample
        max_summary_id (int): Max summary ID
    """
    for fin in file_handles:
        process_input_file(
            fin,
            outloss_mean,
            outloss_sample,
            summary_ids,
            occ_map,
            use_return_period,
            num_sidxs,
            max_summary_id,
        )


def run(
    run_dir,
    subfolder,
    ept_output_file=None,
    psept_output_file=None,
    agg_full_uncertainty=False,
    agg_wheatsheaf=False,
    agg_sample_mean=False,
    agg_wheatsheaf_mean=False,
    occ_full_uncertainty=False,
    occ_wheatsheaf=False,
    occ_sample_mean=False,
    occ_wheatsheaf_mean=False,
    use_return_period=False,
    noheader=False,
    output_format="csv",
):
    """Runs LEC calculations
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
        subfolder (str): Workspace subfolder inside <run_dir>/work/<subfolder>
        ept_output_file (str, optional): Path to EPT output file. Defaults to None
        psept_output_file (str, optional): Path to PSEPT output file. Defaults to None
        agg_full_uncertainty (bool, optional): Aggregate Full Uncertainty. Defaults to False.
        agg_wheatsheaf (bool, optional): Aggregate Wheatsheaf. Defaults to False.
        agg_sample_mean (bool, optional): Aggregate Sample Mean. Defaults to False.
        agg_wheatsheaf_mean (bool, optional): Aggregate Wheatsheaf Mean. Defaults to False.
        occ_full_uncertainty (bool, optional): Occurrence Full Uncertainty. Defaults to False.
        occ_wheatsheaf (bool, optional): Occurrence Wheatsheaf. Defaults to False.
        occ_sample_mean (bool, optional): Occurrence Sample Mean. Defaults to False.
        occ_wheatsheaf_mean (bool, optional): Occurrence Wheatsheaf Mean. Defaults to False.
        use_return_period (bool, optional): Use Return Period file. Defaults to False.
        noheader (bool): Boolean value to skip header in output file
        output_format (str): Output format extension. Defaults to "csv".
    """
    outmap = {
        "ept": {
            "compute": ept_output_file is not None,
            "file_path": ept_output_file,
            "fmt": EPT_fmt,
            "headers": EPT_headers,
            "file": None,
            "dtype": EPT_dtype,
        },
        "psept": {
            "compute": psept_output_file is not None,
            "file_path": psept_output_file,
            "fmt": PSEPT_fmt,
            "headers": PSEPT_headers,
            "file": None,
            "dtype": PSEPT_dtype,
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

    if not all([v["compute"] for v in outmap.values()]):
        logger.warning("No output files specified")

    with ExitStack() as stack:
        workspace_folder = Path(run_dir, "work", subfolder)
        if not workspace_folder.is_dir():
            raise RuntimeError(f"Error: Unable to open directory {workspace_folder}")

        # work folder for lec files
        lec_files_folder = Path(workspace_folder, "lec_files")
        lec_files_folder.mkdir(parents=False, exist_ok=True)

        # Find summary binary files
        files = [file for file in workspace_folder.glob("*.bin")]
        file_handles = [np.memmap(file, mode="r", dtype="u1") for file in files]

        streams_in, (stream_source_type, stream_agg_type, sample_size) = init_streams_in(files, stack)
        if stream_source_type != SUMMARY_STREAM_ID:
            raise RuntimeError(f"Error: Not a summary stream type {stream_source_type}")

        max_summary_id = get_max_summary_id(file_handles)

        file_data, use_return_period, agg_wheatsheaf_mean, occ_wheatsheaf_mean = read_input_files(
            run_dir,
            use_return_period,
            agg_wheatsheaf_mean,
            occ_wheatsheaf_mean,
            sample_size,
        )

        output_flags = [
            agg_full_uncertainty,
            agg_wheatsheaf,
            agg_sample_mean,
            agg_wheatsheaf_mean,
            occ_full_uncertainty,
            occ_wheatsheaf,
            occ_sample_mean,
            occ_wheatsheaf_mean,
        ]

        # Check output_flags against output files
        handles_agg = [AGG_FULL_UNCERTAINTY, AGG_SAMPLE_MEAN, AGG_WHEATSHEAF_MEAN]
        handles_occ = [OCC_FULL_UNCERTAINTY, OCC_SAMPLE_MEAN, OCC_WHEATSHEAF_MEAN]
        handles_psept = [AGG_WHEATSHEAF, OCC_WHEATSHEAF]
        hasAGG = any([output_flags[idx] for idx in handles_agg])
        hasOCC = any([output_flags[idx] for idx in handles_occ])
        hasEPT = hasAGG or hasOCC
        hasPSEPT = any([output_flags[idx] for idx in handles_psept])
        outmap["ept"]["compute"] = outmap["ept"]["compute"] and hasEPT
        outmap["psept"]["compute"] = outmap["psept"]["compute"] and hasPSEPT

        if not outmap["ept"]["compute"]:
            logger.warning("WARNING: no valid output stream to fill EPT file")
        if not outmap["psept"]["compute"]:
            logger.warning("WARNING: no valid output stream to fill PSEPT file")
        if not (outmap["ept"]["compute"] or outmap["psept"]["compute"]):
            return

        # Create outloss array maps
        # outloss_mean has only -1 SIDX
        outloss_mean_file = Path(lec_files_folder, "lec_outloss_mean.bdat")
        outloss_mean = np.memmap(
            outloss_mean_file,
            dtype=OUTLOSS_DTYPE,
            mode="w+",
            shape=(file_data["no_of_periods"] * max_summary_id),
        )

        # outloss_sample has all SIDXs plus -2 and -3
        num_sidxs = sample_size + 2
        outloss_sample_file = Path(lec_files_folder, "lec_outloss_sample.bdat")
        outloss_sample = np.memmap(
            outloss_sample_file,
            dtype=OUTLOSS_DTYPE,
            mode="w+",
            shape=(file_data["no_of_periods"] * num_sidxs * max_summary_id),
        )

        # Array of Summary IDs found
        summary_ids = np.zeros((max_summary_id), dtype=np.bool_)

        # Run LEC calculations to populate outloss and summary_ids arrays
        run_lec(
            file_handles,
            outloss_mean,
            outloss_sample,
            summary_ids,
            file_data["occ_map"],
            use_return_period,
            num_sidxs,
            max_summary_id,
        )

        # Initialise output files LEC
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
                temp_out_data = np.zeros(DEFAULT_BUFFER_SIZE, dtype=outmap[out_type]["dtype"])
                temp_df = pd.DataFrame(temp_out_data, columns=outmap[out_type]["headers"])
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

        # Output aggregate reports to CSVs
        agg = AggReports(
            outmap,
            outloss_mean,
            outloss_sample,
            file_data["period_weights"],
            max_summary_id,
            sample_size,
            file_data["no_of_periods"],
            num_sidxs,
            use_return_period,
            file_data["returnperiods"],
            lec_files_folder,
            output_binary,
            output_parquet,
        )

        # Output Mean Damage Ratio
        if outmap["ept"]["compute"]:
            if hasOCC:
                agg.output_mean_damage_ratio(OEP, OEPTVAR, "max_out_loss")
            if hasAGG:
                agg.output_mean_damage_ratio(AEP, AEPTVAR, "agg_out_loss")

        # Output Full Uncertainty
        if output_flags[OCC_FULL_UNCERTAINTY]:
            agg.output_full_uncertainty(OEP, OEPTVAR, "max_out_loss")
        if output_flags[AGG_FULL_UNCERTAINTY]:
            agg.output_full_uncertainty(AEP, AEPTVAR, "agg_out_loss")

        # Output Wheatsheaf and Wheatsheaf Mean
        if output_flags[OCC_WHEATSHEAF] or output_flags[OCC_WHEATSHEAF_MEAN]:
            agg.output_wheatsheaf_and_wheatsheafmean(
                OEP, OEPTVAR, "max_out_loss",
                output_flags[OCC_WHEATSHEAF], output_flags[OCC_WHEATSHEAF_MEAN]
            )
        if output_flags[AGG_WHEATSHEAF] or output_flags[AGG_WHEATSHEAF_MEAN]:
            agg.output_wheatsheaf_and_wheatsheafmean(
                AEP, AEPTVAR, "agg_out_loss",
                output_flags[AGG_WHEATSHEAF], output_flags[AGG_WHEATSHEAF_MEAN]
            )

        # Output Sample Mean
        if output_flags[OCC_SAMPLE_MEAN]:
            agg.output_sample_mean(OEP, OEPTVAR, "max_out_loss")
        if output_flags[AGG_SAMPLE_MEAN]:
            agg.output_sample_mean(AEP, AEPTVAR, "agg_out_loss")


@redirect_logging(exec_name='lecpy')
def main(
    run_dir='.',
    subfolder=None,
    ept=None,
    psept=None,
    agg_full_uncertainty=False,
    agg_wheatsheaf=False,
    agg_sample_mean=False,
    agg_wheatsheaf_mean=False,
    occ_full_uncertainty=False,
    occ_wheatsheaf=False,
    occ_sample_mean=False,
    occ_wheatsheaf_mean=False,
    use_return_period=False,
    noheader=False,
    ext="csv",
    **kwargs
):
    run(
        run_dir,
        subfolder,
        ept_output_file=ept,
        psept_output_file=psept,
        agg_full_uncertainty=agg_full_uncertainty,
        agg_wheatsheaf=agg_wheatsheaf,
        agg_sample_mean=agg_sample_mean,
        agg_wheatsheaf_mean=agg_wheatsheaf_mean,
        occ_full_uncertainty=occ_full_uncertainty,
        occ_wheatsheaf=occ_wheatsheaf,
        occ_sample_mean=occ_sample_mean,
        occ_wheatsheaf_mean=occ_wheatsheaf_mean,
        use_return_period=use_return_period,
        noheader=noheader,
        output_format=ext,
    )
