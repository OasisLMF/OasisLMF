# lec/manager.py

import logging
import numpy as np
import numba as nb
from contextlib import ExitStack
from pathlib import Path

from oasislmf.pytools.common.data import (oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import MAX_LOSS_IDX, MEAN_IDX, NUMBER_OF_AFFECTED_RISK_IDX, SUMMARY_STREAM_ID, init_streams_in, mv_read
from oasislmf.pytools.common.input_files import PERIODS_FILE, read_occurrence, read_periods, read_return_periods
from oasislmf.pytools.lec.aggreports import AggReports
from oasislmf.pytools.utils import redirect_logging


logger = logging.getLogger(__name__)


AGG_FULL_UNCERTAINTY = 0
AGG_WHEATSHEAF = 1
AGG_SAMPLE_MEAN = 2
AGG_WHEATSHEAF_MEAN = 3
OCC_FULL_UNCERTAINTY = 4
OCC_WHEATSHEAF = 5
OCC_SAMPLE_MEAN = 6
OCC_WHEATSHEAF_MEAN = 7

OEP = 1
OEPTVAR = 2
AEP = 3
AEPTVAR = 4

EPT_output = [
    ('SummaryId', oasis_int, '%d'),
    ('EPCalc', oasis_int, '%d'),
    ('EPType', oasis_int, '%d'),
    ('ReturnPeriod', oasis_float, '%.6f'),
    ('Loss', oasis_float, '%.6f'),
]

PSEPT_output = [
    ('SummaryId', oasis_int, '%d'),
    ('SampleId', oasis_int, '%d'),
    ('EPType', oasis_int, '%d'),
    ('ReturnPeriod', oasis_float, '%.6f'),
    ('Loss', oasis_float, '%.6f'),
]

_OUTLOSS_DTYPE = np.dtype([
    ("row_used", np.bool_),
    ("agg_out_loss", oasis_float),
    ("max_out_loss", oasis_float),
])


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
    periods_fp = Path(run_dir, PERIODS_FILE)
    # TODO: test period_weights by commenting this out
    # if not periods_fp.exists():
    #     period_weights = np.array([], dtype=period_weights.dtype)
    # else:  # Normalise period weights
    #     period_weights["weighting"] /= sample_size
    returnperiods, use_return_period = read_return_periods(use_return_period, input_dir)

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
def _remap_sidx(sidx):
    if sidx == -2:
        return 0
    if sidx == -3:
        return 1
    if sidx >= 1:
        return sidx + 1
    raise ValueError(f"Invalid sidx value: {sidx}")


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def _get_outloss_mean_idx(period_no, summary_id, max_summary_id):
    return ((period_no - 1) * max_summary_id) + summary_id


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def _get_outloss_sample_idx(period_no, sidx, summary_id, num_sidxs, max_summary_id):
    remapped_sidx = _remap_sidx(sidx)
    return ((period_no - 1) * num_sidxs * max_summary_id) + (remapped_sidx * max_summary_id) + summary_id


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
    for row in filtered_occ_map:
        period_no = row["period_no"]
        if sidx == MEAN_IDX:
            idx = _get_outloss_mean_idx(period_no, summary_id, max_summary_id)
            outloss = outloss_mean
        else:
            idx = _get_outloss_sample_idx(period_no, sidx, summary_id, num_sidxs, max_summary_id)
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
    # Set cursor to end of stream header (stream_type, sample_size, summary_set_id)
    cursor = oasis_int_size * 3

    # Read all samples
    valid_buff = len(fin)
    while cursor < valid_buff:
        event_id, cursor = mv_read(fin, cursor, oasis_int, oasis_int_size)
        summary_id, cursor = mv_read(fin, cursor, oasis_int, oasis_int_size)
        expval, cursor = mv_read(fin, cursor, oasis_float, oasis_float_size)

        filtered_occ_map = occ_map[occ_map["event_id"] == event_id]
        # Discard samples if event_id not found
        if len(filtered_occ_map) == 0:
            while cursor < valid_buff:
                sidx, cursor = mv_read(fin, cursor, oasis_int, oasis_int_size)
                _, cursor = mv_read(fin, cursor, oasis_float, oasis_float_size)
                if sidx == 0:
                    break
            continue

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
    """
    output_ept = ept_output_file is not None
    output_psept = psept_output_file is not None

    with ExitStack() as stack:
        workspace_folder = Path(run_dir, "work", subfolder)
        if not workspace_folder.is_dir():
            raise RuntimeError(f"Error: Unable to open directory {workspace_folder}")

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
        hasEPT = any([output_flags[idx] for idx in handles_agg + handles_occ])
        hasPSEPT = any([output_flags[idx] for idx in handles_psept])
        output_ept = output_ept and hasEPT
        output_psept = output_psept and hasPSEPT

        # Create outloss array maps
        # outloss_mean has only -1 SIDX
        outloss_mean = np.zeros(
            (file_data["no_of_periods"] * max_summary_id),
            dtype=_OUTLOSS_DTYPE
        )
        # outloss_sample has all SIDXs plus -2 and -3
        num_sidxs = sample_size + 2
        outloss_sample = np.zeros(
            (file_data["no_of_periods"] * num_sidxs * max_summary_id),
            dtype=_OUTLOSS_DTYPE
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

        output_files = {}
        if output_ept:
            ept_file = stack.enter_context(open(ept_output_file, "w"))
            if not noheader:
                EPT_headers = ",".join([c[0] for c in EPT_output])
                ept_file.write(EPT_headers + "\n")
            output_files["ept"] = ept_file
        if output_psept:
            psept_file = stack.enter_context(open(psept_output_file, "w"))
            if not noheader:
                PSEPT_headers = ",".join([c[0] for c in PSEPT_output])
                psept_file.write(PSEPT_headers + "\n")
            output_files["psept"] = psept_file

        EPT_fmt = ','.join([c[2] for c in EPT_output])
        PSEPT_fmt = ','.join([c[2] for c in PSEPT_output])

        agg = AggReports(
            ept_output_file,
            psept_output_file,
            outloss_mean,
            outloss_sample,
            file_data["period_weights"],
            max_summary_id,
            num_sidxs,
        )

        if hasEPT:
            agg.output_mean_damage_ratio(
                OEP,
                OEPTVAR,
                "max_out_loss",
            )
            agg.output_mean_damage_ratio(
                AEP,
                AEPTVAR,
                "agg_out_loss",
            )

        # TODO: rest of aggreports outputs

        print(file_data["no_of_periods"], (sample_size + 2), max_summary_id)
        print(max_summary_id)
        print(outloss_mean)
        print(outloss_sample)


@redirect_logging(exec_name='aalpy')
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
    )
