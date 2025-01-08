# lec/manager.py

import logging
from contextlib import ExitStack
from pathlib import Path

from oasislmf.pytools.common.input_files import read_occurrence, read_periods, read_return_periods
from oasislmf.pytools.utils import redirect_logging


logger = logging.getLogger(__name__)


def read_input_files(
    run_dir,
    use_return_period,
    agg_wheatsheaf_mean,
    occ_wheatsheaf_mean,
):
    """Reads all input files and returns a dict of relevant data
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
        use_return_period (bool): Use Return Period file.
        agg_wheatsheaf_mean (bool): Aggregate Wheatsheaf Mean.
        occ_wheatsheaf_mean (bool): Occurrence Wheatsheaf Mean.
    Returns:
        file_data (Dict[str, Any]): A dict of relevent data extracted from files
        use_return_period (bool): Use Return Period file.
        agg_wheatsheaf_mean (bool): Aggregate Wheatsheaf Mean.
        occ_wheatsheaf_mean (bool): Occurrence Wheatsheaf Mean.
    """
    occ_map, date_algorithm, granular_date, no_of_periods = read_occurrence(Path(run_dir, "input"))
    period_weights = read_periods(no_of_periods, Path(run_dir, "input"))
    returnperiods, use_return_period = read_return_periods(use_return_period)

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

        file_data, use_return_period, agg_wheatsheaf_mean, occ_wheatsheaf_mean = read_input_files(
            run_dir,
            use_return_period,
            agg_wheatsheaf_mean,
            occ_wheatsheaf_mean,
        )

        # DONE: aggreports class which sets output flags and reads more files
        # TODO: outloss vector of maps, fileIDs_occ and agg vector, set of summary_ids
        # TODO: setupoutputfiles for file headers and csv files based on flags
        # TODO:
        #   if no subfolder (len subfolder 0):
        #       processinputfile, outputaggreports, return
        #   else:
        #       read files, generate filelist, filehandles, idxfiles, samplesize, set of summaryids
        #   NOTE: not sure why above if condition exists, if no subfolder, it uses stdin,
        #         but the documentation says no standard input stream
        #   TODO: drop support for standard input, seems to be just left there as a byproduct
        # TODO:
        #   if len idx files == len files:
        #       read idx files, add to summary_file_to_offset map, write to summaries.idx
        #       read input files by summary id/fileidx, similar to the run_aal loop
        #       dolecoutputaggssummary inside this loop
        #       outputaggreports
        #   else:
        #       processinputfile on each file, then outputaggreports at the end
        #   NOTE: should I implement the condition where idx files exist, might be slower
        #   NOTE: summaries are once again ordered!!!, just no period_no. However, they don't seem
        #         to be ordered when there are no idx files, we just seem to read summaries as they come.
        #   TODO: forget the if condition and just read each summary file in original order
        # TODO: Output CSVS, part of outputaggreports, maybe write to buffer and output occasionally,
        #       similar to aalpy which saves all data then outputs once at the end


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
