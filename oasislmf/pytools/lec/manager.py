# lec/manager.py

import logging
from contextlib import ExitStack
from pathlib import Path

from oasislmf.pytools.common.input_files import read_occurrence, read_periods
from oasislmf.pytools.utils import redirect_logging


logger = logging.getLogger(__name__)


def read_input_files(run_dir):
    """Reads all input files and returns a dict of relevant data
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
    Returns:
        file_data (Dict[str, Any]): A dict of relevent data extracted from files
    """
    occ_map, date_algorithm, granular_date, no_of_periods = read_occurrence(Path(run_dir, "input"))
    period_weights = read_periods(no_of_periods, Path(run_dir, "input"))

    file_data = {
        "occ_map": occ_map,
        "date_algorithm": date_algorithm,
        "granular_date": granular_date,
        "no_of_periods": no_of_periods,
        "period_weights": period_weights,
    }
    return file_data


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
        noheader (bool): Boolean value to skip header in output file
    """
    output_ept = ept_output_file is not None
    output_psept = psept_output_file is not None

    with ExitStack() as stack:
        workspace_folder = Path(run_dir, "work", subfolder)
        if not workspace_folder.is_dir():
            raise RuntimeError(f"Error: Unable to open directory {workspace_folder}")

        file_data = read_input_files(run_dir)


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
        noheader=noheader,
    )
