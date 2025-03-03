# katpy/manager.py

import logging

from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


def run(
    out_file,
    files_in=None,
    dir_in=None,
    concatenate_selt=False,
    concatenate_melt=False,
    concatenate_qelt=False,
    concatenate_splt=False,
    concatenate_mplt=False,
    concatenate_qplt=False,
    unsorted=False,
):
    """Concatenate CSV files (optionally sorted)
    Args:
        out_file (str | os.PathLike): Output Concatenated CSV file.
        files_in (str | os.PathLike, optional): Individual CSV file paths to concatenate. Defaults to None.
        dir_in (str | os.PathLike, optional): Path to the directory containing files for concatenation. Defaults to None.
        concatenate_selt (bool, optional): Concatenate SELT CSV file. Defaults to False.
        concatenate_melt (bool, optional): Concatenate MELT CSV file. Defaults to False.
        concatenate_qelt (bool, optional): Concatenate QELT CSV file. Defaults to False.
        concatenate_splt (bool, optional): Concatenate SPLT CSV file. Defaults to False.
        concatenate_mplt (bool, optional): Concatenate MPLT CSV file. Defaults to False.
        concatenate_qplt (bool, optional): Concatenate QPLT CSV file. Defaults to False.
        unsorted (bool, optional): Do not sort by event/period ID. Defaults to False.
    """
    pass


@redirect_logging(exec_name='eltpy')
def main(
    out_file=None,
    files_in=None,
    dir_in=None,
    selt=False,
    melt=False,
    qelt=False,
    splt=False,
    mplt=False,
    qplt=False,
    unsorted=False,
    **kwargs
):
    run(
        out_file=out_file,
        files_in=files_in,
        dir_in=dir_in,
        concatenate_selt=selt,
        concatenate_melt=melt,
        concatenate_qelt=qelt,
        concatenate_splt=splt,
        concatenate_mplt=mplt,
        concatenate_qplt=qplt,
        unsorted=unsorted,
    )
