#!/usr/bin/env python

from contextlib import ExitStack
import logging

from oasislmf.pytools.common.data import resolve_file
from oasislmf.pytools.converters.csvtobin.utils import (
    amplifications_tobin,
    complex_items_tobin,
    coverages_tobin,
    damagebin_tobin,
    fm_tobin,
    footprint_tobin,
    gul_tobin,
    lossfactors_tobin,
    occurrence_tobin,
    returnperiods_tobin,
    summarycalc_tobin,
    vulnerability_tobin,
)
from oasislmf.pytools.converters.csvtobin.utils.common import read_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO

logger = logging.getLogger(__name__)


TOBIN_FUNC_MAP = {
    "amplifications": amplifications_tobin,
    "complex_items": complex_items_tobin,
    "coverages": coverages_tobin,
    "damagebin": damagebin_tobin,
    "fm": fm_tobin,
    "footprint": footprint_tobin,
    "gul": gul_tobin,
    "lossfactors": lossfactors_tobin,
    "occurrence": occurrence_tobin,
    "returnperiods": returnperiods_tobin,
    "summarycalc": summarycalc_tobin,
    "vulnerability": vulnerability_tobin,
}


def default_tobin(stack, file_in, file_out, file_type):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    data = read_csv_as_ndarray(stack, file_in, headers, dtype)
    data.tofile(file_out)


def csvtobin(file_in, file_out, file_type, **kwargs):
    """Convert csv file to bin file based on file type
    Args:
        file_in (str | os.PathLike): Input file path
        file_out (str | os.PathLike): Output file path
        file_type (str): File type str from SUPPORTED_CSVTOBIN
    """
    with ExitStack() as stack:
        file_out = resolve_file(file_out, "wb", stack)

        tobin_func = TOBIN_FUNC_MAP.get(file_type, default_tobin)
        tobin_func(stack, file_in, file_out, file_type, **kwargs)
