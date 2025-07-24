#!/usr/bin/env python

from contextlib import ExitStack
import logging

from oasislmf.pytools.common.data import resolve_file
from oasislmf.pytools.converters.csvtobin.utils import (
    amplifications_tobin,
    complex_items_tobin,
    coverages_tobin,
    fm_tobin,
    footprint_tobin,
    gul_tobin,
    lossfactors_tobin,
    occurrence_tobin,
    returnperiods_tobin,
    summarycalc_tobin,
)
from oasislmf.pytools.converters.csvtobin.utils.common import read_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO

logger = logging.getLogger(__name__)


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

        tobin_func = default_tobin
        if file_type == "amplifications":
            tobin_func = amplifications_tobin
        elif file_type == "complex_items":
            tobin_func = complex_items_tobin
        elif file_type == "coverages":
            tobin_func = coverages_tobin
        elif file_type == "fm":
            tobin_func = fm_tobin
        elif file_type == "footprint":
            tobin_func = footprint_tobin
        elif file_type == "gul":
            tobin_func = gul_tobin
        elif file_type == "lossfactors":
            tobin_func = lossfactors_tobin
        elif file_type == "occurrence":
            tobin_func = occurrence_tobin
        elif file_type == "returnperiods":
            tobin_func = returnperiods_tobin
        elif file_type == "summarycalc":
            tobin_func = summarycalc_tobin

        tobin_func(stack, file_in, file_out, file_type, **kwargs)
