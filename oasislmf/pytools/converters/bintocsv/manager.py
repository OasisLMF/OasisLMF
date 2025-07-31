#!/usr/bin/env python

from contextlib import ExitStack
import logging
import sys
import numpy as np

from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, write_ndarray_to_fmt_csv
from oasislmf.pytools.converters.bintocsv.utils import (
    amplifications_tocsv,
    cdf_tocsv,
    complex_items_tocsv,
    coverages_tocsv,
    fm_tocsv,
    footprint_tocsv,
    gul_tocsv,
    lossfactors_tocsv,
    occurrence_tocsv,
    vulnerability_tocsv,
)
from oasislmf.pytools.common.data import resolve_file
from oasislmf.pytools.converters.data import TOOL_INFO

logger = logging.getLogger(__name__)


TOCSV_FUNC_MAP = {
    "amplifications": amplifications_tocsv,
    "cdf": cdf_tocsv,
    "complex_items": complex_items_tocsv,
    "coverages": coverages_tocsv,
    "fm": fm_tocsv,
    "footprint": footprint_tocsv,
    "gul": gul_tocsv,
    "lossfactors": lossfactors_tocsv,
    "occurrence": occurrence_tocsv,
    "vulnerability": vulnerability_tocsv,
}


def default_tocsv(stack, file_in, file_out, file_type, noheader):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    fmt = TOOL_INFO[file_type]["fmt"]

    file_in = resolve_file(file_in, "rb", stack)
    if file_in == sys.stdin.buffer:
        data = np.frombuffer(file_in.read(), dtype=dtype)
    else:
        data = np.fromfile(file_in, dtype=dtype)
    num_rows = data.shape[0]

    if not noheader:
        file_out.write(",".join(headers) + "\n")

    buffer_size = DEFAULT_BUFFER_SIZE
    for start in range(0, num_rows, buffer_size):
        end = min(start + buffer_size, num_rows)
        buffer_data = data[start:end]
        write_ndarray_to_fmt_csv(file_out, buffer_data, headers, fmt)


def bintocsv(file_in, file_out, file_type, noheader=False, **kwargs):
    """Convert bin file to csv file based on file type
    Args:
        file_in (str | os.PathLike): Input file path
        file_out (str | os.PathLike): Output file path
        file_type (str): File type str from SUPPORTED_BINTOCSV
        noheader (bool): Bool to not output header. Defaults to False.
    """
    with ExitStack() as stack:
        file_out = resolve_file(file_out, "w", stack)

        tocsv_func = TOCSV_FUNC_MAP.get(file_type, default_tocsv)
        tocsv_func(stack, file_in, file_out, file_type, noheader, **kwargs)
