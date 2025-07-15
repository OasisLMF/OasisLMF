#!/usr/bin/env python

from contextlib import ExitStack
import logging
import sys
import numpy as np

from oasislmf.pytools.common.data import write_ndarray_to_fmt_csv
from oasislmf.pytools.converters.bintocsv.utils import (
    amplifications_tocsv,
    complex_items_tocsv,
    coverages_tocsv,
    fm_tocsv,
    lossfactors_tocsv,
    occurrence_tocsv,
)
from oasislmf.pytools.converters.bintocsv.utils.common import resolve_file
from oasislmf.pytools.converters.data import TYPE_MAP

logger = logging.getLogger(__name__)


def default_tocsv(stack, file_in, file_out, file_type, noheader):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    fmt = TYPE_MAP[file_type]["fmt"]

    file_in = resolve_file(file_in, "rb", stack)
    if file_in == sys.stdin.buffer:
        data = np.frombuffer(file_in.read(), dtype=dtype)
    else:
        data = np.fromfile(file_in, dtype=dtype)
    num_rows = data.shape[0]

    if not noheader:
        file_out.write(",".join(headers) + "\n")

    buffer_size = 1000000
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

        tocsv_func = default_tocsv
        if file_type == "amplifications":
            tocsv_func = amplifications_tocsv
        elif file_type == "complex_items":
            tocsv_func = complex_items_tocsv
        elif file_type == "coverages":
            tocsv_func = coverages_tocsv
        elif file_type == "fm":
            # TODO: fix input with pipes, current checks don't work, and then try to open "-" as a file in init_streams
            # TODO: Implement fmtobin with sample size and stream type arg options, this will need to be done for gultobin as well
            tocsv_func = fm_tocsv
        elif file_type == "lossfactors":
            tocsv_func = lossfactors_tocsv
        elif file_type == "occurrence":
            tocsv_func = occurrence_tocsv

        tocsv_func(stack, file_in, file_out, file_type, noheader, **kwargs)
