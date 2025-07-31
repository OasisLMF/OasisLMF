#!/usr/bin/env python

from contextlib import ExitStack
import logging

from oasislmf.pytools.common.data import resolve_file
from oasislmf.pytools.converters.data import TOOL_INFO
import pandas as pd

logger = logging.getLogger(__name__)


def default_tobin(stack, file_in, file_out, file_type):
    dtype = TOOL_INFO[file_type]["dtype"]

    file_in = resolve_file(file_in, "rb", stack)

    df = pd.read_parquet(file_in)
    data = df.to_records(index=False).astype(dtype)
    data.tofile(file_out)


def parquettobin(file_in, file_out, file_type, **kwargs):
    """Convert parquet file to bin file based on file type
    Args:
        file_in (str | os.PathLike): Input file path
        file_out (str | os.PathLike): Output file path
        file_type (str): File type str from SUPPORTED_PARQUETTOBIN
    """
    with ExitStack() as stack:
        file_out = resolve_file(file_out, "wb", stack)
        default_tobin(stack, file_in, file_out, file_type, **kwargs)
