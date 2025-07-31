#!/usr/bin/env python

from contextlib import ExitStack
import logging
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oasislmf.pytools.common.data import resolve_file
from oasislmf.pytools.converters.data import TOOL_INFO

logger = logging.getLogger(__name__)


def default_toparquet(stack, file_in, file_out, file_type):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]

    file_in = resolve_file(file_in, "rb", stack)
    if file_in == sys.stdin.buffer:
        data = np.frombuffer(file_in.read(), dtype=dtype)
    else:
        data = np.fromfile(file_in, dtype=dtype)

    df = pd.DataFrame(data, columns=headers)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_out)


def bintoparquet(file_in, file_out, file_type, **kwargs):
    """Convert bin file to parquet file based on file type
    Args:
        file_in (str | os.PathLike): Input file path
        file_out (str | os.PathLike): Output file path
        file_type (str): File type str from SUPPORTED_BINTOPARQUET
    """
    with ExitStack() as stack:
        file_out = resolve_file(file_out, "wb", stack)
        default_toparquet(stack, file_in, file_out, file_type, **kwargs)
