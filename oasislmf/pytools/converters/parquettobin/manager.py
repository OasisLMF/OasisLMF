#!/usr/bin/env python

from contextlib import ExitStack
import logging
import numpy as np
import pyarrow.parquet as pq

from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, resolve_file
from oasislmf.pytools.converters.data import TOOL_INFO

logger = logging.getLogger(__name__)


def default_tobin(stack, file_in, file_out, file_type):
    dtype = TOOL_INFO[file_type]["dtype"]

    file_in = resolve_file(file_in, "rb", stack)

    for batch in pq.ParquetFile(file_in).iter_batches(batch_size=DEFAULT_BUFFER_SIZE):
        data = np.empty(len(batch), dtype=dtype)
        for col in dtype.names:
            data[col] = batch.column(col).to_numpy(zero_copy_only=False)
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
