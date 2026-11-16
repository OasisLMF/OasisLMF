#!/usr/bin/env python

from contextlib import ExitStack
import logging
import numpy as np
import pyarrow.parquet as pq

from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, resolve_file
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.utils.exceptions import OasisException

logger = logging.getLogger(__name__)


def default_tobin(stack, file_in, file_out, file_type):
    dtype = TOOL_INFO[file_type]["dtype"]

    file_in = resolve_file(file_in, "rb", stack)

    try:
        parquet_file = pq.ParquetFile(file_in)
    except OSError as e:
        if "Illegal seek" in str(e):
            raise OasisException(
                "Error: parquet input must be a seekable file, not a pipe or stdin "
                "(the parquet footer is read by seeking to the end of the file)."
            ) from e
        raise

    for batch in parquet_file.iter_batches(batch_size=DEFAULT_BUFFER_SIZE):
        data = np.empty(len(batch), dtype=dtype)
        for col in dtype.names:
            arrow_col = batch.column(col)
            # A null in a non-float column casts through NaN into the target dtype's bit
            # pattern (e.g. INT_MIN for an int column) instead of raising. A float column's
            # null legitimately becomes NaN, so only non-float targets are rejected here.
            if arrow_col.null_count and dtype[col].kind not in "fc":
                raise OasisException(
                    f"Error: column '{col}' has {arrow_col.null_count} null value(s); cannot convert to binary."
                )
            data[col] = arrow_col.to_numpy(zero_copy_only=False)
        file_out.write(data.tobytes())


def parquettobin(file_in, file_out, file_type, **kwargs):
    """Convert parquet file to bin file based on file type

    Args:
        file_in (str | os.PathLike): Input file path
        file_out (str | os.PathLike): Output file path
        file_type (str): File type str from SUPPORTED_PARQUETTOBIN
        **kwargs: additional keyword arguments forwarded to the file type specific converter
    """
    with ExitStack() as stack:
        file_out = resolve_file(file_out, "wb", stack)
        default_tobin(stack, file_in, file_out, file_type, **kwargs)
