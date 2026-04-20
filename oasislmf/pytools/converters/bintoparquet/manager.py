#!/usr/bin/env python

from contextlib import ExitStack
import logging
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, resolve_file
from oasislmf.pytools.converters.data import TOOL_INFO

logger = logging.getLogger(__name__)


def default_toparquet(stack, file_in, file_out, file_type):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]

    file_in = resolve_file(file_in, "rb", stack)
    item_size = dtype.itemsize
    chunk_bytes = DEFAULT_BUFFER_SIZE * item_size

    writer = None
    try:
        while True:
            if file_in == sys.stdin.buffer:
                raw = file_in.read(chunk_bytes)
            else:
                raw = file_in.read(chunk_bytes)
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype=dtype)
            arrays = [pa.array(chunk[col]) for col in headers]
            if writer is None:
                schema = pa.schema([(col, arr.type) for col, arr in zip(headers, arrays)])
                writer = pq.ParquetWriter(file_out, schema)
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
    finally:
        if writer is not None:
            writer.close()


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
