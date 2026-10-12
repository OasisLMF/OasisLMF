#!/usr/bin/env python

from contextlib import ExitStack
import logging
import os
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
    chunk_bytes = DEFAULT_BUFFER_SIZE * dtype.itemsize

    schema = pa.schema([(name, pa.from_numpy_dtype(dtype[name])) for name in dtype.names])
    writer = pq.ParquetWriter(file_out, schema)
    try:
        while True:
            raw = file_in.read(chunk_bytes)
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype=dtype)
            writer.write_table(pa.Table.from_arrays([pa.array(chunk[col]) for col in headers], schema=schema))
    finally:
        writer.close()


def bintoparquet(file_in, file_out, file_type, **kwargs):
    """Convert bin file to parquet file based on file type

    Args:
        file_in (str | os.PathLike): Input file path
        file_out (str | os.PathLike): Output file path
        file_type (str): File type str from SUPPORTED_BINTOPARQUET
        **kwargs: additional keyword arguments forwarded to the file type specific converter
    """
    try:
        with ExitStack() as stack:
            resolved_out = resolve_file(file_out, "wb", stack)
            default_toparquet(stack, file_in, resolved_out, file_type, **kwargs)
    except Exception:
        # parquet is self-validating per row group, so a failed conversion that stops partway
        # through still closes into a structurally valid, readable file with silently fewer rows
        # -- unlike a truncated flat binary file, which is visibly wrong on the next read. Remove
        # it so a failure doesn't leave behind something that looks like a successful, if partial,
        # conversion.
        if str(file_out) != "-" and os.path.exists(file_out):
            os.remove(file_out)
        raise
