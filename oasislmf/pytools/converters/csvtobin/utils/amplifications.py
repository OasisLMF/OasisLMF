import numpy as np
from oasislmf.pytools.converters.csvtobin.utils.common import iter_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO


def amplifications_write_bin(data, file_out, *, _write_header=True, _prev_item_id=None):
    """Validate and write amplification data to a binary file.

    Args:
        data (np.ndarray): Structured array with an 'item_id' field.
        file_out: Writable binary file object.
        _write_header: Write the 4-byte zero header before data. Internal use only.
        _prev_item_id: Last item_id from the previous chunk for cross-chunk contiguity
            validation. None means this is the first (or only) call. Internal use only.

    Raises:
        ValueError: If item IDs do not start from 1 or are not contiguous.
    """
    if len(data) > 0:
        if _prev_item_id is None and data["item_id"][0] != 1:
            raise ValueError(f'First item ID is {data["item_id"][0]}. Expected 1.')
        if _prev_item_id is not None and data["item_id"][0] != _prev_item_id + 1:
            raise ValueError('Item IDs are not contiguous')
        if len(data) > 1 and not np.all(data["item_id"][1:] - data["item_id"][:-1] == 1):
            raise ValueError('Item IDs are not contiguous')

    if _write_header:
        np.array([0], dtype="i4").tofile(file_out)
    data.tofile(file_out)


def amplifications_tobin(stack, file_in, file_out, file_type):
    dtype = TOOL_INFO[file_type]["dtype"]

    prev_item_id = None
    header_written = False
    for chunk in iter_csv_as_ndarray(stack, file_in, dtype):
        amplifications_write_bin(
            chunk, file_out,
            _write_header=not header_written,
            _prev_item_id=prev_item_id,
        )
        header_written = True
        if len(chunk) > 0:
            prev_item_id = int(chunk["item_id"][-1])

    if not header_written:
        # Truly empty file (no header row) — still write the 4-byte header
        amplifications_write_bin(np.empty(0, dtype=dtype), file_out)
