import msgpack
import numpy as np
from oasislmf.pytools.common.data import resolve_file
from oasislmf.pytools.converters.csvtobin.utils.common import df_to_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO
import pandas as pd


def complex_items_write_bin(items_df, file_out, header_dtype, chunk_size=10_000):
    """Write complex items DataFrame to a binary file.

    Args:
        items_df (pd.DataFrame): DataFrame with item_id, coverage_id, group_id, model_data columns.
        file_out: Writable binary file object.
        header_dtype (np.dtype): Structured dtype for the per-row header record.
        chunk_size (int): Number of rows to process at a time.
    """
    for start in range(0, len(items_df), chunk_size):
        chunk = items_df.iloc[start:start + chunk_size]
        packed_data_list = [msgpack.packb(md) for md in chunk["model_data"]]

        header_df = pd.DataFrame({
            "item_id": chunk["item_id"].astype(int),
            "coverage_id": chunk["coverage_id"].astype(int),
            "group_id": chunk["group_id"].astype(int),
            "model_data_len": [len(p) for p in packed_data_list],
        })
        headers = df_to_ndarray(header_df, header_dtype)

        for i in range(len(headers)):
            file_out.write(headers[i].tobytes())
            file_out.write(packed_data_list[i])


def complex_items_tobin(stack, file_in, file_out, file_type):
    header_dtype = TOOL_INFO[file_type]["dtype"]

    file_in = resolve_file(file_in, "r", stack)

    try:
        items_df = pd.read_csv(file_in)
    except pd.errors.EmptyDataError:
        np.empty(0, dtype=header_dtype).tofile(file_out)
        return

    # CSV may parse group_id as float (e.g. "3.0"); coerce to int before writing
    items_df["group_id"] = items_df["group_id"].astype(float).astype(int)

    complex_items_write_bin(items_df, file_out, header_dtype)
