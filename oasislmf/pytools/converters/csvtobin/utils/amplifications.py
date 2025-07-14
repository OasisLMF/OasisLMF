import numpy as np
from oasislmf.pytools.converters.csvtobin.utils.common import read_csv_as_ndarray
from oasislmf.pytools.converters.data import TYPE_MAP


def amplifications_tobin(file_in, file_out, file_type):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    data = read_csv_as_ndarray(file_in, headers, dtype)

    # Check item IDs start from 1 and are contiguous
    if len(data) > 0 and data["item_id"][0] != 1:
        raise ValueError(f'First item ID is {data["item_id"][0]}. Expected 1.')
    if len(data) > 0 and not np.all(data["item_id"][1:] - data["item_id"][:-1] == 1):
        raise ValueError(f'Item IDs in {file_in} are not contiguous')

    with open(file_out, "wb") as fout:
        # Write the 4-byte zero header
        np.array([0], dtype="i4").tofile(fout)
        data.tofile(fout)
