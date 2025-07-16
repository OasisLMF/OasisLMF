import numpy as np
from oasislmf.pytools.converters.csvtobin.utils.common import read_csv_as_ndarray
from oasislmf.pytools.converters.data import TYPE_MAP


def gul_tobin(stack, file_in, file_out, file_type, stream_type, max_sample_index):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    item_id_col_name = "item_id"
    data = read_csv_as_ndarray(stack, file_in, headers, dtype)

    stream_agg_type = 1
    stream_info = (stream_type << 24 | stream_agg_type)
    # Write stream info byte
    np.array([stream_info], dtype="i4").tofile(file_out)
    # Write sample len byte
    np.array([max_sample_index], dtype="i4").tofile(file_out)

    curr_event_id = -1
    curr_item_id = -1
    sidx_losses = []
    for row in data:
        event_id = row["event_id"]
        item_id = row[item_id_col_name]
        sidx = row["sidx"]
        loss = row["loss"]
        if (event_id != curr_event_id) or (item_id != curr_item_id):
            if curr_event_id != -1:
                np.array([curr_event_id, curr_item_id], dtype=np.int32).tofile(file_out)
                np.array(sidx_losses, dtype=np.dtype([("sidx", np.int32), ("loss", np.float32)])).tofile(file_out)
                np.array([0], dtype=np.int32).tofile(file_out)
                np.array([0], dtype=np.float32).tofile(file_out)
            curr_event_id = event_id
            curr_item_id = item_id
            sidx_losses = []
        if sidx <= max_sample_index:
            sidx_losses.append((sidx, loss))
    if curr_event_id != -1:
        np.array([curr_event_id, curr_item_id], dtype=np.int32).tofile(file_out)
        np.array(sidx_losses, dtype=np.dtype([("sidx", np.int32), ("loss", np.float32)])).tofile(file_out)
        np.array([0], dtype=np.int32).tofile(file_out)
        np.array([0], dtype=np.float32).tofile(file_out)
