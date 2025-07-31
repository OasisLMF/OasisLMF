import numpy as np
from oasislmf.pytools.common.data import oasis_float
from oasislmf.pytools.converters.csvtobin.utils.common import read_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO


def gul_tobin(stack, file_in, file_out, file_type, stream_type, max_sample_index):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
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
    sidx_loss_dtype = np.dtype([
        ("sidx", np.int32),
        ("loss", oasis_float)]
    )
    for row in data:
        event_id = row["event_id"]
        item_id = row["item_id"]
        sidx = row["sidx"]
        loss = row["loss"]
        if (event_id != curr_event_id) or (item_id != curr_item_id):
            if curr_event_id != -1:
                np.array([curr_event_id, curr_item_id], dtype=np.int32).tofile(file_out)
                np.array(sidx_losses, dtype=sidx_loss_dtype).tofile(file_out)
                np.array([0], dtype=np.int32).tofile(file_out)
                np.array([0], dtype=oasis_float).tofile(file_out)
            curr_event_id = event_id
            curr_item_id = item_id
            sidx_losses = []
        if sidx <= max_sample_index:
            sidx_losses.append((sidx, loss))
    if curr_event_id != -1:
        np.array([curr_event_id, curr_item_id], dtype=np.int32).tofile(file_out)
        np.array(sidx_losses, dtype=sidx_loss_dtype).tofile(file_out)
        np.array([0], dtype=np.int32).tofile(file_out)
        np.array([0], dtype=oasis_float).tofile(file_out)
