import numba as nb
import numpy as np
from oasislmf.pytools.converters.csvtobin.utils.common import read_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO


@nb.njit(cache=True, error_model="numpy")
def _count_gul_groups_and_valid(event_ids, item_ids, sidxs, max_sample_index):
    """Count (event_id, item_id) groups and sidx-filtered rows for buffer sizing."""
    n = len(event_ids)
    if n == 0:
        return np.int64(0), np.int64(0)
    G = np.int64(1)
    n_valid = np.int64(1) if sidxs[0] <= max_sample_index else np.int64(0)
    for i in range(1, n):
        if event_ids[i] != event_ids[i - 1] or item_ids[i] != item_ids[i - 1]:
            G += 1
        if sidxs[i] <= max_sample_index:
            n_valid += 1
    return G, n_valid


@nb.njit(cache=True, error_model="numpy")
def _fill_gul_stream(event_ids, item_ids, sidxs, losses_i32, max_sample_index, out):
    """Fill the pre-allocated int32 output buffer.

    losses_i32 is the float32 loss column reinterpreted as int32 (bit-identical).
    Each group is written as: [event_id, item_id, sidx_0, loss_0, ..., 0, 0].
    """
    n = len(event_ids)
    pos = np.int64(0)

    out[pos] = event_ids[0]
    out[pos + 1] = item_ids[0]
    pos += 2

    for i in range(n):
        if i > 0 and (event_ids[i] != event_ids[i - 1] or item_ids[i] != item_ids[i - 1]):
            out[pos] = np.int32(0)
            out[pos + 1] = np.int32(0)
            pos += 2
            out[pos] = event_ids[i]
            out[pos + 1] = item_ids[i]
            pos += 2

        if sidxs[i] <= max_sample_index:
            out[pos] = sidxs[i]
            out[pos + 1] = losses_i32[i]
            pos += 2

    out[pos] = np.int32(0)
    out[pos + 1] = np.int32(0)


def gul_tobin(stack, file_in, file_out, file_type, stream_type, max_sample_index):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    data = read_csv_as_ndarray(stack, file_in, headers, dtype)

    stream_agg_type = 1
    stream_info = (stream_type << 24 | stream_agg_type)
    np.array([stream_info], dtype="i4").tofile(file_out)
    np.array([max_sample_index], dtype="i4").tofile(file_out)

    if len(data) == 0:
        return

    event_ids = np.ascontiguousarray(data["event_id"])
    item_ids = np.ascontiguousarray(data["item_id"])
    sidxs = np.ascontiguousarray(data["sidx"])
    losses_i32 = data["loss"].astype(np.float32).view(np.int32)

    G, n_valid = _count_gul_groups_and_valid(event_ids, item_ids, sidxs, max_sample_index)
    if G == 0:
        return

    # Each group occupies 4 int32s (2 header + 2 terminator) plus 2 per valid row
    out = np.empty(G * 4 + n_valid * 2, dtype=np.int32)
    _fill_gul_stream(event_ids, item_ids, sidxs, losses_i32, max_sample_index, out)
    out.tofile(file_out)
