import numba as nb
import numpy as np
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE
from oasislmf.pytools.converters.csvtobin.utils.common import iter_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO

# Worst case: every input row opens a new group (2 term + 2 header + 2 data = 6 int32s)
_CHUNK_OUT_SIZE = DEFAULT_BUFFER_SIZE * 6 + 4


@nb.njit(cache=True, error_model="numpy")
def _fill_fm_chunk(event_ids, item_ids, sidxs, losses_i32,
                   max_sample_index, out, pos,
                   prev_event_id, prev_item_id):
    for i in range(len(event_ids)):
        if event_ids[i] != prev_event_id or item_ids[i] != prev_item_id:
            if prev_event_id != np.int32(-1):
                out[pos] = np.int32(0)
                out[pos + 1] = np.int32(0)
                pos += 2
            out[pos] = event_ids[i]
            out[pos + 1] = item_ids[i]
            pos += 2
            prev_event_id = event_ids[i]
            prev_item_id = item_ids[i]
        if sidxs[i] <= max_sample_index:
            out[pos] = sidxs[i]
            out[pos + 1] = losses_i32[i]
            pos += 2
    return pos, prev_event_id, prev_item_id


def fm_tobin(stack, file_in, file_out, file_type, stream_type, max_sample_index):
    dtype = TOOL_INFO[file_type]["dtype"]

    stream_agg_type = 1
    stream_info = (stream_type << 24 | stream_agg_type)
    np.array([stream_info], dtype="i4").tofile(file_out)
    np.array([max_sample_index], dtype="i4").tofile(file_out)

    buf = np.empty(_CHUNK_OUT_SIZE, dtype=np.int32)
    prev_event_id = np.int32(-1)
    prev_item_id = np.int32(-1)

    for chunk in iter_csv_as_ndarray(stack, file_in, dtype):
        event_ids = np.ascontiguousarray(chunk["event_id"])
        item_ids = np.ascontiguousarray(chunk["output_id"])
        sidxs = np.ascontiguousarray(chunk["sidx"])
        losses_i32 = chunk["loss"].astype(np.float32).view(np.int32)

        pos, prev_event_id, prev_item_id = _fill_fm_chunk(
            event_ids, item_ids, sidxs, losses_i32,
            max_sample_index, buf, np.int64(0),
            prev_event_id, prev_item_id,
        )
        buf[:pos].tofile(file_out)

    if prev_event_id != np.int32(-1):
        np.array([0, 0], dtype=np.int32).tofile(file_out)
