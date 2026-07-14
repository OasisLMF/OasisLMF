import numba as nb
import numpy as np
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE
from oasislmf.pytools.common.event_stream import mv_write, mv_write_delimiter, mv_write_item_header, mv_write_sidx_loss
from oasislmf.pytools.converters.csvtobin.utils.common import iter_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.common.data import loss_pair_dtype, item_header_dtype, oasis_int, def_to_type_and_size_str, oasis_float, oasis_int_size, oasis_float_size

# Worst case: every input row opens a new group (2 header + 2 data + 2 termination = 6 int32s)
_CHUNK_OUT_SIZE = DEFAULT_BUFFER_SIZE * (item_header_dtype.itemsize + loss_pair_dtype.itemsize + oasis_int.itemsize * 2)

event_id_dtype, event_id_size = def_to_type_and_size_str('event_id')
item_id_dtype, item_id_size = def_to_type_and_size_str('item_id')


@nb.njit(cache=True, error_model="numpy")
def _fill_fm_chunk(event_ids, item_ids, sidxs, losses,
                   max_sample_index, out, cursor,
                   prev_event_id, prev_item_id):
    for i in range(len(event_ids)):
        if event_ids[i] != prev_event_id or item_ids[i] != prev_item_id:
            if prev_event_id != event_id_dtype.type(-1):
                cursor = mv_write(out, cursor, oasis_int, oasis_int_size, 0)
                cursor = mv_write(out, cursor, oasis_float, oasis_float_size, 0)
            cursor = mv_write_item_header(out, cursor, event_ids[i], item_ids[i])
            prev_event_id = event_ids[i]
            prev_item_id = item_ids[i]
        if sidxs[i] <= max_sample_index:
            cursor = mv_write_sidx_loss(out, cursor, sidxs[i], losses[i])
    return cursor, prev_event_id, prev_item_id


def fm_tobin(stack, file_in, file_out, file_type, stream_type, max_sample_index):
    dtype = TOOL_INFO[file_type]["dtype"]

    stream_agg_type = 1
    stream_info = (stream_type << 24 | stream_agg_type)
    np.array([stream_info], dtype="i4").tofile(file_out)
    np.array([max_sample_index], dtype="i4").tofile(file_out)

    buf = np.empty(_CHUNK_OUT_SIZE, dtype='b')
    prev_event_id = event_id_dtype.type(-1)
    prev_item_id = item_id_dtype.type(-1)

    for chunk in iter_csv_as_ndarray(stack, file_in, dtype):
        event_ids = np.ascontiguousarray(chunk["event_id"])
        item_ids = np.ascontiguousarray(chunk["output_id"])
        sidxs = np.ascontiguousarray(chunk["sidx"])
        losses = np.ascontiguousarray(chunk["loss"])

        pos, prev_event_id, prev_item_id = _fill_fm_chunk(
            event_ids, item_ids, sidxs, losses,
            max_sample_index, buf, np.int64(0),
            prev_event_id, prev_item_id,
        )
        buf[:pos].tofile(file_out)

    if prev_event_id != event_id_dtype.type(-1):
        np.array([0], dtype=oasis_int).tofile(file_out)
        np.array([0], dtype=oasis_float).tofile(file_out)
