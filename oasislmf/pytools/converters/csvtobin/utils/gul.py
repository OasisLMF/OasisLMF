import numba as nb
import numpy as np
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE
from oasislmf.pytools.common.event_stream import (ITEM_PACKED_STREAM, ITEM_STREAM,
                                                  mv_write_item_header, mv_write_sidx_loss)
from oasislmf.pytools.converters.csvtobin.utils.common import iter_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.common.data import loss_pair_dtype, item_header_dtype, def_to_type_and_size

# Worst case: every input row opens a new group (2 header + 2 data + 2 termination)
_CHUNK_OUT_SIZE = DEFAULT_BUFFER_SIZE * (item_header_dtype.itemsize + loss_pair_dtype.itemsize * 2)

event_id_dtype, event_id_size = def_to_type_and_size('event_id')
item_id_dtype, item_id_size = def_to_type_and_size('item_id')


@nb.njit(cache=True, error_model="numpy")
def _fill_gul_chunk(event_ids, item_ids, sidxs, losses,
                    max_sidx, out, cursor,
                    prev_event_id, prev_item_id, event_id_dtype,
                    ):
    for i in range(len(event_ids)):
        if event_ids[i] != prev_event_id or item_ids[i] != prev_item_id:
            if prev_event_id != event_id_dtype.type(-1):
                cursor = mv_write_sidx_loss(out, cursor, 0, 0.)  # delimiter
            cursor = mv_write_item_header(out, cursor, event_ids[i], item_ids[i])
            prev_event_id = event_ids[i]
            prev_item_id = item_ids[i]
        if sidxs[i] <= max_sidx:
            cursor = mv_write_sidx_loss(out, cursor, sidxs[i], losses[i])
    return cursor, prev_event_id, prev_item_id


def gul_tobin(stack, file_in, file_out, file_type, stream_type, max_sample_index,
              packed_buildings=1):
    """Write a gul loss stream from a csv of (event_id, item_id, sidx, loss) rows.

    Args:
        stack (contextlib.ExitStack): stack the input file is opened on.
        file_in (str | os.PathLike): input csv path.
        file_out: binary output file object.
        file_type (str): key into TOOL_INFO for the row dtype.
        stream_type (int): loss stream source type, 1 or 2.
        max_sample_index (int): the logical sample size S, written in the header. Rows whose
            sidx exceeds the stream's valid range are dropped.
        packed_buildings (int): the most building blocks any one item carries. 1 for an ordinary
            stream. Above 1 the stream declares ITEM_PACKED_STREAM and the sidx range widens to
            ``S * packed_buildings``, since buildings 2..N encode their samples above S.
    """
    dtype = TOOL_INFO[file_type]["dtype"]

    stream_agg_type = ITEM_PACKED_STREAM if packed_buildings > 1 else ITEM_STREAM
    stream_info = (stream_type << 24 | stream_agg_type)
    file_out.write(np.array([stream_info], dtype="i4").tobytes())
    # The header carries the LOGICAL sample size either way; a consumer recovers the building
    # from the sidx. Only the filter below needs the widened range.
    file_out.write(np.array([max_sample_index], dtype="i4").tobytes())
    max_sidx = max_sample_index * packed_buildings

    buf = np.empty(_CHUNK_OUT_SIZE, dtype='b')
    prev_event_id = event_id_dtype.type(-1)
    prev_item_id = item_id_dtype.type(-1)

    for chunk in iter_csv_as_ndarray(stack, file_in, dtype):
        event_ids = np.ascontiguousarray(chunk["event_id"])
        item_ids = np.ascontiguousarray(chunk["item_id"])
        sidxs = np.ascontiguousarray(chunk["sidx"])
        losses = np.ascontiguousarray(chunk["loss"])

        cursor, prev_event_id, prev_item_id = _fill_gul_chunk(
            event_ids, item_ids, sidxs, losses,
            max_sidx, buf, np.int64(0),
            prev_event_id, prev_item_id, event_id_dtype,
        )
        file_out.write(buf[:cursor].tobytes())

    if prev_event_id != event_id_dtype.type(-1):
        file_out.write(np.array([0], dtype=loss_pair_dtype).tobytes())  # final delimiter
