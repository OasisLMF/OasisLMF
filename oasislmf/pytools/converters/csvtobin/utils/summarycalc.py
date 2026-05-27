import numba as nb
import numpy as np
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE
from oasislmf.pytools.common.event_stream import SUMMARY_STREAM_ID
from oasislmf.pytools.converters.csvtobin.utils.common import iter_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO

_CHUNK_OUT_SIZE = DEFAULT_BUFFER_SIZE * 7 + 5


@nb.njit(cache=True, error_model="numpy")
def _fill_summarycalc_chunk(event_ids, summary_ids, expvals_i32, sidxs, losses_i32,
                            max_sample_index, out, pos,
                            prev_event_id, prev_summary_id, prev_expval_i32):
    for i in range(len(event_ids)):
        if (event_ids[i] != prev_event_id or summary_ids[i] != prev_summary_id
                or expvals_i32[i] != prev_expval_i32):
            if prev_event_id != np.int32(-1):
                out[pos] = np.int32(0)
                out[pos + 1] = np.int32(0)
                pos += 2
            out[pos] = event_ids[i]
            out[pos + 1] = summary_ids[i]
            out[pos + 2] = expvals_i32[i]
            pos += 3
            prev_event_id = event_ids[i]
            prev_summary_id = summary_ids[i]
            prev_expval_i32 = expvals_i32[i]
        if sidxs[i] <= max_sample_index:
            out[pos] = sidxs[i]
            out[pos + 1] = losses_i32[i]
            pos += 2
    return pos, prev_event_id, prev_summary_id, prev_expval_i32


def summarycalc_tobin(stack, file_in, file_out, file_type, max_sample_index, summary_set_id):
    dtype = TOOL_INFO[file_type]["dtype"]

    stream_agg_type = 1
    stream_info = (SUMMARY_STREAM_ID << 24 | stream_agg_type)
    np.array([stream_info], dtype="i4").tofile(file_out)
    np.array([max_sample_index], dtype="i4").tofile(file_out)
    np.array([summary_set_id], dtype="i4").tofile(file_out)

    buf = np.empty(_CHUNK_OUT_SIZE, dtype=np.int32)
    prev_event_id = np.int32(-1)
    prev_summary_id = np.int32(-1)
    prev_expval_i32 = np.int32(-1)

    for chunk in iter_csv_as_ndarray(stack, file_in, dtype):
        event_ids = np.ascontiguousarray(chunk["EventId"])
        summary_ids = np.ascontiguousarray(chunk["SummaryId"])
        expvals_i32 = chunk["ImpactedExposure"].astype(np.float32).view(np.int32)
        sidxs = np.ascontiguousarray(chunk["SampleId"])
        losses_i32 = chunk["Loss"].astype(np.float32).view(np.int32)

        pos, prev_event_id, prev_summary_id, prev_expval_i32 = _fill_summarycalc_chunk(
            event_ids, summary_ids, expvals_i32, sidxs, losses_i32,
            max_sample_index, buf, np.int64(0),
            prev_event_id, prev_summary_id, prev_expval_i32
        )
        buf[:pos].tofile(file_out)

    if prev_event_id != np.int32(-1):
        np.array([0, 0], dtype=np.int32).tofile(file_out)
