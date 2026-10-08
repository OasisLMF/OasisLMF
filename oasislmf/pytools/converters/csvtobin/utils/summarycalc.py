import numba as nb
import numpy as np
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, def_to_type_and_size, loss_pair_dtype
from oasislmf.pytools.common.event_stream import SUMMARY_STREAM_ID, mv_write_summary_header, mv_write_sidx_loss
from oasislmf.pytools.converters.csvtobin.utils.common import iter_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO

summaryset_id_dtype, _ = def_to_type_and_size('summaryset_id')
event_id_dtype, event_id_size = def_to_type_and_size('event_id')
summary_id_dtype, summary_id_size = def_to_type_and_size('summary_id')
# Loss and ImpactedExposure share the oasis_float wire type/size (f4 or f8, per OASIS_FLOAT)
loss_dtype, loss_size = def_to_type_and_size('loss')

# Worst case: every input row opens a new group (header + delimiter + data pair)
_HEADER_SIZE = event_id_size + summary_id_size + loss_size
_CHUNK_OUT_SIZE = DEFAULT_BUFFER_SIZE * (_HEADER_SIZE + loss_pair_dtype.itemsize * 2)


@nb.njit(cache=True, error_model="numpy")
def _fill_summarycalc_chunk(event_ids, summary_ids, expvals, sidxs, losses,
                            max_sample_index, out, cursor,
                            prev_event_id, prev_summary_id, prev_expval, event_id_dtype):
    for i in range(len(event_ids)):
        if (event_ids[i] != prev_event_id or summary_ids[i] != prev_summary_id
                or expvals[i] != prev_expval):
            if prev_event_id != event_id_dtype.type(-1):
                cursor = mv_write_sidx_loss(out, cursor, 0, 0.)  # delimiter
            cursor = mv_write_summary_header(out, cursor, event_ids[i], summary_ids[i], expvals[i])
            prev_event_id = event_ids[i]
            prev_summary_id = summary_ids[i]
            prev_expval = expvals[i]
        if sidxs[i] <= max_sample_index:
            cursor = mv_write_sidx_loss(out, cursor, sidxs[i], losses[i])
    return cursor, prev_event_id, prev_summary_id, prev_expval


def summarycalc_tobin(stack, file_in, file_out, file_type, max_sample_index, summary_set_id):
    dtype = TOOL_INFO[file_type]["dtype"]

    stream_agg_type = 1
    stream_info = (SUMMARY_STREAM_ID << 24 | stream_agg_type)
    file_out.write(np.array([stream_info], dtype="i4").tobytes())
    file_out.write(np.array([max_sample_index], dtype="i4").tobytes())
    file_out.write(np.array([summary_set_id], dtype=summaryset_id_dtype).tobytes())

    buf = np.empty(_CHUNK_OUT_SIZE, dtype='b')
    prev_event_id = event_id_dtype.type(-1)
    prev_summary_id = summary_id_dtype.type(-1)
    prev_expval = loss_dtype.type(-1)

    for chunk in iter_csv_as_ndarray(stack, file_in, dtype):
        event_ids = np.ascontiguousarray(chunk["EventId"])
        summary_ids = np.ascontiguousarray(chunk["SummaryId"])
        expvals = np.ascontiguousarray(chunk["ImpactedExposure"])
        sidxs = np.ascontiguousarray(chunk["SampleId"])
        losses = np.ascontiguousarray(chunk["Loss"])

        cursor, prev_event_id, prev_summary_id, prev_expval = _fill_summarycalc_chunk(
            event_ids, summary_ids, expvals, sidxs, losses,
            max_sample_index, buf, np.int64(0),
            prev_event_id, prev_summary_id, prev_expval, event_id_dtype
        )
        file_out.write(buf[:cursor].tobytes())

    if prev_event_id != event_id_dtype.type(-1):
        file_out.write(np.array([0], dtype=loss_pair_dtype).tobytes())  # final delimiter
