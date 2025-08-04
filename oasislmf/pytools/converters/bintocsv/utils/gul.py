
import logging
import numba as nb
import numpy as np
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, oasis_int, oasis_int_size, oasis_float, oasis_float_size, write_ndarray_to_fmt_csv
from oasislmf.pytools.common.event_stream import (
    GUL_STREAM_ID, LOSS_STREAM_ID, EventReader, init_streams_in, mv_read
)
from oasislmf.pytools.converters.data import TOOL_INFO

logger = logging.getLogger(__name__)


class GulReader(EventReader):
    def __init__(self, len_sample, data_dtype):
        self.logger = logger
        self.data = np.zeros(DEFAULT_BUFFER_SIZE, dtype=data_dtype)
        self.idx = np.zeros(1, dtype=np.int64)

        read_buffer_state_dtype = np.dtype([
            ('len_sample', oasis_int),
            ('reading_losses', np.bool_),
        ])

        self.state = np.zeros(1, dtype=read_buffer_state_dtype)[0]
        self.state["len_sample"] = len_sample

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, **kwargs):
        cursor, event_id, item_id, ret = read_buffer(
            byte_mv,
            cursor,
            valid_buff,
            event_id,
            item_id,
            self.data,
            self.idx,
            self.state
        )
        return cursor, event_id, item_id, ret


@nb.njit(cache=True)
def read_buffer(byte_mv, cursor, valid_buff, event_id, item_id, data, idxs, state):
    last_event_id = event_id
    idx = idxs[0]

    def _reset_state():
        state["reading_losses"] = False

    def _update_idxs():
        idxs[0] = idx

    while cursor < valid_buff:
        if not state["reading_losses"]:
            # Read summary header
            if valid_buff - cursor >= 2 * oasis_int_size:
                event_id_new, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                if last_event_id != 0 and event_id_new != last_event_id:
                    # New event, return to process the previous event
                    _update_idxs()
                    return cursor - oasis_int_size, last_event_id, item_id, 1
                event_id = event_id_new
                item_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                state["reading_losses"] = True
            else:
                break  # Not enough for whole summary header

        if state["reading_losses"]:
            if valid_buff - cursor < oasis_int_size + oasis_float_size:
                break  # Not enough for whole record

            # Read sidx
            sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            if sidx == 0:  # sidx == 0, end of record
                cursor += oasis_float_size  # Read extra 0 for end of record
                _reset_state()
                continue

            # Read loss
            loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
            data[idx]["event_id"] = event_id
            data[idx]["item_id"] = item_id
            data[idx]["sidx"] = sidx
            data[idx]["loss"] = loss
            idx += 1
        else:
            pass  # Should never reach here

    # Update the indices
    _update_idxs()
    return cursor, event_id, item_id, 0


def gul_tocsv(stack, file_in, file_out, file_type, noheader):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    fmt = TOOL_INFO[file_type]["fmt"]

    if str(file_in) == "-":
        file_in = None  # init_streams checks for None to read from sys.stdin.buffer

    streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(file_in, stack)
    if stream_source_type not in [GUL_STREAM_ID, LOSS_STREAM_ID]:
        raise Exception(f"unsupported stream type {stream_source_type}, {stream_agg_type}")

    if not noheader:
        file_out.write(",".join(headers) + "\n")

    gul_reader = GulReader(len_sample=len_sample, data_dtype=dtype)

    for event_id in gul_reader.read_streams(streams_in):
        idx = gul_reader.idx
        data = gul_reader.data[:idx[0]]
        write_ndarray_to_fmt_csv(
            file_out,
            data,
            headers,
            fmt,
        )
        idx[0] = 0
