
import logging
import numba as nb
import numpy as np
from oasislmf.pytools.common.data import (DEFAULT_BUFFER_SIZE, def_to_type_and_size, oasis_int, loss_pair_dtype, loss_pair_size,
                                          write_ndarray_to_fmt_csv)
from oasislmf.pytools.common.event_stream import (
    GUL_STREAM_ID, LOSS_STREAM_ID, EventReader, init_streams_in, mv_read
)
from oasislmf.pytools.converters.data import TOOL_INFO

logger = logging.getLogger(__name__)

event_id_dtype, event_id_dtype_size = def_to_type_and_size("event_id")
item_id_dtype, item_id_dtype_size = def_to_type_and_size("item_id")


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

    print('Inside read buffer')
    while cursor < valid_buff:
        if not state["reading_losses"]:
            # Read summary header
            print('Reading summary header')
            if valid_buff - cursor >= event_id_dtype_size + item_id_dtype_size:
                event_id_new, cursor = mv_read(byte_mv, cursor, event_id_dtype, event_id_dtype_size)
                if last_event_id != 0 and event_id_new != last_event_id:
                    # New event, return to process the previous event
                    idxs[0] = idx
                    return cursor - event_id_dtype_size, last_event_id, item_id, 1
                event_id = event_id_new
                item_id, cursor = mv_read(byte_mv, cursor, item_id_dtype, item_id_dtype_size)
                print(f'Found event_id {event_id_new}, item_id {item_id}')
                state["reading_losses"] = True
            else:
                print("Not enough for summary header")
                break  # Not enough for whole summary header

        if state["reading_losses"]:
            # View the whole remaining (sidx, loss) payload once as a packed
            # structured array instead of two mv_read slice+casts per pair.
            print('Reading losses')
            n_pairs = (valid_buff - cursor) // loss_pair_size
            if n_pairs == 0:
                break  # Not enough for whole record

            print(loss_pair_dtype)
            sidx_loss_view = byte_mv[cursor:cursor + n_pairs * loss_pair_size].view(loss_pair_dtype)
            for k in range(n_pairs):
                sidx = sidx_loss_view[k]["sidx"]
                if sidx == 0:  # sidx == 0, end of record (loss field is the trailing 0)
                    #                    breakpoint()
                    cursor += (k + 1) * loss_pair_size
                    state["reading_losses"] = False
                    break

                data[idx]["event_id"] = event_id
                data[idx]["item_id"] = item_id
                data[idx]["sidx"] = sidx
                data[idx]["loss"] = sidx_loss_view[k]["loss"]
                idx += 1
                if idx >= data.shape[0]:
                    print("Output array is full")
#                    breakpoint()
                    # Output array is full
                    cursor += (k + 1) * loss_pair_size
                    idxs[0] = idx
                    return cursor, event_id, item_id, 1
            else:
                print("Incrementing cursor in else")
                cursor += n_pairs * loss_pair_size
        else:
            pass  # Should never reach here

    # Update the indices
    idxs[0] = idx
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

    print("Reading gul reader streams")
    for event_id in gul_reader.read_streams(streams_in):
        print(event_id)
        idx = gul_reader.idx
        data = gul_reader.data[:idx[0]]
        print(data)
        write_ndarray_to_fmt_csv(
            file_out,
            data,
            headers,
            fmt,
        )
        idx[0] = 0
