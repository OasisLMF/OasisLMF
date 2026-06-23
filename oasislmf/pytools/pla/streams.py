import numba as nb
import numpy as np
import logging

from oasislmf.pytools.common.data import oasis_int, oasis_int_size, loss_pair_dtype, loss_pair_size
from oasislmf.pytools.common.event_stream import (EventReader, get_and_check_header_in, stream_info_to_bytes, write_mv_to_stream,
                                                  mv_read, PIPE_CAPACITY)

logger = logging.getLogger(__name__)


@nb.jit(nopython=True, cache=True)
def read_buffer(byte_mv, cursor, valid_buff, event_id, item_id, items_amps, plafactors, default_factor, out_byte_mv, out_cursor):
    """
    read the gul loss stream, apply the post loss amplification factor and load it into out_byte_mv buffer
    This modified version of the read_buffer template return result when the whole input buffer is read and not when an event is read.
    therefore it cannot be used to read multiple stream at a time because events would be mixed up.

    Args:
        byte_mv: input byte array
        cursor: read cursor
        valid_buff: valid part of input array
        event_id: last event id
        item_id: last item_id
        items_amps (numpy array): amplification IDs where indexes correspond to item IDs
        plafactors (dict): event ID and amplification ID pairs mapped to loss factors
        default_factor (float): post loss reduction/amplification factor to be used if loss factor not found in plafactors
        out_byte_mv: output byte arrau
        out_cursor: single value array to store valid part of out_byte_mv

    """
    if item_id:
        factor = plafactors.get((event_id, items_amps[item_id]), default_factor)
    while True:
        if item_id:
            # View the whole remaining (sidx, loss) payload once and apply the
            # factor in place, instead of mv_read sidx + mv_read loss + mv_write
            # per pair. The view shares memory with byte_mv, so writing the loss
            # field updates the buffer that is bulk-copied to out_byte_mv below.
            n_pairs = (valid_buff - cursor) // loss_pair_size
            if n_pairs == 0:
                break
            sidx_loss_view = byte_mv[cursor:cursor + n_pairs * loss_pair_size].view(loss_pair_dtype)
            for k in range(n_pairs):
                sidx = sidx_loss_view[k]['sidx']
                if not sidx:
                    ##### do item exit ####
                    ##########
                    cursor += (k + 1) * loss_pair_size
                    item_id = 0
                    break

                loss = sidx_loss_view[k]['loss']
                loss = 0 if np.isnan(loss) else loss

                ###### do loss read ######
                sidx_loss_view[k]['loss'] = loss * factor
                ##########
            else:
                cursor += n_pairs * loss_pair_size
        else:
            if valid_buff - cursor < 2 * oasis_int_size:
                break
            event_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            item_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            ##### do new item setup #####
            factor = plafactors.get((event_id, items_amps[item_id]), default_factor)
            ##########
    out_byte_mv[:cursor] = byte_mv[:cursor]
    out_cursor[0] = cursor
    return cursor, event_id, item_id, 1


@nb.jit(nopython=True, cache=True)
def read_buffer_uniform(byte_mv, cursor, valid_buff, event_id, item_id, items_amps, plafactors, default_factor, out_byte_mv, out_cursor):
    while True:
        if item_id:
            # View the whole remaining (sidx, loss) payload once and apply the
            # uniform factor in place (see read_buffer for details).
            n_pairs = (valid_buff - cursor) // loss_pair_size
            if n_pairs == 0:
                break
            sidx_loss_view = byte_mv[cursor:cursor + n_pairs * loss_pair_size].view(loss_pair_dtype)
            for k in range(n_pairs):
                sidx = sidx_loss_view[k]['sidx']
                if not sidx:
                    ##### do item exit ####
                    ##########
                    cursor += (k + 1) * loss_pair_size
                    item_id = 0
                    break

                loss = sidx_loss_view[k]['loss']
                loss = 0 if np.isnan(loss) else loss

                ###### do loss read ######
                sidx_loss_view[k]['loss'] = loss * default_factor
                ##########
            else:
                cursor += n_pairs * loss_pair_size
        else:
            if valid_buff - cursor < 2 * oasis_int_size:
                break
            event_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            item_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
    out_byte_mv[:cursor] = byte_mv[:cursor]
    out_cursor[0] = cursor
    return cursor, event_id, item_id, 1


class PlaReader(EventReader):
    def __init__(self, items_amps, plafactors, default_factor):
        self.items_amps = items_amps
        self.plafactors = plafactors
        self.default_factor = default_factor
        self.out_byte_mv = np.empty(PIPE_CAPACITY, dtype='b')
        self.out_cursor = np.empty(1, dtype='i4')

        self.event_id = 0
        self.item_id = 0

        if self.items_amps is None and self.plafactors is None:
            self.reader = read_buffer_uniform
        else:
            self.reader = read_buffer

        self.logger = logger

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, **kwargs):
        cursor, self.event_id, self.item_id, yield_event = self.reader(
            byte_mv, cursor, valid_buff, self.event_id, self.item_id,
            self.items_amps, self.plafactors, self.default_factor, self.out_byte_mv, self.out_cursor
        )
        return cursor, self.event_id, self.item_id, yield_event


def read_and_write_streams(
    stream_in, stream_out, items_amps, plafactors, default_factor
):
    """
    Read input stream from gulpy or gulcalc, determine amplification ID from
    item ID, determine loss factor from event ID and amplification ID pair,
    multiply losses by relevant factors, and write to output stream.

    Input stream is binary file with layout:
        stream type (oasis_int), maximum sidx value (oasis_int),
        event ID 1 (oasis_int), item ID 1 (oasis_int),
        sample ID/sidx 1 (oasis_int), loss for sidx 1 (oasis_float),
        ...
        sample ID/sidx n (oasis_int), loss for sidx n (oasis_float),
        0 (oasis_int), 0.0 (4-byte float),
        event ID 1 (oasis_int), item ID 2 (oasis_int),
        ...
        event ID M (oasis_int), item ID N (oasis_int),
        sample ID/sidx 1 (oasis_int), loss for sidx 1 (oasis_float),
        ...
        sample ID/sidx n (oasis_int), loss for sidx n (oasis_float)

    Sample ID/sidx of 0 indicates start of next event ID-item ID pair. Output
    stream has same format as input stream.

    Args:
        stream_in (buffer): input stream
        stream_out (buffer): output stream
        items_amps (numpy array): amplification IDs where indexes correspond to item IDs
        plafactors (dict): event ID and amplification ID pairs mapped to loss factors
        default_factor (float): post loss reduction/amplification factor to be used if loss factor not found in plafactors

    """
    stream_source_type, stream_agg_type, len_sample = get_and_check_header_in(stream_in)
    stream_out.write(stream_info_to_bytes(stream_source_type, stream_agg_type))
    stream_out.write(len_sample.tobytes())

    pla_reader = PlaReader(items_amps, plafactors, default_factor)
    for _ in pla_reader.read_streams(stream_in):
        write_mv_to_stream(stream_out, pla_reader.out_byte_mv, pla_reader.out_cursor[0])
