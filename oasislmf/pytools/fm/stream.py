"""
Module responsible for reading and writing FM stream.

Some detail on the data layout.
event id fron the stream can be:
 - negative for special values like the mean or the standard deviation.
 - positive for normal sample id
in the loss array:
 - sidx -3 correspond to index 0
 - sidx -2 is filtered
 - all other sidx egual their index
 - value are arranged so the index in the array mostly correspond to the sample id,
   the special values -1 is place at the end therefor.
   ie: [sidx_-3, sidx_1, sidx_2, ...sidx_len_sample, sidx_-1]
   here we use python ability to understand negative indexes to equate sidx and index simplifying index management


"""

from .common import float_equal_precision, EXTRA_VALUES

import sys
from numba import njit
import numpy as np
import selectors
from select import select
import logging
logger = logging.getLogger(__name__)


# gul_header = np.int32(1 << 24).tobytes()
fm_header = np.int32(1 | 2 << 24).tobytes()


buff_size = 65536
event_agg_dtype = np.dtype([('event_id', 'i4'), ('agg_id', 'i4')])
sidx_loss_dtype = np.dtype([('sidx', 'i4'), ('loss', 'f4')])
number_size = 8
nb_number = buff_size // number_size


@njit(cache=True)
def stream_to_loss_table(event_agg, sidx_loss, valid_buf, cursor, event_id, agg_id, loss_index, nodes_array, losses, index, computes):
    """valid len must be divisible par 2*4 bytes
    cursor, event_id, agg_id, loss_index, yield_event
    """
    valid_len = valid_buf // number_size

    if agg_id:
        last_event_id = event_id
        node = nodes_array[agg_id]
        index[node['loss']: node['loss'] + node['layer_len']] = loss_index
        computes[loss_index] = agg_id
    else:
        last_event_id = event_id

    while cursor < valid_len:
        if agg_id:
            sidx, loss = sidx_loss[cursor]['sidx'], sidx_loss[cursor]['loss']
            cursor += 1

            if sidx:
                if sidx == -2:
                    continue
                elif sidx < -3:
                    continue
                elif sidx == -3:
                    sidx = 0
                losses[loss_index, sidx] = 0 if np.isnan(loss) else loss
            else:
                agg_id = 0
                loss_index += 1
        else:
            event_id, agg_id = event_agg[cursor]['event_id'], event_agg[cursor]['agg_id']
            cursor += 1
            if event_id != last_event_id:
                if last_event_id:
                    return cursor - 1, last_event_id, 0, loss_index, 1
                else:
                    last_event_id = event_id
            node = nodes_array[agg_id]
            index[node['loss']: node['loss'] + node['layer_len']] = loss_index
            computes[loss_index] = agg_id
            losses[loss_index].fill(0)

    return cursor, event_id, agg_id, loss_index, 0


def read_stream_header(stream_obj):
    stream_type = stream_obj.read(4)
    len_sample = np.frombuffer(stream_obj.read(4), dtype=np.int32)[0]
    return stream_type, len_sample


def read_event(stream, nodes_array, losses, index, computes, main_selector,
               stream_selector, mv, event_agg, sidx_loss, cursor, valid_buf):
    event_id = 0
    agg_id = 0
    loss_index = 0

    while True:
        if valid_buf < buff_size:
            len_read = stream.readinto1(mv[valid_buf:])
            valid_buf += len_read

            if len_read == 0:
                stream_selector.close()
                main_selector.unregister(stream)
                if event_id:
                    if agg_id:
                        loss_index += 1
                    return event_id, loss_index, cursor, valid_buf

                break
        cursor, event_id, agg_id, loss_index, yield_event = stream_to_loss_table(event_agg, sidx_loss, valid_buf,
                                                                                 cursor,
                                                                                 event_id, agg_id, loss_index,
                                                                                 nodes_array, losses, index, computes)

        if yield_event:
            if number_size * cursor == valid_buf:
                valid_buf = 0
            return event_id, loss_index, cursor, valid_buf
        else:
            cursor_buf = number_size * cursor
            mv[:valid_buf - cursor_buf] = mv[cursor_buf: valid_buf]
            valid_buf -= cursor_buf
            cursor = 0
            stream_selector.select()


def register_streams_in(selector_class, streams_in):
    """
    Data from input process is generally sent by event block, meaning once a stream receive data, the complete event is
    going to be sent in a short amount of time.
    Therefore, we can focus on each stream one by one using their specific selector 'stream_selector'.

    """
    main_selector = selector_class()
    stream_data = []
    for stream_in in streams_in:
        mv = memoryview(bytearray(buff_size))

        event_agg = np.ndarray(nb_number, buffer=mv, dtype=event_agg_dtype)
        sidx_loss = np.ndarray(nb_number, buffer=mv, dtype=sidx_loss_dtype)

        stream_selector = selector_class()
        stream_selector.register(stream_in, selectors.EVENT_READ)
        data = {'mv': mv,
                'event_agg': event_agg,
                'sidx_loss': sidx_loss,
                'cursor': 0,
                'valid_buf': 0,
                'stream_selector': stream_selector
               }
        stream_data.append(data)
        main_selector.register(stream_in, selectors.EVENT_READ, data)
    return main_selector, stream_data


def read_streams(streams_in, nodes_array, losses, index, computes):
    try:
        main_selector, stream_data = register_streams_in(selectors.DefaultSelector, streams_in)
        logger.debug("Streams read with DefaultSelector")
    except PermissionError: # Fall back option if stream_in contain regular files
        main_selector, stream_data = register_streams_in(selectors.SelectSelector, streams_in)
        logger.debug("Streams read with SelectSelector")
    try:
        while main_selector.get_map():
            for sKey, _ in main_selector.select():
                event = read_event(sKey.fileobj, nodes_array, losses, index, computes, main_selector, **sKey.data)
                if event:
                    event_id, loss_index, cursor, valid_buf = event
                    sKey.data['cursor'] = cursor
                    sKey.data['valid_buf'] = valid_buf
                    logger.debug(f'event_id: {event_id}, loss_index :{loss_index}')
                    yield event_id, loss_index

        # Stream is read, we need to check if there is remaining event to be parsed
        for data in stream_data:
            if data['cursor'] < data['valid_buf']:
                event_agg = data['event_agg']
                sidx_loss = data['sidx_loss']
                cursor = data['cursor']
                valid_buf = data['valid_buf']
                yield_event = True
                while yield_event:
                    cursor, event_id, agg_id, loss_index, yield_event = stream_to_loss_table(event_agg,
                                                                                             sidx_loss,
                                                                                             valid_buf,
                                                                                             cursor,
                                                                                             0, 0,
                                                                                             0,
                                                                                             nodes_array, losses, index,
                                                                                             computes)

                    if event_id:
                        if not yield_event and agg_id:
                            loss_index += 1
                        logger.debug(f'event_id: {event_id}, loss_index :{loss_index}')
                        yield event_id, loss_index

    finally:
        main_selector.close()


@njit(cache=True)
def load_event(event_agg, sidx_loss, event_id, nodes_array, losses, loss_indexes, computes, output_array, compute_i, i_layer, i_index, nb_values):
    cursor = 0
    top_sidx_range = losses.shape[1] - EXTRA_VALUES + 1

    while computes[compute_i]:
        node = nodes_array[computes[compute_i]]
        for layer in range(i_layer, node['layer_len']):
            output_id = output_array[node['output_ids']+layer]
            if output_id:  # if output is not in xref output_id is 0
                loss_index = loss_indexes[node['ba'] + layer]
                # print(computes[compute_i], output_id, loss_index, losses[loss_index])
                while cursor < nb_values:
                    if i_index == 0:
                        event_agg[cursor]['event_id'], event_agg[cursor]['agg_id'] = event_id, output_id
                        cursor += 1
                        i_index = -3

                    elif i_index == -3:
                        sidx_loss[cursor]['sidx'], sidx_loss[cursor]['loss'] = -3, losses[loss_index, 0]
                        cursor += 1
                        i_index = -1
                    elif i_index == -1:
                        sidx_loss[cursor]['sidx'], sidx_loss[cursor]['loss'] = -1, losses[loss_index, -1]
                        cursor += 1
                        i_index = 1

                    elif i_index == top_sidx_range:
                        sidx_loss[cursor]['sidx'], sidx_loss[cursor]['loss'] = 0, 0
                        cursor += 1
                        i_index = 0
                        i_layer = 0
                        break

                    else:
                        if losses[loss_index, i_index] > float_equal_precision:
                            sidx_loss[cursor]['sidx'], sidx_loss[cursor]['loss'] = i_index, losses[loss_index, i_index]
                            cursor += 1
                        i_index += 1
                else:
                    return cursor * number_size, compute_i, layer, i_index
        compute_i += 1

    return cursor * number_size, compute_i, 0, i_index


class EventWriter:
    def __init__(self, files_out, nodes_array, output_array, losses, loss_indexes, computes, len_sample):
        self.files_out = files_out
        self.nodes_array = nodes_array
        self.losses = losses
        self.loss_indexes = loss_indexes
        self.computes = computes
        self.output_array = output_array

        self.len_sample = len_sample
        self.loss_shape_1 = len_sample + EXTRA_VALUES # all normal sidx plus the extra values

        self.mv = memoryview(bytearray(nb_number * number_size))

        self.event_agg = np.ndarray(nb_number, buffer=self.mv, dtype=event_agg_dtype)
        self.sidx_loss = np.ndarray(nb_number, buffer=self.mv, dtype=sidx_loss_dtype)

    def __enter__(self):
        if self.files_out is None:
            self.stream_out = sys.stdout.buffer
        else:
            self.stream_out = open(self.files_out, 'wb')
        # prepare output buffer
        self.stream_out.write(fm_header)
        self.stream_out.write(np.int32(self.len_sample).tobytes())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.files_out:
            self.stream_out.close()

    def write(self, event_id, compute_i):
        i_index= 0
        i_layer = 0
        stream_out = [self.stream_out]
        while self.computes[compute_i]:
            cursor, compute_i, i_layer, i_index = load_event(self.event_agg,
                                                              self.sidx_loss,
                                                              event_id,
                                                              self.nodes_array,
                                                              self.losses,
                                                              self.loss_indexes,
                                                              self.computes,
                                                              self.output_array,
                                                              compute_i,
                                                              i_layer,
                                                              i_index, nb_number)

            _, writable, exceptional = select([], stream_out, stream_out)
            if exceptional:
                raise IOError(f'error with input stream, {exceptional}')
            writable[0].write(self.mv[:cursor])
        return compute_i


@njit(cache=True)
def get_compute_end(computes, compute_i):
    while computes[compute_i]:
        compute_i += 1
    return compute_i


class EventWriterOrderedOutput(EventWriter):
    def write(self, event_id, compute_i):
        compute_end = get_compute_end(self.computes, compute_i)
        self.computes[compute_i: compute_end] = np.sort(self.computes[compute_i: compute_end])
        return super().write(event_id, compute_i)
