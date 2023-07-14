# sparse array inspired from https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h

import sys
import numpy as np
from numba import jit
import selectors
from select import select
import logging

logger = logging.getLogger(__name__)


remove_empty = False

fm_header = np.int32(1 | 2 << 24).tobytes()
buff_size = 65536
event_agg_dtype = np.dtype([('event_id', 'i4'), ('agg_id', 'i4')])
sidx_loss_dtype = np.dtype([('sidx', 'i4'), ('loss', 'f4')])
number_size = 8
nb_number = buff_size // number_size


@jit(cache=True, nopython=True)
def reset_empty_items(compute_idx, sidx_indptr, sidx_val, loss_val, computes):
    if remove_empty:
        if sidx_indptr[compute_idx['next_compute_i']] == sidx_indptr[compute_idx['next_compute_i'] - 1]:

            computes[compute_idx['next_compute_i']] = 0
            compute_idx['next_compute_i'] -= 1
    else:
        if sidx_indptr[compute_idx['next_compute_i']] == sidx_indptr[compute_idx['next_compute_i'] - 1]:
            sidx_val[sidx_indptr[compute_idx['next_compute_i']]] = -3
            loss_val[sidx_indptr[compute_idx['next_compute_i']]] = 0
            sidx_indptr[compute_idx['next_compute_i']] += 1


@jit(cache=True, nopython=True)
def add_new_loss(sidx, loss, compute_i, sidx_indptr, sidx_val, loss_val):
    if ((sidx_indptr[compute_i - 1] == sidx_indptr[compute_i])
            or (sidx_val[sidx_indptr[compute_i] - 1] < sidx)):
        insert_i = sidx_indptr[compute_i]
    else:
        insert_i = np.searchsorted(sidx_val[sidx_indptr[compute_i - 1]: sidx_indptr[compute_i]], sidx) + sidx_indptr[compute_i - 1]
        if sidx_val[insert_i] == sidx:
            raise ValueError("duplicated sidx in input stream")
        sidx_val[insert_i + 1: sidx_indptr[compute_i] + 1] = sidx_val[insert_i: sidx_indptr[compute_i]]
        loss_val[insert_i + 1: sidx_indptr[compute_i] + 1] = loss_val[insert_i: sidx_indptr[compute_i]]
    sidx_val[insert_i] = sidx
    loss_val[insert_i] = loss
    sidx_indptr[compute_i] += 1


def read_stream_header(stream_obj):
    stream_type = stream_obj.read(4)
    len_sample = np.frombuffer(stream_obj.read(4), dtype=np.int32)[0]
    return stream_type, len_sample


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


@jit(cache=True, nopython=True)
def stream_to_loss_sparse(event_agg, sidx_loss, valid_buf, cursor, event_id, agg_id, nodes_array,
                          sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, pass_through,
                          computes, compute_idx):
    """
    we use a slithly modified version of the CSR sparse matrix where
    the column indices for row i are stored in indices[indptr[i]:indptr[i+1]]
    and their corresponding values are stored in data[indptr[i]:indptr[i+1]].

    nodes_array: array containing all the static information on the nodes
    loss_indptr: array containing the indexes of the beginning and end of samples of an item
    loss_sidx: array containing the sidx of the samples
    loss_val: array containing the loss of the samples
    # """
    # sidx_indexes = financial_structure.sidx_indexes
    # sidx_indptr = financial_structure.sidx_indptr
    # sidx_val = financial_structure.sidx_val
    # loss_indptr = financial_structure.loss_indptr
    # loss_val = financial_structure.loss_val

    valid_len = valid_buf // number_size
    last_event_id = event_id

    while cursor < valid_len:
        if agg_id:  # we set agg_id to 0 if we expect a new set of event and agg
            sidx, loss = sidx_loss[cursor]['sidx'], sidx_loss[cursor]['loss']
            loss = 0 if np.isnan(loss) else loss
            cursor += 1
            if sidx:
                if loss != 0 and not np.isnan(loss):
                    if sidx == -2:  # standard deviation
                        pass
                    elif sidx == -4:  # chance of loss
                        pass_through[compute_idx['next_compute_i']] = loss
                    else:
                        add_new_loss(sidx, loss, compute_idx['next_compute_i'], sidx_indptr, sidx_val, loss_val)
            else:
                reset_empty_items(compute_idx, sidx_indptr, sidx_val, loss_val, computes)
                agg_id = 0

        else:
            event_id, agg_id = event_agg[cursor]['event_id'], event_agg[cursor]['agg_id']
            cursor += 1

            if event_id != last_event_id:
                if last_event_id:
                    return cursor - 1, last_event_id, 0, 1
                else:
                    last_event_id = event_id
            node = nodes_array[agg_id]

            sidx_indexes[node['node_id']] = compute_idx['next_compute_i']
            loss_indptr[node['loss']: node['loss'] + node['layer_len']] = sidx_indptr[compute_idx['next_compute_i']]
            sidx_indptr[compute_idx['next_compute_i'] + 1] = sidx_indptr[compute_idx['next_compute_i']]
            computes[compute_idx['next_compute_i']] = agg_id
            compute_idx['next_compute_i'] += 1

    return cursor, event_id, agg_id, 0


def read_event_sparse(stream, nodes_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, pass_through,
                      computes, compute_idx, main_selector, stream_selector, mv, event_agg, sidx_loss, cursor, valid_buf):
    event_id = 0
    agg_id = 0

    while True:
        if valid_buf < buff_size:
            len_read = stream.readinto1(mv[valid_buf:])
            valid_buf += len_read

            if len_read == 0:
                stream_selector.close()
                main_selector.unregister(stream)
                if event_id:
                    reset_empty_items(compute_idx, sidx_indptr, sidx_val, loss_val, computes)
                    return event_id, cursor, valid_buf

                break

        cursor, event_id, agg_id, yield_event = stream_to_loss_sparse(
            event_agg, sidx_loss, valid_buf, cursor,
            event_id, agg_id, nodes_array,
            sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val,
            pass_through, computes, compute_idx)

        if yield_event:
            if number_size * cursor == valid_buf:
                valid_buf = 0
            return event_id, cursor, valid_buf
        else:
            cursor_buf = number_size * cursor
            mv[:valid_buf - cursor_buf] = mv[cursor_buf: valid_buf]
            valid_buf -= cursor_buf
            cursor = 0
            stream_selector.select()


def event_log_msg(event_id, sidx_indptr, len_array, node_count):
    return f"event_id: {event_id}, node_count: {node_count}, sparsity: {100 * sidx_indptr[node_count] / node_count / len_array}"


def read_streams_sparse(streams_in, nodes_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, pass_through,
                        len_array, computes, compute_idx):
    try:
        main_selector, stream_data = register_streams_in(selectors.DefaultSelector, streams_in)
        logger.debug("Streams read with DefaultSelector")
    except PermissionError:  # Fall back option if stream_in contain regular files
        main_selector, stream_data = register_streams_in(selectors.SelectSelector, streams_in)
        logger.debug("Streams read with SelectSelector")
    try:
        while main_selector.get_map():
            for sKey, _ in main_selector.select():
                event = read_event_sparse(sKey.fileobj, nodes_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr,
                                          loss_val, pass_through, computes, compute_idx, main_selector, **sKey.data)

                if event:
                    event_id, cursor, valid_buf = event
                    sKey.data['cursor'] = cursor
                    sKey.data['valid_buf'] = valid_buf
                    logger.debug(event_log_msg(event_id, sidx_indptr, len_array, compute_idx['next_compute_i']))
                    yield event_id

        # Stream is read, we need to check if there is remaining event to be parsed
        for data in stream_data:
            if data['cursor'] < data['valid_buf']:
                event_agg = data['event_agg']
                sidx_loss = data['sidx_loss']
                cursor = data['cursor']
                valid_buf = data['valid_buf']
                yield_event = True
                while yield_event:
                    cursor, event_id, agg_id, yield_event = stream_to_loss_sparse(
                        event_agg,
                        sidx_loss,
                        valid_buf,
                        cursor,
                        0, 0,
                        nodes_array,
                        sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val,
                        pass_through, computes, compute_idx)

                    if event_id:
                        reset_empty_items(compute_idx, sidx_indptr, sidx_val, loss_val, computes)
                        logger.debug(event_log_msg(event_id, sidx_indptr, len_array, compute_idx['next_compute_i']))
                        yield event_id

    finally:
        main_selector.close()


@jit(cache=True, nopython=True)
def load_event(event_agg_view, sidx_loss_view, event_id, nodes_array,
               sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, pass_through,
               computes, compute_idx, output_array, i_layer, i_index, nb_values):
    cursor = 0
    node_id = computes[compute_idx['level_start_compute_i']]

    while node_id:
        node = nodes_array[node_id]
        node_sidx_start = sidx_indptr[sidx_indexes[node['node_id']]]
        node_sidx_end = sidx_indptr[sidx_indexes[node['node_id']] + 1]
        node_val_len = node_sidx_end - node_sidx_start
        if node_id < pass_through.shape[0]:
            pass_through_loss = pass_through[node_id]
        else:
            pass_through_loss = 0
        for layer in range(i_layer, node['layer_len']):
            output_id = output_array[node['output_ids'] + layer]
            node_loss_start = loss_indptr[node['loss'] + layer]
            # print('output_id', output_id)
            # print('    ', sidx_val[node_sidx_start: node_sidx_end])
            # print('    ', loss_val[node_loss_start: node_loss_start + node_val_len])
            if output_id and node_val_len:  # if output is not in xref output_id is 0
                if i_index == -1:
                    if nb_values - cursor < 5:  # header + -5, -3, -1 sample
                        return cursor * number_size, node_id, layer, i_index
                    else:
                        # write the header
                        event_agg_view[cursor]['event_id'], event_agg_view[cursor]['agg_id'] = event_id, output_id
                        i_index += 1
                        cursor += 1

                        if sidx_val[node_sidx_start + i_index] == -5:
                            sidx_loss_view[cursor]['sidx'], sidx_loss_view[cursor]['loss'] = -5, loss_val[node_loss_start + i_index]
                            i_index += 1
                        else:
                            sidx_loss_view[cursor]['sidx'], sidx_loss_view[cursor]['loss'] = -5, 0
                        cursor += 1

                        # write -4 sidx
                        if pass_through_loss:
                            sidx_loss_view[cursor]['sidx'], sidx_loss_view[cursor]['loss'] = -4, pass_through_loss
                            cursor += 1

                        # write -3 sidx
                        if sidx_val[node_sidx_start + i_index] == -3:
                            sidx_loss_view[cursor]['sidx'], sidx_loss_view[cursor]['loss'] = -3, loss_val[node_loss_start + i_index]
                            i_index += 1
                        else:
                            sidx_loss_view[cursor]['sidx'], sidx_loss_view[cursor]['loss'] = -3, 0
                        cursor += 1

                        # write -1 sidx
                        if sidx_val[node_sidx_start + i_index] == -1 and i_index < node_val_len:
                            sidx_loss_view[cursor]['sidx'], sidx_loss_view[cursor]['loss'] = -1, loss_val[node_loss_start + i_index]
                            i_index += 1
                        else:
                            sidx_loss_view[cursor]['sidx'], sidx_loss_view[cursor]['loss'] = -1, 0
                        cursor += 1

                while cursor < nb_values:
                    if i_index == node_val_len:
                        sidx_loss_view[cursor]['sidx'], sidx_loss_view[cursor]['loss'] = 0, 0
                        cursor += 1
                        i_index = -1
                        i_layer = 0
                        break
                    else:
                        if loss_val[node_loss_start + i_index]:
                            sidx_loss_view[cursor]['sidx'] = sidx_val[node_sidx_start + i_index]
                            sidx_loss_view[cursor]['loss'] = loss_val[node_loss_start + i_index]
                            cursor += 1
                        i_index += 1

                else:
                    return cursor * number_size, node_id, layer, i_index
        compute_idx['level_start_compute_i'] += 1
        node_id = computes[compute_idx['level_start_compute_i']]

    return cursor * number_size, node_id, 0, i_index


class EventWriterSparse:
    def __init__(self, files_out, nodes_array, output_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val,
                 pass_through, len_sample, computes):
        self.files_out = files_out
        self.nodes_array = nodes_array
        self.sidx_indexes = sidx_indexes
        self.sidx_indptr = sidx_indptr
        self.sidx_val = sidx_val
        self.loss_indptr = loss_indptr
        self.loss_val = loss_val
        self.pass_through = pass_through
        self.len_sample = len_sample
        self.computes = computes
        self.output_array = output_array

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

    def write(self, event_id, compute_idx):
        i_index = -1
        i_layer = 0
        node_id = 1
        stream_out = [self.stream_out]
        while node_id:
            cursor, node_id, i_layer, i_index = load_event(
                self.event_agg,
                self.sidx_loss,
                event_id,
                self.nodes_array,
                self.sidx_indexes, self.sidx_indptr, self.sidx_val, self.loss_indptr, self.loss_val,
                self.pass_through,
                self.computes,
                compute_idx,
                self.output_array,
                i_layer,
                i_index, nb_number)

            _, writable, exceptional = select([], stream_out, stream_out)
            if exceptional:
                raise IOError(f'error with input stream, {exceptional}')
            writable[0].write(self.mv[:cursor])


@jit(cache=True, nopython=True)
def get_compute_end(computes, compute_idx):
    compute_start = compute_end = compute_idx['level_start_compute_i']
    while computes[compute_end]:
        compute_end += 1
    return compute_start, compute_end


class EventWriterOrderedOutputSparse(EventWriterSparse):
    def write(self, event_id, compute_idx):
        compute_start, compute_end = get_compute_end(self.computes, compute_idx)
        self.computes[compute_start: compute_end] = np.sort(self.computes[compute_start: compute_end], kind='stable')
        return super().write(event_id, compute_idx)
