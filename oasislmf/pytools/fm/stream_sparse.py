# https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h
import numpy as np
from numba import jit
import selectors
from select import select
import logging
logger = logging.getLogger(__name__)


from .stream import EventWriter, EventWriterOrderedOutput,nb_number, number_size, event_agg_dtype, sidx_loss_dtype, register_streams_in, buff_size
remove_empty = False


@jit(cache=True)
def reset_empty_items(loss_index, sidx_indptr, sidx_val, loss_val, computes):
    if remove_empty:
        if sidx_indptr[loss_index] == sidx_indptr[loss_index - 1]:

            computes[loss_index] = 0
            return loss_index - 1
        else:
            return loss_index
    else:
        if sidx_indptr[loss_index] == sidx_indptr[loss_index - 1]:
            sidx_val[sidx_indptr[loss_index]] = -3
            loss_val[sidx_indptr[loss_index]] = 0
            sidx_indptr[loss_index] += 1
        return loss_index


@jit(cache=True)
def add_new_loss(sidx, loss, loss_index, sidx_indptr, sidx_val, loss_val):
    if ((sidx_indptr[loss_index - 1] == sidx_indptr[loss_index])
            or (sidx_val[sidx_indptr[loss_index] - 1] < sidx)):
        insert_i = sidx_indptr[loss_index]
    else:
        insert_i = np.searchsorted(sidx_val[sidx_indptr[loss_index - 1]: sidx_indptr[loss_index]], sidx) + sidx_indptr[loss_index - 1]
        if sidx_val[insert_i] == sidx:
            raise ValueError("duplicated sidx in input stream")
        sidx_val[insert_i + 1: sidx_indptr[loss_index] + 1] = sidx_val[insert_i: sidx_indptr[loss_index]]
        loss_val[insert_i + 1: sidx_indptr[loss_index] + 1] = loss_val[insert_i: sidx_indptr[loss_index]]
    sidx_val[insert_i] = sidx
    loss_val[insert_i] = loss
    sidx_indptr[loss_index] += 1

@jit(cache=True)
def stream_to_loss_sparse(event_agg, sidx_loss, valid_buf, cursor, event_id, agg_id, loss_index, nodes_array,
                          sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val,
                          computes):
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
        if agg_id:# we set agg_id to 0 if we expect a new set of event and agg
            sidx, loss = sidx_loss[cursor]['sidx'], sidx_loss[cursor]['loss']
            cursor += 1
            if sidx:
                loss = 0 if np.isnan(loss) else loss
                if loss and sidx != -2:
                    add_new_loss(sidx, loss, loss_index, sidx_indptr, sidx_val, loss_val)
            else:
                loss_index = reset_empty_items(loss_index, sidx_indptr, sidx_val, loss_val, computes)
                agg_id = 0

        else:
            event_id, agg_id = event_agg[cursor]['event_id'], event_agg[cursor]['agg_id']
            cursor += 1

            if event_id != last_event_id:
                if last_event_id:
                    return cursor - 1, last_event_id, 0, loss_index, 1
                else:
                    last_event_id = event_id
            node = nodes_array[agg_id]
            sidx_indexes[node['node_id']] = loss_index
            loss_indptr[node['loss']: node['loss'] + node['layer_len']] = sidx_indptr[loss_index]
            sidx_indptr[loss_index + 1] = sidx_indptr[loss_index]
            computes[loss_index] = agg_id
            loss_index += 1

    return cursor, event_id, agg_id, loss_index, 0


def read_event_sparse(stream, nodes_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, computes, main_selector,
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
                    loss_index = reset_empty_items(loss_index, sidx_indptr, sidx_val, loss_val, computes)
                    return event_id, loss_index, cursor, valid_buf

                break

        cursor, event_id, agg_id, loss_index, yield_event = stream_to_loss_sparse(event_agg, sidx_loss, valid_buf, cursor,
                                                                                  event_id, agg_id, loss_index, nodes_array,
                                                                                  sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, computes)

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


def read_streams_sparse(streams_in, nodes_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, len_array, computes):
    try:
        main_selector, stream_data = register_streams_in(selectors.DefaultSelector, streams_in)
        logger.debug("Streams read with DefaultSelector")
    except PermissionError: # Fall back option if stream_in contain regular files
        main_selector, stream_data = register_streams_in(selectors.SelectSelector, streams_in)
        logger.debug("Streams read with SelectSelector")
    try:
        while main_selector.get_map():
            for sKey, _ in main_selector.select():
                event = read_event_sparse(sKey.fileobj, nodes_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, computes, main_selector, **sKey.data)

                if event:
                    event_id, loss_index, cursor, valid_buf = event
                    sKey.data['cursor'] = cursor
                    sKey.data['valid_buf'] = valid_buf
                    logger.debug(f'event_id: {event_id}, loss_index: {loss_index}'
                                 f', sparsity: {100 * sidx_indptr[loss_index]/loss_index/len_array}')
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
                    cursor, event_id, agg_id, loss_index, yield_event = stream_to_loss_sparse(event_agg,
                                                                                             sidx_loss,
                                                                                             valid_buf,
                                                                                             cursor,
                                                                                             0, 0,
                                                                                             0,
                                                                                             nodes_array,
                                                                                             sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val,
                                                                                             computes)

                    if event_id:
                        loss_index = reset_empty_items(loss_index, sidx_indptr, sidx_val, loss_val, computes)
                        logger.debug(f'event_id: {event_id}, loss_index: {loss_index}'
                                 f', sparsity: {100 * sidx_indptr[loss_index]/loss_index/len_array}')
                        yield event_id, loss_index

    finally:
        main_selector.close()



@jit(cache=True)
def load_event(event_agg_view, sidx_loss_view, event_id, nodes_array,
               sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val,
               computes, output_array, compute_i, i_layer, i_index, nb_values):
    cursor = 0
    # sidx_indexes = financial_structure.sidx_indexes
    # sidx_indptr = financial_structure.sidx_indptr
    # sidx_val = financial_structure.sidx_val
    # loss_indptr = financial_structure.loss_indptr
    # loss_val = financial_structure.loss_val
    while computes[compute_i]:
        node = nodes_array[computes[compute_i]]
        node_sidx_start = sidx_indptr[sidx_indexes[node['node_id']]]
        node_sidx_end = sidx_indptr[sidx_indexes[node['node_id']] + 1]
        node_val_len = node_sidx_end - node_sidx_start
        for layer in range(i_layer, node['layer_len']):
            output_id = output_array[node['output_ids']+layer]
            node_loss_start = loss_indptr[node['ba'] + layer]
            if output_id and node_val_len:  # if output is not in xref output_id is 0
                if i_index == -1:
                    if nb_values - cursor < 3: # header + -3 and -1 sample
                        return cursor * number_size, compute_i, layer, i_index
                    else:
                        # write the header
                        event_agg_view[cursor]['event_id'], event_agg_view[cursor]['agg_id'] = event_id, output_id
                        i_index += 1
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
                    return cursor * number_size, compute_i, layer, i_index
        compute_i += 1

    return cursor * number_size, compute_i, 0, i_index


class EventWriterSparse(EventWriter):
    def __init__(self, files_out, nodes_array, output_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, len_sample, computes):
        self.files_out = files_out
        self.nodes_array = nodes_array
        self.sidx_indexes = sidx_indexes
        self.sidx_indptr = sidx_indptr
        self.sidx_val = sidx_val
        self.loss_indptr = loss_indptr
        self.loss_val = loss_val
        self.len_sample = len_sample
        self.computes = computes
        self.output_array = output_array

        self.mv = memoryview(bytearray(nb_number * number_size))

        self.event_agg = np.ndarray(nb_number, buffer=self.mv, dtype=event_agg_dtype)
        self.sidx_loss = np.ndarray(nb_number, buffer=self.mv, dtype=sidx_loss_dtype)

    def write(self, event_id, compute_i):
        i_index= -1
        i_layer = 0
        stream_out = [self.stream_out]
        while self.computes[compute_i]:
            cursor, compute_i, i_layer, i_index = load_event(self.event_agg,
                                                             self.sidx_loss,
                                                             event_id,
                                                             self.nodes_array,
                                                             self.sidx_indexes, self.sidx_indptr, self.sidx_val, self.loss_indptr, self.loss_val,
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


class EventWriterOrderedOutputSparse(EventWriterOrderedOutput, EventWriterSparse):
    pass
