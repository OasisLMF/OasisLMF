# sparse array inspired from https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h

import sys
import numpy as np
import numba as nb
import logging

from oasislmf.pytools.common.event_stream import (stream_info_to_bytes, LOSS_STREAM_ID, ITEM_STREAM, PIPE_CAPACITY, EventReader,
                                                  MAX_LOSS_IDX, CHANCE_OF_LOSS_IDX, TIV_IDX, MEAN_IDX,
                                                  mv_read, mv_write_item_header, mv_write_sidx_loss, mv_write_delimiter, write_mv_to_stream)
from oasislmf.pytools.common.data import oasis_int, oasis_int_size, oasis_float, oasis_float_size

logger = logging.getLogger(__name__)


remove_empty = False


# buff_size = 65536
# event_agg_dtype = np.dtype([('event_id', 'i4'), ('agg_id', 'i4')])
# sidx_loss_dtype = np.dtype([('sidx', 'i4'), ('loss', 'f4')])
# number_size = 8
# nb_number = buff_size // number_size

SPECIAL_SIDX_COUNT = 6  # 0 is included as a special sidx
ITEM_HEADER_SIZE = 2 * oasis_int_size + SPECIAL_SIDX_COUNT * (oasis_int_size + oasis_float_size)
SIDX_LOSS_WRITE_SIZE = oasis_int_size + oasis_float_size


@nb.jit(cache=True, nopython=True)
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


@nb.jit(cache=True, nopython=True)
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


def event_log_msg(event_id, sidx_indptr, len_array, node_count):
    return f"event_id: {event_id}, node_count: {node_count}, sparsity: {100 * sidx_indptr[node_count] / node_count / len_array}"


@nb.njit(cache=True)
def read_buffer(byte_mv, cursor, valid_buff, event_id, item_id,
                nodes_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, pass_through,
                computes, compute_idx
                ):
    last_event_id = event_id
    while True:
        if item_id:
            if valid_buff - cursor < (oasis_int_size + oasis_float_size):
                break
            sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            if sidx:
                loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
                loss = 0 if np.isnan(loss) else loss

                ###### do loss read ######
                if loss != 0:
                    if sidx == -2:  # standard deviation
                        pass
                    elif sidx == -4:  # chance of loss
                        pass_through[compute_idx['next_compute_i']] = loss
                    else:
                        add_new_loss(sidx, loss, compute_idx['next_compute_i'], sidx_indptr, sidx_val, loss_val)
                ##########
            else:
                ##### do item exit ####
                reset_empty_items(compute_idx, sidx_indptr, sidx_val, loss_val, computes)
                ##########
                cursor += oasis_float_size
                item_id = 0
        else:
            if valid_buff - cursor < 2 * oasis_int_size:
                break
            event_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            if event_id != last_event_id:
                if last_event_id:  # we have a new event we return the one we just finished
                    return cursor - oasis_int_size, last_event_id, 0, 1
                else:  # first pass we store the event we are reading
                    last_event_id = event_id
            item_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)

            ##### do new item setup #####
            node = nodes_array[item_id]

            sidx_indexes[node['node_id']] = compute_idx['next_compute_i']
            loss_indptr[node['loss']: node['loss'] + node['layer_len']] = sidx_indptr[compute_idx['next_compute_i']]
            sidx_indptr[compute_idx['next_compute_i'] + 1] = sidx_indptr[compute_idx['next_compute_i']]
            computes[compute_idx['next_compute_i']] = item_id
            compute_idx['next_compute_i'] += 1
            ##########
    return cursor, event_id, item_id, 0


class FMReader(EventReader):
    """
    when reading the stream we store relenvant value into a slithly modified version of the CSR sparse matrix where
    the column indices for row i are stored in indices[indptr[i]:indptr[i+1]]
    and their corresponding values are stored in data[indptr[i]:indptr[i+1]].

    nodes_array: array containing all the static information on the nodes
    loss_indptr: array containing the indexes of the beginning and end of samples of an item
    loss_sidx: array containing the sidx of the samples
    loss_val: array containing the loss of the samples
    """

    def __init__(self, nodes_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, pass_through,
                 len_array, computes, compute_idx):
        self.nodes_array = nodes_array
        self.sidx_indexes = sidx_indexes
        self.sidx_indptr = sidx_indptr
        self.sidx_val = sidx_val
        self.loss_indptr = loss_indptr
        self.loss_val = loss_val
        self.pass_through = pass_through
        self.len_array = len_array
        self.computes = computes
        self.compute_idx = compute_idx
        self.logger = logger

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id,):
        return read_buffer(
            byte_mv, cursor, valid_buff, event_id, item_id,
            self.nodes_array, self.sidx_indexes, self.sidx_indptr,
            self.sidx_val, self.loss_indptr, self.loss_val, self.pass_through,
            self.computes, self.compute_idx
        )

    def item_exit(self):
        reset_empty_items(self.compute_idx, self.sidx_indptr, self.sidx_val, self.loss_val, self.computes)

    def event_read_log(self, event_id):
        logger.debug(event_log_msg(event_id, self.sidx_indptr, self.len_array, self.compute_idx['next_compute_i']))


@nb.jit(cache=True, nopython=True)
def load_event(byte_mv, event_id, nodes_array,
               sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, pass_through,
               computes, compute_idx, output_array, i_layer, i_index):
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
                    if cursor < PIPE_CAPACITY - ITEM_HEADER_SIZE:  # header + -5, -3, -1 sample
                        cursor = mv_write_item_header(byte_mv, cursor, event_id, output_id)
                        i_index += 1

                        if sidx_val[node_sidx_start + i_index] == MAX_LOSS_IDX:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, MAX_LOSS_IDX, loss_val[node_loss_start + i_index])
                            i_index += 1
                        else:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, MAX_LOSS_IDX, 0)

                        # write CHANCE_OF_LOSS_IDX == -4 sidx
                        if pass_through_loss:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, CHANCE_OF_LOSS_IDX, pass_through_loss)

                        # write TIV_IDX == -3 sidx
                        if sidx_val[node_sidx_start + i_index] == TIV_IDX:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, TIV_IDX, loss_val[node_loss_start + i_index])
                            i_index += 1
                        else:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, TIV_IDX, 0)

                        # write MEAN_IDX == -1 sidx
                        if sidx_val[node_sidx_start + i_index] == MEAN_IDX and i_index < node_val_len:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, MEAN_IDX, loss_val[node_loss_start + i_index])
                            i_index += 1
                        else:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, MEAN_IDX, 0)
                    else:
                        return cursor, node_id, layer, i_index

                while cursor < PIPE_CAPACITY - SIDX_LOSS_WRITE_SIZE:
                    if i_index == node_val_len:
                        cursor = mv_write_delimiter(byte_mv, cursor)
                        i_index = -1
                        i_layer = 0
                        break
                    else:
                        if loss_val[node_loss_start + i_index]:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, sidx_val[node_sidx_start + i_index], loss_val[node_loss_start + i_index])
                        i_index += 1

                else:
                    return cursor, node_id, layer, i_index
        compute_idx['level_start_compute_i'] += 1
        node_id = computes[compute_idx['level_start_compute_i']]

    return cursor, node_id, 0, i_index


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

        self.byte_mv = np.frombuffer(buffer=memoryview(bytearray(PIPE_CAPACITY)), dtype='b')

    def __enter__(self):
        if self.files_out is None:
            self.stream_out = sys.stdout.buffer
        else:
            self.stream_out = open(self.files_out, 'wb')
        # prepare output buffer
        self.stream_out.write(stream_info_to_bytes(LOSS_STREAM_ID, ITEM_STREAM))
        self.stream_out.write(np.int32(self.len_sample).tobytes())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.files_out:
            self.stream_out.close()

    def write(self, event_id, compute_idx):
        i_index = -1
        i_layer = 0
        node_id = 1
        while node_id:
            cursor, node_id, i_layer, i_index = load_event(
                self.byte_mv,
                event_id,
                self.nodes_array,
                self.sidx_indexes, self.sidx_indptr, self.sidx_val, self.loss_indptr, self.loss_val,
                self.pass_through,
                self.computes,
                compute_idx,
                self.output_array,
                i_layer,
                i_index)

            write_mv_to_stream(self.stream_out, self.byte_mv, cursor)


@nb.jit(cache=True, nopython=True)
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
