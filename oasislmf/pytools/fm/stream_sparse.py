"""
FM Stream I/O with Sparse Array Storage
========================================

This module handles reading loss data from the input stream (GUL or previous FM)
and writing computed losses to the output stream. Data is stored using a
CSR (Compressed Sparse Row) inspired format for memory efficiency.

Stream Format
-------------
Input/Output streams use the Oasis binary format:
- Header: stream_type (4 bytes) + max_sidx (4 bytes)
- Per item: event_id (4) + item_id (4) + [sidx (4) + loss (4)]* + delimiter (8)

Special sample indices (sidx):
- -5 (MAX_LOSS_IDX): Maximum possible loss
- -4 (CHANCE_OF_LOSS_IDX): Probability of non-zero loss (pass-through value)
- -3 (TIV_IDX): Total Insured Value
- -2: Standard deviation (ignored in FM)
- -1 (MEAN_IDX): Mean/expected loss
- 1..N: Monte Carlo sample indices

Sparse Storage
--------------
The CSR-inspired format stores data compactly:
- sidx_indptr[i] points to the start of node i's data in sidx_val
- sidx_val contains the actual sidx values (only non-zero losses)
- loss_val contains corresponding loss values

This is memory-efficient because most samples may have zero loss,
and we only store the non-zero values.

Reading Flow
------------
1. Read event_id + item_id header
2. For each (sidx, loss) pair until delimiter:
   - Add to sparse arrays maintaining sorted sidx order
3. On new event or end of stream: signal event completion

Writing Flow
------------
1. Write stream header
2. For each output node with losses:
   - Write item header (event_id, output_id, special sidx)
   - Write each (sidx, loss) pair where loss > 0
   - Write delimiter
"""

import sys
import numpy as np
import numba as nb
import logging

from oasislmf.pytools.common.event_stream import (stream_info_to_bytes, LOSS_STREAM_ID, ITEM_STREAM, PIPE_CAPACITY, EventReader,
                                                  MAX_LOSS_IDX, CHANCE_OF_LOSS_IDX, TIV_IDX, MEAN_IDX,
                                                  mv_read, mv_write_item_header, mv_write_sidx_loss, mv_write_delimiter, write_mv_to_stream)
from oasislmf.pytools.common.data import oasis_int, oasis_int_size, oasis_float, oasis_float_size

logger = logging.getLogger(__name__)


# Special sidx values: -5, -4, -3, -2, -1, 0 (delimiter)
SPECIAL_SIDX_COUNT = 6
# Size of item header: event_id + output_id + 6 special (sidx, loss) pairs
ITEM_HEADER_SIZE = 2 * oasis_int_size + SPECIAL_SIDX_COUNT * (oasis_int_size + oasis_float_size)
# Size of one (sidx, loss) pair
SIDX_LOSS_WRITE_SIZE = oasis_int_size + oasis_float_size


@nb.jit(cache=True, nopython=True)
def reset_empty_items(compute_idx, sidx_indptr, sidx_val, loss_val, computes):
    """
    Handle items that received no loss data (all samples were zero).

    If an item's sidx range is empty (no non-zero losses), we still need
    a placeholder entry to maintain array consistency. This adds a single
    TIV_IDX (-3) entry with zero loss.
    """
    if sidx_indptr[compute_idx['next_compute_i']] == sidx_indptr[compute_idx['next_compute_i'] - 1]:
        sidx_val[sidx_indptr[compute_idx['next_compute_i']]] = -3
        loss_val[sidx_indptr[compute_idx['next_compute_i']]] = 0
        sidx_indptr[compute_idx['next_compute_i']] += 1


@nb.jit(cache=True, nopython=True)
def add_new_loss(sidx, loss, compute_i, sidx_indptr, sidx_val, loss_val):
    """
    Insert a (sidx, loss) pair into the sparse arrays, maintaining sorted sidx order.

    The sidx values must be stored in sorted order for efficient lookup during
    computation. This function handles three cases:

    1. First value for this node: insert at current position
    2. Sidx > last sidx: append at end (common case, O(1))
    3. Sidx < last sidx: binary search for position, shift existing values (O(n))

    Raises ValueError if duplicate sidx is detected (stream corruption).

    Args:
        sidx: Sample index to insert
        loss: Loss value for this sample
        compute_i: Current computation index (node being populated)
        sidx_indptr: CSR pointers into sidx_val
        sidx_val: Sample index values
        loss_val: Loss values
    """
    # Fast path: empty or append at end (sidx values usually arrive in order)
    if ((sidx_indptr[compute_i - 1] == sidx_indptr[compute_i])
            or (sidx_val[sidx_indptr[compute_i] - 1] < sidx)):
        insert_i = sidx_indptr[compute_i]
    else:
        # Slow path: need to insert in middle, shift existing values
        insert_i = np.searchsorted(sidx_val[sidx_indptr[compute_i - 1]: sidx_indptr[compute_i]], sidx) + sidx_indptr[compute_i - 1]
        if sidx_val[insert_i] == sidx:
            raise ValueError("duplicated sidx in input stream")
        # Shift values to make room for insertion
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
    """
    Parse a buffer of stream data, populating sparse loss arrays.

    This is the core stream parsing function. It handles the state machine for
    reading the Oasis binary stream format:

    State: item_id == 0 (reading header)
        - Read event_id, item_id
        - Initialize node storage pointers
        - Add node to compute queue

    State: item_id != 0 (reading item data)
        - Read (sidx, loss) pairs
        - sidx == 0: delimiter, item complete, return to header state
        - sidx == -2: standard deviation (ignored)
        - sidx == -4: chance of loss (stored in pass_through)
        - other: regular loss sample, add to sparse arrays

    The function processes until:
    - Buffer exhausted (returns with done=0, will be called again)
    - New event detected (returns with done=1, event complete)

    Args:
        byte_mv: Memory view of the input buffer
        cursor: Current read position in buffer
        valid_buff: Number of valid bytes in buffer
        event_id: Current event ID (0 on first call)
        item_id: Current item ID (0 when reading header)
        nodes_array: Node metadata for mapping item_id to storage
        sidx_indexes: Maps node_id to sidx array position
        sidx_indptr: CSR pointers for sidx
        sidx_val: Sample index values
        loss_indptr: CSR pointers for loss
        loss_val: Loss values
        pass_through: Chance-of-loss values per item
        computes: Queue of items to compute
        compute_idx: Computation state pointers

    Returns:
        (cursor, event_id, item_id, done): Updated state
        - done=1: Event complete, ready for computation
        - done=0: Need more data, call again with next buffer
    """
    last_event_id = event_id
    while True:
        if item_id:
            # --- Reading (sidx, loss) pairs for current item ---
            if valid_buff - cursor < (oasis_int_size + oasis_float_size):
                break  # Need more data
            sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            if sidx:
                loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
                loss = 0 if np.isnan(loss) else loss

                # Process the (sidx, loss) pair
                if loss != 0:
                    if sidx == -2:
                        pass  # Standard deviation - ignored in FM
                    elif sidx == -4:
                        # Chance of loss - store separately for pass-through
                        pass_through[compute_idx['next_compute_i']] = loss
                    else:
                        # Regular sample or special index (-5, -3, -1, 1..N)
                        add_new_loss(sidx, loss, compute_idx['next_compute_i'], sidx_indptr, sidx_val, loss_val)
            else:
                # sidx == 0: Item delimiter reached
                reset_empty_items(compute_idx, sidx_indptr, sidx_val, loss_val, computes)
                cursor += oasis_float_size  # Skip the delimiter's loss value
                item_id = 0  # Return to header-reading state
        else:
            # --- Reading event_id + item_id header ---
            if valid_buff - cursor < 2 * oasis_int_size:
                break  # Need more data
            event_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            if event_id != last_event_id:
                if last_event_id:
                    # New event started - return to process the completed event
                    # Rewind cursor so the new event_id is read again next time
                    return cursor - oasis_int_size, last_event_id, 0, 1
                else:
                    # First event in stream - record it
                    last_event_id = event_id
            item_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)

            # Initialize storage for this item
            node = nodes_array[item_id]
            # Map node_id to its position in the compute arrays
            sidx_indexes[node['node_id']] = compute_idx['next_compute_i']
            # Initialize loss pointers for all layers to current position
            loss_indptr[node['loss']: node['loss'] + node['layer_len']] = sidx_indptr[compute_idx['next_compute_i']]
            # Prepare next sidx_indptr entry
            sidx_indptr[compute_idx['next_compute_i'] + 1] = sidx_indptr[compute_idx['next_compute_i']]
            # Add to compute queue
            computes[compute_idx['next_compute_i']] = item_id
            compute_idx['next_compute_i'] += 1

    return cursor, event_id, item_id, 0  # Need more data


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

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, **kwargs):
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
               computes, compute_idx, output_array, layer_i, val_i):
    cursor = 0
    node_id = computes[compute_idx['level_start_compute_i']]

    while node_id:
        node = nodes_array[node_id]
        node_sidx_start = sidx_indptr[sidx_indexes[node['node_id']]]
        node_sidx_end = sidx_indptr[sidx_indexes[node['node_id']] + 1]
        node_val_count = node_sidx_end - node_sidx_start
        if node_id < pass_through.shape[0]:
            pass_through_loss = pass_through[node_id]
        else:
            pass_through_loss = 0
        for layer_cur_i in range(layer_i, node['layer_len']):
            output_id = output_array[node['output_ids'] + layer_cur_i]
            node_loss_start = loss_indptr[node['loss'] + layer_cur_i]

            assert node_loss_start + node_val_count < loss_val.shape[0], "loss index is out of range, this must not happen"

            # print('output_id', output_id)
            # print('    ', sidx_val[node_sidx_start: node_sidx_end])
            # print('    ', loss_val[node_loss_start: node_loss_start + node_val_count])
            if output_id and node_val_count:  # if output is not in xref output_id is 0
                if val_i == -1:
                    if cursor < PIPE_CAPACITY - ITEM_HEADER_SIZE:  # header + -5, -3, -1 sample
                        cursor = mv_write_item_header(byte_mv, cursor, event_id, output_id)
                        val_i += 1

                        # write MAX_LOSS_IDX == -5 sidx
                        if sidx_val[node_sidx_start + val_i] == MAX_LOSS_IDX:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, MAX_LOSS_IDX, loss_val[node_loss_start + val_i])
                            val_i += 1
                        else:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, MAX_LOSS_IDX, 0)

                        # write CHANCE_OF_LOSS_IDX == -4 sidx
                        if pass_through_loss:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, CHANCE_OF_LOSS_IDX, pass_through_loss)

                        # write TIV_IDX == -3 sidx
                        if sidx_val[node_sidx_start + val_i] == TIV_IDX and val_i < node_val_count:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, TIV_IDX, loss_val[node_loss_start + val_i])
                            val_i += 1
                        else:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, TIV_IDX, 0)

                        # write MEAN_IDX == -1 sidx
                        if sidx_val[node_sidx_start + val_i] == MEAN_IDX and val_i < node_val_count:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, MEAN_IDX, loss_val[node_loss_start + val_i])
                            val_i += 1
                        else:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, MEAN_IDX, 0)
                    else:
                        return cursor, node_id, layer_cur_i, val_i

                while cursor < PIPE_CAPACITY - SIDX_LOSS_WRITE_SIZE:
                    if val_i == node_val_count:
                        cursor = mv_write_delimiter(byte_mv, cursor)
                        val_i = -1
                        layer_i = 0
                        break
                    else:
                        if loss_val[node_loss_start + val_i]:
                            cursor = mv_write_sidx_loss(byte_mv, cursor, sidx_val[node_sidx_start + val_i], loss_val[node_loss_start + val_i])
                        val_i += 1

                else:
                    return cursor, node_id, layer_cur_i, val_i
        compute_idx['level_start_compute_i'] += 1
        node_id = computes[compute_idx['level_start_compute_i']]

    return cursor, node_id, 0, val_i


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
        val_i = -1
        layer_i = 0
        node_id = 1
        while node_id:
            cursor, node_id, layer_i, val_i = load_event(
                self.byte_mv,
                event_id,
                self.nodes_array,
                self.sidx_indexes, self.sidx_indptr, self.sidx_val, self.loss_indptr, self.loss_val,
                self.pass_through,
                self.computes,
                compute_idx,
                self.output_array,
                layer_i,
                val_i)

            write_mv_to_stream(self.stream_out, self.byte_mv, cursor)


@nb.jit(cache=True, nopython=True)
def get_compute_end(computes, compute_idx):
    compute_start_i = compute_end_i = compute_idx['level_start_compute_i']
    while computes[compute_end_i]:
        compute_end_i += 1
    return compute_start_i, compute_end_i


class EventWriterOrderedOutputSparse(EventWriterSparse):
    def write(self, event_id, compute_idx):
        compute_start_i, compute_end_i = get_compute_end(self.computes, compute_idx)
        self.computes[compute_start_i: compute_end_i] = np.sort(self.computes[compute_start_i: compute_end_i], kind='stable')
        return super().write(event_id, compute_idx)
