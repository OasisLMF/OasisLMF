from numba import njit
import numpy as np

from .common import (
    N_PAIRS,
    BUFFER_SIZE,
    DATA_SIZE,
    event_item_dtype,
    sidx_loss_dtype
)


@njit(cache=True, fastmath=True)
def read_and_write_sidx_loss(cursor, factor, sidx_loss_in, sidx_loss_out):
    """
    Read sample ID (sidx) and loss from input stream. Write sidx and loss after
    Post Loss Amplification (PLA) to output stream.

    Args:
        cursor (int): position in buffer
        factor (float): PLA factor
        sidx_loss_in (numpy.ndarray): input array of sidx-loss pairs
        sidx_loss_out (numpy.ndarray): output array of sidx-loss pairs

    Returns:
        cursor (int): position in buffer
        sidx (int): sample ID
    """

    sidx = sidx_loss_in[cursor]['sidx']
    loss = sidx_loss_in[cursor]['loss'] * factor
    sidx_loss_out[cursor]['sidx'] = sidx
    sidx_loss_out[cursor]['loss'] = loss
    cursor += 1

    return cursor, sidx


@njit(cache=True)
def read_and_write_event_item_ids_and_plafactor(
    cursor, event_item_in, event_item_out, plafactors, items_amps
):
    """
    Read event ID, item ID and Post Loss Amplification (PLA) factor from input
    stream. Write event ID and item ID to output stream.

    Args:
        cursor (int): position in buffer
        event_item_in (numpy.ndarray): input array of event ID-item ID pairs
        event_item_out (numpy.ndarray): output array of event ID-item ID pairs
        plafactors (numba.typed.typeddict.Dict): PLA factors dictionary mapped
            to event ID-item ID pair
        items_amps (numpy.ndarray): array of amplification IDs, where index
            corresponds to item ID

    Returns:
        cursor (int): position in buffer
        sidx (int): sample ID
        factor (float): PLA factor
    """
    event_id = event_item_in[cursor]['event_id']
    item_id = event_item_in[cursor]['item_id']
    event_item_out[cursor]['event_id'] = event_id
    event_item_out[cursor]['item_id'] = item_id
    cursor += 1
    sidx = -1

    # loss factor defaults to 1.0 if missing (i.e. no change)
    factor = plafactors.get((event_id, items_amps[item_id]), 1.0)

    return cursor, sidx, factor


@njit(cache=True)
def read_and_write_buffers(
    cursor, valid_length, sidx, factor, sidx_loss_in, sidx_loss_out,
    event_item_in, event_item_out, plafactors, items_amps
):
    """
    Cycle through data in input buffer. Call functions to extract event IDs,
    item IDs, sample IDs (sidx) and losses. Call functions to write event IDs,
    items IDs, sidx and Post Loss Amplification (PLA) losses to output buffer.

    Args:
        cursor (int): position in buffer
        valid_length (int): length of buffer as a multiple of DATA_SIZE
        sidx (int): sample ID
        factor (float): PLA factor
        sidx_loss_in (numpy.ndarray): input array of sidx-loss pairs
        sidx_loss_out (numpy.ndarray): output array of sidx-loss pairs
        event_item_in (numpy.ndarray): input array of event ID-item ID pairs
        event_item_out (numpy.ndarray): output array of event ID-item ID pairs
        plafactors (numba.typed.typeddict.Dict): PLA factors dictionary mapped
            to event ID-item ID pair
        items_amps (numpy.ndarray): array of amplification IDs, where index
            corresponds to item ID

    Returns:
        cursor (int): position in buffer
        sidx (int): sample ID
        factor (float): PLA factor
    """
    while cursor < valid_length:

        if sidx != 0:   # end of samples
            cursor, sidx = read_and_write_sidx_loss(
                cursor, factor, sidx_loss_in, sidx_loss_out
            )

        else:
            cursor, sidx, factor = read_and_write_event_item_ids_and_plafactor(
                cursor, event_item_in, event_item_out, plafactors, items_amps
            )

    return cursor, sidx, factor


def read_and_write_streams(stream_in, stream_out, items_amps, plafactors):
    """
    Read input stream from gulpy or gulcalc, determine amplification ID from
    item ID, determine loss factor from event ID and amplification ID pair,
    multiply losses by relevant factors, and write to output stream.

    Input stream is binary file with layout:
        stream type (4-byte int), maximum sidx value (4-byte int),
        event ID 1 (4-byte int), item ID 1 (4-byte int),
        sample ID/sidx 1 (4-byte int), loss for sidx 1 (4-byte float),
        ...
        sample ID/sidx n (4-byte int), loss for sidx n (4-byte float),
        0 (4-byte int), 0.0 (4-byte float),
        event ID 1 (4-byte int), item ID 2 (4-byte int),
        ...
        event ID M (4-byte int), item ID N (4-byte int),
        sample ID/sidx 1 (4-byte int), loss for sidx 1 (4-byte float),
        ...
        sample ID/sidx n (4-byte int), loss for sidx n (4-byte float)

    Sample ID/sidx of 0 indicates start of next event ID-item ID pair. Output
    stream has same format as input stream.

    Args:
        stream_in (buffer): input stream
        stream_out (buffer): output stream
        items_amps (numpy array): amplification IDs where indexes correspond to
            item IDs
        plafactors (dict): event ID and amplification ID pairs mapped to loss
            factors
    """
    # Read and write 8-byte (two 4-byte integers) header
    stream_type_and_max_sidx_val = stream_in.read(8)
    stream_out.write(stream_type_and_max_sidx_val)

    read_buffer = memoryview(bytearray(BUFFER_SIZE))
    event_item_in = np.ndarray(
        N_PAIRS, buffer=read_buffer, dtype=event_item_dtype
    )
    sidx_loss_in = np.ndarray(
        N_PAIRS, buffer=read_buffer, dtype=sidx_loss_dtype
    )

    write_buffer = memoryview(bytearray(N_PAIRS * DATA_SIZE))
    event_item_out = np.ndarray(
        N_PAIRS, buffer=write_buffer, dtype=event_item_dtype
    )
    sidx_loss_out = np.ndarray(
        N_PAIRS, buffer=write_buffer, dtype=sidx_loss_dtype
    )

    cursor = 0
    valid_buffer = 0
    sidx = 0
    factor = 1
    while True:
        len_read = stream_in.readinto1(read_buffer[valid_buffer:])
        valid_buffer += len_read

        if len_read == 0:
            break

        valid_length = valid_buffer // DATA_SIZE

        cursor, sidx, factor = read_and_write_buffers(
            cursor, valid_length, sidx, factor, sidx_loss_in, sidx_loss_out,
            event_item_in, event_item_out, plafactors, items_amps
        )

        stream_out.write(write_buffer[:cursor * DATA_SIZE])
        valid_buffer = 0
        cursor = 0

    stream_in.close()
    stream_out.close()
