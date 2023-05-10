import numpy as np

from .common import (
    N_PAIRS,
    BUFFER_SIZE,
    DATA_SIZE,
    event_item_dtype,
    sidx_loss_dtype
)


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

        while cursor < valid_length:

            if sidx != 0:   # end of samples
                sidx = sidx_loss_in[cursor]['sidx']
                loss = sidx_loss_in[cursor]['loss'] * factor
                stream_out.write(np.int32(sidx).tobytes())
                stream_out.write(np.float32(loss).tobytes())
                cursor += 1

            else:
                event_id = event_item_in[cursor]['event_id']
                item_id = event_item_in[cursor]['item_id']
                # loss factor defaults to 1.0 if missing (i.e. no change)
                factor = plafactors.get((event_id, items_amps[item_id]), 1.0)
                stream_out.write(np.int32(event_id).tobytes())
                stream_out.write(np.int32(item_id).tobytes())
                cursor += 1
                sidx = -1

        valid_buffer = 0
        cursor = 0

    stream_in.close()
    stream_out.close()
