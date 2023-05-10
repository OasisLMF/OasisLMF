import numpy as np
import os

from .common import (
    BUFFER_SIZE,
    DATA_SIZE,
    N_PAIRS,
    FILE_HEADER_SIZE,
    event_count_dtype,
    amp_factor_dtype,
    ITEMS_AMPLIFICATIONS_FILE_NAME,
    LOSS_FACTORS_FILE_NAME
)


def get_items_amplifications(path):
    """
    Get array of amplification IDs from itemsamplifications.bin, where index
    corresponds to item ID.

    itemsamplifications.bin is binary file with layout:
        reserved header (4-byte int),
        item ID 1 (4-byte int), amplification ID a_1 (4-byte int),
        ...
        item ID n (4-byte int), amplification ID a_n (4-byte int)

    Args:
        path (str): path to itemsamplifications.bin file

    Returns:
        items_amps (numpy array): array of amplification IDs, where index
            corresponds to item ID
    """
    # Assume item IDs start from 1 and are contiguous
    items_amps = np.fromfile(
        os.path.join(path, ITEMS_AMPLIFICATIONS_FILE_NAME), dtype=np.int32,
        offset=FILE_HEADER_SIZE
    )[1::2]
    items_amps = np.concatenate((np.array([0]), items_amps))

    return items_amps


def get_post_loss_amplification_factors(path):
    """
    Get Post Loss Amplification (PLA) factors mapped to event ID-item ID pair.

    lossfactors.bin is binary file with layout:
        reserved header (4-byte int),
        event ID 1 (4-byte int), number of amplification IDs for event ID 1 (4-byte int),
        amplification ID 1 (4-byte int), loss factor for amplification ID 1 (4-byte float),
        ...
        amplification ID n (4-byte int), loss factor for amplification ID n (4-byte float),
        event ID 2 (4-byte int), number of amplification IDs for event ID 2 (4-byte int),
        ...
        event ID N (4-byte int), number of amplification IDs for event ID N (4-byte int),
        amplification ID 1 (4-byte int), loss factor for amplification ID 1 (4-byte float),
        ...
        amplification ID n (4-byte int), loss factor for amplification ID n (4-byte float)

    Args:
        path (str): path to lossfactors.bin file

    Returns:
        plafactors (dict): event ID-item ID pairs mapped to amplification IDs
    """
    plafactors = {}

    with open(os.path.join(path, LOSS_FACTORS_FILE_NAME), 'rb') as f:
        factors_buffer = memoryview(bytearray(BUFFER_SIZE))
        event_count = np.ndarray(
            N_PAIRS, buffer=factors_buffer, dtype=event_count_dtype
        )
        amp_factor = np.ndarray(
            N_PAIRS, buffer=factors_buffer, dtype=amp_factor_dtype
        )
        f.readinto1(factors_buffer[:FILE_HEADER_SIZE])   # Ignore first 4 bytes

        cursor = 0
        valid_buffer = 0
        count = 0
        while True:
            len_read = f.readinto1(factors_buffer[valid_buffer:])
            valid_buffer += len_read

            if len_read == 0:
                break

            valid_length = valid_buffer // DATA_SIZE

            while cursor < valid_length:

                if count == 0:
                    event_id = event_count[cursor]['event_id']
                    count = event_count[cursor]['count']
                    cursor += 1

                else:
                    amplification_id = amp_factor[cursor]['amplification_id']
                    loss_factor = amp_factor[cursor]['factor']
                    plafactors[(event_id, amplification_id)] = loss_factor
                    cursor += 1
                    count -= 1

            valid_buffer = 0
            cursor = 0


    return plafactors
