import logging
from numba import njit
from numba.core import types
from numba.typed import Dict
import numpy as np
import os

from oasis_data_manager.filestore.backends.base import BaseStorage
from .common import (
    BUFFER_SIZE,
    DATA_SIZE,
    N_PAIRS,
    FILE_HEADER_SIZE,
    event_count_dtype,
    amp_factor_dtype,
    AMPLIFICATIONS_FILE_NAME,
    LOSS_FACTORS_FILE_NAME
)

logger = logging.getLogger(__name__)


def get_items_amplifications(path):
    """
    Get array of amplification IDs from amplifications.bin, where index
    corresponds to item ID.

    amplifications.bin is binary file with layout:
        reserved header (4-byte int),
        item ID 1 (4-byte int), amplification ID a_1 (4-byte int),
        ...
        item ID n (4-byte int), amplification ID a_n (4-byte int)

    Args:
        path (str): path to amplifications.bin file

    Returns:
        items_amps (numpy.ndarray): array of amplification IDs, where index
            corresponds to item ID
    """
    try:
        items_amps = np.fromfile(
            os.path.join(path, AMPLIFICATIONS_FILE_NAME), dtype=np.int32,
            offset=FILE_HEADER_SIZE
        )
    except FileNotFoundError:
        logger.error('amplifications.bin not found')
        raise SystemExit(1)

    # Check item IDs start from 1 and are contiguous
    if items_amps[0] != 1:
        logger.error(f'First item ID is {items_amps[0]}. Expected 1.')
        raise SystemExit(1)
    items_amps = items_amps.reshape(len(items_amps) // 2, 2)
    if not np.all(items_amps[1:, 0] - items_amps[:-1, 0] == 1):
        logger.error(f'Item IDs in {os.path.join(path, AMPLIFICATIONS_FILE_NAME)} are not contiguous')
        raise SystemExit(1)

    items_amps = np.concatenate((np.array([0]), items_amps[:, 1]))

    return items_amps


@njit(cache=True)
def fill_post_loss_amplification_factors(
    event_id, count, cursor, valid_length, event_count, amp_factor, plafactors,
    secondary_factor
):
    """
    Fill Post Loss Amplification (PLA) factors dictionary mapped to
    event ID-item ID pair.

    Args:
        event_id (int): current event ID
        count (int): number of remaning amplification IDs associated with
            current event ID
        cursor (int): position in buffer
        valid_length (int): length of buffer in 8-bit chunks
        event_count (numpy.ndarray): array of event ID-count pairs
        amp_factor (numpy.ndarray): array of amplification ID-loss pairs
        plafactors (numba.typed.typeddict.Dict): PLA factors dictionary
        secondary_factor (float): secondary factor to apply to post loss
          amplification

    Returns:
        event_id (int): current event ID
        count (int): number of remaining amplification IDs associated with
            current event ID
        plafactors (numba.typed.typeddict.Dict): PLA factors dictionary
    """
    while cursor < valid_length:

        if count == 0:
            event_id = event_count[cursor]['event_id']
            count = event_count[cursor]['count']
            cursor += 1

        else:
            amplification_id = amp_factor[cursor]['amplification_id']
            loss_factor = max(
                1 + (amp_factor[cursor]['factor'] - 1) * secondary_factor, 0.0
            )   # Losses cannot be negative
            plafactors[(event_id, amplification_id)] = loss_factor
            cursor += 1
            count -= 1

    return event_id, count, plafactors


def get_post_loss_amplification_factors(storage: BaseStorage, secondary_factor, uniform_factor, ignore_file_type=set()):
    """
    Get Post Loss Amplification (PLA) factors mapped to event ID-item ID pair.
    Returns empty dictionary if uniform factor to apply across all losses has
    been given.

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
        storage: (BaseStorage) the storage connector for fetching the model data
        secondary_factor (float): secondary factor to apply to post loss
          amplification
        uniform_factor (float): uniform factor to apply across all losses
        ignore_file_type: set(str) file extension to ignore when loading

    Returns:
        plafactors (dict): event ID-item ID pairs mapped to amplification IDs
    """
    plafactors = Dict.empty(
        key_type=types.UniTuple(types.int64, 2), value_type=types.float64
    )
    if uniform_factor > 0.0:
        return plafactors

    input_files = set(storage.listdir())
    if LOSS_FACTORS_FILE_NAME in input_files and 'bin' not in ignore_file_type:
        logger.debug(f"loading {storage.get_storage_url(LOSS_FACTORS_FILE_NAME, encode_params=False)[1]}")
        with storage.with_fileno(LOSS_FACTORS_FILE_NAME) as f:
            factors_buffer = memoryview(bytearray(BUFFER_SIZE))
            event_count = np.ndarray(
                N_PAIRS, buffer=factors_buffer, dtype=event_count_dtype
            )
            amp_factor = np.ndarray(
                N_PAIRS, buffer=factors_buffer, dtype=amp_factor_dtype
            )
            f.readinto(factors_buffer[:FILE_HEADER_SIZE])   # Ignore first 4 bytes

            cursor = 0
            valid_buffer = 0
            count = 0
            event_id = 0
            while True:
                len_read = f.readinto(factors_buffer[valid_buffer:])
                valid_buffer += len_read

                if len_read == 0:
                    break

                valid_length = valid_buffer // DATA_SIZE

                event_id, count, plafactors = fill_post_loss_amplification_factors(
                    event_id, count, cursor, valid_length, event_count, amp_factor,
                    plafactors, secondary_factor
                )

                valid_buffer = 0
                cursor = 0

            return plafactors
    else:
        raise FileNotFoundError(f"lossfactors.bin file not found at {storage.get_storage_url('', encode_params=False)[1]}")
