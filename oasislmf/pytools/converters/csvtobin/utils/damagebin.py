import numpy as np
from oasislmf.pytools.converters.csvtobin.utils.common import iter_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.gulmc.common import VALID_DAMAGE_TYPE
from oasislmf.utils.exceptions import OasisException


def _validate_chunk(data, logger, first_chunk, prev_last_bin_index, row_offset):
    if len(data) == 0:
        return

    # Check first bin
    if first_chunk:
        if data[0]["bin_from"] != 0:
            logger.warning(f"Warning: Lower limit of first bin is not 0. Are you sure this is what you want?: bin_from={data[0]['bin_from']}")
        if data[0]["bin_index"] != 1:
            raise OasisException(f"Error: First bin index must be 1, bin_index found: {data[0]['bin_index']}")

    # Check contiguous bin_index
    if prev_last_bin_index is not None and data[0]["bin_index"] != prev_last_bin_index + 1:
        raise OasisException(f"Error: Non-contiguous bin_indices found in csv. ({prev_last_bin_index}, {data[0]['bin_index']})")

    if len(data) > 1:
        indices = data["bin_index"]
        diffs = np.diff(indices)
        if not np.all(diffs == 1):
            idx = np.where(diffs != 1)[0][0]
            v1, v2 = indices[idx], indices[idx + 1]
            raise OasisException(f"Error: Non-contiguous bin_indices found in csv. ({v1}, {v2})")

    # Check interpolation damage values within range
    bin_froms = data["bin_from"]
    bin_tos = data["bin_to"]
    interpolations = data["interpolation"]
    mask_low = interpolations < bin_froms
    mask_high = interpolations > bin_tos
    mask_invalid = mask_low | mask_high
    if np.any(mask_invalid):
        bad_rows = np.where(mask_invalid)[0]
        error_msg = "\n".join(
            f"\tRow {row_offset + i}: interpolation={interpolations[i]}, bin_from={bin_froms[i]}, bin_to={bin_tos[i]}"
            for i in bad_rows
        )
        raise OasisException(f"Error: Interpolation damage value outside of range.\n {error_msg}")

    # Check valid damage_type
    damages = data['damage_type']
    invalid_mask = ~np.isin(damages, list(VALID_DAMAGE_TYPE))
    if invalid_mask.any():
        invalid_indices = np.where(invalid_mask)[0]
        invalid_values = damages[invalid_mask]
        warning_msg = "\n".join(
            f"Row {row_offset + i}: damage_type={val}, damage_type must be in {VALID_DAMAGE_TYPE}"
            for i, val in zip(invalid_indices, invalid_values)
        )
        logger.warning(f"Error: Invalid damage_type values found:\n{warning_msg}")


def damagebin_tobin(stack, file_in, file_out, file_type, no_validation):
    from oasislmf.pytools.converters.csvtobin.manager import logger
    dtype = TOOL_INFO[file_type]["dtype"]

    first_chunk = True
    prev_last_bin_index = None
    last_row = None
    row_offset = 0

    for chunk in iter_csv_as_ndarray(stack, file_in, dtype):
        if not no_validation:
            _validate_chunk(chunk, logger, first_chunk, prev_last_bin_index, row_offset)

        chunk.tofile(file_out)
        first_chunk = False

        if len(chunk) > 0:
            prev_last_bin_index = int(chunk["bin_index"][-1])
            last_row = chunk[-1:]
            row_offset += len(chunk)

    # Check last bin
    if not no_validation and last_row is not None:
        if last_row[0]["bin_to"] != 1:
            logger.warning(f"Warning: Upper limit of last bin is not 1. Are you sure this is what you want?: bin_to={last_row[0]['bin_to']}")
