import numpy as np
from oasislmf.pytools.converters.csvtobin.utils.common import read_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.gulmc.common import VALID_DAMAGE_TYPE
from oasislmf.utils.exceptions import OasisException


def _validate(data):
    from oasislmf.pytools.converters.csvtobin.manager import logger
    # Check first and last bin
    first_bin_index = data[0]["bin_index"]
    first_bin_from = data[0]["bin_from"]
    if first_bin_from != 0:
        logger.warning(f"Warning: Lower limit of first bin is not 0. Are you sure this is what you want?: bin_from={first_bin_from}")
    if first_bin_index != 1:
        raise OasisException(f"Error: First bin index must be 1, bin_index found: {first_bin_index}")

    last_bin_to = data[-1]["bin_to"]
    if last_bin_to != 1:
        logger.warning(f"Warning: Upper limit of last bin is not 1. Are you sure this is what you want?: bin_to={last_bin_to}")

    # Check contiguous bin_index
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
            f"\tRow {i}: interpolation={interpolations[i]}, bin_from={bin_froms[i]}, bin_to={bin_tos[i]}"
            for i in bad_rows
        )
        raise OasisException(f"Error: Interpolation damage value outside of range.\n {error_msg}")

    # Check valid damage_type
    damages = data['damage_type']
    invalid_mask = ~np.isin(damages, list(VALID_DAMAGE_TYPE))
    invalid_indices = np.where(invalid_mask)[0]
    if invalid_indices.size > 0:
        invalid_values = damages[invalid_mask]
        warning_msg = "\n".join(
            f"Row {i}: damage_type={val}, damage_type must be in {VALID_DAMAGE_TYPE}" for i, val in zip(invalid_indices, invalid_values)
        )
        logger.warning(f"Error: Invalid damage_type values found:\n{warning_msg}")


def damagebin_tobin(stack, file_in, file_out, file_type, no_validation):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    data = read_csv_as_ndarray(stack, file_in, headers, dtype)

    if not no_validation:
        _validate(data)

    data.tofile(file_out)
