import zlib
import numba as nb
import numpy as np

from oasislmf.pytools.common.data import resolve_file
from oasislmf.pytools.converters.csvtobin.utils.common import read_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.getmodel.common import Event_dtype, EventIndexBin_dtype, EventIndexBinZ_dtype
from oasislmf.utils.exceptions import OasisException


@nb.njit(cache=True, error_model="numpy")
def _check_sorted(event_ids, areaperil_ids):
    """Returns first out-of-order row index, or -1 if sorted."""
    for i in range(1, len(event_ids)):
        if event_ids[i] < event_ids[i - 1]:
            return i
        if event_ids[i] == event_ids[i - 1] and areaperil_ids[i] < areaperil_ids[i - 1]:
            return i
    return -1


@nb.njit(cache=True, error_model="numpy")
def _check_prob_sums(event_ids, areaperil_ids, probs, atol=1e-6):
    """Single-pass probability sum check assuming sorted data.
    Returns the last-row index of the first bad group, or -1 if all valid."""
    if len(event_ids) == 0:
        return -1
    group_sum = np.float64(probs[0])
    for i in range(1, len(event_ids)):
        if event_ids[i] != event_ids[i - 1] or areaperil_ids[i] != areaperil_ids[i - 1]:
            if abs(group_sum - 1.0) > atol:
                return i - 1
            group_sum = np.float64(probs[i])
        else:
            group_sum += probs[i]
    if abs(group_sum - 1.0) > atol:
        return len(event_ids) - 1
    return -1


@nb.njit(cache=True, error_model="numpy")
def _check_duplicates(event_ids, areaperil_ids, intensity_bin_ids):
    """Single-pass duplicate intensity_bin_id check assuming sorted data.
    Returns first duplicate row index, or -1 if no duplicates."""
    for i in range(1, len(event_ids)):
        if (event_ids[i] == event_ids[i - 1]
                and areaperil_ids[i] == areaperil_ids[i - 1]
                and intensity_bin_ids[i] == intensity_bin_ids[i - 1]):
            return i
    return -1


@nb.njit(cache=True, error_model="numpy")
def _exceeds_max_intensity(intensity_bin_ids, max_val):
    """Early-exit check for any intensity_bin_id exceeding max_val."""
    for v in intensity_bin_ids:
        if v > max_val:
            return True
    return False


def _validate(data):
    idx = _check_sorted(data["event_id"], data["areaperil_id"])
    if idx != -1:
        raise OasisException(
            f"IDs not in ascending order at row {idx}: {data[idx]}"
        )

    idx = _check_prob_sums(data["event_id"], data["areaperil_id"], data["probability"])
    if idx != -1:
        raise OasisException(
            f"Probabilities do not sum to 1 for group ending at row {idx}: "
            f"event_id={data['event_id'][idx]}, areaperil_id={data['areaperil_id'][idx]}"
        )

    idx = _check_duplicates(data["event_id"], data["areaperil_id"], data["intensity_bin_id"])
    if idx != -1:
        raise OasisException(
            f"Duplicate intensity_bin_id at row {idx}: "
            f"event_id={data['event_id'][idx]}, areaperil_id={data['areaperil_id'][idx]}, "
            f"intensity_bin_id={data['intensity_bin_id'][idx]}"
        )


def footprint_tobin(
    stack, file_in, file_out, file_type,
    idx_file_out,
    zip_files,
    max_intensity_bin_idx,
    no_intensity_uncertainty,
    decompressed_size,
    no_validation
):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    idx_file_out = resolve_file(idx_file_out, "wb", stack)
    data = read_csv_as_ndarray(stack, file_in, headers, dtype)

    if not no_validation:
        _validate(data)
        # _validate confirmed sort order; reuse that guarantee — skip re-sorting
        sorted_data = data
        unique_events, group_starts = np.unique(data["event_id"], return_index=True)
    else:
        sort_idx = np.argsort(data["event_id"], kind="stable")
        sorted_data = data[sort_idx]
        unique_events, group_starts = np.unique(sorted_data["event_id"], return_index=True)

    group_ends = np.append(group_starts[1:], len(sorted_data))

    # Write bin file header
    np.array([max_intensity_bin_idx], dtype=np.int32).tofile(file_out)
    zip_opts = decompressed_size << 1 | (not no_intensity_uncertainty)
    np.array([zip_opts], dtype=np.int32).tofile(file_out)

    offset = np.dtype(np.int32).itemsize * 2

    idx_dtype = EventIndexBinZ_dtype if decompressed_size else EventIndexBin_dtype
    idx_entries = np.empty(len(unique_events), dtype=idx_dtype)

    for i, (event_id, start, end) in enumerate(zip(unique_events, group_starts, group_ends)):
        event_data = sorted_data[start:end]

        bin_data = np.empty(end - start, dtype=Event_dtype)
        bin_data["areaperil_id"] = event_data["areaperil_id"]
        bin_data["intensity_bin_id"] = event_data["intensity_bin_id"]
        bin_data["probability"] = event_data["probability"]

        if _exceeds_max_intensity(bin_data["intensity_bin_id"], max_intensity_bin_idx):
            raise OasisException(
                f"Error: Found intensity_bin_idx in data larger than max_intensity_bin_idx: {max_intensity_bin_idx}"
            )

        bin_bytes = bin_data.tobytes()
        dsize = len(bin_bytes)
        if zip_files:
            bin_bytes = zlib.compress(bin_bytes)

        file_out.write(bin_bytes)
        size = len(bin_bytes)
        if decompressed_size:
            idx_entries[i] = (event_id, offset, size, dsize)
        else:
            idx_entries[i] = (event_id, offset, size)
        offset += size

    idx_entries.tofile(idx_file_out)
