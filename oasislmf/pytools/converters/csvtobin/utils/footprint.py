import zlib
import numpy as np
import pandas as pd

from oasislmf.pytools.common.data import resolve_file
from oasislmf.pytools.converters.csvtobin.utils.common import read_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.getmodel.common import Event_dtype, EventIndexBin_dtype, EventIndexBinZ_dtype
from oasislmf.utils.exceptions import OasisException


def _validate(data):
    df = pd.DataFrame(data)

    # Check probability sums to 1 for each (event_id, areaperil_id) group
    prob_sums = df.groupby(["event_id", "areaperil_id"])["probability"].sum()
    invalid_sums = prob_sums[~np.isclose(prob_sums, 1, atol=1e-6)]
    if not invalid_sums.empty:
        error_msg = "\n".join([
            f"Group (event_id={idx[0]}, areaperil_id={idx[1]}) has prob sum = {val:.6f}"
            for idx, val in invalid_sums.items()
        ])
        raise OasisException(f"Error: Probabilities do not sum to 1 for the following groups: \n{error_msg}")

    # Check sorted by event_id, areaperil_id
    expected_order = df.sort_values(['event_id', 'areaperil_id']).reset_index(drop=True)
    if not df[['event_id', 'areaperil_id']].equals(expected_order[['event_id', 'areaperil_id']]):
        unordered_rows = df[['event_id', 'areaperil_id']].ne(expected_order[['event_id', 'areaperil_id']]).any(axis=1)
        mismatch_indices = df.index[unordered_rows].tolist()
        raise OasisException(f"IDs not in ascending order. First few mismatched indices: \n{df.iloc[mismatch_indices[:10]]}")

    # Check intensity bin uniqueness for each (event_id, areaperil_id) group
    duplicates = df.duplicated(subset=['event_id', 'areaperil_id', 'intensity_bin_id'], keep=False)
    if duplicates.any():
        dup_rows = df[duplicates]
        error_msg = dup_rows[['event_id', 'areaperil_id', 'intensity_bin_id']].drop_duplicates(keep="last").to_string()
        raise OasisException(f"Error: Duplicate intensity bins found: \n{error_msg}")


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

    # Write bin file header
    np.array([max_intensity_bin_idx], dtype=np.int32).tofile(file_out)
    zip_opts = decompressed_size << 1 | (not no_intensity_uncertainty)
    np.array([zip_opts], dtype=np.int32).tofile(file_out)

    offset = np.dtype(np.int32).itemsize * 2

    unique_events = np.unique(data["event_id"])
    for event_id in unique_events:
        event_mask = data["event_id"] == event_id
        event_data = data[event_mask]

        bin_data = np.empty(len(event_data), dtype=Event_dtype)
        bin_data["areaperil_id"] = event_data["areaperil_id"]
        bin_data["intensity_bin_id"] = event_data["intensity_bin_id"]
        bin_data["probability"] = event_data["probability"]

        if any(bin_data["intensity_bin_id"] > max_intensity_bin_idx):
            raise OasisException(f"Error: Found intensity_bin_idx in data larger than max_intensity_bin_idx: {max_intensity_bin_idx}")

        bin_data = bin_data.tobytes()
        dsize = len(bin_data)
        if zip_files:
            bin_data = zlib.compress(bin_data)

        file_out.write(bin_data)
        size = len(bin_data)
        if decompressed_size:
            np.array([(event_id, offset, size, dsize)], dtype=EventIndexBinZ_dtype).tofile(idx_file_out)
        else:
            np.array([(event_id, offset, size)], dtype=EventIndexBin_dtype).tofile(idx_file_out)
        offset += size
