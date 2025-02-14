import numba as nb
import numpy as np

from oasislmf.pytools.lec.utils import get_sample_idx_data, get_wheatsheaf_items_idx


@nb.njit(cache=True, error_model="numpy")
def fill_wheatsheaf_items(
    items,
    items_start_end,
    row_used_indices,
    outloss_vals,
    period_weights,
    max_summary_id,
    num_sidxs,
):
    # Track number of entries per summary_id
    summary_sidx_counts = np.zeros(max_summary_id * num_sidxs, dtype=np.int32)

    # First pass to count how many times each summary_id appears
    for idx in row_used_indices:
        summary_id, sidx, period_no = get_sample_idx_data(idx, max_summary_id, num_sidxs)
        summary_sidx_idx = get_wheatsheaf_items_idx(summary_id, sidx, num_sidxs)
        summary_sidx_counts[summary_sidx_idx] += 1

    # Compute cumulative start indices
    pos = 0
    for idx in range(max_summary_id * num_sidxs):
        if summary_sidx_counts[idx] > 0:
            items_start_end[idx][0] = pos  # Start index
            pos += summary_sidx_counts[idx]
            items_start_end[idx][1] = pos  # End index

    # Reset summary counts for inserting data
    summary_sidx_counts[:] = 0

    # Track which period_no are used if period_weights exists
    is_weighted = len(period_weights) > 0
    used_period_no = np.zeros(len(period_weights), dtype=np.bool_)

    # Second pass to populate the data array
    for idx in row_used_indices:
        summary_id, sidx, period_no = get_sample_idx_data(idx, max_summary_id, num_sidxs)
        summary_sidx_idx = get_wheatsheaf_items_idx(summary_id, sidx, num_sidxs)

        # Compute position in the flat array
        insert_idx = items_start_end[summary_sidx_idx][0] + summary_sidx_counts[summary_sidx_idx]

        # Store values
        items[insert_idx]["summary_id"] = summary_id
        items[insert_idx]["sidx"] = sidx
        items[insert_idx]["value"] = outloss_vals[idx]
        if is_weighted:
            # Fast lookup period_weights as they are numbered 1 to no_of_periods
            items[insert_idx]["period_weighting"] = period_weights[period_no - 1]["weighting"]
            items[insert_idx]["period_no"] = period_no
            used_period_no[period_no - 1] = True

        summary_sidx_counts[summary_sidx_idx] += 1

    return is_weighted, used_period_no
