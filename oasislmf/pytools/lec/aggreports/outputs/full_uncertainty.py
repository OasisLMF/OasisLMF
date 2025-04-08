import numba as nb
import numpy as np

from oasislmf.pytools.lec.utils import get_sample_idx_data


@nb.njit(cache=True, error_model="numpy")
def output_full_uncertainty(
    items,
    items_start_end,
    row_used_indices,
    outloss_vals,
    period_weights,
    max_summary_id,
    num_sidxs,
):
    # Track number of entries per summary_id
    summary_counts = np.zeros(max_summary_id, dtype=np.int32)

    # First pass to count how many times each summary_id appears
    for idx in row_used_indices:
        summary_id, sidx, period_no = get_sample_idx_data(idx, max_summary_id, num_sidxs)
        summary_counts[summary_id - 1] += 1

    # Compute cumulative start indices
    pos = 0
    for idx in range(max_summary_id):
        if summary_counts[idx] > 0:
            items_start_end[idx][0] = pos  # Start index
            pos += summary_counts[idx]
            items_start_end[idx][1] = pos  # End index

    # Reset summary counts for inserting data
    summary_counts[:] = 0

    # Track which period_no are used if period_weights exists
    is_weighted = len(period_weights) > 0
    used_period_no = np.zeros(len(period_weights), dtype=np.bool_)

    # Second pass to populate the data array
    for idx in row_used_indices:
        summary_id, sidx, period_no = get_sample_idx_data(idx, max_summary_id, num_sidxs)

        # Compute position in the flat array
        insert_idx = items_start_end[summary_id - 1][0] + summary_counts[summary_id - 1]

        # Store values
        items[insert_idx]["summary_id"] = summary_id
        items[insert_idx]["value"] = outloss_vals[idx]
        if is_weighted:
            # Fast lookup period_weights as they are numbered 1 to no_of_periods
            items[insert_idx]["period_weighting"] = period_weights[period_no - 1]["weighting"]
            items[insert_idx]["period_no"] = period_no
            used_period_no[period_no - 1] = True

        summary_counts[summary_id - 1] += 1

    return is_weighted, used_period_no
