import numba as nb
import numpy as np

from oasislmf.pytools.lec.utils import get_wheatsheaf_items_idx_data


@nb.njit(cache=True, error_model="numpy")
def get_wheatsheaf_max_count(
    wheatsheaf_items,
    wheatsheaf_items_start_end,
    max_summary_id,
):
    maxcount = np.full((max_summary_id), -1, dtype=np.int64)
    for start, end in wheatsheaf_items_start_end:
        summary_id = wheatsheaf_items[start]["summary_id"]
        size = end - start
        if size < maxcount[summary_id - 1]:
            continue
        maxcount[summary_id - 1] = size
    return maxcount


@nb.njit(cache=True, error_model="numpy")
def fill_wheatsheaf_mean_items(
    wheatsheaf_mean_items,
    wheatsheaf_items,
    wheatsheaf_items_start_end,
    maxcounts,
    max_summary_id,
    num_sidxs,
):
    # Track start and end indices for each summary_id
    items_start_end = np.full((max_summary_id, 2), -1, dtype=np.int32)

    # Compute cumulative start indices
    pos = 0
    for idx in range(max_summary_id):
        if maxcounts[idx] > 0:
            items_start_end[idx][0] = pos  # Start index
            pos += maxcounts[idx]
            items_start_end[idx][1] = pos  # End index

    for idx in range(max_summary_id * num_sidxs):
        ws_start, ws_end = wheatsheaf_items_start_end[idx]
        if ws_start == -1:
            continue
        sidx, summary_id = get_wheatsheaf_items_idx_data(idx, num_sidxs)
        filtered_items = wheatsheaf_items[ws_start:ws_end]
        sorted_idxs = np.argsort(filtered_items["value"])[::-1]
        sorted_items = filtered_items[sorted_idxs]

        wsm_start, wsm_end = items_start_end[summary_id - 1]

        for i, item in enumerate(sorted_items):
            # Compute position in the flat array
            insert_idx = wsm_start + i

            # Store values
            wheatsheaf_mean_items[insert_idx]["summary_id"] = summary_id
            wheatsheaf_mean_items[insert_idx]["value"] += item["value"]

    return items_start_end
