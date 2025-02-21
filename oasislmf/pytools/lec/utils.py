import numba as nb
import numpy as np


@nb.njit(cache=True, error_model="numpy")
def create_empty_array(mydtype, init_size=8):
    return np.zeros(init_size, dtype=mydtype)


@nb.njit(cache=True, error_model="numpy")
def resize_array(array, current_size):
    if current_size >= len(array):  # Resize if the array is full
        new_array = np.empty(len(array) * 2, dtype=array.dtype)
        new_array[:len(array)] = array
        array = new_array
    return array


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def _remap_sidx(sidx):
    if sidx == -2:
        return 0
    if sidx == -3:
        return 1
    if sidx >= 1:
        return sidx + 1
    raise ValueError(f"Invalid sidx value: {sidx}")


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def get_outloss_mean_idx(period_no, summary_id, max_summary_id):
    return ((period_no - 1) * max_summary_id) + (summary_id - 1)


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def get_outloss_sample_idx(period_no, sidx, summary_id, num_sidxs, max_summary_id):
    remapped_sidx = _remap_sidx(sidx)
    return ((period_no - 1) * num_sidxs * max_summary_id) + (remapped_sidx * max_summary_id) + (summary_id - 1)


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def _inverse_remap_sidx(remapped_sidx):
    if remapped_sidx == 0:
        return -2
    if remapped_sidx == 1:
        return -3
    if remapped_sidx >= 2:
        return remapped_sidx - 1
    raise ValueError(f"Invalid remapped_sidx value: {remapped_sidx}")


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def get_mean_idx_data(idx, max_summary_id):
    summary_id = (idx % max_summary_id) + 1
    period_no = (idx // max_summary_id) + 1
    return summary_id, period_no


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def get_sample_idx_data(idx, max_summary_id, num_sidxs):
    summary_id = (idx % max_summary_id) + 1
    idx //= max_summary_id
    remapped_sidx = idx % num_sidxs
    period_no = (idx // num_sidxs) + 1
    sidx = _inverse_remap_sidx(remapped_sidx)
    return summary_id, sidx, period_no


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def get_wheatsheaf_items_idx(summary_id, sidx, num_sidxs):
    remapped_sidx = _remap_sidx(sidx)
    return ((summary_id - 1) * num_sidxs) + (remapped_sidx)


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def get_wheatsheaf_items_idx_data(idx, num_sidxs):
    summary_id = (idx // num_sidxs) + 1
    remapped_sidx = (idx % num_sidxs)
    sidx = _inverse_remap_sidx(remapped_sidx)
    return sidx, summary_id


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def get_sample_mean_outlosses_idx(summary_id, period_no, no_of_periods):
    return ((summary_id - 1) * no_of_periods) + (period_no - 1)


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def get_sample_mean_outlosses_idx_data(idx, no_of_periods):
    summary_id = (idx // no_of_periods) + 1
    period_no = (idx % no_of_periods) + 1
    return period_no, summary_id
