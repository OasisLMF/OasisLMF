import numba as nb
import numpy as np

from oasislmf.pytools.common.data import oasis_float, oasis_int


MEANDR = 1
FULL = 2
PERSAMPLEMEAN = 3
MEANSAMPLE = 4

MEANS = 0
SAMPLES = 1

WHEATSHEAF = 0
WHEATSHEAF_MEAN = 1


LOSSVEC2MAP_dtype = np.dtype([
    ("summary_id", oasis_int),
    ("period_no", np.int32),
    ("period_weighting", np.float64),
    ("value", oasis_float),
])


class AggReports():
    def __init__(
        self,
        ept_output_file,
        psept_output_file,
        outloss_mean,
        outloss_sample,
        period_weights,
        max_summary_id,
        num_sidxs,
    ):
        self.ept_output_file = ept_output_file
        self.psept_output_file = psept_output_file
        self.outloss_mean = outloss_mean
        self.outloss_sample = outloss_sample
        self.period_weights = period_weights
        self.max_summary_id = max_summary_id
        self.num_sidxs = num_sidxs

    def output_mean_damage_ratio(self, eptype, eptype_tvar, outloss_type):
        # TODO: Fix this function, keeps seg faulting

        epcalc = MEANDR
        has_weights, items, used_period_no = output_mean_damage_ratio(
            self.outloss_mean,
            outloss_type,
            self.period_weights,
            self.max_summary_id,
        )
        mask = ~np.isin(self.period_weights["period_no"], list(set(used_period_no)))
        unused_period_weights = self.period_weights[mask]
        print(has_weights)
        print(items)
        print(unused_period_weights)


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
def _get_mean_idx_data(idx, max_summary_id):
    summary_id = (idx % max_summary_id) + 1
    period_no = (idx // max_summary_id) + 1
    return summary_id, period_no


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def _get_sample_idx_data(idx, max_summary_id, num_sidxs):
    summary_id = (idx % max_summary_id) + 1
    idx //= max_summary_id
    remapped_sidx = idx % num_sidxs
    period_no = (idx // num_sidxs) + 1
    sidx = _inverse_remap_sidx(remapped_sidx)
    return summary_id, sidx, period_no


@nb.njit(cache=True, error_model="numpy")
def output_mean_damage_ratio(
    outloss_mean,
    outloss_type,
    period_weights,
    max_summary_id,
):
    num_rows = len(outloss_mean[outloss_mean["row_used"] == True])
    items = np.zeros(num_rows, dtype=LOSSVEC2MAP_dtype)
    i = 0
    used_period_no = np.zeros(outloss_mean["row_used"].sum(), dtype=np.int32)
    if len(period_weights) == 0:  # Mean Damage Ratio
        for idx, outloss in enumerate(outloss_mean):
            if not outloss["row_used"]:
                continue
            summary_id, _ = _get_mean_idx_data(idx, max_summary_id)
            items[i]["summary_id"] = summary_id
            if outloss_type == "agg_out_loss":
                items[i]["value"] = outloss["agg_out_loss"]
            elif outloss_type == "max_out_loss":
                items[i]["value"] = outloss["max_out_loss"]
            i += 1
        return False, items, used_period_no
    else:  # Mean Damage Ratio with Weighting
        used_count = 0
        for idx, outloss in enumerate(outloss_mean):
            if not outloss["row_used"]:
                continue
            summary_id, period_no = _get_mean_idx_data(idx, max_summary_id)
            period_weighting = period_weights[period_weights["period_no"] == period_no][0]["weighting"]
            items[i]["summary_id"] = summary_id
            items[i]["period_no"] = period_no
            items[i]["period_weighting"] = period_weighting
            if outloss_type == "agg_out_loss":
                items[i]["value"] = outloss["agg_out_loss"]
            elif outloss_type == "max_out_loss":
                items[i]["value"] = outloss["max_out_loss"]
            used_period_no[used_count] = period_no
            used_count += 1
            i += 1
        used_period_no = used_period_no[:used_count]
        return True, items, used_period_no
