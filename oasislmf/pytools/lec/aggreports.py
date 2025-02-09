import logging
from pathlib import Path
from line_profiler import profile
import numba as nb
import numpy as np

from oasislmf.pytools.common.data import oasis_float, oasis_int, nb_oasis_int

logger = logging.getLogger(__name__)

# Output flags
AGG_FULL_UNCERTAINTY = 0
AGG_WHEATSHEAF = 1
AGG_SAMPLE_MEAN = 2
AGG_WHEATSHEAF_MEAN = 3
OCC_FULL_UNCERTAINTY = 4
OCC_WHEATSHEAF = 5
OCC_SAMPLE_MEAN = 6
OCC_WHEATSHEAF_MEAN = 7

# EPCalcs
MEANDR = 1
FULL = 2
PERSAMPLEMEAN = 3
MEANSAMPLE = 4

# EPTypes
OEP = 1
OEPTVAR = 2
AEP = 3
AEPTVAR = 4


EPT_output = [
    ('SummaryId', oasis_int, '%d'),
    ('EPCalc', oasis_int, '%d'),
    ('EPType', oasis_int, '%d'),
    ('ReturnPeriod', oasis_float, '%.6f'),
    ('Loss', oasis_float, '%.6f'),
]

PSEPT_output = [
    ('SummaryId', oasis_int, '%d'),
    ('SampleId', oasis_int, '%d'),
    ('EPType', oasis_int, '%d'),
    ('ReturnPeriod', oasis_float, '%.6f'),
    ('Loss', oasis_float, '%.6f'),
]

EPT_dtype = np.dtype([(c[0], c[1]) for c in EPT_output])
PSEPT_dtype = np.dtype([(c[0], c[1]) for c in PSEPT_output])
EPT_fmt = ','.join([c[2] for c in EPT_output])
PSEPT_fmt = ','.join([c[2] for c in PSEPT_output])

LOSSVEC2MAP_dtype = np.dtype([
    ("summary_id", oasis_int),
    ("period_no", np.int32),
    ("period_weighting", np.float64),
    ("value", oasis_float),
])

WHEATKEYITEMS_dtype = np.dtype([
    ("summary_id", oasis_int),
    ("sidx", oasis_int),
    ("period_no", np.int32),
    ("period_weighting", np.float64),
    ("value", oasis_float),
])

MEANMAP_dtype = np.dtype([
    ("retperiod", np.float64),
    ("mean", np.float64),
    ("count", np.int32),
])

# For Dict of summary_id (oasis_int) to nb_Tail_valtype
TAIL_valtype = np.dtype([
    ("retperiod", np.float64),
    ("tvar", oasis_float),
])
nb_TAIL_valtype = nb.types.Array(nb.from_dtype(TAIL_valtype), 1, "C")


class AggReports():
    def __init__(
        self,
        output_files,
        outloss_mean,
        outloss_sample,
        period_weights,
        max_summary_id,
        sample_size,
        no_of_periods,
        num_sidxs,
        use_return_period,
        returnperiods,
        lec_files_folder,
    ):
        self.output_files = output_files
        self.outloss_mean = outloss_mean
        self.outloss_sample = outloss_sample
        self.period_weights = period_weights
        self.max_summary_id = max_summary_id
        self.sample_size = sample_size
        self.no_of_periods = no_of_periods
        self.num_sidxs = num_sidxs
        self.use_return_period = use_return_period
        self.returnperiods = returnperiods
        self.lec_files_folder = lec_files_folder

    @profile
    def output_mean_damage_ratio(self, eptype, eptype_tvar, outloss_type):
        epcalc = MEANDR

        # Get row indices that are used
        row_used_indices = np.where(self.outloss_mean["row_used"])[0]

        # Allocate storage for the flat data array
        items_fp = Path(self.lec_files_folder, f"lec_mean_damage_ratio-{outloss_type}-items.bdat")
        items = np.memmap(items_fp, dtype=LOSSVEC2MAP_dtype, mode="w+", shape=(len(row_used_indices),))

        # Select the correct outloss values based on type
        # Required if-else condition as njit cannot resolve outloss_type inside []
        if outloss_type == "agg_out_loss":
            outloss_vals = self.outloss_mean["agg_out_loss"]
        elif outloss_type == "max_out_loss":
            outloss_vals = self.outloss_mean["max_out_loss"]
        else:
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")

        has_weights, items_start_end, used_period_no = output_mean_damage_ratio(
            items,
            row_used_indices,
            outloss_vals,
            self.period_weights,
            self.max_summary_id,
        )
        unused_periods_to_weights = self.period_weights[~used_period_no]

        if has_weights:
            gen = write_ept_weighted(
                items,
                items_start_end,
                self.sample_size,
                epcalc,
                eptype,
                eptype_tvar,
                unused_periods_to_weights,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id
            )
        else:
            gen = write_ept(
                items,
                items_start_end,
                self.no_of_periods,
                epcalc,
                eptype,
                eptype_tvar,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id

            )

        for data in gen:
            np.savetxt(self.output_files["ept"], data, delimiter=",", fmt=EPT_fmt)

    @profile
    def output_full_uncertainty(self, eptype, eptype_tvar, outloss_type):
        epcalc = FULL

        # Get row indices that are used
        row_used_indices = np.where(self.outloss_sample["row_used"])[0]

        # Allocate storage for the flat data array
        items_fp = Path(self.lec_files_folder, f"lec_full_uncertainty-{outloss_type}-items.bdat")
        items = np.memmap(items_fp, dtype=LOSSVEC2MAP_dtype, mode="w+", shape=(len(row_used_indices),))

        # Select the correct outloss values based on type
        # Required if-else condition as njit cannot resolve outloss_type inside []
        if outloss_type == "agg_out_loss":
            outloss_vals = self.outloss_sample["agg_out_loss"]
        elif outloss_type == "max_out_loss":
            outloss_vals = self.outloss_sample["max_out_loss"]
        else:
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")

        has_weights, items_start_end, used_period_no = output_full_uncertainty(
            items,
            row_used_indices,
            outloss_vals,
            self.period_weights,
            self.max_summary_id,
            self.num_sidxs,
        )
        unused_periods_to_weights = self.period_weights[~used_period_no]

        if has_weights:
            gen = write_ept_weighted(
                items,
                items_start_end,
                1,
                epcalc,
                eptype,
                eptype_tvar,
                unused_periods_to_weights,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id
            )
        else:
            gen = write_ept(
                items,
                items_start_end,
                self.no_of_periods * self.sample_size,
                epcalc,
                eptype,
                eptype_tvar,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id
            )

        for data in gen:
            np.savetxt(self.output_files["ept"], data, delimiter=",", fmt=EPT_fmt)

    @profile
    def output_wheatsheaf_and_wheatsheafmean(self, eptype, eptype_tvar, outloss_type, output_wheatsheaf, output_wheatsheaf_mean):
        epcalc = PERSAMPLEMEAN

        # Get row indices that are used
        row_used_indices = np.where(self.outloss_sample["row_used"])[0]

        wheatsheaf_items_file = Path(self.lec_files_folder, f"lec_wheatsheaf-items-{outloss_type}.bdat")
        wheatsheaf_items = np.memmap(
            wheatsheaf_items_file,
            dtype=WHEATKEYITEMS_dtype,
            mode="w+",
            shape=(len(row_used_indices)),
        )

        # Select the correct outloss values based on type
        # Required if-else condition as njit cannot resolve outloss_type inside []
        if outloss_type == "agg_out_loss":
            outloss_vals = self.outloss_sample["agg_out_loss"]
        elif outloss_type == "max_out_loss":
            outloss_vals = self.outloss_sample["max_out_loss"]
        else:
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")

        has_weights, wheatsheaf_items_start_end, used_period_no = fill_wheatsheaf_items(
            wheatsheaf_items,
            row_used_indices,
            outloss_vals,
            self.period_weights,
            self.max_summary_id,
            self.num_sidxs,
        )
        unused_periods_to_weights = self.period_weights[~used_period_no]

        if has_weights:
            mean_map = None

            if output_wheatsheaf_mean:
                mean_map_file = Path(self.lec_files_folder, f"lec_wheatsheaf_mean-map-{outloss_type}.bdat")
                mean_map = np.memmap(
                    mean_map_file,
                    dtype=MEANMAP_dtype,
                    mode="w+",
                    shape=(self.max_summary_id, len(self.returnperiods)),
                )

            if output_wheatsheaf:
                gen = write_psept_weighted(
                    wheatsheaf_items,
                    wheatsheaf_items_start_end,
                    self.no_of_periods,
                    eptype,
                    eptype_tvar,
                    unused_periods_to_weights,
                    self.use_return_period,
                    self.returnperiods,
                    self.max_summary_id,
                    self.num_sidxs,
                    self.sample_size,
                    mean_map=mean_map,
                )
                for data in gen:
                    np.savetxt(self.output_files["psept"], data, delimiter=",", fmt=PSEPT_fmt)

            if output_wheatsheaf_mean:
                gen = write_wheatsheaf_mean(
                    mean_map,
                    eptype,
                    epcalc,
                    self.max_summary_id,
                )
                for data in gen:
                    np.savetxt(self.output_files["ept"], data, delimiter=",", fmt=EPT_fmt)
        else:
            if output_wheatsheaf:
                gen = write_psept(
                    wheatsheaf_items,
                    wheatsheaf_items_start_end,
                    self.no_of_periods,
                    eptype,
                    eptype_tvar,
                    self.use_return_period,
                    self.returnperiods,
                    self.max_summary_id,
                    self.num_sidxs,
                )
                for data in gen:
                    np.savetxt(self.output_files["psept"], data, delimiter=",", fmt=PSEPT_fmt)

            if not output_wheatsheaf_mean:
                return

            maxcounts = get_wheatsheaf_max_count(
                wheatsheaf_items,
                wheatsheaf_items_start_end,
                self.max_summary_id,
            )

            wheatsheaf_mean_items_file = Path(self.lec_files_folder, f"lec_wheatsheaf_mean-items-{outloss_type}.bdat")
            wheatsheaf_mean_items = np.memmap(
                wheatsheaf_mean_items_file,
                dtype=LOSSVEC2MAP_dtype,
                mode="w+",
                shape=(np.sum(maxcounts[maxcounts != -1])),
            )

            wheatsheaf_mean_items_start_end = fill_wheatsheaf_mean_items(
                wheatsheaf_mean_items,
                wheatsheaf_items,
                wheatsheaf_items_start_end,
                maxcounts,
                self.max_summary_id,
                self.num_sidxs,
            )

            gen = write_ept(
                wheatsheaf_mean_items,
                wheatsheaf_mean_items_start_end,
                self.no_of_periods,
                epcalc,
                eptype,
                eptype_tvar,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id,
                sample_size=self.sample_size
            )

            for data in gen:
                np.savetxt(self.output_files["ept"], data, delimiter=",", fmt=EPT_fmt)

    @profile
    def output_sample_mean(self, eptype, eptype_tvar, outloss_type):
        if self.sample_size == 0:
            logger.warning("aggreports.output_sample_mean, self.sample_size is 0, not outputting any sample mean")
            return
        epcalc = MEANSAMPLE

        # outloss_sample has all SIDXs plus -2 and -3
        reordered_outlosses_file = Path(self.lec_files_folder, f"lec_sample_mean-reordered_outlosses-{outloss_type}.bdat")
        reordered_outlosses = np.memmap(
            reordered_outlosses_file,
            dtype=np.dtype([
                ("row_used", np.bool_),
                ("value", oasis_float),
            ]),
            mode="w+",
            shape=(self.no_of_periods * self.max_summary_id),
        )

        # Select the correct outloss values based on type
        # Required if-else condition as njit cannot resolve outloss_type inside []
        if outloss_type == "agg_out_loss":
            outloss_vals = self.outloss_sample["agg_out_loss"]
        elif outloss_type == "max_out_loss":
            outloss_vals = self.outloss_sample["max_out_loss"]
        else:
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")

        # Get row indices that are used
        row_used_indices = np.where(self.outloss_sample["row_used"])[0]

        # Reorder outlosses by summary_id and period_no
        reorder_losses_by_summary_and_period(
            reordered_outlosses,
            row_used_indices,
            outloss_vals,
            self.max_summary_id,
            self.no_of_periods,
            self.num_sidxs,
            self.sample_size,
        )

        # Get row indices that are used
        row_used_indices = np.where(reordered_outlosses["row_used"])[0]

        # Allocate storage for the flat data array
        items_fp = Path(self.lec_files_folder, f"lec_sample_mean-{outloss_type}-items.bdat")
        items = np.memmap(items_fp, dtype=LOSSVEC2MAP_dtype, mode="w+", shape=(len(row_used_indices),))

        has_weights, items_start_end, used_period_no = output_sample_mean(
            items,
            row_used_indices,
            reordered_outlosses["value"],
            self.period_weights,
            self.max_summary_id,
            self.no_of_periods,
            self.num_sidxs,
        )
        unused_periods_to_weights = self.period_weights[~used_period_no]

        if has_weights:
            gen = write_ept_weighted(
                items,
                items_start_end,
                self.sample_size,
                epcalc,
                eptype,
                eptype_tvar,
                unused_periods_to_weights,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id
            )
        else:
            gen = write_ept(
                items,
                items_start_end,
                self.no_of_periods,
                epcalc,
                eptype,
                eptype_tvar,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id
            )

        for data in gen:
            np.savetxt(self.output_files["ept"], data, delimiter=",", fmt=EPT_fmt)


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


@nb.njit(cache=True, error_model="numpy")
def get_loss(
    next_retperiod,
    last_retperiod,
    last_loss,
    curr_retperiod,
    curr_loss
):
    if curr_retperiod == 0:
        return 0
    if curr_loss == 0:
        return 0
    if curr_retperiod == next_retperiod:
        return curr_loss
    if curr_retperiod < next_retperiod:
        # Linear Interpolate
        return (
            (next_retperiod - curr_retperiod) * (last_loss - curr_loss) /
            (last_retperiod - curr_retperiod) + curr_loss
        )
    return -1


@nb.njit(cache=True, error_model="numpy")
def fill_tvar(
    tail,
    tail_sizes,
    summary_id,
    next_retperiod,
    tvar
):
    if summary_id not in tail:
        tail[summary_id] = create_empty_array(TAIL_valtype)
        tail_sizes[summary_id] = 0
    tail_arr = tail[summary_id]
    tail_arr = resize_array(tail_arr, tail_sizes[summary_id])
    tail_current_size = tail_sizes[summary_id]
    tail_arr[tail_current_size]["retperiod"] = next_retperiod
    tail_arr[tail_current_size]["tvar"] = tvar
    tail[summary_id] = tail_arr
    tail_sizes[summary_id] += 1

    return tail, tail_sizes


@nb.njit(cache=True, error_model="numpy")
def fill_tvar_wheatsheaf(
    tail,
    tail_sizes,
    summary_id,
    sidx,
    num_sidxs,
    next_retperiod,
    tvar
):
    idx = get_wheatsheaf_items_idx(summary_id, sidx, num_sidxs)
    if idx not in tail:
        tail[idx] = create_empty_array(TAIL_valtype)
        tail_sizes[idx] = 0
    tail_arr = tail[idx]
    tail_arr = resize_array(tail_arr, tail_sizes[idx])
    tail_current_size = tail_sizes[idx]
    tail_arr[tail_current_size]["retperiod"] = next_retperiod
    tail_arr[tail_current_size]["tvar"] = tvar
    tail[idx] = tail_arr
    tail_sizes[idx] += 1

    return tail, tail_sizes


@nb.njit(cache=True, error_model="numpy")
def write_return_period_out(
    next_returnperiod_idx,
    last_computed_rp,
    last_computed_loss,
    curr_retperiod,
    curr_loss,
    summary_id,
    eptype,
    epcalc,
    max_retperiod,
    counter,
    tvar,
    tail,
    tail_sizes,
    returnperiods,
    mean_map=None,
    is_wheatsheaf=False,
    num_sidxs=-1,
):
    next_retperiod = 0
    rets = []
    while True:
        if next_returnperiod_idx >= len(returnperiods):
            return rets, tail, tail_sizes, next_returnperiod_idx, last_computed_rp, last_computed_loss

        next_retperiod = returnperiods[next_returnperiod_idx]

        if curr_retperiod > next_retperiod:
            break

        if max_retperiod < next_retperiod:
            next_returnperiod_idx += 1
            continue

        loss = get_loss(
            next_retperiod,
            last_computed_rp,
            last_computed_loss,
            curr_retperiod,
            curr_loss
        )

        rets.append((summary_id, epcalc, eptype, next_retperiod, loss))

        if mean_map is not None:
            mean_map[summary_id - 1][next_returnperiod_idx]["retperiod"] = next_retperiod
            mean_map[summary_id - 1][next_returnperiod_idx]["mean"] += loss
            mean_map[summary_id - 1][next_returnperiod_idx]["count"] += 1

        if curr_retperiod != 0:
            tvar = tvar - ((tvar - loss) / counter)
            if is_wheatsheaf:
                tail, tail_sizes = fill_tvar_wheatsheaf(
                    tail,
                    tail_sizes,
                    summary_id,
                    epcalc,
                    num_sidxs,
                    next_retperiod,
                    tvar,
                )
            else:
                tail, tail_sizes = fill_tvar(
                    tail,
                    tail_sizes,
                    summary_id,
                    next_retperiod,
                    tvar,
                )

        next_returnperiod_idx += 1
        counter += 1

        if curr_retperiod > next_retperiod:
            break

    if curr_retperiod > 0:
        last_computed_rp = curr_retperiod
        last_computed_loss = curr_loss
    return rets, tail, tail_sizes, next_returnperiod_idx, last_computed_rp, last_computed_loss


@nb.njit(cache=True, error_model="numpy")
def write_tvar(
    epcalc,
    eptype_tvar,
    tail,
    tail_sizes,
):
    rets = []
    for summary_id in sorted(tail.keys()):
        vals = tail[summary_id][:tail_sizes[summary_id]]
        for row in vals:
            rets.append((summary_id, epcalc, eptype_tvar, row["retperiod"], row["tvar"]))
    return rets


@nb.njit(cache=True, error_model="numpy")
def write_tvar_wheatsheaf(
    num_sidxs,
    eptype_tvar,
    tail,
    tail_sizes,
):
    rets = []
    for idx in sorted(tail.keys()):
        sidx, summary_id = get_wheatsheaf_items_idx_data(idx, num_sidxs)
        vals = tail[idx][:tail_sizes[idx]]
        for row in vals:
            rets.append((summary_id, sidx, eptype_tvar, row["retperiod"], row["tvar"]))
    return rets


@nb.njit(cache=True, error_model="numpy")
def write_ept(
    items,
    items_start_end,
    max_retperiod,
    epcalc,
    eptype,
    eptype_tvar,
    use_return_period,
    returnperiods,
    max_summary_id,
    sample_size=1
):
    buffer = np.zeros(1000000, dtype=EPT_dtype)
    bidx = 0

    if len(items) == 0 or sample_size == 0:
        return

    tail = nb.typed.Dict.empty(nb_oasis_int, nb_TAIL_valtype)
    tail_sizes = nb.typed.Dict.empty(nb_oasis_int, nb.types.int64)

    for summary_id in range(1, max_summary_id + 1):
        start, end = items_start_end[summary_id - 1]
        if start == -1:
            continue
        filtered_items = items[start:end]
        sorted_idxs = np.argsort(filtered_items["value"])[::-1]
        sorted_items = filtered_items[sorted_idxs]
        next_returnperiod_idx = 0
        last_computed_rp = 0
        last_computed_loss = 0
        tvar = 0
        i = 1
        for item in sorted_items:
            value = item["value"] / sample_size
            retperiod = max_retperiod / i

            if use_return_period:
                if next_returnperiod_idx < len(returnperiods):
                    rets, tail, tail_sizes, next_returnperiod_idx, last_computed_rp, last_computed_loss = write_return_period_out(
                        next_returnperiod_idx,
                        last_computed_rp,
                        last_computed_loss,
                        retperiod,
                        value,
                        summary_id,
                        eptype,
                        epcalc,
                        max_retperiod,
                        i,
                        tvar,
                        tail,
                        tail_sizes,
                        returnperiods,
                    )
                    for ret in rets:
                        if bidx >= len(buffer):
                            yield buffer[:bidx]
                            bidx = 0
                        buffer[bidx]["SummaryId"] = ret[0]
                        buffer[bidx]["EPCalc"] = ret[1]
                        buffer[bidx]["EPType"] = ret[2]
                        buffer[bidx]["ReturnPeriod"] = ret[3]
                        buffer[bidx]["Loss"] = ret[4]
                        bidx += 1
                tvar = tvar - ((tvar - value) / i)
            else:
                tvar = tvar - ((tvar - value) / i)

                if summary_id not in tail:
                    tail[summary_id] = create_empty_array(TAIL_valtype)
                    tail_sizes[summary_id] = 0
                tail_arr = tail[summary_id]
                tail_arr = resize_array(tail_arr, tail_sizes[summary_id])
                tail_current_size = tail_sizes[summary_id]
                tail_arr[tail_current_size]["retperiod"] = retperiod
                tail_arr[tail_current_size]["tvar"] = tvar
                tail[summary_id] = tail_arr
                tail_sizes[summary_id] += 1

                if bidx >= len(buffer):
                    yield buffer[:bidx]
                    bidx = 0
                buffer[bidx]["SummaryId"] = summary_id
                buffer[bidx]["EPCalc"] = epcalc
                buffer[bidx]["EPType"] = eptype
                buffer[bidx]["ReturnPeriod"] = retperiod
                buffer[bidx]["Loss"] = value
                bidx += 1

            i += 1

        if use_return_period:
            while True:
                retperiod = max_retperiod / i
                if returnperiods[next_returnperiod_idx] <= 0:
                    retperiod = 0
                if next_returnperiod_idx < len(returnperiods):
                    rets, tail, tail_sizes, next_returnperiod_idx, last_computed_rp, last_computed_loss = write_return_period_out(
                        next_returnperiod_idx,
                        last_computed_rp,
                        last_computed_loss,
                        retperiod,
                        0,
                        summary_id,
                        eptype,
                        epcalc,
                        max_retperiod,
                        i,
                        tvar,
                        tail,
                        tail_sizes,
                        returnperiods,
                    )
                    for ret in rets:
                        if bidx >= len(buffer):
                            yield buffer[:bidx]
                            bidx = 0
                        buffer[bidx]["SummaryId"] = ret[0]
                        buffer[bidx]["EPCalc"] = ret[1]
                        buffer[bidx]["EPType"] = ret[2]
                        buffer[bidx]["ReturnPeriod"] = ret[3]
                        buffer[bidx]["Loss"] = ret[4]
                        bidx += 1
                tvar = tvar - (tvar / i)
                i += 1
                if next_returnperiod_idx >= len(returnperiods):
                    break

    rets = write_tvar(
        epcalc,
        eptype_tvar,
        tail,
        tail_sizes,
    )
    for ret in rets:
        if bidx >= len(buffer):
            yield buffer[:bidx]
            bidx = 0
        buffer[bidx]["SummaryId"] = ret[0]
        buffer[bidx]["EPCalc"] = ret[1]
        buffer[bidx]["EPType"] = ret[2]
        buffer[bidx]["ReturnPeriod"] = ret[3]
        buffer[bidx]["Loss"] = ret[4]
        bidx += 1
    yield buffer[:bidx]


@nb.njit(cache=True, error_model="numpy")
def write_ept_weighted(
    items,
    items_start_end,
    cum_weight_constant,
    epcalc,
    eptype,
    eptype_tvar,
    unused_periods_to_weights,
    use_return_period,
    returnperiods,
    max_summary_id,
    sample_size=1
):
    buffer = np.zeros(1000000, dtype=EPT_dtype)
    bidx = 0

    if len(items) == 0 or sample_size == 0:
        return

    tail = nb.typed.Dict.empty(nb_oasis_int, nb_TAIL_valtype)
    tail_sizes = nb.typed.Dict.empty(nb_oasis_int, nb.types.int64)

    for summary_id in range(1, max_summary_id + 1):
        start, end = items_start_end[summary_id - 1]
        if start == -1:
            continue
        filtered_items = items[start:end]
        sorted_idxs = np.argsort(filtered_items["value"])[::-1]
        sorted_items = filtered_items[sorted_idxs]
        next_returnperiod_idx = 0
        last_computed_rp = 0
        last_computed_loss = 0
        tvar = 0
        i = 1
        cumulative_weighting = 0
        max_retperiod = 0
        largest_loss = False

        for item in sorted_items:
            value = item["value"] / sample_size
            cumulative_weighting += (item["period_weighting"] * cum_weight_constant)
            retperiod = max_retperiod / i

            if item["period_weighting"]:
                retperiod = 1 / cumulative_weighting

                if not largest_loss:
                    max_retperiod = retperiod + 0.0001  # Add for floating point errors
                    largest_loss = True

                if use_return_period:
                    if next_returnperiod_idx < len(returnperiods):
                        rets, tail, tail_sizes, next_returnperiod_idx, last_computed_rp, last_computed_loss = write_return_period_out(
                            next_returnperiod_idx,
                            last_computed_rp,
                            last_computed_loss,
                            retperiod,
                            value,
                            summary_id,
                            eptype,
                            epcalc,
                            max_retperiod,
                            i,
                            tvar,
                            tail,
                            tail_sizes,
                            returnperiods,
                        )
                        for ret in rets:
                            if bidx >= len(buffer):
                                yield buffer[:bidx]
                                bidx = 0
                            buffer[bidx]["SummaryId"] = ret[0]
                            buffer[bidx]["EPCalc"] = ret[1]
                            buffer[bidx]["EPType"] = ret[2]
                            buffer[bidx]["ReturnPeriod"] = ret[3]
                            buffer[bidx]["Loss"] = ret[4]
                            bidx += 1
                    tvar = tvar - ((tvar - (value)) / i)
                else:
                    tvar = tvar - ((tvar - (value)) / i)

                    if summary_id not in tail:
                        tail[summary_id] = create_empty_array(TAIL_valtype)
                        tail_sizes[summary_id] = 0
                    tail_arr = tail[summary_id]
                    tail_arr = resize_array(tail_arr, tail_sizes[summary_id])
                    tail_current_size = tail_sizes[summary_id]
                    tail_arr[tail_current_size]["retperiod"] = retperiod
                    tail_arr[tail_current_size]["tvar"] = tvar
                    tail[summary_id] = tail_arr
                    tail_sizes[summary_id] += 1

                    if bidx >= len(buffer):
                        yield buffer[:bidx]
                        bidx = 0
                    buffer[bidx]["SummaryId"] = summary_id
                    buffer[bidx]["EPCalc"] = epcalc
                    buffer[bidx]["EPType"] = eptype
                    buffer[bidx]["ReturnPeriod"] = retperiod
                    buffer[bidx]["Loss"] = value
                    bidx += 1

                i += 1
        if use_return_period:
            unused_ptw_idx = 0
            while True:
                retperiod = 0
                if unused_ptw_idx < len(unused_periods_to_weights):
                    cumulative_weighting += (
                        unused_periods_to_weights[unused_ptw_idx]["weighting"] * cum_weight_constant
                    )
                    retperiod = 1 / cumulative_weighting
                    unused_ptw_idx += 1
                if next_returnperiod_idx < len(returnperiods):
                    rets, tail, tail_sizes, next_returnperiod_idx, last_computed_rp, last_computed_loss = write_return_period_out(
                        next_returnperiod_idx,
                        last_computed_rp,
                        last_computed_loss,
                        retperiod,
                        0,
                        summary_id,
                        eptype,
                        epcalc,
                        max_retperiod,
                        i,
                        tvar,
                        tail,
                        tail_sizes,
                        returnperiods,
                    )
                    for ret in rets:
                        if bidx >= len(buffer):
                            yield buffer[:bidx]
                            bidx = 0
                        buffer[bidx]["SummaryId"] = ret[0]
                        buffer[bidx]["EPCalc"] = ret[1]
                        buffer[bidx]["EPType"] = ret[2]
                        buffer[bidx]["ReturnPeriod"] = ret[3]
                        buffer[bidx]["Loss"] = ret[4]
                        bidx += 1
                tvar = tvar - (tvar / i)
                i += 1
                if next_returnperiod_idx >= len(returnperiods):
                    break

    rets = write_tvar(
        epcalc,
        eptype_tvar,
        tail,
        tail_sizes,
    )
    for ret in rets:
        if bidx >= len(buffer):
            yield buffer[:bidx]
            bidx = 0
        buffer[bidx]["SummaryId"] = ret[0]
        buffer[bidx]["EPCalc"] = ret[1]
        buffer[bidx]["EPType"] = ret[2]
        buffer[bidx]["ReturnPeriod"] = ret[3]
        buffer[bidx]["Loss"] = ret[4]
        bidx += 1
    yield buffer[:bidx]


@nb.njit(cache=True, error_model="numpy")
def write_psept(
    items,
    items_start_end,
    max_retperiod,
    eptype,
    eptype_tvar,
    use_return_period,
    returnperiods,
    max_summary_id,
    num_sidxs
):
    buffer = np.zeros(1000000, dtype=PSEPT_dtype)
    bidx = 0

    if len(items) == 0:
        return

    tail = nb.typed.Dict.empty(nb_oasis_int, nb_TAIL_valtype)
    tail_sizes = nb.typed.Dict.empty(nb_oasis_int, nb.types.int64)

    for idx in range(max_summary_id * num_sidxs):
        start, end = items_start_end[idx]
        if start == -1:
            continue
        sidx, summary_id = get_wheatsheaf_items_idx_data(idx, num_sidxs)
        filtered_items = items[start:end]
        sorted_idxs = np.argsort(filtered_items["value"])[::-1]
        sorted_items = filtered_items[sorted_idxs]
        next_returnperiod_idx = 0
        last_computed_rp = 0
        last_computed_loss = 0
        tvar = 0
        i = 1
        for item in sorted_items:
            value = item["value"]
            retperiod = max_retperiod / i

            if use_return_period:
                if next_returnperiod_idx < len(returnperiods):
                    rets, tail, tail_sizes, next_returnperiod_idx, last_computed_rp, last_computed_loss = write_return_period_out(
                        next_returnperiod_idx,
                        last_computed_rp,
                        last_computed_loss,
                        retperiod,
                        value,
                        summary_id,
                        eptype,
                        sidx,
                        max_retperiod,
                        i,
                        tvar,
                        tail,
                        tail_sizes,
                        returnperiods,
                        is_wheatsheaf=True,
                        num_sidxs=num_sidxs,
                    )
                    for ret in rets:
                        if bidx >= len(buffer):
                            yield buffer[:bidx]
                            bidx = 0
                        buffer[bidx]["SummaryId"] = ret[0]
                        buffer[bidx]["SampleId"] = ret[1]
                        buffer[bidx]["EPType"] = ret[2]
                        buffer[bidx]["ReturnPeriod"] = ret[3]
                        buffer[bidx]["Loss"] = ret[4]
                        bidx += 1
                tvar = tvar - ((tvar - value) / i)
            else:
                tvar = tvar - ((tvar - value) / i)

                if idx not in tail:
                    tail[idx] = create_empty_array(TAIL_valtype)
                    tail_sizes[idx] = 0
                tail_arr = tail[idx]
                tail_arr = resize_array(tail_arr, tail_sizes[idx])
                tail_current_size = tail_sizes[idx]
                tail_arr[tail_current_size]["retperiod"] = retperiod
                tail_arr[tail_current_size]["tvar"] = tvar
                tail[idx] = tail_arr
                tail_sizes[idx] += 1

                if bidx >= len(buffer):
                    yield buffer[:bidx]
                    bidx = 0
                buffer[bidx]["SummaryId"] = summary_id
                buffer[bidx]["SampleId"] = sidx
                buffer[bidx]["EPType"] = eptype
                buffer[bidx]["ReturnPeriod"] = retperiod
                buffer[bidx]["Loss"] = value
                bidx += 1

            i += 1

        if use_return_period:
            while True:
                retperiod = max_retperiod / i
                if returnperiods[next_returnperiod_idx] <= 0:
                    retperiod = 0
                if next_returnperiod_idx < len(returnperiods):
                    rets, tail, tail_sizes, next_returnperiod_idx, last_computed_rp, last_computed_loss = write_return_period_out(
                        next_returnperiod_idx,
                        last_computed_rp,
                        last_computed_loss,
                        retperiod,
                        0,
                        summary_id,
                        eptype,
                        sidx,
                        max_retperiod,
                        i,
                        tvar,
                        tail,
                        tail_sizes,
                        returnperiods,
                        is_wheatsheaf=True,
                        num_sidxs=num_sidxs,
                    )
                    for ret in rets:
                        if bidx >= len(buffer):
                            yield buffer[:bidx]
                            bidx = 0
                        buffer[bidx]["SummaryId"] = ret[0]
                        buffer[bidx]["SampleId"] = ret[1]
                        buffer[bidx]["EPType"] = ret[2]
                        buffer[bidx]["ReturnPeriod"] = ret[3]
                        buffer[bidx]["Loss"] = ret[4]
                        bidx += 1
                tvar = tvar - (tvar / i)
                i += 1
                if next_returnperiod_idx >= len(returnperiods):
                    break

    rets = write_tvar_wheatsheaf(
        num_sidxs,
        eptype_tvar,
        tail,
        tail_sizes,
    )
    for ret in rets:
        if bidx >= len(buffer):
            yield buffer[:bidx]
            bidx = 0
        buffer[bidx]["SummaryId"] = ret[0]
        buffer[bidx]["SampleId"] = ret[1]
        buffer[bidx]["EPType"] = ret[2]
        buffer[bidx]["ReturnPeriod"] = ret[3]
        buffer[bidx]["Loss"] = ret[4]
        bidx += 1
    yield buffer[:bidx]


@nb.njit(cache=True, error_model="numpy")
def write_psept_weighted(
    items,
    items_start_end,
    max_retperiod,
    eptype,
    eptype_tvar,
    unused_periods_to_weights,
    use_return_period,
    returnperiods,
    max_summary_id,
    num_sidxs,
    sample_size,
    mean_map=None
):
    buffer = np.zeros(1000000, dtype=PSEPT_dtype)
    bidx = 0

    if len(items) == 0:
        return

    tail = nb.typed.Dict.empty(nb_oasis_int, nb_TAIL_valtype)
    tail_sizes = nb.typed.Dict.empty(nb_oasis_int, nb.types.int64)

    for idx in range(max_summary_id * num_sidxs):
        start, end = items_start_end[idx]
        if start == -1:
            continue
        sidx, summary_id = get_wheatsheaf_items_idx_data(idx, num_sidxs)
        filtered_items = items[start:end]
        sorted_idxs = np.argsort(filtered_items["value"])[::-1]
        sorted_items = filtered_items[sorted_idxs]
        next_returnperiod_idx = 0
        last_computed_rp = 0
        last_computed_loss = 0
        tvar = 0
        i = 1
        cumulative_weighting = 0
        max_retperiod = 0
        largest_loss = False

        for item in sorted_items:
            value = item["value"]
            cumulative_weighting += (item["period_weighting"] * sample_size)
            retperiod = max_retperiod / i

            if item["period_weighting"]:
                retperiod = 1 / cumulative_weighting

                if not largest_loss:
                    max_retperiod = retperiod + 0.0001  # Add for floating point errors
                    largest_loss = True

                if use_return_period:
                    if next_returnperiod_idx < len(returnperiods):
                        rets, tail, tail_sizes, next_returnperiod_idx, last_computed_rp, last_computed_loss = write_return_period_out(
                            next_returnperiod_idx,
                            last_computed_rp,
                            last_computed_loss,
                            retperiod,
                            value,
                            summary_id,
                            eptype,
                            sidx,
                            max_retperiod,
                            i,
                            tvar,
                            tail,
                            tail_sizes,
                            returnperiods,
                            mean_map=mean_map,
                            is_wheatsheaf=True,
                            num_sidxs=num_sidxs,
                        )
                        for ret in rets:
                            if bidx >= len(buffer):
                                yield buffer[:bidx]
                                bidx = 0
                            buffer[bidx]["SummaryId"] = ret[0]
                            buffer[bidx]["SampleId"] = ret[1]
                            buffer[bidx]["EPType"] = ret[2]
                            buffer[bidx]["ReturnPeriod"] = ret[3]
                            buffer[bidx]["Loss"] = ret[4]
                            bidx += 1
                    tvar = tvar - ((tvar - (value)) / i)
                else:
                    tvar = tvar - ((tvar - (value)) / i)

                    if idx not in tail:
                        tail[idx] = create_empty_array(TAIL_valtype)
                        tail_sizes[idx] = 0
                    tail_arr = tail[idx]
                    tail_arr = resize_array(tail_arr, tail_sizes[idx])
                    tail_current_size = tail_sizes[idx]
                    tail_arr[tail_current_size]["retperiod"] = retperiod
                    tail_arr[tail_current_size]["tvar"] = tvar
                    tail[idx] = tail_arr
                    tail_sizes[idx] += 1

                    if bidx >= len(buffer):
                        yield buffer[:bidx]
                        bidx = 0
                    buffer[bidx]["SummaryId"] = summary_id
                    buffer[bidx]["SampleId"] = sidx
                    buffer[bidx]["EPType"] = eptype
                    buffer[bidx]["ReturnPeriod"] = retperiod
                    buffer[bidx]["Loss"] = value
                    bidx += 1

                i += 1
        if use_return_period:
            unused_ptw_idx = 0
            while True:
                retperiod = 0
                if unused_ptw_idx < len(unused_periods_to_weights):
                    cumulative_weighting += (
                        unused_periods_to_weights[unused_ptw_idx]["weighting"] * sample_size
                    )
                    retperiod = 1 / cumulative_weighting
                    unused_ptw_idx += 1
                if next_returnperiod_idx < len(returnperiods):
                    rets, tail, tail_sizes, next_returnperiod_idx, last_computed_rp, last_computed_loss = write_return_period_out(
                        next_returnperiod_idx,
                        last_computed_rp,
                        last_computed_loss,
                        retperiod,
                        0,
                        summary_id,
                        eptype,
                        sidx,
                        max_retperiod,
                        i,
                        tvar,
                        tail,
                        tail_sizes,
                        returnperiods,
                        mean_map=mean_map,
                        is_wheatsheaf=True,
                        num_sidxs=num_sidxs,
                    )
                    for ret in rets:
                        if bidx >= len(buffer):
                            yield buffer[:bidx]
                            bidx = 0
                        buffer[bidx]["SummaryId"] = ret[0]
                        buffer[bidx]["SampleId"] = ret[1]
                        buffer[bidx]["EPType"] = ret[2]
                        buffer[bidx]["ReturnPeriod"] = ret[3]
                        buffer[bidx]["Loss"] = ret[4]
                        bidx += 1
                tvar = tvar - (tvar / i)
                i += 1
                if next_returnperiod_idx >= len(returnperiods):
                    break

    rets = write_tvar_wheatsheaf(
        num_sidxs,
        eptype_tvar,
        tail,
        tail_sizes,
    )
    for ret in rets:
        if bidx >= len(buffer):
            yield buffer[:bidx]
            bidx = 0
        buffer[bidx]["SummaryId"] = ret[0]
        buffer[bidx]["SampleId"] = ret[1]
        buffer[bidx]["EPType"] = ret[2]
        buffer[bidx]["ReturnPeriod"] = ret[3]
        buffer[bidx]["Loss"] = ret[4]
        bidx += 1
    yield buffer[:bidx]


@nb.njit(cache=True, error_model="numpy")
def write_wheatsheaf_mean(
    mean_map,
    eptype,
    epcalc,
    max_summary_id,
):
    if len(mean_map) == 0:
        return

    buffer = np.zeros(1000000, dtype=EPT_dtype)
    bidx = 0

    for summary_id in range(1, max_summary_id + 1):
        if np.sum(mean_map[summary_id - 1]["count"]) == 0:
            continue
        for mc in mean_map[summary_id - 1]:
            if bidx >= len(buffer):
                yield buffer[:bidx]
                bidx = 0
            buffer[bidx]["SummaryId"] = summary_id
            buffer[bidx]["EPCalc"] = epcalc
            buffer[bidx]["EPType"] = eptype
            buffer[bidx]["ReturnPeriod"] = mc["retperiod"]
            buffer[bidx]["Loss"] = mc["mean"] / max(mc["count"], 1)
            bidx += 1
    yield buffer[:bidx]


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


@nb.njit(cache=True, error_model="numpy")
def output_mean_damage_ratio(
    items,
    row_used_indices,
    outloss_vals,
    period_weights,
    max_summary_id,
):
    # Track start and end indices for each summary_id
    items_start_end = np.full((max_summary_id, 2), -1, dtype=np.int32)

    # Track number of entries per summary_id
    summary_counts = np.zeros(max_summary_id, dtype=np.int32)

    # First pass to count how many times each summary_id appears
    for idx in row_used_indices:
        summary_id, period_no = get_mean_idx_data(idx, max_summary_id)
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
        summary_id, period_no = get_mean_idx_data(idx, max_summary_id)

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

    return is_weighted, items_start_end, used_period_no


@nb.njit(cache=True, error_model="numpy")
def output_full_uncertainty(
    items,
    row_used_indices,
    outloss_vals,
    period_weights,
    max_summary_id,
    num_sidxs,
):
    # Track start and end indices for each summary_id
    items_start_end = np.full((max_summary_id, 2), -1, dtype=np.int32)

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

    return is_weighted, items_start_end, used_period_no


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


@nb.njit(cache=True, error_model="numpy")
def fill_wheatsheaf_items(
    wheatsheaf_items,
    row_used_indices,
    outloss_vals,
    period_weights,
    max_summary_id,
    num_sidxs,
):
    # Track start and end indices for each summary_id
    items_start_end = np.full((max_summary_id * num_sidxs, 2), -1, dtype=np.int32)

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
        wheatsheaf_items[insert_idx]["summary_id"] = summary_id
        wheatsheaf_items[insert_idx]["sidx"] = sidx
        wheatsheaf_items[insert_idx]["value"] = outloss_vals[idx]
        if is_weighted:
            # Fast lookup period_weights as they are numbered 1 to no_of_periods
            wheatsheaf_items[insert_idx]["period_weighting"] = period_weights[period_no - 1]["weighting"]
            wheatsheaf_items[insert_idx]["period_no"] = period_no
            used_period_no[period_no - 1] = True

        summary_sidx_counts[summary_sidx_idx] += 1

    return is_weighted, items_start_end, used_period_no


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


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def get_sample_mean_outlosses_idx(summary_id, period_no, no_of_periods):
    return ((summary_id - 1) * no_of_periods) + (period_no - 1)


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def get_sample_mean_outlosses_idx_data(idx, no_of_periods):
    summary_id = (idx // no_of_periods) + 1
    period_no = (idx % no_of_periods) + 1
    return period_no, summary_id


@nb.njit(cache=True, error_model="numpy")
def reorder_losses_by_summary_and_period(
    reordered_outlosses,
    row_used_indices,
    outloss_vals,
    max_summary_id,
    no_of_periods,
    num_sidxs,
    sample_size,
):
    for idx in row_used_indices:
        summary_id, sidx, period_no = get_sample_idx_data(idx, max_summary_id, num_sidxs)
        outloss_idx = get_sample_mean_outlosses_idx(summary_id, period_no, no_of_periods)
        reordered_outlosses[outloss_idx]["row_used"] = True
        reordered_outlosses[outloss_idx]["value"] += (outloss_vals[idx] / sample_size)


@nb.njit(cache=True, error_model="numpy")
def output_sample_mean(
    items,
    row_used_indices,
    reordered_outlosses,
    period_weights,
    max_summary_id,
    no_of_periods,
    num_sidxs,
):
    # Track start and end indices for each summary_id
    items_start_end = np.full((max_summary_id, 2), -1, dtype=np.int32)

    # Track number of entries per summary_id
    summary_counts = np.zeros(max_summary_id, dtype=np.int32)

    # First pass to count how many times each summary_id appears
    for idx in row_used_indices:
        period_no, summary_id = get_sample_mean_outlosses_idx_data(idx, no_of_periods)
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
        period_no, summary_id = get_sample_mean_outlosses_idx_data(idx, no_of_periods)

        # Compute position in the flat array
        insert_idx = items_start_end[summary_id - 1][0] + summary_counts[summary_id - 1]

        # Store values
        items[insert_idx]["summary_id"] = summary_id
        items[insert_idx]["value"] = reordered_outlosses[idx]
        if is_weighted:
            # Fast lookup period_weights as they are numbered 1 to no_of_periods
            items[insert_idx]["period_weighting"] = period_weights[period_no - 1]["weighting"]
            items[insert_idx]["period_no"] = period_no
            used_period_no[period_no - 1] = True

        summary_counts[summary_id - 1] += 1

    return is_weighted, items_start_end, used_period_no
