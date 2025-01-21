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

LOSSVEC2MAP_dtype = np.dtype([
    ("summary_id", oasis_int),
    ("period_no", np.int32),
    ("period_weighting", np.float64),
    ("value", oasis_float),
])

WRITE_EPT_STATE_dtype = np.dtype([
    ("next_returnperiod_idx", np.int64),
    ("last_computed_rp", np.float64),
    ("last_computed_loss", oasis_float),
])

TAIL_dtype = np.dtype([
    ("summary_id", oasis_int),
    ("retperiod", np.float64),
    ("tvar", oasis_float),
])


class AggReports():
    def __init__(
        self,
        output_files,
        ept_fmt,
        psept_fmt,
        outloss_mean,
        outloss_sample,
        period_weights,
        max_summary_id,
        sample_size,
        no_of_periods,
        num_sidxs,
        use_return_period,
        returnperiods,
    ):
        self.output_files = output_files
        self.ept_fmt = ept_fmt
        self.psept_fmt = psept_fmt
        self.outloss_mean = outloss_mean
        self.outloss_sample = outloss_sample
        self.period_weights = period_weights
        self.max_summary_id = max_summary_id
        self.sample_size = sample_size
        self.no_of_periods = no_of_periods
        self.num_sidxs = num_sidxs
        self.use_return_period = use_return_period
        self.returnperiods = returnperiods

        self.EPT_dtype = np.dtype([(c[0], c[1]) for c in EPT_output])
        self.PSEPT_dtype = np.dtype([(c[0], c[1]) for c in PSEPT_output])

    def output_mean_damage_ratio(self, eptype, eptype_tvar, outloss_type):
        epcalc = MEANDR
        has_weights, items, used_period_no = output_mean_damage_ratio(
            self.outloss_mean,
            outloss_type,
            self.period_weights,
            self.max_summary_id,
            self.no_of_periods,
        )
        mask = ~np.isin(self.period_weights["period_no"], list(set(used_period_no)))
        unused_periods_to_weights = self.period_weights[mask]

        if has_weights:
            gen = write_exceedance_probability_table_weighted(
                items,
                self.sample_size,
                epcalc,
                eptype,
                eptype_tvar,
                unused_periods_to_weights,
                self.use_return_period,
                self.returnperiods
            )
        else:
            gen = write_exceedance_probability_table(
                items,
                self.no_of_periods,
                epcalc,
                eptype,
                eptype_tvar,
                self.use_return_period,
                self.returnperiods
            )
        for v in gen:
            np.savetxt(self.output_files["ept"], [v], delimiter=",", fmt=self.ept_fmt)

    def output_full_uncertainty(self, eptype, eptype_tvar, outloss_type):
        epcalc = FULL
        has_weights, items, used_period_no = output_full_uncertainty(
            self.outloss_sample,
            outloss_type,
            self.period_weights,
            self.max_summary_id,
            self.no_of_periods,
            self.num_sidxs,
        )
        mask = ~np.isin(self.period_weights["period_no"], list(set(used_period_no)))
        unused_periods_to_weights = self.period_weights[mask]

        if has_weights:
            gen = write_exceedance_probability_table_weighted(
                items,
                1,
                epcalc,
                eptype,
                eptype_tvar,
                unused_periods_to_weights,
                self.use_return_period,
                self.returnperiods
            )
        else:
            gen = write_exceedance_probability_table(
                items,
                self.no_of_periods * self.sample_size,
                epcalc,
                eptype,
                eptype_tvar,
                self.use_return_period,
                self.returnperiods
            )
        for v in gen:
            np.savetxt(self.output_files["ept"], [v], delimiter=",", fmt=self.ept_fmt)


@nb.njit(cache=True, error_model="numpy")
def resize_tail(tail, tail_idx):
    if tail_idx >= len(tail):
        new_size = len(tail) * 2
        new_tail = np.zeros(new_size, dtype=TAIL_dtype)
        new_tail[:tail_idx] = tail
        return new_tail
    return tail


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
    tail_idx,
    summary_id,
    next_retperiod,
    tvar
):
    tail = resize_tail(tail, tail_idx)
    tail[tail_idx]["summary_id"] = summary_id
    tail[tail_idx]["retperiod"] = next_retperiod
    tail[tail_idx]["tvar"] = tvar
    tail_idx += 1
    return tail, tail_idx


@nb.njit(cache=True, error_model="numpy")
def write_return_period_out(
    state,
    curr_retperiod,
    curr_loss,
    summary_id,
    eptype,
    epcalc,
    max_retperiod,
    counter,
    tvar,
    tail,
    tail_idx,
    returnperiods,
    mean_map=None,
):
    next_retperiod = 0
    rets = []
    while True:
        if state["next_returnperiod_idx"] >= len(returnperiods):
            return rets, tail, tail_idx

        next_retperiod = returnperiods[state["next_returnperiod_idx"]]

        if curr_retperiod > next_retperiod:
            break

        if max_retperiod < next_retperiod:
            state["next_returnperiod_idx"] += 1
            continue

        loss = get_loss(
            next_retperiod,
            state["last_computed_rp"],
            state["last_computed_loss"],
            curr_retperiod,
            curr_loss
        )

        rets.append((summary_id, epcalc, eptype, next_retperiod, loss))

        if mean_map:
            # TODO: implement mean map case
            pass

        if curr_retperiod != 0:
            tvar = tvar - ((tvar - loss) / counter)
            tail, tail_idx = fill_tvar(
                tail,
                tail_idx,
                summary_id,
                next_retperiod,
                tvar,
            )

        state["next_returnperiod_idx"] += 1
        counter += 1

        if curr_retperiod > next_retperiod:
            break

    if curr_retperiod > 0:
        state["last_computed_rp"] = curr_retperiod
        state["last_computed_loss"] = curr_loss
    return rets, tail, tail_idx


@nb.njit(cache=True, error_model="numpy")
def write_tvar(
    epcalc,
    eptype_tvar,
    tail,
):
    rets = []
    unique_ids = np.unique(tail["summary_id"])
    unique_ids = np.sort(unique_ids)
    for summary_id in unique_ids:
        vals = tail[tail["summary_id"] == summary_id]
        for row in vals:
            rets.append((summary_id, epcalc, eptype_tvar, row["retperiod"], row["tvar"]))
    return rets


@nb.njit(cache=True, error_model="numpy")
def write_exceedance_probability_table(
    items,
    max_retperiod,
    epcalc,
    eptype,
    eptype_tvar,
    use_return_period,
    returnperiods,
    sample_size=1
):
    if len(items) == 0 or sample_size == 0:
        return

    tail = np.zeros(16, dtype=TAIL_dtype)
    tail_idx = 0

    unique_ids = np.unique(items["summary_id"])
    for summary_id in unique_ids:
        values = items[items['summary_id'] == summary_id]['value']
        sorted_values = np.sort(values)[::-1]
        state = np.zeros(1, dtype=WRITE_EPT_STATE_dtype)[0]
        tvar = 0
        i = 1
        for value in sorted_values:
            retperiod = max_retperiod / i

            if use_return_period:
                rets, tail, tail_idx = write_return_period_out(
                    state,
                    retperiod,
                    value / sample_size,
                    summary_id,
                    eptype,
                    epcalc,
                    max_retperiod,
                    i,
                    tvar,
                    tail,
                    tail_idx,
                    returnperiods,
                )
                for ret in rets:
                    yield ret
                tvar = tvar - ((tvar - (value / sample_size)) / i)
            else:
                tvar = tvar - ((tvar - (value / sample_size)) / i)
                tail = resize_tail(tail, tail_idx)
                tail[tail_idx]["summary_id"] = summary_id
                tail[tail_idx]["retperiod"] = retperiod
                tail[tail_idx]["tvar"] = tvar
                tail_idx += 1
                yield summary_id, epcalc, eptype, retperiod, value / sample_size

            i += 1

        if use_return_period:
            while True:
                retperiod = max_retperiod / i
                if returnperiods[state["next_returnperiod_idx"]] <= 0:
                    retperiod = 0
                rets, tail, tail_idx = write_return_period_out(
                    state,
                    retperiod,
                    0,
                    summary_id,
                    eptype,
                    epcalc,
                    max_retperiod,
                    i,
                    tvar,
                    tail,
                    tail_idx,
                    returnperiods,
                )
                for ret in rets:
                    yield ret
                tvar = tvar - (tvar / i)
                i += 1
                if state["next_returnperiod_idx"] >= len(returnperiods):
                    break

    rets = write_tvar(
        epcalc,
        eptype_tvar,
        tail[:tail_idx],
    )
    for ret in rets:
        yield ret


@nb.njit(cache=True, error_model="numpy")
def write_exceedance_probability_table_weighted(
    items,
    cum_weight_constant,
    epcalc,
    eptype,
    eptype_tvar,
    unused_periods_to_weights,
    use_return_period,
    returnperiods,
    sample_size=1
):
    if len(items) == 0 or sample_size == 0:
        return

    tail = np.zeros(16, dtype=TAIL_dtype)
    tail_idx = 0

    unique_ids = np.unique(items["summary_id"])
    for summary_id in unique_ids:
        filtered_items = items[items['summary_id'] == summary_id]
        sorted_idxs = np.argsort(filtered_items["value"])[::-1]
        sorted_items = filtered_items[sorted_idxs]
        state = np.zeros(1, dtype=WRITE_EPT_STATE_dtype)[0]
        tvar = 0
        i = 1
        cumulative_weighting = 0
        max_retperiod = 0
        largest_loss = False

        for item in sorted_items:
            cumulative_weighting += (item["period_weighting"] * cum_weight_constant)
            retperiod = max_retperiod / i

            if item["period_weighting"]:
                retperiod = 1 / cumulative_weighting

                if not largest_loss:
                    max_retperiod = retperiod + 0.0001  # Add for floating point errors
                    largest_loss = True

                if use_return_period:
                    rets, tail, tail_idx = write_return_period_out(
                        state,
                        retperiod,
                        item["value"] / sample_size,
                        summary_id,
                        eptype,
                        epcalc,
                        max_retperiod,
                        i,
                        tvar,
                        tail,
                        tail_idx,
                        returnperiods,
                    )
                    for ret in rets:
                        yield ret
                    tvar = tvar - ((tvar - (item["value"] / sample_size)) / i)
                else:
                    tvar = tvar - ((tvar - (item["value"] / sample_size)) / i)
                    tail = resize_tail(tail, tail_idx)
                    tail[tail_idx]["summary_id"] = summary_id
                    tail[tail_idx]["retperiod"] = retperiod
                    tail[tail_idx]["tvar"] = tvar
                    tail_idx += 1
                    yield summary_id, epcalc, eptype, retperiod, item["value"] / sample_size

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
                rets, tail, tail_idx = write_return_period_out(
                    state,
                    retperiod,
                    0,
                    summary_id,
                    eptype,
                    epcalc,
                    max_retperiod,
                    i,
                    tvar,
                    tail,
                    tail_idx,
                    returnperiods,
                )
                for ret in rets:
                    yield ret
                tvar = tvar - (tvar / i)
                i += 1
                if state["next_returnperiod_idx"] >= len(returnperiods):
                    break

    rets = write_tvar(
        epcalc,
        eptype_tvar,
        tail[:tail_idx],
    )
    for ret in rets:
        yield ret


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
    no_of_periods,
):
    row_used_indices = np.where(outloss_mean["row_used"])[0]
    num_rows = len(row_used_indices)

    items = np.zeros(num_rows, dtype=LOSSVEC2MAP_dtype)
    used_period_no = np.zeros(num_rows, dtype=np.int32)

    # Required if-else condition as njit cannot resolve outloss_type inside []
    if outloss_type == "agg_out_loss":
        outloss_vals = outloss_mean["agg_out_loss"]
    elif outloss_type == "max_out_loss":
        outloss_vals = outloss_mean["max_out_loss"]
    else:
        raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")

    is_weighted = len(period_weights) > 0

    # Fast lookup period_weights
    is_weighted = len(period_weights) > 0
    if is_weighted:
        period_weight_map = np.zeros(no_of_periods + 1, dtype=np.float64)
        for p in range(len(period_weights)):
            period_weight_map[period_weights[p]["period_no"]] = period_weights[p]["weighting"]

    used_count = 0
    i = 0
    for i in range(num_rows):
        idx = row_used_indices[i]
        summary_id, period_no = _get_mean_idx_data(idx, max_summary_id)

        items[i]["summary_id"] = summary_id
        items[i]["value"] = outloss_vals[idx]
        if is_weighted:  # Mean Damage Ratio with weighting
            period_weighting = period_weight_map[period_no]
            items[i]["period_no"] = period_no
            items[i]["period_weighting"] = period_weighting
            used_period_no[used_count] = period_no
            used_count += 1
        i += 1

    return is_weighted, items, used_period_no[:used_count]


@nb.njit(cache=True, error_model="numpy")
def output_full_uncertainty(
    outloss_sample,
    outloss_type,
    period_weights,
    max_summary_id,
    no_of_periods,
    num_sidxs,
):
    row_used_indices = np.where(outloss_sample["row_used"])[0]
    num_rows = len(row_used_indices)

    items = np.zeros(num_rows, dtype=LOSSVEC2MAP_dtype)
    used_period_no = np.zeros(num_rows, dtype=np.int32)

    # Required if-else condition as njit cannot resolve outloss_type inside []
    if outloss_type == "agg_out_loss":
        outloss_vals = outloss_sample["agg_out_loss"]
    elif outloss_type == "max_out_loss":
        outloss_vals = outloss_sample["max_out_loss"]
    else:
        raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")

    # Fast lookup period_weights
    is_weighted = len(period_weights) > 0
    if is_weighted:
        period_weight_map = np.zeros(no_of_periods + 1, dtype=np.float64)
        for p in range(len(period_weights)):
            period_weight_map[period_weights[p]["period_no"]] = period_weights[p]["weighting"]

    used_count = 0
    for i in range(num_rows):
        idx = row_used_indices[i]
        summary_id, sidx, period_no = _get_sample_idx_data(idx, max_summary_id, num_sidxs)

        items[i]["summary_id"] = summary_id
        items[i]["value"] = outloss_vals[idx]
        if is_weighted:  # Full Uncertainty with weighting
            period_weighting = period_weight_map[period_no]
            items[i]["period_no"] = period_no
            items[i]["period_weighting"] = period_weighting
            used_period_no[used_count] = period_no
            used_count += 1

    return is_weighted, items, used_period_no[:used_count]
