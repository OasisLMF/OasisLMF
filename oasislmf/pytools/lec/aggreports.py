from line_profiler import profile
import numba as nb
import numpy as np

from oasislmf.pytools.common.data import oasis_float, oasis_int, nb_oasis_int


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

    @profile
    def output_mean_damage_ratio(self, eptype, eptype_tvar, outloss_type):
        epcalc = MEANDR
        has_weights, items, items_start_end, used_period_no = output_mean_damage_ratio(
            self.outloss_mean,
            outloss_type,
            self.period_weights,
            self.max_summary_id,
        )
        unused_periods_to_weights = self.period_weights[~used_period_no]

        if has_weights:
            gen = write_exceedance_probability_table_weighted(
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
            gen = write_exceedance_probability_table(
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
        has_weights, items, items_start_end, used_period_no = output_full_uncertainty(
            self.outloss_sample,
            outloss_type,
            self.period_weights,
            self.max_summary_id,
            self.num_sidxs,
        )
        unused_periods_to_weights = self.period_weights[~used_period_no]

        if has_weights:
            gen = write_exceedance_probability_table_weighted(
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
            gen = write_exceedance_probability_table(
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

        if mean_map:
            # TODO: implement mean map case
            pass

        if curr_retperiod != 0:
            tvar = tvar - ((tvar - loss) / counter)
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
def write_exceedance_probability_table(
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
def write_exceedance_probability_table_weighted(
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
    # Get row indices that are used
    row_used_indices = np.where(outloss_mean["row_used"])[0]
    num_rows = len(row_used_indices)

    # Allocate storage for the flat data array (max possible size)
    items = np.zeros(num_rows, dtype=LOSSVEC2MAP_dtype)

    # Track start and end indices for each summary_id
    items_start_end = np.full((max_summary_id, 2), -1, dtype=np.int32)

    # Track number of entries per summary_id
    summary_counts = np.zeros(max_summary_id, dtype=np.int32)

    # Select the correct outloss values based on type
    # Required if-else condition as njit cannot resolve outloss_type inside []
    if outloss_type == "agg_out_loss":
        outloss_vals = outloss_mean["agg_out_loss"]
    elif outloss_type == "max_out_loss":
        outloss_vals = outloss_mean["max_out_loss"]
    else:
        raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")

    # First pass to count how many times each summary_id appears
    for i in range(num_rows):
        idx = row_used_indices[i]
        summary_id, period_no = _get_mean_idx_data(idx, max_summary_id)
        summary_counts[summary_id - 1] += 1

    # Compute cumulative start indices
    pos = 0
    for summary_id in range(max_summary_id):
        if summary_counts[summary_id] > 0:
            items_start_end[summary_id][0] = pos  # Start index
            pos += summary_counts[summary_id]
            items_start_end[summary_id][1] = pos  # End index

    # Reset summary counts for inserting data
    summary_counts[:] = 0

    # Track which period_no are used if period_weights exists
    is_weighted = len(period_weights) > 0
    used_period_no = np.zeros(len(period_weights), dtype=np.bool_)

    # Second pass to populate the data array
    for i in range(num_rows):
        idx = row_used_indices[i]
        summary_id, period_no = _get_mean_idx_data(idx, max_summary_id)

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

    return is_weighted, items, items_start_end, used_period_no


@nb.njit(cache=True, error_model="numpy")
def output_full_uncertainty(
    outloss_sample,
    outloss_type,
    period_weights,
    max_summary_id,
    num_sidxs,
):
    # Get row indices that are used
    row_used_indices = np.where(outloss_sample["row_used"])[0]
    num_rows = len(row_used_indices)

    # Allocate storage for the flat data array (max possible size)
    items = np.zeros(num_rows, dtype=LOSSVEC2MAP_dtype)

    # Track start and end indices for each summary_id
    items_start_end = np.full((max_summary_id, 2), -1, dtype=np.int32)

    # Track number of entries per summary_id
    summary_counts = np.zeros(max_summary_id, dtype=np.int32)

    # Select the correct outloss values based on type
    # Required if-else condition as njit cannot resolve outloss_type inside []
    if outloss_type == "agg_out_loss":
        outloss_vals = outloss_sample["agg_out_loss"]
    elif outloss_type == "max_out_loss":
        outloss_vals = outloss_sample["max_out_loss"]
    else:
        raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")

    # First pass to count how many times each summary_id appears
    for i in range(num_rows):
        idx = row_used_indices[i]
        summary_id, sidx, period_no = _get_sample_idx_data(idx, max_summary_id, num_sidxs)
        summary_counts[summary_id - 1] += 1

    # Compute cumulative start indices
    pos = 0
    for summary_id in range(max_summary_id):
        if summary_counts[summary_id] > 0:
            items_start_end[summary_id][0] = pos  # Start index
            pos += summary_counts[summary_id]
            items_start_end[summary_id][1] = pos  # End index

    # Reset summary counts for inserting data
    summary_counts[:] = 0

    # Track which period_no are used if period_weights exists
    is_weighted = len(period_weights) > 0
    used_period_no = np.zeros(len(period_weights), dtype=np.bool_)

    # Second pass to populate the data array
    for i in range(num_rows):
        idx = row_used_indices[i]
        summary_id, sidx, period_no = _get_sample_idx_data(idx, max_summary_id, num_sidxs)

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

    return is_weighted, items, items_start_end, used_period_no
