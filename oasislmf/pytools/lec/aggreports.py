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
        self.use_return_period= use_return_period
        self.returnperiods= returnperiods

        self.EPT_dtype = np.dtype([(c[0], c[1]) for c in EPT_output])
        self.PSEPT_dtype = np.dtype([(c[0], c[1]) for c in PSEPT_output])


    def output_mean_damage_ratio(self, eptype, eptype_tvar, outloss_type):
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

        gen = write_exceedance_probability_table(
            items,
            self.no_of_periods,
            epcalc,
            eptype,
            eptype_tvar,
            self.use_return_period,
            self.returnperiods
        )
        print("GEN RESULTS")
        for v in gen:
            np.savetxt(self.output_files["ept"], [v], delimiter=",", fmt=self.ept_fmt)


@nb.njit(cache=True, error_model="numpy")
def get_loss(
    next_retperiod_value,
    last_retperiod,
    last_loss,
    curr_retperiod,
    curr_loss
):
    # TODO: implement
    pass


@nb.njit(cache=True, error_model="numpy")
def fill_tvar(
    tail,
    tail_idx,
    summary_id,
    epcalc,
    next_retperiod_value,
    tvar
):
    # TODO: implement
    # NOTE: tail is updated here, is the current max_tail_size enough?
    pass


@nb.njit(cache=True, error_model="numpy")
def write_return_period_out(
    next_retperiod_idx,
    last_retperiod,
    last_loss,
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
    while True:
        if next_retperiod_idx >= len(returnperiods):
            return next_retperiod_idx, last_retperiod, last_loss, tvar, None
        
        next_retperiod_value = returnperiods[next_retperiod_idx]

        if curr_retperiod > next_retperiod_value:
            break

        if max_retperiod < next_retperiod_value:
            next_retperiod_idx += 1
            continue
        
        loss = get_loss(
            next_retperiod_value,
            last_retperiod,
            last_loss,
            curr_retperiod,
            curr_loss
        )

        yield next_retperiod_idx, last_retperiod, last_loss, tvar, \
            (summary_id, epcalc, eptype, next_retperiod_value, loss)

        if mean_map:
            #TODO: implement mean map case
            pass
        
        if curr_retperiod > 0:
            tvar = tvar - ((tvar - loss) / counter)
            fill_tvar(
                tail,
                tail_idx,
                summary_id,
                epcalc,
                next_retperiod_value,
            )
        
        next_retperiod_idx += 1
        counter += 1

        if curr_retperiod <= next_retperiod_value:
            break
    
    if curr_retperiod > 0:
        last_ret_period = curr_retperiod
        last_loss = curr_loss
    return next_retperiod_idx, last_retperiod, last_loss, tvar, None


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
    
    max_tail_size = len(items)
    tail = np.zeros(max_tail_size, dtype=TAIL_dtype)
    tail_idx = 0

    unique_ids = np.unique(items["summary_id"])
    for summary_id in unique_ids:
        values = items[items['summary_id'] == summary_id]['value']
        sorted_values = np.sort(values)[::-1]
        next_returnperiod_idx = 0
        last_computed_rp = 0
        last_computed_loss = 0
        tvar = 0
        i = 1
        for value in sorted_values:
            retperiod = max_retperiod / i
            
            if use_return_period:
                yield from write_return_period_out(
                    next_returnperiod_idx,
                    last_computed_rp,
                    last_computed_loss,
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
                )
                tvar = tvar - ((tvar - (value / sample_size)) / i)
            else:
                tvar = tvar - ((tvar - (value / sample_size)) / i)
                tail[tail_idx]["summary_id"] = summary_id
                tail[tail_idx]["retperiod"] = retperiod
                tail[tail_idx]["tvar"] = tvar
                yield summary_id, epcalc, eptype, retperiod, value / sample_size

            i += 1
        
        if use_return_period:
            retperiod = max_retperiod / i
            # TODO: dowhile loop
            pass
                


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
    used_period_no = np.zeros(outloss_mean["row_used"].sum(), dtype=np.int32)

    is_weighted = len(period_weights) > 0
    used_count = 0
    i = 0

    for idx in range(len(outloss_mean)):
        if not outloss_mean[idx]["row_used"]:
            continue

        summary_id, period_no = _get_mean_idx_data(idx, max_summary_id)
        
        # Mean Damage Ratio
        items[i]["summary_id"] = summary_id
        # Required if-else condition as njit cannot resolve outloss_type inside []
        if outloss_type == "agg_out_loss":
            items[i]["value"] = outloss_mean[idx]["agg_out_loss"]
        elif outloss_type == "max_out_loss":
            items[i]["value"] = outloss_mean[idx]["max_out_loss"]
        else:
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")
        
        if is_weighted:  # Mean Damage Ratio with weighting
            period_weighting = period_weights[period_weights["period_no"] == period_no][0]["weighting"]
            items[i]["period_no"] = period_no
            items[i]["period_weighting"] = period_weighting
            used_period_no[used_count] = period_no
            used_count += 1
        i += 1
    
    if is_weighted:
        used_period_no = used_period_no[:used_count]
        return True, items, used_period_no
    return False, items, used_period_no
