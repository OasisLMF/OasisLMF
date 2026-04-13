import numba as nb
import numpy as np

from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE
from oasislmf.pytools.lec.data import (EPT_dtype, PSEPT_dtype, TAIL_valtype)
from oasislmf.pytools.lec.utils import (get_wheatsheaf_items_idx, get_wheatsheaf_items_idx_data)


@nb.njit(cache=True, error_model="numpy")
def get_loss(
    next_retperiod,
    last_retperiod,
    last_loss,
    curr_retperiod,
    curr_loss
):
    """Get loss based on current and next return period
    Args:
        next_retperiod (float): Next return period
        last_retperiod (float): Previous return period
        last_loss (float): Previous Loss value
        curr_retperiod (float): Current return period
        curr_loss (float): Current Loss value
    Returns:
        loss (float): Loss Value
    """
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
    tail_offsets,
    summary_id,
    next_retperiod,
    tvar
):
    """Populate the Tail with retperiod and tvar values for summary_id
    Args:
        tail (ndarray[TAIL_valtype]): Flat array of (return period, tvar) values
        tail_sizes (ndarray[int64]): Array of current fill size per summary_id
        tail_offsets (ndarray[int64]): Array of start positions per summary_id in tail
        summary_id (int): Summary ID
        next_retperiod (float): Next Return Period
        tvar (float): Tail Value at Risk
    Returns:
        tail (ndarray[TAIL_valtype]): Flat array of (return period, tvar) values
        tail_sizes (ndarray[int64]): Array of current fill size per summary_id
    """
    idx = summary_id - 1
    insert_pos = tail_offsets[idx] + tail_sizes[idx]
    tail[insert_pos]["retperiod"] = next_retperiod
    tail[insert_pos]["tvar"] = tvar
    tail_sizes[idx] += 1
    return tail, tail_sizes


@nb.njit(cache=True, error_model="numpy")
def fill_tvar_wheatsheaf(
    tail,
    tail_sizes,
    tail_offsets,
    summary_id,
    sidx,
    num_sidxs,
    next_retperiod,
    tvar
):
    """Populate the Tail with retperiod and tvar values for (summary_id, sidx) pair
    Args:
        tail (ndarray[TAIL_valtype]): Flat array of (return period, tvar) values
        tail_sizes (ndarray[int64]): Array of current fill size per (summary_id, sidx) idx
        tail_offsets (ndarray[int64]): Array of start positions per idx in tail
        summary_id (int): Summary ID
        sidx (int): Sample ID
        num_sidxs (int): Number of sidxs to consider
        next_retperiod (float): Next Return Period
        tvar (float): Tail Value at Risk
    Returns:
        tail (ndarray[TAIL_valtype]): Flat array of (return period, tvar) values
        tail_sizes (ndarray[int64]): Array of current fill size per (summary_id, sidx) idx
    """
    idx = get_wheatsheaf_items_idx(summary_id, sidx, num_sidxs)
    insert_pos = tail_offsets[idx] + tail_sizes[idx]
    tail[insert_pos]["retperiod"] = next_retperiod
    tail[insert_pos]["tvar"] = tvar
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
    tail_offsets,
    returnperiods,
    mean_map=None,
    is_wheatsheaf=False,
    num_sidxs=-1,
):
    """Processes return periods and computes losses for a given summary, updating TVaR and mean map if required.
    Args:
        next_returnperiod_idx (int): Index of the next return period to process.
        last_computed_rp (float): Last computed return period
        last_computed_loss (float): Last computed loss value
        curr_retperiod (float): Current return period being processed.
        curr_loss (float): Loss associated with the current return period.
        summary_id (int): Identifier for the current summary.
        eptype (int): Type of exceedance probability (0 = OEP, 1 = AEP).
        epcalc (int): Type of exceedance probability calculation.
        max_retperiod (int): Maximum return period to be used in calculations
        counter (int): Counter used for updating TVaR
        tvar (float): Tail Value at Risk
        tail (ndarray[TAIL_valtype]): Flat array of (return period, tvar) values
        tail_sizes (ndarray[int64]): Array of current fill size per summary/idx
        tail_offsets (ndarray[int64]): Array of start positions per summary/idx in tail
        returnperiods (ndarray[np.int32]): Return Periods array
        mean_map (ndarray[MEANMAP_dtype], optional): An array mapping used for mean loss calculations per Summary ID. Used for EPT output later. Defaults to None.
        is_wheatsheaf (bool, optional): If True, update the wheatsheaf TVaR structure.
        num_sidxs (int, optional): Number of sidxs to consider. Defaults to -1 if not is_wheatsheaf.
    Returns:
        rets (list[EPT_dtype]): Return period and Loss EPT data
        tail (ndarray[TAIL_valtype]): Flat array of (return period, tvar) values
        tail_sizes (ndarray[int64]): Array of current fill size per summary/idx
        last_computed_rp (float): Last computed return period
        last_computed_loss (float): Last computed loss value
    """
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
                    tail_offsets,
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
                    tail_offsets,
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
    tail_offsets,
    max_summary_id,
):
    """Get TVaR values for EPT output from tail
    Args:
        epcalc (int): Type of exceedance probability calculation.
        eptype_tvar (int): Type of Tail Value-at-Risk (TVAR) to calculate (0 = OEP TVAR, 1 = AEP TVAR).
        tail (ndarray[TAIL_valtype]): Flat array of (return period, tvar) values
        tail_sizes (ndarray[int64]): Array of current fill size per summary_id
        tail_offsets (ndarray[int64]): Array of start positions per summary_id in tail
        max_summary_id (int): Maximum summary ID
    Returns:
        rets (ndarray[EPT_dtype]): Return period and Loss EPT data
    """
    rets = np.empty(np.sum(tail_sizes), dtype=EPT_dtype)
    pos = 0
    for i in range(max_summary_id):
        size = tail_sizes[i]
        if size == 0:
            continue
        summary_id = i + 1
        for j in range(size):
            p = tail_offsets[i] + j
            rets[pos]["SummaryId"] = summary_id
            rets[pos]["EPCalc"] = epcalc
            rets[pos]["EPType"] = eptype_tvar
            rets[pos]["ReturnPeriod"] = tail[p]["retperiod"]
            rets[pos]["Loss"] = tail[p]["tvar"]
            pos += 1
    return rets


@nb.njit(cache=True, error_model="numpy")
def write_tvar_wheatsheaf(
    num_sidxs,
    eptype_tvar,
    tail,
    tail_sizes,
    tail_offsets,
    total_idxs,
):
    """Get TVaR values for PSEPT output from tail
    Args:
        num_sidxs (int): Number of sidxs to consider.
        eptype_tvar (int): Type of Tail Value-at-Risk (TVAR) to calculate (0 = OEP TVAR, 1 = AEP TVAR).
        tail (ndarray[TAIL_valtype]): Flat array of (return period, tvar) values
        tail_sizes (ndarray[int64]): Array of current fill size per (summary_id, sidx) idx
        tail_offsets (ndarray[int64]): Array of start positions per idx in tail
        total_idxs (int): Total number of (summary_id, sidx) index entries
    Returns:
        rets (ndarray[PSEPT_dtype]): Return period and Loss PSEPT data
    """
    rets = np.empty(np.sum(tail_sizes), dtype=PSEPT_dtype)
    pos = 0
    for idx in range(total_idxs):
        size = tail_sizes[idx]
        if size == 0:
            continue
        sidx, summary_id = get_wheatsheaf_items_idx_data(idx, num_sidxs)
        for j in range(size):
            p = tail_offsets[idx] + j
            rets[pos]["SummaryId"] = summary_id
            rets[pos]["SampleId"] = sidx
            rets[pos]["EPType"] = eptype_tvar
            rets[pos]["ReturnPeriod"] = tail[p]["retperiod"]
            rets[pos]["Loss"] = tail[p]["tvar"]
            pos += 1
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
    """Generate Loss Exceedance Curve values and Tail Value at Risk values based on items and epcalc/eptype/eptype_tvar

    The loss calculation follows these principles:
    - For Aggregate Loss Exceedance Curves (AEP): The sum of all losses within a period is calculated.
    - For Occurrence Loss Exceedance Curves (OEP): The maximum loss within a period is taken.
    - TVAR (Tail Conditional Expectation): Calculated as the average of losses exceeding a given return period.
    Args:
        items (ndarray[LOSSVEC2MAP_dtype]): Array mapping summary_id to loss value (and period_no/period_weighting where applicable)
        items_start_end (ndarray[np.int32]): An array marking where the start and end idxs are for each summary_id in the items array
        max_retperiod (int): Maximum return period to be used in calculations
        epcalc (int): Specifies the calculation method (mean damage loss, full uncertainty, per sample mean, sample mean).
        eptype (int): Type of exceedance probability (0 = OEP, 1 = AEP).
        eptype_tvar (int): Type of Tail Value-at-Risk (TVAR) to calculate (0 = OEP TVAR, 1 = AEP TVAR).
        use_return_period (bool): Use Return Period file.
        returnperiods (ndarray[np.int32]): Return Periods array
        max_summary_id (int): Maximum summary ID
        sample_size (int, optional): Sample Size. Defaults to 1.
    Yields:
        buffer (ndarray[EPT_dtype]): Buffered chunks of EPT data
    """
    buffer = np.zeros(DEFAULT_BUFFER_SIZE, dtype=EPT_dtype)
    bidx = 0

    if len(items) == 0 or sample_size == 0:
        return

    tail_sizes = np.zeros(max_summary_id, dtype=np.int64)
    tail_offsets = np.zeros(max_summary_id + 1, dtype=np.int64)
    if use_return_period:
        rp_len = len(returnperiods) if len(returnperiods) > 0 else 1
        for s in range(max_summary_id + 1):
            tail_offsets[s] = s * rp_len
        tail = np.zeros(max_summary_id * rp_len, dtype=TAIL_valtype)
    else:
        for s in range(max_summary_id):
            start, end = items_start_end[s]
            if start != -1:
                tail_offsets[s + 1] = tail_offsets[s] + (end - start)
            else:
                tail_offsets[s + 1] = tail_offsets[s]
        total_tail = tail_offsets[max_summary_id]
        tail = np.zeros(total_tail if total_tail > 0 else 1, dtype=TAIL_valtype)

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
                        tail_offsets,
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
                insert_pos = tail_offsets[summary_id - 1] + tail_sizes[summary_id - 1]
                tail[insert_pos]["retperiod"] = retperiod
                tail[insert_pos]["tvar"] = tvar
                tail_sizes[summary_id - 1] += 1

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
                if next_returnperiod_idx >= len(returnperiods):
                    break
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
                        tail_offsets,
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

    rets = write_tvar(
        epcalc,
        eptype_tvar,
        tail,
        tail_sizes,
        tail_offsets,
        max_summary_id,
    )
    for ret in rets:
        if bidx >= len(buffer):
            yield buffer[:bidx]
            bidx = 0
        buffer[bidx]["SummaryId"] = ret["SummaryId"]
        buffer[bidx]["EPCalc"] = ret["EPCalc"]
        buffer[bidx]["EPType"] = ret["EPType"]
        buffer[bidx]["ReturnPeriod"] = ret["ReturnPeriod"]
        buffer[bidx]["Loss"] = ret["Loss"]
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
    unused_period_weights,
    use_return_period,
    returnperiods,
    max_summary_id,
    sample_size=1
):
    """Generate Loss Exceedance Curve values and Tail Value at Risk values based on items and epcalc/eptype/eptype_tvar.

    This function calculates weighted exceedance probability tables using cumulative period weightings (`period_weighting`),
    which impact the calculation of return periods. The weighting allows for more accurate representation of losses when
    event periods have different probabilities or frequencies of occurrence.

    The loss calculation follows these principles:
    - For Aggregate Loss Exceedance Curves (AEP): The sum of all losses within a period is calculated.
    - For Occurrence Loss Exceedance Curves (OEP): The maximum loss within a period is taken.
    - TVAR (Tail Conditional Expectation): Calculated as the average of losses exceeding a given return period.
    Args:
        items (ndarray[LOSSVEC2MAP_dtype]): Array mapping summary_id to loss value (and period_no/period_weighting where applicable)
        items_start_end (ndarray[np.int32]): An array marking where the start and end idxs are for each summary_id in the items array
        cum_weight_constant (float): Constant factor for scaling cumulative period weights.
        epcalc (int): Specifies the calculation method (mean damage loss, full uncertainty, per sample mean, sample mean).
        eptype (int): Type of exceedance probability (0 = OEP, 1 = AEP).
        eptype_tvar (int): Type of Tail Value-at-Risk (TVAR) to calculate (0 = OEP TVAR, 1 = AEP TVAR).
        unused_period_weights (ndarray[float]): Array of unused period weights
        use_return_period (bool): Use Return Period file.
        returnperiods (ndarray[np.int32]): Return Periods array
        max_summary_id (int): Maximum summary ID
        sample_size (int, optional): Sample Size. Defaults to 1.
    Yields:
        buffer (ndarray[EPT_dtype]): Buffered chunks of EPT data
    """
    buffer = np.zeros(DEFAULT_BUFFER_SIZE, dtype=EPT_dtype)
    bidx = 0

    if len(items) == 0 or sample_size == 0:
        return

    tail_sizes = np.zeros(max_summary_id, dtype=np.int64)
    tail_offsets = np.zeros(max_summary_id + 1, dtype=np.int64)
    if use_return_period:
        rp_len = len(returnperiods) if len(returnperiods) > 0 else 1
        for s in range(max_summary_id + 1):
            tail_offsets[s] = s * rp_len
        tail = np.zeros(max_summary_id * rp_len, dtype=TAIL_valtype)
    else:
        for s in range(max_summary_id):
            start, end = items_start_end[s]
            if start != -1:
                tail_offsets[s + 1] = tail_offsets[s] + (end - start)
            else:
                tail_offsets[s + 1] = tail_offsets[s]
        total_tail = tail_offsets[max_summary_id]
        tail = np.zeros(total_tail if total_tail > 0 else 1, dtype=TAIL_valtype)

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
                            tail_offsets,
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
                    insert_pos = tail_offsets[summary_id - 1] + tail_sizes[summary_id - 1]
                    tail[insert_pos]["retperiod"] = retperiod
                    tail[insert_pos]["tvar"] = tvar
                    tail_sizes[summary_id - 1] += 1

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
            unused_pw_idx = 0
            while True:
                if next_returnperiod_idx >= len(returnperiods):
                    break
                retperiod = 0
                if unused_pw_idx < len(unused_period_weights):
                    cumulative_weighting += (
                        unused_period_weights[unused_pw_idx]["weighting"] * cum_weight_constant
                    )
                    retperiod = 1 / cumulative_weighting
                    unused_pw_idx += 1
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
                        tail_offsets,
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

    rets = write_tvar(
        epcalc,
        eptype_tvar,
        tail,
        tail_sizes,
        tail_offsets,
        max_summary_id,
    )
    for ret in rets:
        if bidx >= len(buffer):
            yield buffer[:bidx]
            bidx = 0
        buffer[bidx]["SummaryId"] = ret["SummaryId"]
        buffer[bidx]["EPCalc"] = ret["EPCalc"]
        buffer[bidx]["EPType"] = ret["EPType"]
        buffer[bidx]["ReturnPeriod"] = ret["ReturnPeriod"]
        buffer[bidx]["Loss"] = ret["Loss"]
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
    """Generate Per Sample Exceedance Probability Tables (PSEPT) for each individual sample, producing a separate loss
    exceedance curve for each sample, eptype, eptype_tvar.
    Args:
        items (ndarray[WHEATKEYITEMS_dtype]): Array mapping (summary_id, sidx) to loss value (and period_no/period_weighting where applicable)
        items_start_end (ndarray[np.int32]): An array marking where the start and end idxs are for each (summary_id, sidx) pair in the items array
        max_retperiod (int): Maximum return period to be used in calculations
        eptype (int): Type of exceedance probability (0 = OEP, 1 = AEP).
        eptype_tvar (int): Type of Tail Value-at-Risk (TVAR) to calculate (0 = OEP TVAR, 1 = AEP TVAR).
        use_return_period (bool): Use Return Period file.
        returnperiods (ndarray[np.int32]): Return Periods array
        max_summary_id (int): Maximum summary ID
        num_sidxs (int): Number of sidxs to consider
    Yields:
        buffer (ndarray[PSEPT_dtype]): Buffered chunks of PSEPT data
    """
    buffer = np.zeros(DEFAULT_BUFFER_SIZE, dtype=PSEPT_dtype)
    bidx = 0

    if len(items) == 0:
        return

    total_idxs = max_summary_id * num_sidxs
    tail_sizes = np.zeros(total_idxs, dtype=np.int64)
    tail_offsets = np.zeros(total_idxs + 1, dtype=np.int64)
    if use_return_period:
        rp_len = len(returnperiods) if len(returnperiods) > 0 else 1
        for s in range(total_idxs + 1):
            tail_offsets[s] = s * rp_len
        tail = np.zeros(total_idxs * rp_len, dtype=TAIL_valtype)
    else:
        for s in range(total_idxs):
            start, end = items_start_end[s]
            if start != -1:
                tail_offsets[s + 1] = tail_offsets[s] + (end - start)
            else:
                tail_offsets[s + 1] = tail_offsets[s]
        total_tail = tail_offsets[total_idxs]
        tail = np.zeros(total_tail if total_tail > 0 else 1, dtype=TAIL_valtype)

    for idx in range(total_idxs):
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
                        tail_offsets,
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
                insert_pos = tail_offsets[idx] + tail_sizes[idx]
                tail[insert_pos]["retperiod"] = retperiod
                tail[insert_pos]["tvar"] = tvar
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
                if next_returnperiod_idx >= len(returnperiods):
                    break
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
                        tail_offsets,
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

    rets = write_tvar_wheatsheaf(
        num_sidxs,
        eptype_tvar,
        tail,
        tail_sizes,
        tail_offsets,
        total_idxs,
    )
    for ret in rets:
        if bidx >= len(buffer):
            yield buffer[:bidx]
            bidx = 0
        buffer[bidx]["SummaryId"] = ret["SummaryId"]
        buffer[bidx]["SampleId"] = ret["SampleId"]
        buffer[bidx]["EPType"] = ret["EPType"]
        buffer[bidx]["ReturnPeriod"] = ret["ReturnPeriod"]
        buffer[bidx]["Loss"] = ret["Loss"]
        bidx += 1
    yield buffer[:bidx]


@nb.njit(cache=True, error_model="numpy")
def write_psept_weighted(
    items,
    items_start_end,
    max_retperiod,
    eptype,
    eptype_tvar,
    unused_period_weights,
    use_return_period,
    returnperiods,
    max_summary_id,
    num_sidxs,
    sample_size,
    mean_map=None
):
    """Generate Per Sample Exceedance Probability Tables (PSEPT) for each individual sample, producing a separate loss
    exceedance curve for each sample, eptype, eptype_tvar.
    Args:
        items (ndarray[WHEATKEYITEMS_dtype]): Array mapping (summary_id, sidx) to loss value (and period_no/period_weighting where applicable)
        items_start_end (ndarray[np.int32]): An array marking where the start and end idxs are for each (summary_id, sidx) pair in the items array
        max_retperiod (int): Maximum return period to be used in calculations
        eptype (int): Type of exceedance probability (0 = OEP, 1 = AEP).
        eptype_tvar (int): Type of Tail Value-at-Risk (TVAR) to calculate (0 = OEP TVAR, 1 = AEP TVAR).
        unused_period_weights (ndarray[float]): Array of unused period weights
        use_return_period (bool): Use Return Period file.
        returnperiods (ndarray[np.int32]): Return Periods array
        max_summary_id (int): Maximum summary ID
        num_sidxs (int): Number of sidxs to consider
        sample_size (int): Sample Size. Defaults to 1.
        mean_map (ndarray[MEANMAP_dtype], optional): An array mapping used for mean loss calculations per Summary ID. Used for EPT output later. Defaults to None.
    Yields:
        buffer (ndarray[PSEPT_dtype]): Buffered chunks of PSEPT data
    """
    buffer = np.zeros(DEFAULT_BUFFER_SIZE, dtype=PSEPT_dtype)
    bidx = 0

    if len(items) == 0:
        return

    total_idxs = max_summary_id * num_sidxs
    tail_sizes = np.zeros(total_idxs, dtype=np.int64)
    tail_offsets = np.zeros(total_idxs + 1, dtype=np.int64)
    if use_return_period:
        rp_len = len(returnperiods) if len(returnperiods) > 0 else 1
        for s in range(total_idxs + 1):
            tail_offsets[s] = s * rp_len
        tail = np.zeros(total_idxs * rp_len, dtype=TAIL_valtype)
    else:
        for s in range(total_idxs):
            start, end = items_start_end[s]
            if start != -1:
                tail_offsets[s + 1] = tail_offsets[s] + (end - start)
            else:
                tail_offsets[s + 1] = tail_offsets[s]
        total_tail = tail_offsets[total_idxs]
        tail = np.zeros(total_tail if total_tail > 0 else 1, dtype=TAIL_valtype)

    for idx in range(total_idxs):
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
                            tail_offsets,
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
                    insert_pos = tail_offsets[idx] + tail_sizes[idx]
                    tail[insert_pos]["retperiod"] = retperiod
                    tail[insert_pos]["tvar"] = tvar
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
            unused_pw_idx = 0
            while True:
                if next_returnperiod_idx >= len(returnperiods):
                    break
                retperiod = 0
                if unused_pw_idx < len(unused_period_weights):
                    cumulative_weighting += (
                        unused_period_weights[unused_pw_idx]["weighting"] * sample_size
                    )
                    retperiod = 1 / cumulative_weighting
                    unused_pw_idx += 1
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
                        tail_offsets,
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

    rets = write_tvar_wheatsheaf(
        num_sidxs,
        eptype_tvar,
        tail,
        tail_sizes,
        tail_offsets,
        total_idxs,
    )
    for ret in rets:
        if bidx >= len(buffer):
            yield buffer[:bidx]
            bidx = 0
        buffer[bidx]["SummaryId"] = ret["SummaryId"]
        buffer[bidx]["SampleId"] = ret["SampleId"]
        buffer[bidx]["EPType"] = ret["EPType"]
        buffer[bidx]["ReturnPeriod"] = ret["ReturnPeriod"]
        buffer[bidx]["Loss"] = ret["Loss"]
        bidx += 1
    yield buffer[:bidx]


@nb.njit(cache=True, error_model="numpy")
def write_wheatsheaf_mean(
    mean_map,
    eptype,
    epcalc,
    max_summary_id,
):
    """Generate Wheatsheaf Mean Exceedance Probability Table (EPT) by averaging losses for each return period
    from a precomputed mean map.
    Args:
        mean_map (ndarray[MEANMAP_dtype]): An array mapping used for mean loss calculations per Summary ID.
        epcalc (int): Specifies the calculation method (mean damage loss, full uncertainty, per sample mean, sample mean).
        eptype (int): Type of exceedance probability (0 = OEP, 1 = AEP).
        max_summary_id (int): Maximum summary ID
    Yields:
        buffer (ndarray[EPT_dtype]): Buffered chunks of EPT data
    """
    if len(mean_map) == 0:
        return

    buffer = np.zeros(DEFAULT_BUFFER_SIZE, dtype=EPT_dtype)
    bidx = 0

    for summary_id in range(1, max_summary_id + 1):
        if np.sum(mean_map[summary_id - 1]["count"]) == 0:
            continue
        for mc in mean_map[summary_id - 1]:
            if mc["count"] == 0:
                continue
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
