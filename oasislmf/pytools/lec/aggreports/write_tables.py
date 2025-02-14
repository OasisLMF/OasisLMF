import numba as nb
import numpy as np

from oasislmf.pytools.common.data import nb_oasis_int
from oasislmf.pytools.lec.data import (EPT_dtype, PSEPT_dtype, TAIL_valtype, NB_TAIL_valtype)
from oasislmf.pytools.lec.utils import (create_empty_array, get_wheatsheaf_items_idx, get_wheatsheaf_items_idx_data, resize_array)


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

    tail = nb.typed.Dict.empty(nb_oasis_int, NB_TAIL_valtype)
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
    unused_period_weights,
    use_return_period,
    returnperiods,
    max_summary_id,
    sample_size=1
):
    buffer = np.zeros(1000000, dtype=EPT_dtype)
    bidx = 0

    if len(items) == 0 or sample_size == 0:
        return

    tail = nb.typed.Dict.empty(nb_oasis_int, NB_TAIL_valtype)
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
            unused_pw_idx = 0
            while True:
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

    tail = nb.typed.Dict.empty(nb_oasis_int, NB_TAIL_valtype)
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
    unused_period_weights,
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

    tail = nb.typed.Dict.empty(nb_oasis_int, NB_TAIL_valtype)
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
            unused_pw_idx = 0
            while True:
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
