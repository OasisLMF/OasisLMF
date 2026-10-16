"""Numba kernels building a dynamic footprint's intensity distributions.

A distribution is a run of rows sharing a key, each row an intensity with its probability, sorted
by intensity. Both kernels walk two or more such runs side by side along their cumulative
probability: at every level where one of them steps, an output row is written, holding each run's
intensity at that level and the width of the level interval as its probability. Pairing the
quantiles this way works whatever the number of rows of each run and their probabilities, and
reduces to pairing by rank when those match.

Cumulative probabilities are compared as integers in units of 1 / LEVEL_SCALE, so that 1/3 + 1/3
and 2/3 meet instead of leaving a sliver of probability between them.
"""
import numba as nb
import numpy as np

LEVEL_SCALE = 10 ** 12
# levels of different sections this close are the same level: probabilities such as 1/3 are not
# exact in units of 1 / LEVEL_SCALE, and a section's rows re-ordered by intensity (an interpolation
# weight outside [0, 1] reverses them) accumulate the rounding the other way round
LEVEL_TOLERANCE = 1000

# section merge options, as passed to merge_sections
MERGE_MAX = 0
MERGE_SUM = 1
MERGE_MAX_INDEPENDENT = 2


@nb.njit(cache=True)
def _fill_levels(probability, start, end, level):
    """Cumulative probability of rows start..end-1, normalised, as integer levels ending at LEVEL_SCALE.

    probability may be float probabilities or integer level widths; for widths, which already add up
    to LEVEL_SCALE, the levels are exact.
    """
    total = 0.0
    for i in range(start, end):
        total += probability[i]
    cumulative = 0.0
    for i in range(start, end):
        cumulative += probability[i]
        level[i] = np.int64(np.rint(cumulative / total * LEVEL_SCALE))
    if end > start:
        level[end - 1] = LEVEL_SCALE


@nb.njit(cache=True)
def is_sorted(section_id, areaperil_id, return_period, intensity):
    """True if the rows are sorted by section_id, areaperil_id, return_period, intensity."""
    for i in range(1, len(section_id)):
        if section_id[i] != section_id[i - 1]:
            if section_id[i] < section_id[i - 1]:
                return False
        elif areaperil_id[i] != areaperil_id[i - 1]:
            if areaperil_id[i] < areaperil_id[i - 1]:
                return False
        elif return_period[i] != return_period[i - 1]:
            if return_period[i] < return_period[i - 1]:
                return False
        elif intensity[i] < intensity[i - 1]:
            return False
    return True


@nb.njit(cache=True)
def _key_before(section_a, areaperil_a, section_b, areaperil_b):
    return section_a < section_b or (section_a == section_b and areaperil_a < areaperil_b)


@nb.njit(cache=True)
def _group_end(section_id, areaperil_id, start):
    end = start
    while end < len(section_id) and section_id[end] == section_id[start] and areaperil_id[end] == areaperil_id[start]:
        end += 1
    return end


@nb.njit(cache=True)
def interpolate_brackets(from_section, from_areaperil, from_intensity, from_probability, from_interpolation,
                         to_section, to_areaperil, to_intensity, to_probability, to_interpolation):
    """Interpolate each (section, areaperil) from its rp_from distribution to its rp_to one.

    The two brackets come as two streams of rows, each sorted by section_id, areaperil_id,
    intensity, and are walked together. A (section, areaperil) missing from one bracket reads it as
    intensity 0 at probability 1. interpolation is the event's weight towards rp_to for the row's
    section.

    Returns:
        tuple: section_id, areaperil_id, intensity (unfloored), and the width of each output row's
            level interval in units of 1 / LEVEL_SCALE (its probability, as an exact integer, so that
            merge_sections re-derives the same levels); sorted by section_id, areaperil_id, intensity.
    """
    n_from, n_to = len(from_section), len(to_section)
    from_level = np.empty(n_from, np.int64)
    to_level = np.empty(n_to, np.int64)
    # a group gives at most one row per row of its two brackets
    n = n_from + n_to
    out_section = np.empty(n, from_section.dtype)
    out_areaperil = np.empty(n, from_areaperil.dtype)
    out_intensity = np.empty(n, np.float64)
    out_width = np.empty(n, np.int64)
    k = 0

    f, t = 0, 0
    while f < n_from or t < n_to:
        # the next (section, areaperil) in either stream, and its rows in each
        if t == n_to or (f < n_from and not _key_before(to_section[t], to_areaperil[t], from_section[f], from_areaperil[f])):
            section, areaperil, weight = from_section[f], from_areaperil[f], from_interpolation[f]
        else:
            section, areaperil, weight = to_section[t], to_areaperil[t], to_interpolation[t]
        f_end = _group_end(from_section, from_areaperil, f) if (
            f < n_from and from_section[f] == section and from_areaperil[f] == areaperil) else f
        t_end = _group_end(to_section, to_areaperil, t) if (
            t < n_to and to_section[t] == section and to_areaperil[t] == areaperil) else t
        _fill_levels(from_probability, f, f_end, from_level)
        _fill_levels(to_probability, t, t_end, to_level)

        # walk both brackets by cumulative level; i over rp_from rows, j over rp_to rows
        i, j, previous = f, t, 0
        while True:
            level_from = from_level[i] if f_end > f else LEVEL_SCALE
            level_to = to_level[j] if t_end > t else LEVEL_SCALE
            current = min(level_from, level_to)
            if current > previous:
                x_from = from_intensity[i] if f_end > f else 0.0
                x_to = to_intensity[j] if t_end > t else 0.0
                out_section[k] = section
                out_areaperil[k] = areaperil
                out_intensity[k] = x_from + (x_to - x_from) * weight
                out_width[k] = current - previous
                k += 1
                previous = current
            if current == LEVEL_SCALE:
                break
            if level_from == current:
                i += 1
            if level_to == current:
                j += 1
        f, t = f_end, t_end

    return out_section[:k], out_areaperil[:k], out_intensity[:k], out_width[:k]


@nb.njit(cache=True)
def collapse(areaperil_id, intensity, probability, return_period):
    """Floor the intensities, merge equal ones, normalise, and order each areaperil for gulmc.

    Each areaperil's rows must be together; they are expected in non-decreasing intensity, as both
    kernels above write them, and sorted here if not. Rows of one areaperil with the same floored intensity and return period become
    one; probabilities are normalised per areaperil; each areaperil is written from the largest
    intensity down, the smaller return period first on a tie, as gulmc reads the event's return
    period off an areaperil's first row.

    Returns:
        tuple: areaperil_id, intensity (int32), probability, return_period of each output row.
    """
    n = len(areaperil_id)
    floored = np.floor(intensity).astype(np.int32)
    return_period = return_period.copy()
    probability = probability.copy()
    out_areaperil = np.empty(n, areaperil_id.dtype)
    out_intensity = np.empty(n, np.int32)
    out_probability = np.empty(n, np.float64)
    out_return_period = np.empty(n, return_period.dtype)
    k = 0

    start = 0
    while start < n:
        end = start
        while end < n and areaperil_id[end] == areaperil_id[start]:
            end += 1
        # in non-decreasing intensity already unless an interpolation weight lies outside [0, 1]
        for i in range(start + 1, end):
            if floored[i] < floored[i - 1]:
                order = start + np.argsort(floored[start:end], kind='mergesort')
                floored[start:end] = floored[order]
                probability[start:end] = probability[order]
                return_period[start:end] = return_period[order]
                break
        total = 0.0
        for i in range(start, end):
            total += probability[i]
        first = k
        # runs of one floored intensity, merged per return period, written in reverse afterwards
        i = start
        while i < end:
            j = i
            while j < end and floored[j] == floored[i]:
                j += 1
            run_start = k
            for r in range(i, j):
                found = False
                for q in range(run_start, k):
                    if out_return_period[q] == return_period[r]:
                        out_probability[q] += probability[r]
                        found = True
                        break
                if not found:
                    out_areaperil[k] = areaperil_id[r]
                    out_intensity[k] = floored[r]
                    out_probability[k] = probability[r]
                    out_return_period[k] = return_period[r]
                    k += 1
            # within a run, the larger return period first, so that it ends up last once reversed
            for a in range(run_start + 1, k):
                b = a
                while b > run_start and out_return_period[b - 1] < out_return_period[b]:
                    out_return_period[b - 1], out_return_period[b] = out_return_period[b], out_return_period[b - 1]
                    out_probability[b - 1], out_probability[b] = out_probability[b], out_probability[b - 1]
                    b -= 1
            i = j
        for q in range(first, k):
            out_probability[q] /= total
        # largest intensity first
        out_return_period[first:k] = out_return_period[first:k][::-1].copy()
        out_probability[first:k] = out_probability[first:k][::-1].copy()
        out_intensity[first:k] = out_intensity[first:k][::-1].copy()
        start = end

    return out_areaperil[:k], out_intensity[:k], out_probability[:k], out_return_period[:k]


@nb.njit(cache=True)
def merge_sections(areaperil_id, section_id, intensity, width, return_period, mode):
    """Merge, per areaperil, the distributions of the sections of one event it lies in.

    Rows must be sorted by areaperil_id, section_id, intensity, and carry their probability as the
    integer level width interpolate_brackets writes. An areaperil in a single section passes
    through unchanged. Otherwise, by mode:

    - MERGE_MAX: walk the sections by cumulative level, keep the largest intensity at each level
      (the sections' uncertainties move together);
    - MERGE_SUM: the same walk, keeping the sum of the intensities;
    - MERGE_MAX_INDEPENDENT: P(max <= x) is the product of the sections' P(intensity <= x),
      evaluated at every intensity any section can take.

    An output row keeps the return period of the section giving its intensity, the larger on a
    tie; for a sum, that of the largest contributor.

    Returns:
        tuple: areaperil_id, intensity, probability, return_period of each output row.
    """
    n = len(areaperil_id)
    level = np.empty(n, np.int64)
    out_areaperil = np.empty(n, areaperil_id.dtype)
    out_intensity = np.empty(n, np.float64)
    out_probability = np.empty(n, np.float64)
    out_return_period = np.empty(n, return_period.dtype)
    # per-section cursors, grown when an areaperil has more sections than seen so far
    seg_start = np.empty(4, np.int64)
    seg_end = np.empty(4, np.int64)
    cursor = np.empty(4, np.int64)
    k = 0

    start = 0
    while start < n:
        end = start
        while end < n and areaperil_id[end] == areaperil_id[start]:
            end += 1

        # the sections of this areaperil
        n_sections = 0
        i = start
        while i < end:
            j = i
            while j < end and section_id[j] == section_id[i]:
                j += 1
            if n_sections == len(seg_start):
                seg_start = np.concatenate((seg_start, np.empty(n_sections, np.int64)))
                seg_end = np.concatenate((seg_end, np.empty(n_sections, np.int64)))
                cursor = np.concatenate((cursor, np.empty(n_sections, np.int64)))
            seg_start[n_sections] = i
            seg_end[n_sections] = j
            cursor[n_sections] = i
            _fill_levels(width, i, j, level)
            n_sections += 1
            i = j

        if n_sections == 1:
            for i in range(start, end):
                out_areaperil[k] = areaperil_id[i]
                out_intensity[k] = intensity[i]
                out_probability[k] = width[i] / LEVEL_SCALE
                out_return_period[k] = return_period[i]
                k += 1

        elif mode == MERGE_MAX_INDEPENDENT:
            # visit the intensities any section can take in increasing order; a section's
            # P(intensity <= v) is the level of its last row at or below v
            previous_cdf = 0.0
            while True:
                value = np.inf
                for s in range(n_sections):
                    if cursor[s] < seg_end[s] and intensity[cursor[s]] < value:
                        value = intensity[cursor[s]]
                if value == np.inf:
                    break
                rp = return_period[start]
                first_rp = True
                cdf = 1.0
                for s in range(n_sections):
                    while cursor[s] < seg_end[s] and intensity[cursor[s]] == value:
                        if first_rp or return_period[cursor[s]] > rp:
                            rp = return_period[cursor[s]]
                            first_rp = False
                        cursor[s] += 1
                    cdf *= level[cursor[s] - 1] / LEVEL_SCALE if cursor[s] > seg_start[s] else 0.0
                if cdf - previous_cdf > 1e-15:
                    out_areaperil[k] = areaperil_id[start]
                    out_intensity[k] = value
                    out_probability[k] = cdf - previous_cdf
                    out_return_period[k] = rp
                    k += 1
                previous_cdf = cdf

        else:
            previous = 0
            while True:
                current = LEVEL_SCALE
                for s in range(n_sections):
                    current = min(current, level[cursor[s]])
                if current > previous:
                    # the largest intensity at this level, the larger return period on a tie
                    winner = cursor[0]
                    total = 0.0
                    for s in range(n_sections):
                        row = cursor[s]
                        total += intensity[row]
                        if intensity[row] > intensity[winner] or (
                                intensity[row] == intensity[winner] and return_period[row] > return_period[winner]):
                            winner = row
                    out_areaperil[k] = areaperil_id[start]
                    out_intensity[k] = total if mode == MERGE_SUM else intensity[winner]
                    out_probability[k] = (current - previous) / LEVEL_SCALE
                    out_return_period[k] = return_period[winner]
                    k += 1
                    previous = current
                if current == LEVEL_SCALE:
                    break
                # a section's last level is exactly LEVEL_SCALE, so its cursor never passes its last row
                for s in range(n_sections):
                    if level[cursor[s]] != LEVEL_SCALE and level[cursor[s]] - current <= LEVEL_TOLERANCE:
                        cursor[s] += 1
        start = end

    return out_areaperil[:k], out_intensity[:k], out_probability[:k], out_return_period[:k]
