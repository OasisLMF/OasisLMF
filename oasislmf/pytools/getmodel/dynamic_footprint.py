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

# section merge options, as passed to merge_sections
MERGE_MAX = 0
MERGE_SUM = 1
MERGE_MAX_INDEPENDENT = 2


@nb.njit(cache=True)
def _fill_levels(probability, start, end, level):
    """Cumulative probability of rows start..end-1, normalised, as integer levels ending at LEVEL_SCALE."""
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
def interpolate_brackets(section_id, areaperil_id, member, intensity, probability, interpolation):
    """Interpolate each (section, areaperil) from its rp_from distribution to its rp_to one.

    Rows must be sorted by section_id, areaperil_id, member, intensity. member 0 is the rp_from
    bracket, member 1 the rp_to bracket. A bracket with no row reads as intensity 0 at
    probability 1. interpolation is the event's weight towards rp_to, read on the group's first row.

    Returns:
        tuple: section_id, areaperil_id, intensity (unfloored), probability of each output row.
    """
    n = len(section_id)
    level = np.empty(n, np.int64)
    # a group gives at most one row per row of its two brackets
    out_section = np.empty(n, section_id.dtype)
    out_areaperil = np.empty(n, areaperil_id.dtype)
    out_intensity = np.empty(n, np.float64)
    out_probability = np.empty(n, np.float64)
    k = 0

    start = 0
    while start < n:
        end = start
        while end < n and section_id[end] == section_id[start] and areaperil_id[end] == areaperil_id[start]:
            end += 1
        split = start
        while split < end and member[split] == 0:
            split += 1
        _fill_levels(probability, start, split, level)
        _fill_levels(probability, split, end, level)
        weight = interpolation[start]

        # walk both brackets; i over rp_from rows start..split-1, j over rp_to rows split..end-1
        i, j, previous = start, split, 0
        while True:
            level_from = level[i] if split > start else LEVEL_SCALE
            level_to = level[j] if end > split else LEVEL_SCALE
            current = min(level_from, level_to)
            if current > previous:
                x_from = intensity[i] if split > start else 0.0
                x_to = intensity[j] if end > split else 0.0
                out_section[k] = section_id[start]
                out_areaperil[k] = areaperil_id[start]
                out_intensity[k] = x_from + (x_to - x_from) * weight
                out_probability[k] = (current - previous) / LEVEL_SCALE
                k += 1
                previous = current
            if current == LEVEL_SCALE:
                break
            if level_from == current:
                i += 1
            if level_to == current:
                j += 1
        start = end

    return out_section[:k], out_areaperil[:k], out_intensity[:k], out_probability[:k]


@nb.njit(cache=True)
def merge_sections(areaperil_id, section_id, intensity, probability, return_period, mode):
    """Merge, per areaperil, the distributions of the sections of one event it lies in.

    Rows must be sorted by areaperil_id, section_id, intensity. An areaperil in a single section
    passes through unchanged. Otherwise, by mode:

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
            _fill_levels(probability, i, j, level)
            n_sections += 1
            i = j

        if n_sections == 1:
            for i in range(start, end):
                out_areaperil[k] = areaperil_id[i]
                out_intensity[k] = intensity[i]
                out_probability[k] = probability[i]
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
                for s in range(n_sections):
                    if level[cursor[s]] == current:
                        cursor[s] += 1
        start = end

    return out_areaperil[:k], out_intensity[:k], out_probability[:k], out_return_period[:k]
