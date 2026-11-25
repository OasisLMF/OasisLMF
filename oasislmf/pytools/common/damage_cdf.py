"""Effective damage cdfs, as gulmc and getmodel compute them.

Both engines take an areaperil's damage distribution for an event from the same pieces, so the cdf
gulpy reads from getmodel is the one gulmc samples under effective damageability:

- ``build_cdf_table``: the vulnerability pdfs as damage cdfs, one contiguous row per (vulnerability,
  intensity bin), built once.
- ``pdf_to_cdf``: the areaperil's hazard records, cut where their cumulative probability reaches 1.
- ``convolve_damage_cdf``: the damage cdf rows weighted by that hazard.
- ``add_cdf_row`` and ``add_no_loss_mass``: a blended areaperil's cdf, its live areaperils' cdfs
  weighted by probability and the rest of the mass on damage bin 0.

A vulnerability is read through a member block: ``vuln_member_ja_offsets[block]`` to
``[block + 1]`` index ``vuln_member_ja_rows`` (rows of the table) and ``vuln_member_ja_weights``.
An ordinary vulnerability is the block holding itself with weight 1, an aggregate one its members
with their weights.
"""
import numba as nb
import numpy as np

from oasislmf.pytools.common.data import oasis_float


@nb.njit(cache=True, fastmath=True)
def get_last_non_empty(cdf, bin_i):
    """Remove empty bucket from the end

    Args:
        cdf: cumulative distribution
        bin_i: last valid bin index

    Returns:
        last bin index with an increased in the cdf
    """
    last_prob = cdf[bin_i]
    while bin_i > 0 and cdf[bin_i - 1] == last_prob:
        bin_i -= 1
    return bin_i


@nb.njit(cache=True, fastmath=True)
def pdf_to_cdf(pdf, n_pdf, cdf):
    """Write the cumulative distribution of ``pdf[:n_pdf]`` into ``cdf`` and return its valid length.

    The cdf stops once it reaches 1 (to float32 precision) and drops trailing bins that add no
    probability, so the returned length can be shorter than ``n_pdf``. It returns a length rather
    than a ``cdf[:n]`` slice so the per-item loop creates no array view (see the refcount note in common/event_stream.py).
    """
    cumsum = 0
    i = 0
    while i < n_pdf:
        cumsum += pdf[i]
        cdf[i] = cumsum
        i += 1
        if cumsum >= 0.999999940:
            break
    return get_last_non_empty(cdf, i - 1) + 1


@nb.njit(cache=True)
def build_cdf_table(pdf):
    """Turn damage pdfs into damage cdfs, one row per condition.

    ``pdf[v, d, c]`` is the probability of damage bin ``d`` for vulnerability ``v`` under condition
    ``c``: an intensity bin for a vulnerability, a source damage bin for a conditional one. The table
    is ``cdf[v, c, d]``, the damage axis last so a row is contiguous. A row is the running sum over
    the whole damage axis, so it can be read at any bin; a condition with no probability at all is
    no damage, a point mass on damage bin 0.

    The second array is each row's length as a cdf: up to the first bin that reaches 1 to float
    precision, less the bins it ends flat on. Sampling searches no further.
    """
    n_rows, n_damage_bins, n_conditions = pdf.shape
    cdf = np.empty((n_rows, n_conditions, max(n_damage_bins, 1)), dtype=oasis_float)
    lengths = np.ones((n_rows, n_conditions), dtype=np.int32)
    for v in range(n_rows):
        for c in range(n_conditions):
            total = 0.
            for d in range(n_damage_bins):
                total += pdf[v, d, c]
            if total <= 0.:
                for d in range(n_damage_bins):
                    cdf[v, c, d] = 1.
                continue
            cumsum = 0.
            length = -1
            for d in range(n_damage_bins):
                cumsum += pdf[v, d, c]
                cdf[v, c, d] = cumsum
                if length < 0 and cumsum >= 0.999999940:
                    length = d
            if length < 0:
                length = n_damage_bins - 1
            last_prob = cdf[v, c, length]
            while length > 0 and cdf[v, c, length - 1] == last_prob:
                length -= 1
            lengths[v, c] = length + 1
    return cdf, lengths


@nb.njit(cache=True, fastmath=True, inline='always')
def cdf_length(cdf, n_bins):
    """The length of a cdf held in its first ``n_bins`` entries.

    Up to the first bin that reaches 1 to float precision, less the bins it ends flat on: the rule
    ``pdf_to_cdf`` applies, for a cdf summed over its whole width.
    """
    bin_i = 0
    while bin_i < n_bins - 1 and cdf[bin_i] < 0.999999940:
        bin_i += 1
    return get_last_non_empty(cdf, bin_i) + 1


@nb.njit(cache=True, fastmath=True, inline='always')
def add_member_cdfs(acc, weight, table, vuln_member_ja_offsets, vuln_member_ja_rows, vuln_member_ja_weights, block, condition,
                    n_bins):
    """Add ``weight`` times a member block's damage cdf under one condition into the first ``n_bins`` of ``acc``."""
    for member in range(vuln_member_ja_offsets[block], vuln_member_ja_offsets[block + 1]):
        member_weight = weight * vuln_member_ja_weights[member]
        row = vuln_member_ja_rows[member]
        for damage_bin_i in range(n_bins):
            acc[damage_bin_i] += member_weight * table[row, condition, damage_bin_i]


@nb.njit(cache=True, fastmath=True, inline='always')
def convolve_damage_cdf(acc, table, table_lengths, vuln_member_ja_offsets, vuln_member_ja_rows, vuln_member_ja_weights,
                        block, condition_ids, condition_probs, n_conditions):
    """Write into ``acc`` a member block's damage cdf weighted by a distribution over its conditions; return its length.

    A condition is an intensity bin for a vulnerability (the distribution is the areaperil's hazard)
    or a source damage bin for a conditional one (the source's damage pmf). ``condition_ids`` are
    1-based, as intensity bin ids are. The sum runs only as far as the longest cdf in it: past its
    length every row has reached 1, so the result can end no later. Only that many entries of
    ``acc`` are written.
    """
    n_bins = 1
    for condition_i in range(n_conditions):
        if condition_probs[condition_i] > 0.:
            condition = condition_ids[condition_i] - 1
            for member in range(vuln_member_ja_offsets[block], vuln_member_ja_offsets[block + 1]):
                n_bins = max(n_bins, table_lengths[vuln_member_ja_rows[member], condition])
    for damage_bin_i in range(n_bins):
        acc[damage_bin_i] = 0.
    for condition_i in range(n_conditions):
        probability = condition_probs[condition_i]
        if probability > 0.:
            add_member_cdfs(acc, probability, table, vuln_member_ja_offsets, vuln_member_ja_rows, vuln_member_ja_weights,
                            block, condition_ids[condition_i] - 1, n_bins)
    return cdf_length(acc, n_bins)


@nb.njit(cache=True, fastmath=True, inline='always')
def add_cdf_row(acc, weight, cdf_rows, row, row_bins, n_bins):
    """Add ``weight`` times a stored cdf into the first ``n_bins`` of ``acc``.

    The cdf is ``cdf_rows[row]``, valid for its first ``row_bins`` entries; past them it reads as its
    last value, the 1 it reached.
    """
    for damage_bin_i in range(n_bins):
        acc[damage_bin_i] += weight * cdf_rows[row, min(damage_bin_i, row_bins - 1)]


@nb.njit(cache=True, fastmath=True, inline='always')
def add_no_loss_mass(acc, live_probability, n_bins):
    """Close a blended areaperil's cdf: the mass no live areaperil carries goes on damage bin 0.

    ``acc`` holds its live areaperils' cdfs weighted by probability (``add_cdf_row``), summing to
    ``live_probability``. The rest -- dead areaperils, placements no areaperil covers, a float weight
    shortfall -- loses nothing. **Damage bin 0 must therefore be the zero-loss bin** of the damage
    bin dictionary, which is the convention every Oasis model follows.
    """
    for damage_bin_i in range(n_bins):
        acc[damage_bin_i] += 1. - live_probability
