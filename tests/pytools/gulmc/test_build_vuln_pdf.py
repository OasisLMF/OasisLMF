"""An aggregate vulnerability's damage cdf: its members' cdfs by normalised weight.

The members' cdfs come from the damage cdf table (``build_cdf_table``) and their weights from
``build_vuln_member_blocks``; ``member_cdf_at`` is what sampling and the effective cdf read.
"""
import numpy as np
import pytest

from oasislmf.pytools.gulmc.common import agg_vuln_idx_weight_dtype
from oasislmf.pytools.gulmc.items import build_vuln_member_blocks, mixed_cdf_group_dtype
from oasislmf.pytools.gulmc.manager import member_cdf_at
from oasislmf.pytools.gulmc.structure import build_cdf_table

NDAMAGE_BINS = 4


def _make_vuln_array(*rows):
    """Build a (n_vuln, NDAMAGE_BINS, 1) vuln_array from one damage-bin row per member."""
    vuln_array = np.zeros((len(rows), NDAMAGE_BINS, 1), dtype=np.float32)
    for i, row in enumerate(rows):
        vuln_array[i, :, 0] = row
    return vuln_array


def _aggregate_cdf(vuln_array, weights, intensity_bin=0):
    """The damage cdf of one aggregate vulnerability whose members are every row of vuln_array."""
    n_vulns = vuln_array.shape[0]
    agg_data = np.zeros(len(weights), dtype=agg_vuln_idx_weight_dtype)
    agg_data['vuln_idx'] = np.arange(len(weights))
    agg_data['weight'] = weights
    groups = np.zeros(1, dtype=mixed_cdf_group_dtype)
    offsets, rows, member_weights, blocks = build_vuln_member_blocks(
        n_vulns, np.full(n_vulns, -1, dtype=np.int64), np.array([0, len(weights)]), agg_data,
        np.zeros(1, dtype=np.int64), groups, np.array([0, 1]))
    cdf, _ = build_cdf_table(vuln_array)
    return np.array([member_cdf_at(cdf, offsets, rows, member_weights, blocks[0], intensity_bin, d)
                     for d in range(NDAMAGE_BINS)])


def test_weighted_sums_to_one_with_undefined_member():
    """A weighted member with no data at this hazard bin counts as no loss."""
    cdf = _aggregate_cdf(_make_vuln_array([0.5, 0.5, 0, 0], [0, 0, 0, 0]), [1.0, 1.0])
    np.testing.assert_allclose(cdf, np.cumsum([0.75, 0.25, 0, 0]))


def test_unweighted_fallback_sums_to_one_with_undefined_member():
    """With no weights given, an undefined member still counts as no loss."""
    cdf = _aggregate_cdf(_make_vuln_array([0.5, 0.5, 0, 0], [0, 0, 0, 0]), [0.0, 0.0])
    np.testing.assert_allclose(cdf, np.cumsum([0.75, 0.25, 0, 0]))


def test_unweighted_fallback_matches_plain_average_when_all_defined():
    """With no weights given and every member defined, it's a plain average."""
    cdf = _aggregate_cdf(_make_vuln_array([0.5, 0.5, 0, 0], [0.2, 0.3, 0.5, 0]), [0.0, 0.0])
    np.testing.assert_allclose(cdf, np.cumsum([0.35, 0.4, 0.25, 0]))


def test_weights_are_normalised_and_zero_weights_left_out():
    """Positive weights are taken over their total; a zero weight takes no part."""
    vuln_array = _make_vuln_array([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0])
    cdf = _aggregate_cdf(vuln_array, [3.0, 1.0, 0.0])
    np.testing.assert_allclose(cdf, np.cumsum([0.75, 0.25, 0, 0]))


@pytest.mark.parametrize("intensity_bin, expected_pdf", [(0, [0.75, 0.25, 0, 0]), (1, [0.6, 0.4, 0, 0])])
def test_unweighted_fallback_handles_undefined_hazard_bins_independently(intensity_bin, expected_pdf):
    """Each hazard bin treats an all-zero vulnerability row as no loss."""
    vuln_array = np.array([
        [[0.5, 0.0], [0.5, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.2], [0.0, 0.8], [0.0, 0.0], [0.0, 0.0]],
    ], dtype=np.float32)
    cdf = _aggregate_cdf(vuln_array, [0.0, 0.0], intensity_bin)
    np.testing.assert_allclose(cdf, np.cumsum(expected_pdf))


def test_an_intensity_bin_with_no_probability_is_no_damage():
    """A vulnerability with nothing at an intensity bin loses nothing there, in every mode.

    Its damage cdf is a point mass on damage bin 0 rather than a cdf that never reaches 1, which
    effective damageability would sample as the top damage bin.
    """
    cdf, lengths = build_cdf_table(_make_vuln_array([0.5, 0.5, 0, 0], [0, 0, 0, 0]))
    np.testing.assert_allclose(cdf[1, 0], [1, 1, 1, 1])
    assert lengths[1, 0] == 1
    np.testing.assert_allclose(cdf[0, 0], [0.5, 1, 1, 1])
    assert lengths[0, 0] == 2
