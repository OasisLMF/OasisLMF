import numpy as np
import pytest

from oasislmf.pytools.gulmc.common import agg_vuln_idx_weight_dtype
from oasislmf.pytools.gulmc.manager import build_vuln_pdf

NDAMAGE_BINS = 4
HAZ_BIN_ID = np.array([1], dtype=np.int64)


def _make_vuln_array(*rows):
    """Build a (n_vuln, NDAMAGE_BINS, 1) vuln_array from one damage-bin row per constituent."""
    vuln_array = np.zeros((len(rows), NDAMAGE_BINS, 1), dtype=np.float32)
    for i, row in enumerate(rows):
        vuln_array[i, :, 0] = row
    return vuln_array


def _make_agg_offsets(n_sub):
    """Offsets of a single aggregate vulnerability block (block 0) of n_sub constituents."""
    return np.array([0, n_sub], dtype=np.int64)


def _run(vuln_array, weights, haz_bin_id=HAZ_BIN_ID):
    """Call build_vuln_pdf with the given constituent weights and hazard bins."""
    offsets = _make_agg_offsets(len(weights))
    ja_data = np.zeros(len(weights), dtype=agg_vuln_idx_weight_dtype)
    for i, w in enumerate(weights):
        ja_data[i]['vuln_idx'] = i
        ja_data[i]['weight'] = w
    vuln_pdf_empty = np.empty((len(haz_bin_id), NDAMAGE_BINS), dtype=np.float32)
    # build_vuln_pdf writes the pdf into rows [:Nhaz_bins] of the buffer rather than returning it.
    # vulnerability_idx 0 is unused: aggregate block 0 is read instead
    build_vuln_pdf(0, 0, len(haz_bin_id), haz_bin_id, vuln_array, NDAMAGE_BINS, offsets, ja_data, vuln_pdf_empty)
    return vuln_pdf_empty[:len(haz_bin_id)]


def test_weighted_sums_to_one_with_undefined_constituent():
    """A weighted constituent with no data at this hazard bin collapses to 100% no-loss."""
    vuln_array = _make_vuln_array([0.5, 0.5, 0, 0], [0, 0, 0, 0])
    pdf = _run(vuln_array, [1.0, 1.0])
    assert pdf.sum() == pytest.approx(1.0)
    assert list(pdf[0]) == pytest.approx([0.75, 0.25, 0, 0])


def test_unweighted_fallback_sums_to_one_with_undefined_constituent():
    """With no weights given, an undefined constituent still collapses to 100% no-loss."""
    vuln_array = _make_vuln_array([0.5, 0.5, 0, 0], [0, 0, 0, 0])
    pdf = _run(vuln_array, [0.0, 0.0])
    assert pdf.sum() == pytest.approx(1.0)
    assert list(pdf[0]) == pytest.approx([0.75, 0.25, 0, 0])


def test_unweighted_fallback_matches_plain_average_when_all_defined():
    """With no weights given and every constituent defined, it's a plain average."""
    vuln_array = _make_vuln_array([0.5, 0.5, 0, 0], [0.2, 0.3, 0.5, 0])
    pdf = _run(vuln_array, [0.0, 0.0])
    assert pdf.sum() == pytest.approx(1.0)
    assert list(pdf[0]) == pytest.approx([0.35, 0.4, 0.25, 0])


def test_unweighted_fallback_handles_undefined_hazard_bins_independently():
    """Each hazard bin treats an all-zero vulnerability row as 100% no-loss."""
    vuln_array = np.array([
        [[0.5, 0.0], [0.5, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.2], [0.0, 0.8], [0.0, 0.0], [0.0, 0.0]],
    ], dtype=np.float32)
    pdf = _run(vuln_array, [0.0, 0.0], np.array([1, 2], dtype=np.int64))
    assert pdf.sum(axis=1) == pytest.approx([1.0, 1.0])
    np.testing.assert_allclose(pdf, [[0.75, 0.25, 0, 0], [0.6, 0.4, 0, 0]])
