import numpy as np
import pytest

from oasislmf.pytools.gulmc.common import agg_vuln_idx_weight_dtype
from oasislmf.pytools.gulmc.manager import build_vuln_pdf

item_dtype = np.dtype([('areaperil_agg_vuln_idx', np.int32), ('vulnerability_idx', np.int32)])

NDAMAGE_BINS = 4
HAZ_BIN_ID = np.array([1], dtype=np.int64)


def _make_vuln_array(*rows):
    """Build a (n_vuln, NDAMAGE_BINS, 1) vuln_array from one damage-bin row per constituent."""
    vuln_array = np.zeros((len(rows), NDAMAGE_BINS, 1), dtype=np.float32)
    for i, row in enumerate(rows):
        vuln_array[i, :, 0] = row
    return vuln_array


def _make_agg_item(n_sub):
    """Build an item using a single aggregate vulnerability block of n_sub constituents."""
    item = np.zeros(1, dtype=item_dtype)[0]
    item['areaperil_agg_vuln_idx'] = 0
    offsets = np.array([0, n_sub], dtype=np.int64)
    return item, offsets


def _run(vuln_array, weights):
    """Call build_vuln_pdf for a single hazard bin with the given constituent weights."""
    item, offsets = _make_agg_item(len(weights))
    ja_data = np.zeros(len(weights), dtype=agg_vuln_idx_weight_dtype)
    for i, w in enumerate(weights):
        ja_data[i]['vuln_idx'] = i
        ja_data[i]['weight'] = w
    vuln_pdf_empty = np.empty((1, NDAMAGE_BINS), dtype=np.float32)
    return build_vuln_pdf(item, 1, HAZ_BIN_ID, vuln_array, NDAMAGE_BINS, offsets, ja_data, vuln_pdf_empty)


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
