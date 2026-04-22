"""
Tests for FootprintParquetDynamic._build_footprint() and the supporting
gulmc functions that handle stochastic/dynamic hazard.
"""
import numpy as np
import pandas as pd
import pytest
from numba.typed import Dict

from oasislmf.pytools.getmodel.footprint import FootprintParquetDynamic
from oasislmf.pytools.getmodel.common import EventDynamic
from oasislmf.pytools.gulmc.manager import process_areaperils_in_footprint
from oasislmf.pytools.gulmc.common import haz_arr_type
from oasislmf.pytools.common.data import areaperil_int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fp():
    """Return a bare FootprintParquetDynamic instance (no file system needed)."""
    return FootprintParquetDynamic.__new__(FootprintParquetDynamic)


def make_event_def(section_id=1, rp_from=10, rp_to=20, interpolation=0.5):
    return pd.DataFrame({
        'event_id': [1],
        'section_id': [section_id],
        'rp_from': [rp_from],
        'rp_to': [rp_to],
        'interpolation': [interpolation],
        'rp': [int(rp_from + interpolation * (rp_to - rp_from))],
    })


def build(haz_case, event_def=None):
    if event_def is None:
        event_def = make_event_def()
    return make_fp()._build_footprint(haz_case, event_def)


# ---------------------------------------------------------------------------
# _build_footprint — deterministic path (no probability column)
# ---------------------------------------------------------------------------

def test_deterministic_returns_one_record_per_areaperil():
    haz = pd.DataFrame({
        'section_id': [1, 1, 1, 1],
        'areaperil_id': [100, 100, 200, 200],
        'return_period': [10, 20, 10, 20],
        'intensity': [4, 8, 6, 10],
    })
    result = build(haz)
    assert len(result) == 2
    assert set(result['areaperil_id']) == {100, 200}


def test_deterministic_interpolates_intensity_correctly():
    # intensity = floor(from + interpolation * (to - from))
    # areaperil 100: floor(4 + 0.5 * (8 - 4)) = floor(6.0) = 6
    haz = pd.DataFrame({
        'section_id': [1, 1],
        'areaperil_id': [100, 100],
        'return_period': [10, 20],
        'intensity': [4, 8],
    })
    result = build(haz)
    assert result[0]['intensity'] == 6


def test_deterministic_probability_is_one():
    haz = pd.DataFrame({
        'section_id': [1, 1],
        'areaperil_id': [100, 100],
        'return_period': [10, 20],
        'intensity': [4, 8],
    })
    result = build(haz)
    assert result[0]['probability'] == pytest.approx(1.0)


def test_deterministic_returns_none_for_empty_footprint():
    haz = pd.DataFrame({
        'section_id': pd.Series([], dtype=int),
        'areaperil_id': pd.Series([], dtype=int),
        'return_period': pd.Series([], dtype=int),
        'intensity': pd.Series([], dtype=int),
    })
    result = build(haz)
    assert result is None


# ---------------------------------------------------------------------------
# _build_footprint — stochastic path (probability column present)
# ---------------------------------------------------------------------------

def _stochastic_haz():
    """Two areaperils with two realisations each.

    areaperil 100 — realisations diverge:
      r0 (lower): from=2, to=4  → floor(2 + 0.5*2) = 3
      r1 (higher): from=6, to=10 → floor(6 + 0.5*4) = 8

    areaperil 200 — realisations converge (same intensity at both RPs):
      r0: from=5, to=8 → floor(5 + 0.5*3) = 6
      r1: from=5, to=8 → same → collapse to one row, probability=1.0
    """
    return pd.DataFrame({
        'section_id': [1, 1, 1, 1, 1, 1, 1, 1],
        'areaperil_id': [100, 100, 100, 100, 200, 200, 200, 200],
        'return_period': [10, 10, 20, 20, 10, 10, 20, 20],
        'intensity': [2, 6, 4, 10, 5, 5, 8, 8],
        'probability': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    })


def test_stochastic_detected_by_probability_column():
    result = build(_stochastic_haz())
    # stochastic path: areaperil 100 → 2 bins, areaperil 200 → 1 bin (converged)
    assert len(result) == 3


def test_stochastic_divergent_areaperil_produces_two_bins():
    result = build(_stochastic_haz())
    ap100 = result[result['areaperil_id'] == 100]
    assert len(ap100) == 2
    assert set(ap100['intensity']) == {3, 8}


def test_stochastic_convergent_areaperil_collapses_to_one_bin():
    result = build(_stochastic_haz())
    ap200 = result[result['areaperil_id'] == 200]
    assert len(ap200) == 1
    assert ap200[0]['intensity'] == 6
    assert ap200[0]['probability'] == pytest.approx(1.0)


def test_stochastic_probabilities_sum_to_one_per_areaperil():
    result = build(_stochastic_haz())
    for ap_id in [100, 200]:
        total = result[result['areaperil_id'] == ap_id]['probability'].sum()
        assert total == pytest.approx(1.0, abs=1e-5)


def test_stochastic_realisation_pairing_by_intensity_rank():
    # Verify correct pairing: lower rank at rp_from pairs with lower rank at rp_to.
    # areaperil 100 r0: from=2, to=4 → floor(2 + 0.5*2) = 3
    # areaperil 100 r1: from=6, to=10 → floor(6 + 0.5*4) = 8
    result = build(_stochastic_haz())
    ap100 = sorted(result[result['areaperil_id'] == 100]['intensity'].tolist())
    assert ap100 == [3, 8]


def test_stochastic_equal_weight_realisations():
    result = build(_stochastic_haz())
    ap100 = result[result['areaperil_id'] == 100]
    for rec in ap100:
        assert rec['probability'] == pytest.approx(0.5)


def test_stochastic_unequal_probabilities():
    haz = pd.DataFrame({
        'section_id': [1, 1, 1, 1],
        'areaperil_id': [100, 100, 100, 100],
        'return_period': [10, 10, 20, 20],
        'intensity': [2, 8, 4, 16],
        'probability': [0.25, 0.75, 0.25, 0.75],
    })
    result = build(haz)
    ap100 = {r['intensity']: r['probability'] for r in result[result['areaperil_id'] == 100]}
    # r0: from=2, to=4 → 3, prob=0.25
    # r1: from=8, to=16 → floor(8+0.5*8)=12, prob=0.75
    assert ap100[3] == pytest.approx(0.25)
    assert ap100[12] == pytest.approx(0.75)


def test_stochastic_missing_rp_from_uses_zero():
    # When a section has no rp_from entry for an areaperil (outer merge),
    # from_intensity fills to 0.
    haz = pd.DataFrame({
        'section_id': [1],
        'areaperil_id': [100],
        'return_period': [20],  # only rp_to present
        'intensity': [8],
        'probability': [1.0],
    })
    result = build(haz)
    # from_intensity=0, to_intensity=8, interp=0.5 → floor(0 + 4) = 4
    assert result is not None
    assert result[0]['intensity'] == 4


# ---------------------------------------------------------------------------
# process_areaperils_in_footprint — return_period stored per bin
# ---------------------------------------------------------------------------

def _make_event_footprint(areaperils_bins):
    """Build an EventDynamic array.

    areaperils_bins: list of (areaperil_id, intensity, probability, return_period)
    rows must be ordered so same areaperil is contiguous.
    """
    rows = []
    for ap, intensity, prob, rp in areaperils_bins:
        rows.append((ap, 0, intensity, prob, rp))
    return np.array(rows, dtype=EventDynamic.dtype)


def _make_present_areaperils(areaperil_ids):
    d = Dict.empty(
        key_type=np.dtype(areaperil_int).type,
        value_type=np.int32,
    )
    for ap in areaperil_ids:
        d[areaperil_int.type(ap)] = np.int32(1)
    return d


def test_process_areaperils_return_period_single_bin():
    fp = _make_event_footprint([
        (100, 5, 1.0, 200),
    ])
    present = _make_present_areaperils([100])
    _, _, _, haz_pdf, haz_arr_ptr, haz_rp = process_areaperils_in_footprint(fp, present, True)

    assert haz_arr_ptr[1] - haz_arr_ptr[0] == 1
    assert haz_rp[0] == 200


def test_process_areaperils_return_period_multi_bin():
    fp = _make_event_footprint([
        (100, 3, 0.5, 50),
        (100, 8, 0.5, 50),
        (200, 6, 1.0, 100),
    ])
    present = _make_present_areaperils([100, 200])
    _, Nhaz, _, haz_pdf, haz_arr_ptr, haz_rp = process_areaperils_in_footprint(fp, present, True)

    assert Nhaz == 2

    # areaperil 100: 2 bins, both with rp=50
    s0, e0 = haz_arr_ptr[0], haz_arr_ptr[1]
    assert e0 - s0 == 2
    assert all(haz_rp[s0:e0] == 50)

    # areaperil 200: 1 bin with rp=100
    s1, e1 = haz_arr_ptr[1], haz_arr_ptr[2]
    assert e1 - s1 == 1
    assert haz_rp[s1] == 100


def test_process_areaperils_return_period_not_populated_for_non_dynamic():
    fp = _make_event_footprint([
        (100, 5, 1.0, 999),
    ])
    present = _make_present_areaperils([100])
    _, _, _, haz_pdf, _, haz_rp = process_areaperils_in_footprint(fp, present, None)
    # non-dynamic: haz_rp is empty and haz_pdf has no return_period field
    assert len(haz_rp) == 0
    assert haz_pdf.dtype == haz_arr_type.dtype


def test_process_areaperils_skips_absent_areaperils():
    fp = _make_event_footprint([
        (100, 5, 1.0, 50),
        (999, 3, 1.0, 50),   # not in present_areaperils
    ])
    present = _make_present_areaperils([100])
    areaperil_ids, Nhaz, _, _, _, _ = process_areaperils_in_footprint(fp, present, True)

    assert Nhaz == 1
    assert list(areaperil_ids) == [100]
