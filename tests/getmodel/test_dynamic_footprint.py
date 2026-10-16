"""
Tests for FootprintParquetDynamic._build_footprint().
"""
import pandas as pd
import pytest

from oasislmf.pytools.getmodel.common import SECTION_MERGE_DEFAULT
from oasislmf.pytools.getmodel.footprint import FootprintParquetDynamic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fp():
    """Return a bare FootprintParquetDynamic instance (no file system needed)."""
    fp = FootprintParquetDynamic.__new__(FootprintParquetDynamic)
    fp.section_merge = SECTION_MERGE_DEFAULT
    return fp


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
    assert len(result) == 1
    assert result[0]['intensity'] == 6


def test_deterministic_probability_is_one():
    haz = pd.DataFrame({
        'section_id': [1, 1],
        'areaperil_id': [100, 100],
        'return_period': [10, 20],
        'intensity': [4, 8],
    })
    result = build(haz)
    assert len(result) == 1
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
    assert len(result) == 3
    ap100 = result[result['areaperil_id'] == 100]
    assert len(ap100) == 2
    assert set(ap100['intensity']) == {3, 8}


def test_stochastic_convergent_areaperil_collapses_to_one_bin():
    result = build(_stochastic_haz())
    assert len(result) == 3
    ap200 = result[result['areaperil_id'] == 200]
    assert len(ap200) == 1
    assert ap200[0]['intensity'] == 6
    assert ap200[0]['probability'] == pytest.approx(1.0)


def test_stochastic_probabilities_sum_to_one_per_areaperil():
    result = build(_stochastic_haz())
    assert len(result) == 3
    for ap_id in [100, 200]:
        total = result[result['areaperil_id'] == ap_id]['probability'].sum()
        assert total == pytest.approx(1.0, abs=1e-5)


def test_stochastic_realisation_pairing_by_intensity_rank():
    # Verify correct pairing: lower rank at rp_from pairs with lower rank at rp_to.
    # areaperil 100 r0: from=2, to=4 → floor(2 + 0.5*2) = 3
    # areaperil 100 r1: from=6, to=10 → floor(6 + 0.5*4) = 8
    result = build(_stochastic_haz())
    assert len(result) == 3
    ap100 = sorted(result[result['areaperil_id'] == 100]['intensity'].tolist())
    assert ap100 == [3, 8]


def test_stochastic_equal_weight_realisations():
    result = build(_stochastic_haz())
    assert len(result) == 3
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
    assert len(result) == 2
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
    assert len(result) == 1
    assert result[0]['intensity'] == 4


# ---------------------------------------------------------------------------
# _build_footprint — brackets holding different numbers of realisations.
# They are paired by cumulative probability level: the j-th quantile at rp_from
# meets the j-th quantile at rp_to. Expected values are worked by hand.
# ---------------------------------------------------------------------------

def distribution(result, areaperil_id=100):
    return {int(r['intensity']): pytest.approx(float(r['probability']))
            for r in result[result['areaperil_id'] == areaperil_id]}


def test_one_realisation_against_three_interpolates_each_quantile():
    # rp_from: 10 @ 1; rp_to: 16/20/24 @ .25/.5/.25; halfway -> 13/15/17 keeping rp_to's weights
    haz = pd.DataFrame({
        'section_id': 1, 'areaperil_id': 100,
        'return_period': [10, 20, 20, 20],
        'intensity': [10, 16, 20, 24],
        'probability': [1.0, 0.25, 0.5, 0.25],
    })
    assert distribution(build(haz)) == {13: 0.25, 15: 0.5, 17: 0.25}


def test_three_realisations_against_five_with_different_probabilities():
    # rp_from: 14/18/22 @ .25/.5/.25 (levels .25 .75 1); rp_to: 34/37/40/43/46 @ .1/.2/.4/.2/.1
    # (levels .1 .3 .7 .9 1). Union of levels .1 .25 .3 .7 .75 .9 1, and at interpolation .25:
    #   (0,.1]   14->34 = 19     (.1,.25]  14->37 = 19.75 -> 19   (.25,.3] 18->37 = 22.75 -> 22
    #   (.3,.7]  18->40 = 23.5   (.7,.75]  18->43 = 24.25 -> 24   (.75,.9] 22->43 = 27.25 -> 27
    #   (.9,1]   22->46 = 28
    haz = pd.DataFrame({
        'section_id': 1, 'areaperil_id': 100,
        'return_period': [10] * 3 + [20] * 5,
        'intensity': [14, 18, 22, 34, 37, 40, 43, 46],
        'probability': [0.25, 0.5, 0.25, 0.1, 0.2, 0.4, 0.2, 0.1],
    })
    result = build(haz, make_event_def(interpolation=0.25))
    assert distribution(result) == {19: 0.25, 22: 0.05, 23: 0.4, 24: 0.05, 27: 0.15, 28: 0.1}


def test_unweighted_duplicate_rows_in_unequal_numbers_keep_their_bracket_shares():
    # No probability column, rows repeated a different number of times per return period, as
    # some model data has it: rp_from 0/1, rp_to 8 four times. At interpolation 0 the footprint
    # is rp_from's, 0 and 1 at one half each. Pairing by rank used to interpolate the two
    # unmatched rp_to rows from 0 instead, giving 0 at 0.75.
    haz = pd.DataFrame({
        'section_id': 1, 'areaperil_id': 100,
        'return_period': [10, 10, 20, 20, 20, 20],
        'intensity': [0, 1, 8, 8, 8, 8],
    })
    assert distribution(build(haz, make_event_def(interpolation=0.0))) == {0: 0.5, 1: 0.5}


def test_missing_bracket_is_one_atom_at_zero_against_any_number_of_realisations():
    haz = pd.DataFrame({
        'section_id': 1, 'areaperil_id': 100,
        'return_period': [20, 20, 20],
        'intensity': [4, 8, 12],
        'probability': [0.25, 0.5, 0.25],
    })
    assert distribution(build(haz)) == {2: 0.25, 4: 0.5, 6: 0.25}


# ---------------------------------------------------------------------------
# _build_footprint — an areaperil lying in several sections of the event (a
# confluence) is merged by the section_merge option. Expected values by hand.
# ---------------------------------------------------------------------------

def build_merged(haz, sections, section_merge):
    """sections: (section_id, rp_from, rp_to, interpolation) of the one event."""
    event_def = pd.DataFrame(sections, columns=['section_id', 'rp_from', 'rp_to', 'interpolation']).assign(event_id=1)
    fp = make_fp()
    fp.section_merge = section_merge
    return fp._build_footprint(haz, event_def)


def atoms(result, areaperil_id=100):
    """(intensity, probability, return_period) of an areaperil, largest intensity first."""
    rows = result[result['areaperil_id'] == areaperil_id]
    return [(int(r['intensity']), pytest.approx(float(r['probability'])), int(r['return_period'])) for r in rows]


def flat(section_id, areaperil_id, values, probabilities=None, return_periods=(10, 100)):
    """A section whose distribution is the same at both brackets, so interpolation leaves it as is."""
    probabilities = probabilities or [1.0 / len(values)] * len(values)
    return pd.DataFrame([(section_id, areaperil_id, rp, v, p)
                         for rp in return_periods for v, p in zip(values, probabilities)],
                        columns=['section_id', 'areaperil_id', 'return_period', 'intensity', 'probability'])


def test_confluence_takes_the_deeper_section():
    # the confluence regression: section 1 interpolates to 158, section 2 to 36. The deeper governs,
    # deterministically, carrying its own return period (section 1's rp_to, 100)
    haz = pd.DataFrame({'section_id': [1, 1, 2, 2], 'areaperil_id': 100, 'return_period': [10, 100, 10, 50],
                        'intensity': [80, 200, 30, 90]})
    for section_merge in ('max', 'max_independent'):
        result = build_merged(haz, [(1, 10, 100, 0.65), (2, 10, 50, 0.10)], section_merge)
        assert atoms(result) == [(158, 1.0, 100)]


def test_confluence_sum_floors_after_adding():
    # 158.5 + 36.5 = 195; flooring each section first would give 158 + 36 = 194
    haz = pd.DataFrame({'section_id': [1, 1, 2, 2], 'areaperil_id': 100, 'return_period': [10, 100, 10, 100],
                        'intensity': [80, 237, 30, 103]})
    result = build_merged(haz, [(1, 10, 100, 0.5), (2, 10, 100, 0.0890411)], 'sum')
    assert atoms(result) == [(195, 1.0, 100)]


@pytest.mark.parametrize('section_merge, expected', [
    # section 1: 140/160/180, section 2: 120/150/190, each at .25/.5/.25
    ('max', [(190, 0.25, 100), (160, 0.5, 100), (140, 0.25, 100)]),                     # mean 162.50
    ('max_independent', [(190, 0.25, 100), (180, 0.1875, 100), (160, 0.375, 100),
                         (150, 0.125, 100), (140, 0.0625, 100)]),                        # mean 168.75
    ('sum', [(370, 0.25, 100), (310, 0.5, 100), (260, 0.25, 100)]),
])
def test_two_uncertain_sections(section_merge, expected):
    haz = pd.concat([flat(1, 100, [140, 160, 180], [.25, .5, .25]), flat(2, 100, [120, 150, 190], [.25, .5, .25])])
    result = build_merged(haz, [(1, 10, 100, 0.5), (2, 10, 100, 0.5)], section_merge)
    assert atoms(result) == expected


@pytest.mark.parametrize('section_merge, expected', [
    # section 1: 10; section 2: 4/16 at .5 each; section 3: 6/8/12/20 at .25 each.
    # comonotone, by level .25 .5 .75 1:  max (10,10,16,20)   sum (20,22,38,46)
    ('max', [(20, 0.25, 100), (16, 0.25, 100), (10, 0.5, 100)]),
    ('sum', [(46, 0.25, 100), (38, 0.25, 100), (22, 0.25, 100), (20, 0.25, 100)]),
    # independent: P(max <= x) = F1 F2 F3 = .25 at 10, .375 at 12, .75 at 16, 1 at 20
    ('max_independent', [(20, 0.25, 100), (16, 0.375, 100), (12, 0.125, 100), (10, 0.25, 100)]),
])
def test_three_sections_with_different_numbers_of_realisations(section_merge, expected):
    haz = pd.concat([flat(1, 100, [10]), flat(2, 100, [4, 16]), flat(3, 100, [6, 8, 12, 20])])
    result = build_merged(haz, [(s, 10, 100, 0.5) for s in (1, 2, 3)], section_merge)
    assert atoms(result) == expected


def test_each_level_keeps_the_return_period_of_the_section_that_wins_it():
    # section 1 (rp_to 50) wins the two lower levels, section 2 (rp_to 100) the top one;
    # on a tie the larger return period is kept
    haz = pd.concat([flat(1, 100, [140, 160, 180], [.25, .5, .25], return_periods=(10, 50)),
                     flat(2, 100, [120, 150, 190], [.25, .5, .25]),
                     flat(1, 200, [30], return_periods=(10, 50)), flat(2, 200, [30])])
    result = build_merged(haz, [(1, 10, 50, 0.5), (2, 10, 100, 0.5)], 'max')
    assert atoms(result) == [(190, 0.25, 100), (160, 0.5, 50), (140, 0.25, 50)]
    assert atoms(result, 200) == [(30, 1.0, 100)]


@pytest.mark.parametrize('section_merge', ['max', 'max_independent', 'sum'])
def test_an_areaperil_in_one_section_is_the_same_under_every_merge(section_merge):
    # areaperil 300 lies in section 1 only, next to a confluence areaperil 100
    haz = pd.concat([flat(1, 100, [140, 160, 180], [.25, .5, .25]), flat(2, 100, [120, 150, 190], [.25, .5, .25]),
                     pd.DataFrame({'section_id': 1, 'areaperil_id': 300, 'return_period': [10, 10, 100, 100],
                                   'intensity': [4, 8, 12, 20], 'probability': .5})])
    result = build_merged(haz, [(1, 10, 100, 0.25), (2, 10, 100, 0.5)], section_merge)
    # 4->12 and 8->20 a quarter of the way: 6 and 11
    assert atoms(result, 300) == [(11, 0.5, 100), (6, 0.5, 100)]


def test_64_bit_areaperil_ids_are_kept_apart():
    """areaperil_id may be 64 bits (OASIS_AREAPERIL_TYPE=u8). Section 1 / areaperil 5 and section 0 /
    areaperil 2**32 + 5 are different places, which packing the pair into one 64-bit key confused."""
    import numpy as np
    from oasislmf.pytools.getmodel.dynamic_footprint import interpolate_brackets

    section_id = np.array([0, 1], np.int64)
    areaperil_id = np.array([2**32 + 5, 5], np.uint64)
    half = np.full(2, 0.5)
    section_id, areaperil_id, intensity, width = interpolate_brackets(
        section_id, areaperil_id, np.array([2.0, 10.0]), np.ones(2), half,   # rp_from bracket
        section_id, areaperil_id, np.array([4.0, 20.0]), np.ones(2), half)   # rp_to bracket
    # width is the probability in units of 1e-12: one interval covering the whole distribution
    assert list(zip(section_id, areaperil_id, intensity, width)) == [(0, 2**32 + 5, 3.0, 10**12), (1, 5, 15.0, 10**12)]


def test_sections_reversed_by_an_extrapolating_weight_merge_without_a_sliver():
    # interpolation weights outside [0, 1] reverse a section's interpolated order; once re-sorted,
    # its levels (thirds, not exact in 1e-12 units) round the other way from the other section's,
    # and must still meet rather than leave a 1e-12 interval
    haz = pd.DataFrame([(1, 1, 10, 8, 2), (3, 1, 50, 6, 1), (3, 1, 50, 2, 1), (1, 1, 10, 18, 1), (3, 1, 10, 5, 2),
                        (1, 1, 20, 3, 1), (1, 1, 20, 14, 1), (3, 1, 50, 12, 1), (1, 1, 20, 11, 1), (1, 1, 10, 18, 0)],
                       columns=['section_id', 'areaperil_id', 'return_period', 'intensity', 'probability'])
    result = build_merged(haz, [(1, 10, 20, 1.5), (3, 20, 50, 2.25)], 'sum')
    assert atoms(result, 1) == [(39, 1 / 3, 50), (25, 1 / 3, 50), (5, 1 / 3, 50)]


def test_a_section_given_twice_with_different_brackets_is_refused():
    from oasislmf.pytools.getmodel.footprint import OasisFootPrintError
    haz = flat(1, 100, [10])
    with pytest.raises(OasisFootPrintError, match=r"section\(s\) \[1\] more than once"):
        build_merged(haz, [(1, 10, 100, 0.5), (1, 10, 100, 0.25)], 'max')
    # the same row twice is harmless
    assert atoms(build_merged(haz, [(1, 10, 100, 0.5), (1, 10, 100, 0.5)], 'max')) == [(10, 1.0, 100)]
