"""Tests for building-packing mode in GUL input preparation.

Building-packing mode (``disaggregation='samples'``) keeps one item per
(location, peril, coverage_type) instead of expanding one row per building, and carries
the per-item building count on the ``correlations`` table (the ``packed_buildings``
column, 1:1 with items) so the buildings can be multiplexed into the sample dimension
downstream (gulmc/gulpy). It is no longer a separate side file.
"""
import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd

from oasislmf.preparation.gul_inputs import (
    build_correlations_frame,
    get_gul_input_items,
    process_group_id_cols,
    write_gul_input_files,
)
from oasislmf.preparation.summaries import get_summary_mapping, _location_tiv_total
from oasislmf.pytools.common.input_files import read_correlations
from oasislmf.utils.profiles import get_oed_hierarchy
from oasislmf.utils.defaults import (DISAGGREGATION_ITEMS, DISAGGREGATION_NONE,
                                     DISAGGREGATION_SAMPLES)


def _correlations_df(gul_inputs_df):
    """Build the correlations frame the way computation/generate/files.py does."""
    df = gul_inputs_df.copy()
    for col in ('peril_correlation_group', 'damage_correlation_value',
                'hazard_group_id', 'hazard_correlation_value'):
        if col not in df.columns:
            df[col] = 0
    return build_correlations_frame(df)


def _loc_df():
    # loc 1: 3-building aggregate; loc 2: single building.
    return pd.DataFrame({
        'loc_id': [1, 2],
        'PortNumber': [1, 1], 'AccNumber': [1, 1], 'LocNumber': ['1', '2'],
        'BuildingTIV': [999.0, 2000.0],
        'NumberOfBuildings': [3, 1], 'IsAggregate': [1, 0],
    })


def _mixed_loc_df():
    """loc 1: 3 buildings, IsAggregate=1. loc 2: 2 buildings, IsAggregate=0."""
    return pd.DataFrame({
        'loc_id': [1, 2],
        'PortNumber': [1, 1], 'AccNumber': [1, 1], 'LocNumber': ['1', '2'],
        'BuildingTIV': [999.0, 2000.0],
        'NumberOfBuildings': [3, 2], 'IsAggregate': [1, 0],
    })


def _keys_df():
    return pd.DataFrame({
        'loc_id': [1, 2], 'peril_id': ['WW1', 'WW1'], 'coverage_type_id': [1, 1],
        'areaperil_id': [1, 1], 'vulnerability_id': [1, 1], 'status': ['success', 'success'],
    })


def _one_loc(n=3, is_aggregate=0, loc_ded=0.0, loc_limit=0.0):
    """Single location with N buildings and optional location-level coverage terms."""
    return pd.DataFrame({
        'loc_id': [1], 'PortNumber': [1], 'AccNumber': [1], 'LocNumber': ['1'],
        'BuildingTIV': [999.0], 'NumberOfBuildings': [n], 'IsAggregate': [is_aggregate],
        'LocDed1Building': [loc_ded], 'LocLimit1Building': [loc_limit],
    })


def _one_key():
    return pd.DataFrame({
        'loc_id': [1], 'peril_id': ['WW1'], 'coverage_type_id': [1],
        'areaperil_id': [1], 'vulnerability_id': [1], 'status': ['success'],
    })


class TestBuildingLevelGroupCols(TestCase):
    """Listing a building-level column in the group_id columns is the supported way to make
    the buildings of a location draw independently instead of sharing the location's draw.

    Omitting them (the default) keeps a location's buildings perfectly correlated. Both
    behaviours are the user's to choose; neither is imposed by the engine.
    """

    def test_building_level_cols_are_accepted(self):
        cols = process_group_id_cols(['building_id', 'loc_id', 'risk_id'], ['loc_id'], False)
        self.assertEqual(sorted(cols), ['building_id', 'loc_id', 'risk_id'])

    def test_unknown_cols_are_still_stripped(self):
        cols = process_group_id_cols(['loc_id', 'NotAColumn'], ['loc_id'], False)
        self.assertEqual(cols, ['loc_id'])

    def test_default_cols_keep_a_locations_buildings_in_one_group(self):
        gul = get_gul_input_items(_mixed_loc_df(), _keys_df(),
                                  damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_ITEMS)
        self.assertTrue((gul.groupby('loc_id')['group_id'].nunique() == 1).all())

    def test_building_id_gives_every_building_its_own_group(self):
        gul = get_gul_input_items(_mixed_loc_df(), _keys_df(),
                                  damage_group_id_cols=['loc_id', 'building_id'], disaggregation=DISAGGREGATION_ITEMS)
        per_loc = gul.groupby('loc_id')['group_id'].nunique()
        self.assertEqual(per_loc.loc[1], 3)
        self.assertEqual(per_loc.loc[2], 2)

    def test_risk_id_splits_aggregate_locations_only(self):
        """risk_id tracks building_id where IsAggregate == 1 and collapses to 1 elsewhere, so it
        separates genuinely distinct risks while leaving one-risk-split-N-ways locations correlated.
        """
        gul = get_gul_input_items(_mixed_loc_df(), _keys_df(),
                                  damage_group_id_cols=['loc_id', 'risk_id'], disaggregation=DISAGGREGATION_ITEMS)
        per_loc = gul.groupby('loc_id')['group_id'].nunique()
        self.assertEqual(per_loc.loc[1], 3)   # IsAggregate == 1 -> risk_id == building_id
        self.assertEqual(per_loc.loc[2], 1)   # IsAggregate == 0 -> risk_id == 1 for every building

    def test_damage_and_hazard_axes_are_chosen_separately(self):
        gul = get_gul_input_items(_mixed_loc_df(), _keys_df(),
                                  damage_group_id_cols=['loc_id'],
                                  hazard_group_id_cols=['loc_id', 'building_id'],
                                  disaggregation=DISAGGREGATION_ITEMS)
        loc1 = gul[gul['loc_id'] == 1]
        self.assertEqual(loc1['group_id'].nunique(), 1)
        self.assertEqual(loc1['hazard_group_id'].nunique(), 3)


class TestBuildingPacking(TestCase):

    def test_packing_keeps_one_item_per_loc_peril_cov(self):
        """A packable location keeps one item however many buildings it has."""
        loc = _loc_df()
        loc['IsAggregate'] = 0  # summed before any term -> packable
        legacy = get_gul_input_items(loc.copy(), _keys_df(), damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_ITEMS)
        packed = get_gul_input_items(loc.copy(), _keys_df(), damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_SAMPLES)

        # legacy expands to one row per building: 3 (loc 1) + 1 (loc 2) = 4 items
        self.assertEqual(len(legacy), 4)
        self.assertEqual(sorted(legacy['building_id'].tolist()), [1, 1, 2, 3])

        # packed keeps one item per (loc, peril, coverage_type): 1 + 1 = 2 items
        self.assertEqual(len(packed), 2)
        self.assertEqual(packed['building_id'].tolist(), [1, 1])

    def test_packing_carries_number_of_buildings(self):
        loc = _loc_df()
        loc['IsAggregate'] = 0
        packed = get_gul_input_items(loc, _keys_df(), damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_SAMPLES)
        self.assertIn('number_of_buildings', packed.columns)
        self.assertEqual(packed.sort_values('loc_id')['number_of_buildings'].tolist(), [3, 1])

    def test_packing_conserves_total_tiv(self):
        """Packed per-item TIV is per-building; total (tiv * N) matches the expanded run."""
        legacy = get_gul_input_items(_loc_df(), _keys_df(), damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_ITEMS)
        packed = get_gul_input_items(_loc_df(), _keys_df(), damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_SAMPLES)
        self.assertAlmostEqual(legacy['tiv'].sum(), (packed['tiv'] * packed['number_of_buildings']).sum(), places=4)

    def test_group_id_is_location_level_not_per_building(self):
        """Packing leaves one row per location, so group_id is necessarily location-level.

        There is nothing for a building-level group column to differentiate here, and nothing is
        lost by that: packing exists to give the buildings their own draws (perfectly correlated
        buildings need no building dimension at all — that is the no-disaggregation mode), so the
        building dimension is separated at sampling time rather than through the hash.
        """
        packed = get_gul_input_items(_loc_df(), _keys_df(), damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_SAMPLES)
        self.assertEqual(packed['group_id'].nunique(), packed['loc_id'].nunique())

    def test_building_count_carried_on_correlations(self):
        """The per-item count rides on correlations.bin; no side file is written."""
        loc = _loc_df()
        loc['IsAggregate'] = 0  # summed before any term -> packable
        for mode, kw, expected in [
            ('packed', dict(disaggregation=DISAGGREGATION_SAMPLES), [1, 3]),
            ('legacy', dict(disaggregation=DISAGGREGATION_ITEMS), [1, 1, 1, 1]),
        ]:
            df = get_gul_input_items(loc.copy(), _keys_df(), damage_group_id_cols=['loc_id'], **kw)
            with TemporaryDirectory() as d:
                write_gul_input_files(df, d, _correlations_df(df), d)
                # items.bin (shared with C++ ktools) is always written and unchanged
                self.assertTrue(os.path.exists(os.path.join(d, 'items.bin')))
                # number_of_buildings is no longer a side file
                self.assertFalse(os.path.exists(os.path.join(d, 'number_of_buildings.bin')))
                self.assertFalse(os.path.exists(os.path.join(d, 'number_of_buildings.csv')))
                # it is present on correlations.bin, 1:1 with items
                corr = read_correlations(d)
                self.assertIn('packed_buildings', corr.dtype.names)
                self.assertEqual(sorted(np.asarray(corr['packed_buildings']).tolist()), expected)


class TestWhichLocationsArePacked(TestCase):
    """What decides packing is IsAggregate, not the presence of location terms.

    The site FM levels aggregate on ('loc_id', 'risk_id') and risk_id == building_id only when
    IsAggregate == 1, so:

    * IsAggregate == 0 -- every building shares risk_id 1, the site levels sum them before
      applying any term, and packing reproduces that exactly. Location terms do not change this.
    * IsAggregate == 1 -- one site node per building, each carrying term/NumberOfRisks, so the
      buildings have to stay separate until those terms have been applied.

    Both are packed: the financial module keeps the buildings of the second kind apart until the
    site levels have applied their terms, then collapses them.
    """

    def test_non_aggregate_location_is_packed(self):
        packed = get_gul_input_items(_one_loc(n=3, is_aggregate=0), _one_key(),
                                     damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_SAMPLES)
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed['number_of_buildings'].tolist(), [3])

    def test_location_terms_do_not_prevent_packing(self):
        """A non-aggregate location sums its buildings before any term, so terms are irrelevant."""
        for term in (dict(loc_ded=500.0), dict(loc_limit=10000.0)):
            with self.subTest(**term):
                packed = get_gul_input_items(_one_loc(n=3, is_aggregate=0, **term), _one_key(),
                                             damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_SAMPLES)
                self.assertEqual(len(packed), 1)
                self.assertEqual(packed['number_of_buildings'].tolist(), [3])

    def test_aggregate_location_keeps_its_buildings_separate(self):
        """Still one item -- the buildings ride in the sample dimension and collapse in fm."""
        packed = get_gul_input_items(_one_loc(n=3, is_aggregate=1, loc_ded=500.0), _one_key(),
                                     damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_SAMPLES)
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed['keep_buildings_separate'].tolist(), [1])
        self.assertEqual(packed['number_of_buildings'].tolist(), [3])

    def test_single_building_is_never_kept_separate(self):
        packed = get_gul_input_items(_one_loc(n=1, is_aggregate=1, loc_ded=500.0), _one_key(),
                                     damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_SAMPLES)
        self.assertEqual(packed['keep_buildings_separate'].tolist(), [0])
        self.assertEqual(packed['number_of_buildings'].tolist(), [1])

    def test_total_tiv_conserved_either_way(self):
        packed = get_gul_input_items(_one_loc(n=4, is_aggregate=0, loc_ded=500.0), _one_key(),
                                     damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_SAMPLES)
        expanded = get_gul_input_items(_one_loc(n=4, is_aggregate=1, loc_ded=500.0), _one_key(),
                                       damage_group_id_cols=['loc_id'], disaggregation=DISAGGREGATION_SAMPLES)
        self.assertAlmostEqual((packed['tiv'] * packed['number_of_buildings']).sum(),
                               (expanded['tiv'] * expanded['number_of_buildings']).sum(), places=4)


class TestSummaryMapTiv(TestCase):
    """The reported TIV must not depend on how the buildings are represented.

    ``tiv`` on the summary map is one building's share. Row disaggregation writes N rows with
    distinct building_id, so de-duplicating on (loc_id, building_id, coverage_type_id) recovers
    the location total. Packing writes ONE row standing for N, so de-duplication alone returns
    1/N of it -- the count has to be multiplied back in.
    """

    def _map(self, disaggregation):
        gul = get_gul_input_items(_one_loc(n=3, is_aggregate=1), _one_key(),
                                  damage_group_id_cols=['loc_id'], disaggregation=disaggregation)
        return get_summary_mapping(gul, get_oed_hierarchy())

    def test_every_mode_reports_the_same_location_tiv(self):
        totals = {mode: _location_tiv_total(self._map(mode))
                  for mode in (DISAGGREGATION_NONE, DISAGGREGATION_ITEMS, DISAGGREGATION_SAMPLES)}
        self.assertAlmostEqual(totals[DISAGGREGATION_NONE], totals[DISAGGREGATION_ITEMS], places=4)
        self.assertAlmostEqual(totals[DISAGGREGATION_NONE], totals[DISAGGREGATION_SAMPLES], places=4,
                               msg=f'packing reports a different TIV: {totals}')

    def test_the_count_is_carried_on_the_map(self):
        """Without it the multiplication above has nothing to work from."""
        for mode, expected in ((DISAGGREGATION_ITEMS, 1), (DISAGGREGATION_SAMPLES, 3)):
            with self.subTest(disaggregation=mode):
                m = self._map(mode)
                self.assertIn('number_of_buildings', m.columns)
                self.assertEqual(sorted(set(m['number_of_buildings'])), [expected])


class TestThreeDisaggregationModes(TestCase):
    """The three modes are mutually exclusive and each answers the building question its own way.

    - no disaggregation: one item carrying the whole location TIV. The buildings are implicitly
      perfectly correlated, which is why they need no building dimension at all.
    - row disaggregation: one item per building, TIV split. Correlation is the user's choice via
      the group_id columns (see TestBuildingLevelGroupCols).
    - building packing: one item per location, TIV per building, N carried per item. The buildings
      are separated in the sample dimension, so they draw independently.

    The per-item building count is what tells the loss side which of these it is: it is > 1 only
    under packing, and gulmc derives the mode from ``items['packed_buildings'].max() > 1``.
    """

    def _run(self, **kwargs):
        return get_gul_input_items(_mixed_loc_df(), _keys_df(),
                                   damage_group_id_cols=['loc_id', 'building_id'], **kwargs)

    def test_no_disaggregation_keeps_one_item_holding_the_whole_location(self):
        gul = self._run(disaggregation=DISAGGREGATION_NONE)
        self.assertEqual(len(gul), 2)
        self.assertEqual(gul['number_of_buildings'].max(), 1)
        self.assertEqual(gul['building_id'].unique().tolist(), [1])

    def test_row_disaggregation_expands_and_keeps_the_count_at_one(self):
        gul = self._run(disaggregation=DISAGGREGATION_ITEMS)
        self.assertEqual(len(gul), 5)                       # 3 + 2 buildings
        self.assertEqual(gul['number_of_buildings'].max(), 1)

    def test_packing_keeps_the_rows_and_carries_the_count(self):
        """Every location keeps one item, whatever its IsAggregate."""
        gul = self._run(disaggregation=DISAGGREGATION_SAMPLES)
        self.assertEqual(len(gul), 2)
        self.assertEqual(sorted(gul['number_of_buildings'].tolist()), [2, 3])
        # only the aggregate location needs its buildings kept apart downstream
        by_loc = gul.set_index('loc_id')['keep_buildings_separate'].to_dict()
        self.assertEqual(by_loc[1], 1)   # IsAggregate = 1
        self.assertEqual(by_loc[2], 0)   # IsAggregate = 0

    def test_every_mode_conserves_total_tiv(self):
        totals = {
            name: (gul['tiv'] * gul['number_of_buildings']).sum()
            for name, gul in (
                ('nothing', self._run(disaggregation=DISAGGREGATION_NONE)),
                ('today', self._run(disaggregation=DISAGGREGATION_ITEMS)),
                ('packing', self._run(disaggregation=DISAGGREGATION_SAMPLES)),
            )
        }
        self.assertAlmostEqual(totals['nothing'], totals['today'], places=4)
        self.assertAlmostEqual(totals['nothing'], totals['packing'], places=4)

    def test_a_degenerate_building_column_shifts_no_grouping(self):
        """Neither of these modes expands rows, so building_id is a constant 1 in the group cols.

        It is then a no-op rather than something needing to be stripped. Under row disaggregation
        it legitimately gives one group per building -- that is the whole point of listing the
        column, and TestBuildingLevelGroupCols covers it.
        """
        for loc, kwargs in ((_mixed_loc_df(), dict(disaggregation=DISAGGREGATION_NONE)),
                            (_mixed_loc_df(), dict(disaggregation=DISAGGREGATION_SAMPLES))):
            with self.subTest(**kwargs):
                with_col = get_gul_input_items(loc.copy(), _keys_df(),
                                               damage_group_id_cols=['loc_id', 'building_id'], **kwargs)
                without = get_gul_input_items(loc.copy(), _keys_df(),
                                              damage_group_id_cols=['loc_id'], **kwargs)
                self.assertEqual(len(with_col), len(loc))
                self.assertEqual(with_col['group_id'].nunique(), without['group_id'].nunique())
                self.assertEqual(with_col['group_id'].nunique(), with_col['loc_id'].nunique())
