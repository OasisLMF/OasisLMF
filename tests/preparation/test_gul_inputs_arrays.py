"""Cover the array building in the GUL inputs: disaggregated building ids and correlations.bin."""

import numpy as np
import pandas as pd
import pytest

from oasislmf.preparation.gul_inputs import get_gul_input_items, write_gul_input_files
from oasislmf.pytools.common.data import correlations_dtype, correlations_headers

COVERAGE_TYPE_IDS = [1, 3]


def make_exposure(number_of_buildings, is_aggregate=None):
    """Build a location frame with the given number of buildings per location."""
    num_locations = len(number_of_buildings)
    return pd.DataFrame({
        'loc_id': np.arange(1, num_locations + 1),
        'PortNumber': '1',
        'AccNumber': '1',
        'LocNumber': [str(loc_id) for loc_id in range(1, num_locations + 1)],
        'BuildingTIV': np.linspace(1000, 1000 * num_locations, num_locations),
        'ContentsTIV': np.linspace(100, 100 * num_locations, num_locations),
        'NumberOfBuildings': number_of_buildings,
        'IsAggregate': is_aggregate if is_aggregate is not None else [1] * num_locations,
    })


def make_keys(exposure, peril_ids=('WTC',)):
    """One keys row per (location, peril, coverage type)."""
    peril_ids = list(peril_ids)
    return pd.DataFrame({
        'loc_id': np.repeat(exposure['loc_id'], len(COVERAGE_TYPE_IDS) * len(peril_ids)),
        'peril_id': np.tile(np.repeat(peril_ids, len(COVERAGE_TYPE_IDS)), len(exposure)),
        'coverage_type_id': np.tile(COVERAGE_TYPE_IDS, len(exposure) * len(peril_ids)),
        'areaperil_id': 1,
        'vulnerability_id': 1,
        'status': 'success',
    })


def reference_building_ids(repeat_counts):
    """The building ids as they were numbered before, one arange per location."""
    return np.concatenate([np.arange(1, count + 1) for count in repeat_counts])


@pytest.mark.parametrize('number_of_buildings', [
    [1, 1, 1],
    [1, 3, 0, 2, 5, 1],
    [4],
    [0, 0],
    [7, 1, 1, 2],
])
def test_building_ids_number_each_locations_buildings(number_of_buildings):
    exposure = make_exposure(number_of_buildings)
    gul_inputs = get_gul_input_items(exposure, make_keys(exposure), damage_group_id_cols=['loc_id'])

    tiv_by_coverage = {1: 'BuildingTIV', 3: 'ContentsTIV'}
    for coverage_type_id in COVERAGE_TYPE_IDS:
        coverage = gul_inputs[gul_inputs['coverage_type_id'] == coverage_type_id]
        expected = reference_building_ids(np.maximum(1, number_of_buildings))
        np.testing.assert_array_equal(coverage['building_id'].to_numpy(), expected)

        # the buildings split the location's TIV between them, they do not each carry the whole of it
        per_location = coverage.groupby('loc_id', observed=True)['tiv'].sum().to_numpy()
        np.testing.assert_allclose(per_location, exposure[tiv_by_coverage[coverage_type_id]].to_numpy())


@pytest.mark.parametrize('is_aggregate', [0, 1])
def test_disaggregation_is_driven_by_the_building_count_not_the_aggregate_flag(is_aggregate):
    """IsAggregate does not gate the building expansion; NumberOfBuildings does.

    Every other test here runs with IsAggregate=1, so the flag is otherwise only ever a
    passed-through column. It feeds risk_id/NumberOfRisks in assign_risk_ids rather than the
    building numbering, and a location splits into one row per building either way.
    """
    number_of_buildings = [1, 3, 2]
    exposure = make_exposure(number_of_buildings, is_aggregate=[is_aggregate] * len(number_of_buildings))
    gul_inputs = get_gul_input_items(exposure, make_keys(exposure), damage_group_id_cols=['loc_id'])

    for coverage_type_id in COVERAGE_TYPE_IDS:
        coverage = gul_inputs[gul_inputs['coverage_type_id'] == coverage_type_id]
        np.testing.assert_array_equal(
            coverage['building_id'].to_numpy(), reference_building_ids(number_of_buildings))
    assert set(gul_inputs['IsAggregate']) == {is_aggregate}


def test_building_ids_are_all_one_without_disaggregation():
    exposure = make_exposure([1, 3, 2])
    gul_inputs = get_gul_input_items(exposure, make_keys(exposure), damage_group_id_cols=['loc_id'],
                                     do_disaggregation=False)

    assert set(gul_inputs['building_id']) == {1}


def test_item_ids_stay_unique_per_building_and_coverage():
    exposure = make_exposure([1, 3, 0, 2, 5, 1])
    gul_inputs = get_gul_input_items(exposure, make_keys(exposure), damage_group_id_cols=['loc_id'])

    keys = gul_inputs[['loc_id', 'peril_id', 'coverage_type_id', 'building_id']]
    assert len(gul_inputs['item_id'].unique()) == len(keys.drop_duplicates())


@pytest.mark.parametrize('peril_ids', [('WTC',), ('WTC', 'WSS')])
def test_item_ids_are_the_contiguous_range_from_one(peril_ids):
    """item_id is exactly 1..n over the returned rows, not merely n distinct values.

    The cardinality assertion above holds for any bijection from rows to ids -- 0-based numbering,
    a reversed assignment, ids offset by 1000 -- so on its own it cannot detect an off-by-one in
    the ``ngroup() + 1`` this is built from. The frame is sorted by item_id on the way out, so the
    ids must line up with the row numbers.
    """
    exposure = make_exposure([1, 3, 0, 2, 5, 1])
    gul_inputs = get_gul_input_items(exposure, make_keys(exposure, peril_ids), damage_group_id_cols=['loc_id'])

    np.testing.assert_array_equal(
        gul_inputs['item_id'].to_numpy(), np.arange(1, len(gul_inputs) + 1))
    assert gul_inputs['item_id'].dtype == np.int32


def test_coverage_ids_are_contiguous_and_shared_across_perils():
    """coverage_id is 1..m over (loc_id, building_id, coverage_type_id), ignoring peril.

    coverage_id is asserted nowhere else, and with a single peril in the keys the fact that it
    collapses perils together is not exercised at all -- every group would be a singleton.
    """
    peril_ids = ('WTC', 'WSS')
    exposure = make_exposure([1, 3, 2])
    gul_inputs = get_gul_input_items(exposure, make_keys(exposure, peril_ids), damage_group_id_cols=['loc_id'])

    # the fixture must really carry both perils, or the property below is vacuous
    assert set(gul_inputs['peril_id']) == set(peril_ids)

    coverage_keys = ['loc_id', 'building_id', 'coverage_type_id']
    per_group = gul_inputs.groupby(coverage_keys, observed=True)['coverage_id']
    assert (per_group.nunique() == 1).all(), 'one coverage_id per (loc, building, coverage type)'
    assert per_group.ngroup().nunique() == gul_inputs['coverage_id'].nunique()

    expected = len(gul_inputs[coverage_keys].drop_duplicates())
    assert sorted(gul_inputs['coverage_id'].unique()) == list(range(1, expected + 1))
    # each coverage is shared by exactly one item per peril
    assert gul_inputs.groupby('coverage_id', observed=True).size().eq(len(peril_ids)).all()


def make_correlations(num_items):
    return pd.DataFrame({
        'item_id': np.arange(1, num_items + 1, dtype='int32'),
        'peril_correlation_group': np.arange(num_items, dtype='uint32') % 3,
        'damage_correlation_value': np.linspace(0, 1, num_items),
        'hazard_group_id': np.arange(num_items, dtype='uint32') % 7,
        'hazard_correlation_value': np.linspace(1, 0, num_items),
        # coverage dependency: 0 = independent. A distinct pattern from the columns above, so the
        # by-name-keying test below can still tell the columns apart.
        'source_item_id': np.arange(num_items, dtype='int32') % 5,
    })[correlations_headers]


def written_correlations(tmp_path, correlations_df):
    exposure = make_exposure([1, 2])
    gul_inputs = get_gul_input_items(exposure, make_keys(exposure), damage_group_id_cols=['loc_id'])
    write_gul_input_files(gul_inputs, str(tmp_path), correlations_df, str(tmp_path))

    return np.fromfile(tmp_path / 'correlations.bin', dtype=correlations_dtype)


def test_correlations_bin_holds_the_correlations_frame(tmp_path):
    correlations_df = make_correlations(50)

    written = written_correlations(tmp_path, correlations_df)

    # the row by row packing this replaced, which read the columns positionally
    expected = np.array([row for row in correlations_df.itertuples(index=False)], dtype=correlations_dtype)
    np.testing.assert_array_equal(written, expected)


def test_correlations_bin_keys_the_columns_by_name_not_position(tmp_path):
    """A frame whose columns are ordered differently from the dtype still packs into the right fields.

    This is the property the switch away from itertuples was made for, and it is otherwise
    untested: make_correlations ends with [correlations_headers], so every other fixture here is
    already in dtype order and positional packing and by-name packing agree.
    """
    correlations_df = make_correlations(50)
    reordered = correlations_df[correlations_headers[::-1]]
    assert list(reordered.columns) != correlations_headers, 'the fixture must really be reordered'

    expected = written_correlations(tmp_path, correlations_df)
    written = written_correlations(tmp_path, reordered)

    np.testing.assert_array_equal(written, expected)
    # the packing this replaced read the columns positionally, so it would have scrambled them
    positional = np.array([row for row in reordered.itertuples(index=False)], dtype=correlations_dtype)
    assert not np.array_equal(positional, expected)


def test_correlations_bin_ignores_a_column_that_is_not_a_field(tmp_path):
    """A column with no matching field is dropped rather than raising.

    The positional packing raised ValueError on a frame with more columns than the dtype has
    fields. Ignoring it is the saner behaviour, but it does mean passing a whole gul_inputs frame
    here by mistake would now quietly succeed, so the contract is worth stating.
    """
    correlations_df = make_correlations(20)

    expected = written_correlations(tmp_path, correlations_df)
    written = written_correlations(tmp_path, correlations_df.assign(loc_id=-1))

    np.testing.assert_array_equal(written, expected)


def test_correlations_bin_is_empty_when_there_are_no_correlations(tmp_path):
    assert len(written_correlations(tmp_path, pd.DataFrame(columns=correlations_headers))) == 0


def test_correlations_bin_is_empty_when_correlations_are_not_given(tmp_path):
    assert len(written_correlations(tmp_path, None)) == 0
