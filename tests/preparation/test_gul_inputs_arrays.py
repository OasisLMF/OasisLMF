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


def make_keys(exposure):
    return pd.DataFrame({
        'loc_id': np.repeat(exposure['loc_id'], len(COVERAGE_TYPE_IDS)),
        'peril_id': 'WTC',
        'coverage_type_id': np.tile(COVERAGE_TYPE_IDS, len(exposure)),
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

    for coverage_type_id in COVERAGE_TYPE_IDS:
        coverage = gul_inputs[gul_inputs['coverage_type_id'] == coverage_type_id]
        expected = reference_building_ids(np.maximum(1, number_of_buildings))
        np.testing.assert_array_equal(coverage['building_id'].to_numpy(), expected)


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


def make_correlations(num_items):
    return pd.DataFrame({
        'item_id': np.arange(1, num_items + 1, dtype='int32'),
        'peril_correlation_group': np.arange(num_items, dtype='uint32') % 3,
        'damage_correlation_value': np.linspace(0, 1, num_items),
        'hazard_group_id': np.arange(num_items, dtype='uint32') % 7,
        'hazard_correlation_value': np.linspace(1, 0, num_items),
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


def test_correlations_bin_is_empty_when_there_are_no_correlations(tmp_path):
    assert len(written_correlations(tmp_path, pd.DataFrame(columns=correlations_headers))) == 0


def test_correlations_bin_is_empty_when_correlations_are_not_given(tmp_path):
    assert len(written_correlations(tmp_path, None)) == 0
