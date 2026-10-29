"""How GUL input generation represents a location of several buildings.

``disaggregation`` says where a location's ``NumberOfBuildings`` is separated:

* ``none`` -- nowhere: one item carries the whole location, its buildings perfectly correlated;
* ``items`` -- into one item per building, each with its share of the TIV. How the buildings
  correlate is the user's choice, through building-level group fields;
* ``samples`` -- into the sample dimension: one item per location at one building's TIV, with
  the building count on ``coverage_buildings``. Negative keeps the buildings separate (an
  ``IsAggregate`` location, whose site terms apply building by building); positive sums them at
  source. Every building draws its own samples either way.

What a packed run computes from these inputs is tested end to end in tests/fm and
tests/pytools/gulmc; the reported TIV and the loss-step checks in tests/computation.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from oasislmf.preparation.gul_inputs import (build_correlations_frame, get_gul_input_items,
                                             validate_source_and_dependent_building_counts,
                                             write_gul_input_files)
from oasislmf.pytools.common.data import items_dtype
from oasislmf.pytools.common.input_files import read_coverage_buildings, read_coverages
from oasislmf.utils.defaults import DISAGGREGATION_ITEMS, DISAGGREGATION_NONE, DISAGGREGATION_SAMPLES
from oasislmf.utils.exceptions import OasisException

LOCATION_TIV = 999.0


def _locations(n_buildings, is_aggregate, **terms):
    """One location per entry of ``n_buildings``, all on one account."""
    n = len(n_buildings)
    return pd.DataFrame({
        'loc_id': range(1, n + 1), 'PortNumber': 1, 'AccNumber': 1, 'LocNumber': [str(i) for i in range(1, n + 1)],
        'BuildingTIV': LOCATION_TIV, 'NumberOfBuildings': n_buildings, 'IsAggregate': is_aggregate, **terms,
    })


def _keys(locations):
    return pd.DataFrame({'loc_id': locations['loc_id'], 'peril_id': 'WW1', 'coverage_type_id': 1,
                         'areaperil_id': 1, 'vulnerability_id': 1, 'status': 'success'})


def _gul_inputs(locations, disaggregation, damage_group_id_cols=('loc_id',), **kwargs):
    return get_gul_input_items(locations, _keys(locations), damage_group_id_cols=list(damage_group_id_cols),
                               disaggregation=disaggregation, **kwargs)


def _write(gul_inputs, target_dir):
    df = gul_inputs.copy()
    for col in ('peril_correlation_group', 'damage_correlation_value', 'hazard_group_id', 'hazard_correlation_value'):
        if col not in df.columns:
            df[col] = 0
    write_gul_input_files(gul_inputs, str(target_dir), build_correlations_frame(df), str(target_dir))


@pytest.mark.parametrize('disaggregation, n_buildings, is_aggregate, terms, n_items, n_building', [
    (DISAGGREGATION_NONE, 3, 1, {}, 1, None),
    (DISAGGREGATION_ITEMS, 3, 1, {}, 3, None),
    (DISAGGREGATION_SAMPLES, 3, 0, {}, 1, 3),
    # site terms apply after an IsAggregate=0 location's buildings are summed, so they change nothing
    (DISAGGREGATION_SAMPLES, 3, 0, {'LocDed1Building': 500.0}, 1, 3),
    (DISAGGREGATION_SAMPLES, 3, 1, {'LocDed1Building': 500.0}, 1, -3),
    (DISAGGREGATION_SAMPLES, 1, 1, {'LocDed1Building': 500.0}, 1, None),
], ids=['none', 'items', 'samples-summed', 'samples-summed-with-site-terms', 'samples-separate', 'samples-one-building'])
def test_how_each_mode_writes_a_location_of_many_buildings(tmp_path, disaggregation, n_buildings, is_aggregate, terms,
                                                           n_items, n_building):
    """``n_building`` is the count written to coverage_buildings, or None where the file is not
    written because every coverage holds one building. Whatever the representation, the TIV
    written adds up to the location's."""
    _write(_gul_inputs(_locations([n_buildings], is_aggregate, **terms), disaggregation), tmp_path)

    assert len(np.fromfile(tmp_path / 'items.bin', dtype=items_dtype)) == n_items
    counts = read_coverage_buildings(tmp_path)
    if n_building is None:
        assert counts.shape[0] == 0
    else:
        assert np.asarray(counts['n_building']).tolist() == [n_building]
    per_coverage = np.abs(np.asarray(counts['n_building'])) if counts.shape[0] else 1
    assert (read_coverages(tmp_path) * per_coverage).sum() == pytest.approx(LOCATION_TIV)


@pytest.mark.parametrize('damage, hazard, damage_groups, hazard_groups', [
    (['loc_id'], ['loc_id'], [1, 1], [1, 1]),
    (['loc_id', 'building_id'], ['loc_id'], [3, 2], [1, 1]),
    # risk_id is building_id where IsAggregate=1 and 1 elsewhere: only distinct risks are split
    (['loc_id', 'risk_id'], ['loc_id'], [3, 1], [1, 1]),
    (['loc_id'], ['loc_id', 'building_id'], [1, 1], [3, 2]),
], ids=['location', 'building_id', 'risk_id', 'hazard_only'])
def test_group_fields_choose_how_the_buildings_of_items_correlate(damage, hazard, damage_groups, hazard_groups):
    """Under ``items`` each building is its own item, and the group fields decide which share a
    draw: location-level fields keep a location's buildings perfectly correlated, building-level
    ones make them independent. Damage and hazard are chosen separately.

    Location 1 has 3 buildings and IsAggregate=1, location 2 has 2 and IsAggregate=0; the counts
    are the distinct groups in each.
    """
    gul = _gul_inputs(_locations([3, 2], [1, 0]), DISAGGREGATION_ITEMS,
                      damage_group_id_cols=damage, hazard_group_id_cols=hazard)
    assert gul.groupby('loc_id')['group_id'].nunique().tolist() == damage_groups
    assert gul.groupby('loc_id')['hazard_group_id'].nunique().tolist() == hazard_groups


@pytest.mark.parametrize('disaggregation, n_notices', [(DISAGGREGATION_NONE, 0), (DISAGGREGATION_SAMPLES, 1)])
@pytest.mark.parametrize('field', ['building_id', 'risk_id'])
def test_a_building_level_group_field_does_nothing_without_items(disaggregation, n_notices, field):
    """Only ``items`` gives a building a row of its own, so elsewhere the field changes no group.

    Under ``samples`` that is worth saying: the buildings already draw independently, and a model
    carrying the field for ``items`` has had the correlation it encodes stop applying.
    """
    locations = _locations([3, 2], [1, 0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with_field = _gul_inputs(locations.copy(), disaggregation, damage_group_id_cols=['loc_id', field])
    without = _gul_inputs(locations.copy(), disaggregation)

    assert with_field['group_id'].nunique() == without['group_id'].nunique() == len(locations)
    notices = [str(w.message) for w in caught if 'no effect under --disaggregation samples' in str(w.message)]
    assert len(notices) == n_notices
    if n_notices:
        assert field in notices[0] and '--disaggregation items' in notices[0]


def test_no_notice_where_the_field_still_works():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _gul_inputs(_locations([3, 2], [1, 0]), DISAGGREGATION_ITEMS, damage_group_id_cols=['loc_id', 'building_id'])
    assert not [w for w in caught if 'no effect under' in str(w.message)]


def test_a_dependent_and_its_source_must_pack_the_same_buildings():
    """gulmc pairs building b of a dependent with building b of its source. The two share a
    location, so the counts agree by construction, and a mismatch means the link crossed
    locations; it is refused rather than paired wrongly."""
    df = pd.DataFrame({'item_id': [1, 2, 3], 'source_item_id': [0, 1, 0], 'number_of_buildings': [4, 4, 7]})
    validate_source_and_dependent_building_counts(df)

    df.loc[df['item_id'] == 2, 'number_of_buildings'] = 3
    with pytest.raises(OasisException, match='different number of buildings'):
        validate_source_and_dependent_building_counts(df)
