"""Cover how the keys rows are joined to the locations and reduced to one row per GUL item."""

import logging

import pandas as pd
import pytest

from oasislmf.preparation.gul_inputs import get_gul_input_items, prepare_sections_df
from oasislmf.utils.exceptions import OasisException


def make_exposure(num_locations=2):
    return pd.DataFrame({
        'loc_id': range(1, num_locations + 1),
        'PortNumber': '1',
        'AccNumber': '1',
        'LocNumber': [str(loc_id) for loc_id in range(1, num_locations + 1)],
        'BuildingTIV': 1000.,
        'ContentsTIV': 100.,
    })


def make_keys(rows):
    """Keys rows given as (loc_id, coverage_type_id, areaperil_id, vulnerability_id, section_id)."""
    keys = pd.DataFrame(rows, columns=['loc_id', 'coverage_type_id', 'areaperil_id', 'vulnerability_id', 'section_id'])
    keys['peril_id'] = 'WTC'
    keys['status'] = 'success'
    return keys


def sections_of(gul_inputs):
    return sorted(prepare_sections_df(gul_inputs)['section_id'].tolist())


def test_one_row_per_item_keeps_sections_unchanged():
    keys = make_keys([(1, 1, 10, 1, 3), (1, 3, 10, 2, 3), (2, 1, 11, 1, 4), (2, 3, 11, 2, 4)])
    gul_inputs = get_gul_input_items(make_exposure(), keys)

    assert len(gul_inputs) == 4
    assert gul_inputs['section_id'].tolist() == [3, 3, 4, 4]
    assert sections_of(gul_inputs) == [3, 4]


def test_rows_differing_only_in_section_are_combined(caplog):
    keys = make_keys([
        (1, 1, 10, 1, 5), (1, 1, 10, 1, 2), (1, 1, 10, 1, 5),
        (1, 3, 10, 2, '7;2'), (1, 3, 10, 2, 9),
        (2, 1, 11, 1, 4),
    ])
    with caplog.at_level(logging.WARNING, logger='oasislmf.preparation.gul_inputs'):
        gul_inputs = get_gul_input_items(make_exposure(), keys)

    assert len(gul_inputs) == 3
    assert gul_inputs['section_id'].astype(str).tolist() == ['2;5', '2;7;9', '4']
    assert sections_of(gul_inputs) == [2, 4, 5, 7, 9]
    assert not caplog.records


def test_rows_differing_in_areaperil_warn_and_keep_the_first(caplog):
    keys = make_keys([(1, 1, 10, 1, 1), (1, 1, 12, 1, 2), (2, 1, 11, 1, 3)])
    with caplog.at_level(logging.WARNING, logger='oasislmf.preparation.gul_inputs'):
        gul_inputs = get_gul_input_items(make_exposure(), keys)

    assert gul_inputs['areaperil_id'].tolist() == [10, 11]
    assert sections_of(gul_inputs) == [1, 2, 3]
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert '1 item(s)' in message
    assert 'loc_id 1 peril WTC coverage type 1 building 1: differing areaperil_id' in message


def test_keys_columns_are_matched_case_insensitively():
    keys = make_keys([(1, 1, 10, 1, 1), (2, 1, 11, 1, 2)]).rename(columns={
        'loc_id': 'LocID', 'peril_id': 'PerilID', 'coverage_type_id': 'CoverageTypeID',
        'areaperil_id': 'AreaPerilID', 'vulnerability_id': 'VulnerabilityID',
    })
    gul_inputs = get_gul_input_items(make_exposure(), keys)

    assert gul_inputs['loc_id'].tolist() == [1, 2]
    assert gul_inputs['areaperil_id'].tolist() == [10, 11]


@pytest.mark.parametrize('which', ['keys', 'location'])
def test_missing_loc_id_names_the_frame_and_its_columns(which):
    exposure = make_exposure()
    keys = make_keys([(1, 1, 10, 1, 1)])
    if which == 'keys':
        keys = keys.rename(columns={'loc_id': 'locnumber'})
    else:
        exposure = exposure.drop(columns='loc_id')

    with pytest.raises(OasisException, match=f"The {which} dataframe has no 'loc_id' column"):
        get_gul_input_items(exposure, keys)


def test_empty_join_reports_both_sides_loc_ids():
    keys = make_keys([(7, 1, 10, 1, 1), (8, 1, 11, 1, 2)])
    with pytest.raises(OasisException, match=r"matched no row.*keys loc_id values: \[7, 8\].*location loc_id values: \[1, 2\]"):
        get_gul_input_items(make_exposure(), keys)
