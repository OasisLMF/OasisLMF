"""Pin the sub-step trigger type lookup against the row-wise assignment it replaced."""

import numpy as np
import pandas as pd
import pytest

from oasislmf.preparation.il_inputs import assign_sub_step_trigger_types as _assign_sub_step_trigger_types
from oasislmf.utils.fm import STEP_TRIGGER_TYPES

COVERAGE_TYPE_IDS = [1, 2, 3, 4, 5]
STEP_TRIGGER_TYPE_IDS = [0, 1, 2, 3, 4, 5, 6]


def reference_sub_step_trigger_type(row):
    """The row-wise assignment replaced by the get_sub_step_trigger_types lookup."""
    try:
        return STEP_TRIGGER_TYPES[row['steptriggertype']]['sub_step_trigger_types'][row['coverage_type_id']]
    except KeyError:
        return row['steptriggertype']


def assign_sub_step_trigger_types(level_df, step_filter=None):
    """Call the assignment, so a regression in it fails these tests.

    Only the reference implementation is duplicated here; the code under test is imported.
    """
    if step_filter is None:
        step_filter = pd.Series(True, index=level_df.index)

    return _assign_sub_step_trigger_types(level_df.copy(), step_filter)['steptriggertype']


# Int32 is what the real pipeline carries: coverage_type_id_df gives steptriggertype a nullable
# extension dtype, which numpy's astype cannot interpret
@pytest.mark.parametrize('dtype', ['int64', 'int32', 'float64', 'object', 'Int32', 'Int64'])
def test_matches_the_row_wise_assignment(dtype):
    rng = np.random.default_rng(3)
    level_df = pd.DataFrame({
        'steptriggertype': pd.Series(rng.choice(STEP_TRIGGER_TYPE_IDS, 200)).astype(dtype),
        'coverage_type_id': pd.Series(rng.choice(COVERAGE_TYPE_IDS, 200)).astype(dtype),
    })

    expected = level_df.apply(reference_sub_step_trigger_type, axis=1)
    assigned = assign_sub_step_trigger_types(level_df)

    np.testing.assert_array_equal(
        assigned.to_numpy().astype('int64'), expected.to_numpy().astype('int64'))
    assert assigned.dtype == level_df['steptriggertype'].dtype


def test_an_object_column_keeps_integers_rather_than_floats():
    """The calc rules table is merged on steptriggertype, and 1.0 does not match the key 1.

    Reindexing onto the MultiIndex introduces NaN, which promotes the lookup to float64. Masking
    the NaN away afterwards does not convert the surviving values back, so an object column would
    silently take floats and the merge would return a NaN calcrule_id for those rows.
    """
    level_df = pd.DataFrame({
        'steptriggertype': pd.Series([5, 5, 4], dtype=object),
        'coverage_type_id': pd.Series([1, 2, 1], dtype=object),
    })

    assigned = assign_sub_step_trigger_types(level_df)

    assert assigned.dtype == object
    assert not any(isinstance(value, float) for value in assigned), assigned.to_list()
    # (5, 1) has sub-type 1; (5, 2) has none so keeps 5; (4, 1) has none so keeps 4
    np.testing.assert_array_equal(assigned.to_numpy().astype('int64'), [1, 5, 4])


def test_a_duplicated_index_is_assigned_positionally():
    """level_df's index is not guaranteed unique, and reindexing onto duplicate labels raises."""
    level_df = pd.DataFrame(
        {'steptriggertype': [5, 5, 5, 1], 'coverage_type_id': [1, 2, 3, 4]}, index=[0, 0, 1, 1])

    assigned = assign_sub_step_trigger_types(level_df)

    # (5, 1) -> 1; (5, 2) has no sub-type so keeps 5; (5, 3) -> 2; (1, 4) -> 0
    np.testing.assert_array_equal(assigned.to_numpy(), [1, 5, 2, 0])


def test_only_the_filtered_rows_are_assigned():
    """Non-step rows keep their trigger type even when the pair has a sub-type."""
    level_df = pd.DataFrame({'steptriggertype': [5, 5], 'coverage_type_id': [1, 2]})
    step_filter = pd.Series([True, False], index=level_df.index)

    assigned = assign_sub_step_trigger_types(level_df, step_filter)

    np.testing.assert_array_equal(assigned.to_numpy(), [1, 5])


def test_covers_every_declared_pair():
    pairs = [(step_trigger_type, coverage_type_id)
             for step_trigger_type, info in STEP_TRIGGER_TYPES.items()
             for coverage_type_id in info['sub_step_trigger_types']]
    level_df = pd.DataFrame(pairs, columns=['steptriggertype', 'coverage_type_id'])

    expected = level_df.apply(reference_sub_step_trigger_type, axis=1)

    np.testing.assert_array_equal(assign_sub_step_trigger_types(level_df).to_numpy(), expected.to_numpy())
    # every declared pair has a sub-type, so none of them keeps the trigger type it came in with
    assert (assign_sub_step_trigger_types(level_df).to_numpy() != level_df['steptriggertype'].to_numpy()).any()


def test_a_trigger_type_without_a_sub_type_is_left_alone():
    level_df = pd.DataFrame({'steptriggertype': [4, 6, 5], 'coverage_type_id': [1, 1, 2]})

    np.testing.assert_array_equal(assign_sub_step_trigger_types(level_df).to_numpy(), [4, 6, 5])


def test_a_missing_trigger_type_is_left_alone():
    level_df = pd.DataFrame({'steptriggertype': [5.0, np.nan], 'coverage_type_id': [1.0, 1.0]})

    assigned = assign_sub_step_trigger_types(level_df).to_numpy()
    assert assigned[0] == 1
    assert np.isnan(assigned[1])
