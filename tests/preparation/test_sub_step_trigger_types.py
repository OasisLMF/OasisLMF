"""Pin the sub-step trigger type lookup against the row-wise assignment it replaced."""

import numpy as np
import pandas as pd
import pytest

from oasislmf.preparation.il_inputs import get_sub_step_trigger_types
from oasislmf.utils.fm import STEP_TRIGGER_TYPES

COVERAGE_TYPE_IDS = [1, 2, 3, 4, 5]
STEP_TRIGGER_TYPE_IDS = [0, 1, 2, 3, 4, 5, 6]


def reference_sub_step_trigger_type(row):
    """The row-wise assignment replaced by the get_sub_step_trigger_types lookup."""
    try:
        return STEP_TRIGGER_TYPES[row['steptriggertype']]['sub_step_trigger_types'][row['coverage_type_id']]
    except KeyError:
        return row['steptriggertype']


def assign_sub_step_trigger_types(level_df):
    """Assign the sub-step trigger types the way assign_level_calcrule_and_profile_ids does."""
    step_rows = level_df[['steptriggertype', 'coverage_type_id']]
    sub_step_trigger_type = get_sub_step_trigger_types().reindex(pd.MultiIndex.from_frame(step_rows))
    has_sub_type = sub_step_trigger_type.notna().to_numpy()

    assigned = level_df.copy()
    assigned.loc[step_rows.index[has_sub_type], 'steptriggertype'] = pd.Series(
        sub_step_trigger_type.to_numpy()[has_sub_type],
        index=step_rows.index[has_sub_type], dtype=assigned['steptriggertype'].dtype)

    return assigned['steptriggertype']


@pytest.mark.parametrize('dtype', ['int64', 'int32', 'float64'])
def test_matches_the_row_wise_assignment(dtype):
    rng = np.random.default_rng(3)
    level_df = pd.DataFrame({
        'steptriggertype': rng.choice(STEP_TRIGGER_TYPE_IDS, 200).astype(dtype),
        'coverage_type_id': rng.choice(COVERAGE_TYPE_IDS, 200).astype(dtype),
    })

    expected = level_df.apply(reference_sub_step_trigger_type, axis=1)

    np.testing.assert_array_equal(
        assign_sub_step_trigger_types(level_df).to_numpy().astype('int64'),
        expected.to_numpy().astype('int64'),
    )


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
