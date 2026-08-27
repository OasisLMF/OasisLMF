"""
Unit tests for the early referential-integrity checks in oasislmf.preparation.il_inputs:
validate_account_location_references() and check_cond_tags().
"""
from unittest import TestCase, main

import pandas as pd

from oasislmf.preparation.il_inputs import (
    check_cond_tags,
    get_cond_info,
    validate_account_location_references,
)
from oasislmf.utils.exceptions import OasisException


def make_locations(**overrides):
    data = {
        'LocNumber': [1, 2, 3],
        'PortNumber': ['1', '1', '1'],
        'AccNumber': ['A1', 'A1', 'A2'],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def make_accounts(**overrides):
    data = {
        'PortNumber': ['1', '1'],
        'AccNumber': ['A1', 'A2'],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestValidateAccountLocationReferences(TestCase):
    def test_none_inputs_are_a_no_op(self):
        # should not raise regardless of which side is missing (e.g. GUL-only portfolios)
        validate_account_location_references(None, None)
        validate_account_location_references(make_locations(), None)
        validate_account_location_references(None, make_accounts())

    def test_valid_data_passes_without_mutating_inputs(self):
        locations_df = make_locations()
        accounts_df = make_accounts()
        loc_snapshot = locations_df.copy(deep=True)
        acc_snapshot = accounts_df.copy(deep=True)

        validate_account_location_references(locations_df, accounts_df)

        pd.testing.assert_frame_equal(locations_df, loc_snapshot)
        pd.testing.assert_frame_equal(accounts_df, acc_snapshot)

    def test_valid_data_with_preexisting_acc_id_column(self):
        # exercises the "acc_id already present" branches on both frames, instead of
        # computing it via get_ids()/merge
        accounts_df = make_accounts(acc_id=[1, 2])
        locations_df = make_locations(acc_id=[1, 1, 2])

        validate_account_location_references(locations_df, accounts_df)

    def test_missing_account_reference_raises(self):
        locations_df = make_locations(AccNumber=['A1', 'A1', 'A9'])
        accounts_df = make_accounts()

        with self.assertRaises(OasisException) as ctx:
            validate_account_location_references(locations_df, accounts_df)

        self.assertIn('PortNumber/AccNumber combination', str(ctx.exception))
        self.assertIn('A9', str(ctx.exception))
        self.assertIn('total=1', str(ctx.exception))

    def test_duplicate_acc_id_mapping_raises(self):
        # a pre-existing (corrupted) acc_id column where the same PortNumber/AccNumber
        # combination maps to two different acc_id values
        locations_df = make_locations(AccNumber=['A1', 'A1', 'A1'])
        accounts_df = make_accounts(AccNumber=['A1', 'A1'], acc_id=[10, 99])

        with self.assertRaises(OasisException) as ctx:
            validate_account_location_references(locations_df, accounts_df)

        self.assertIn('more than one acc_id', str(ctx.exception))
        self.assertIn('total=2', str(ctx.exception))

    def test_valid_matching_condtag_on_both_sides_does_not_raise(self):
        # exercises the CondTag branches in validate_account_location_references itself
        # (as opposed to check_cond_tags directly), on both the location and account side
        locations_df = make_locations(acc_id=[1, 1, 2], CondTag=['C1', '0', '0'])
        accounts_df = make_accounts(acc_id=[1, 2], CondTag=['C1', ''])

        validate_account_location_references(locations_df, accounts_df)

    def test_missing_condtag_reference_raises(self):
        locations_df = make_locations(CondTag=['C1', '0', '0'])
        accounts_df = make_accounts()

        with self.assertRaises(OasisException) as ctx:
            validate_account_location_references(locations_df, accounts_df)

        self.assertIn('condtag', str(ctx.exception))
        self.assertIn('C1', str(ctx.exception))


class TestCheckCondTags(TestCase):
    def test_no_condtag_columns_does_not_raise(self):
        check_cond_tags(make_locations(acc_id=[1, 1, 2]), make_accounts(acc_id=[1, 2]))

    def test_condtag_in_locations_only_raises(self):
        locations_df = make_locations(acc_id=[1, 1, 2], CondTag=['C1', '0', '0'])
        accounts_df = make_accounts(acc_id=[1, 2])

        with self.assertRaises(OasisException):
            check_cond_tags(locations_df, accounts_df)

    def test_matching_condtag_on_both_sides_does_not_raise(self):
        locations_df = make_locations(acc_id=[1, 1, 2], CondTag=['C1', '0', '0'])
        accounts_df = make_accounts(acc_id=[1, 2], CondTag=['C1', ''])

        check_cond_tags(locations_df, accounts_df)

    def test_condtag_missing_from_accounts_condtag_column_raises(self):
        # accounts_df declares a CondTag column, but not the one referenced by the location
        locations_df = make_locations(acc_id=[1, 1, 2], CondTag=['C1', '0', '0'])
        accounts_df = make_accounts(acc_id=[1, 2], CondTag=['C2', ''])

        with self.assertRaises(OasisException):
            check_cond_tags(locations_df, accounts_df)


class TestGetCondInfo(TestCase):
    def test_get_cond_info_still_calls_check_cond_tags_first(self):
        # get_cond_info() is only ever reached when the account file already declares
        # both CondTag and CondNumber (see get_levels' gating condition), so build minimal
        # fixtures for that shape directly, rather than depending on end-to-end fixtures.
        locations_df = pd.DataFrame({
            'loc_id': [1, 2],
            'acc_id': [10, 10],
            'CondTag': ['C1', '0'],
        })
        accounts_df = pd.DataFrame({
            'acc_id': [10],
            'layer_id': [1],
            'PolNumber': ['P1'],
            'LayerNumber': [1],
            'acc_idx': [0],
            'CondTag': ['C1'],
            'CondNumber': [1],
        })

        level_conds, extra_accounts = get_cond_info(locations_df, accounts_df)

        self.assertEqual(level_conds, {1: {(10, '0'), (10, 'C1')}})

    def test_get_cond_info_raises_via_check_cond_tags_on_mismatch(self):
        locations_df = pd.DataFrame({
            'loc_id': [1],
            'acc_id': [10],
            'CondTag': ['C9'],
        })
        accounts_df = pd.DataFrame({
            'acc_id': [10],
            'layer_id': [1],
            'PolNumber': ['P1'],
            'LayerNumber': [1],
            'acc_idx': [0],
            'CondTag': ['C1'],
            'CondNumber': [1],
        })

        with self.assertRaises(OasisException):
            get_cond_info(locations_df, accounts_df)


if __name__ == "__main__":
    main()
