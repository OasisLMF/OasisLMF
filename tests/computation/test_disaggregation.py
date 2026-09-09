"""Tests for the disaggregation parameter and the deprecation of do_disaggregation.

``disaggregation`` names where a location's ``NumberOfBuildings`` is separated -- nowhere, into
separate items, or into separate samples. It replaces the boolean ``do_disaggregation``, which
could only express the first two and so had no way to ask for the sample dimension.
"""
import warnings
from unittest import TestCase

from oasislmf.utils.data import resolve_disaggregation
from oasislmf.utils.defaults import DISAGGREGATION_MODES
from oasislmf.utils.exceptions import OasisException


class TestResolveDisaggregation(TestCase):

    def test_each_mode_passes_through_unchanged(self):
        """One string is threaded all the way through generation -- no re-encoding."""
        for mode in DISAGGREGATION_MODES:
            with self.subTest(mode=mode):
                self.assertEqual(resolve_disaggregation(mode, None), mode)

    def test_nothing_given_keeps_the_historical_default(self):
        """One item per building, which is what do_disaggregation defaulted to."""
        self.assertEqual(resolve_disaggregation(None, None), 'items')

    def test_the_deprecated_boolean_is_still_honoured(self):
        for do_disaggregation, expected in ((True, 'items'), (False, 'none')):
            with self.subTest(do_disaggregation=do_disaggregation):
                with self.assertWarns(DeprecationWarning):
                    self.assertEqual(resolve_disaggregation(None, do_disaggregation), expected)

    def test_the_new_parameter_wins_and_says_so(self):
        with self.assertWarns(DeprecationWarning) as caught:
            self.assertEqual(resolve_disaggregation('samples', True), 'samples')
        self.assertIn('ignored', str(caught.warning))

    def test_the_new_parameter_alone_is_not_deprecated(self):
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            resolve_disaggregation('samples', None)
        self.assertEqual(raised, [])

    def test_the_deprecated_packing_boolean_maps_to_samples(self):
        """building_packing was the unreleased second boolean; still accepted, still warns."""
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(
                resolve_disaggregation(None, None, building_packing=True), 'samples')

    def test_the_warning_survives_a_hostile_filter(self):
        """DeprecationWarning is ignored by default outside __main__, and this fires from inside
        the computation layer, so the notice has to force its own visibility."""
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("ignore")
            resolve_disaggregation(None, True)
        self.assertEqual(len(raised), 1)
        self.assertIs(raised[0].category, DeprecationWarning)

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(OasisException) as caught:
            resolve_disaggregation('sideways', None)
        for mode in DISAGGREGATION_MODES:
            self.assertIn(mode, str(caught.exception))
