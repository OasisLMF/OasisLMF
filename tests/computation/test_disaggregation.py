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

    def test_nothing_given_is_samples(self):
        """'samples' is the default, and the API agrees with the command line.

        It was 'items' -- what do_disaggregation defaulted to -- until packing made 'samples' the
        better representation of a multi-building location. The deprecated boolean still names the
        two modes it could, below.
        """
        self.assertEqual(resolve_disaggregation(None, None), 'samples')

    def test_the_deprecated_boolean_is_still_honoured(self):
        for do_disaggregation, expected in ((True, 'items'), (False, 'none')):
            with self.subTest(do_disaggregation=do_disaggregation):
                with self.assertWarns(FutureWarning):
                    self.assertEqual(resolve_disaggregation(None, do_disaggregation), expected)

    def test_the_new_parameter_wins_and_says_so(self):
        with self.assertWarns(FutureWarning) as caught:
            self.assertEqual(resolve_disaggregation('samples', True), 'samples')
        self.assertIn('ignored', str(caught.warning))

    def test_the_new_parameter_alone_is_not_deprecated(self):
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            resolve_disaggregation('samples', None)
        self.assertEqual(raised, [])

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(OasisException) as caught:
            resolve_disaggregation('sideways', None)
        for mode in DISAGGREGATION_MODES:
            self.assertIn(mode, str(caught.exception))
