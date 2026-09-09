"""Tests for the disaggregation parameter and the deprecation of do_disaggregation.

``disaggregation`` names where a location's ``NumberOfBuildings`` is separated -- nowhere, into
separate items, or into separate samples. It replaces the boolean ``do_disaggregation``, which
could only express the first two and so had no way to ask for the sample dimension.
"""
import logging
from unittest import TestCase

from oasislmf.utils.data import resolve_disaggregation
from oasislmf.utils.defaults import DISAGGREGATION_MODES
from oasislmf.utils.exceptions import OasisException


class _CapturingLogger(logging.Logger):
    def __init__(self):
        super().__init__('capture')
        self.messages = []

    def warning(self, msg, *args, **kwargs):
        self.messages.append(msg % args if args else msg)

    def info(self, msg, *args, **kwargs):
        self.messages.append(msg % args if args else msg)


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
                log = _CapturingLogger()
                self.assertEqual(resolve_disaggregation(None, do_disaggregation, logger=log), expected)
                self.assertTrue(any('deprecated' in m for m in log.messages))

    def test_the_new_parameter_wins_and_says_so(self):
        log = _CapturingLogger()
        self.assertEqual(resolve_disaggregation('samples', True, logger=log), 'samples')
        self.assertTrue(any('ignored' in m for m in log.messages))

    def test_the_new_parameter_alone_is_not_deprecated(self):
        log = _CapturingLogger()
        resolve_disaggregation('samples', None, logger=log)
        self.assertEqual(log.messages, [])

    def test_the_deprecated_packing_boolean_maps_to_samples(self):
        """building_packing was the unreleased second boolean; still accepted, still warns."""
        log = _CapturingLogger()
        self.assertEqual(
            resolve_disaggregation(None, None, building_packing=True, logger=log), 'samples')
        self.assertTrue(any('deprecated' in m for m in log.messages))

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(OasisException) as caught:
            resolve_disaggregation('sideways', None)
        for mode in DISAGGREGATION_MODES:
            self.assertIn(mode, str(caught.exception))
