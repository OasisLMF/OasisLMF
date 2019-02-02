# -*- coding: utf-8 -*-

import io
import json

from unittest import TestCase

import pandas as pd
import pytest

from hypothesis import (
    given,
    HealthCheck,
    reproduce_failure,
    settings,
)

from hypothesis.strategies import (
    dictionaries,
    integers,
    floats,
    just,
    lists,
    sampled_from,
    text,
    tuples,
)

from mock import patch, Mock

from tempfile import NamedTemporaryFile

from oasislmf.exposures.manager import OasisExposuresManager
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.fm import (
    unified_canonical_fm_profile_by_level,
    unified_canonical_fm_profile_by_level_and_term_group,
    get_layer_calcrule_id,
    get_layer_level_fm_terms,
    get_sub_layer_calcrule_id,
    get_coverage_level_fm_terms,
    get_sub_layer_non_coverage_level_fm_terms,
    get_policytc_ids,
)
from oasislmf.utils.metadata import (
    DEDUCTIBLE_TYPES,
    FM_TERMS,
    OASIS_COVERAGE_TYPES,
    OASIS_FM_LEVELS,
    OASIS_KEYS_STATUS,
    OASIS_PERILS,
    OED_COVERAGE_TYPES,
    OED_FM_LEVELS,
    OED_PERILS,
)
from tests.data import (
    calcrule_ids,
    canonical_accounts,
    canonical_accounts_profile,
    canonical_exposure,
    canonical_exposure_profile,
    canonical_oed_accounts,
    canonical_oed_accounts_profile,
    canonical_oed_exposure,
    canonical_oed_exposure_profile,
    oasis_fm_agg_profile,
    deductible_types,
    fm_input_items,
    fm_levels_simple,
)

class CanonicalProfilesFmTermsGroupedByLevel(TestCase):

    def setUp(self):
        self.exposure_profile = canonical_exposure_profile
        self.accounts_profile = canonical_accounts_profile

    # Adapted from a solution by Martijn Pieters
    # https://stackoverflow.com/a/23499088
    def _depth(self, d, level=1):
        if not d or not isinstance(d, dict):
            return level
        return max(self._depth(d[k], level + 1) for k in d)

    def test_no_canonical_profiles_or_profiles_paths_provided__oasis_exception_is_raised(self):
        with self.assertRaises(OasisException):
            unified_canonical_fm_profile_by_level()

    #@pytest.mark.skip(reason="inconsistent output from unified canonical profile constructor")
    def test_only_canonical_profiles_provided(self):

        profiles = (self.exposure_profile, self.accounts_profile,)

        cpft = unified_canonical_fm_profile_by_level(profiles=profiles)

        self.assertEqual(self._depth(cpft), 4)

        fm_levels = set(p[l].get('FMLevel') for p in profiles for l in p if p[l].get('FMLevel'))

        self.assertEqual(fm_levels, set(cpft.keys()))

        for l in fm_levels:
            for _, v in cpft[l].items():
                self.assertTrue(set(v.keys()).issuperset(['FMLevel', 'FMLevelName', 'FMTermGroupID', 'FMTermType']))
                self.assertEqual(l, v['FMLevel'])

        matching_profile_term = lambda t: (
            [
                cpft[l][_t] for l in cpft for _t in cpft[l] if t.lower() == _t.lower()
            ][0] if [cpft[l][_t] for l in cpft for _t in cpft[l] if t.lower() == _t.lower()]
            else None
        )

        non_fm_terms = set(t for p in profiles for t in p if 'FMLevel' not in p[t])

        for t in (_t for p in profiles for _t in p):
            pt = matching_profile_term(t)
            self.assertIsNotNone(pt) if t not in non_fm_terms else self.assertIsNone(pt)


    def test_only_canonical_profile_paths_provided(self):
        profiles = (self.exposure_profile, self.accounts_profile,)

        with NamedTemporaryFile('w') as exposure_profile_file, NamedTemporaryFile('w') as accounts_profile_file:
            with io.open(exposure_profile_file.name, 'w', encoding='utf-8') as f1, io.open(accounts_profile_file.name, 'w', encoding='utf-8') as f2:
                f1.write(u'{}'.format(json.dumps(self.exposure_profile)))
                f2.write(u'{}'.format(json.dumps(self.accounts_profile)))

            paths = (exposure_profile_file.name, accounts_profile_file.name,)

            cpft = unified_canonical_fm_profile_by_level(profile_paths=paths)

        self.assertEqual(self._depth(cpft), 4)

        fm_levels = set(p[l].get('FMLevel') for p in profiles for l in p if p[l].get('FMLevel'))

        self.assertEqual(fm_levels, set(cpft.keys()))

        for l in fm_levels:
            for _, v in cpft[l].items():
                self.assertTrue(set(v.keys()).issuperset(['FMLevel', 'FMLevelName', 'FMTermGroupID', 'FMTermType']))
                self.assertEqual(l, v['FMLevel'])

        matching_profile_term = lambda t: (
            [
                cpft[l][_t] for l in cpft for _t in cpft[l] if t.lower() == _t.lower()
            ][0] if [cpft[l][_t] for l in cpft for _t in cpft[l] if t.lower() == _t.lower()]
            else None
        )

        non_fm_terms = set(t for p in profiles for t in p if 'FMLevel' not in p[t])

        for t in (_t for p in profiles for _t in p):
            pt = matching_profile_term(t)
            self.assertIsNotNone(pt) if t not in non_fm_terms else self.assertIsNone(pt)


    def test_canonical_profile_and_profiles_paths_provided(self):
        profiles = (self.exposure_profile, self.accounts_profile,)

        with NamedTemporaryFile('w') as exposure_profile_file, NamedTemporaryFile('w') as accounts_profile_file:
            with io.open(exposure_profile_file.name, 'w', encoding='utf-8') as f1, io.open(accounts_profile_file.name, 'w', encoding='utf-8') as f2:
                f1.write(u'{}'.format(json.dumps(self.exposure_profile)))
                f2.write(u'{}'.format(json.dumps(self.accounts_profile)))

            paths = (exposure_profile_file.name, accounts_profile_file.name,)

            cpft = unified_canonical_fm_profile_by_level(profiles=profiles, profile_paths=paths)

        self.assertEqual(self._depth(cpft), 4)

        fm_levels = set(p[l].get('FMLevel') for p in profiles for l in p if p[l].get('FMLevel'))

        self.assertEqual(fm_levels, set(cpft.keys()))

        for l in fm_levels:
            for _, v in cpft[l].items():
                self.assertTrue(set(v.keys()).issuperset(['FMLevel', 'FMLevelName', 'FMTermGroupID', 'FMTermType']))
                self.assertEqual(l, v['FMLevel'])

        matching_profile_term = lambda t: (
            [
                cpft[l][_t] for l in cpft for _t in cpft[l] if t.lower() == _t.lower()
            ][0] if [cpft[l][_t] for l in cpft for _t in cpft[l] if t.lower() == _t.lower()]
            else None
        )

        non_fm_terms = set(t for p in profiles for t in p if 'FMLevel' not in p[t])

        for t in (_t for p in profiles for _t in p):
            pt = matching_profile_term(t)
            self.assertIsNotNone(pt) if t not in non_fm_terms else self.assertIsNone(pt)


class CanonicalProfilesFmTermsGroupedByLevelAndTermGroup(TestCase):

    def setUp(self):
        self.exposure_profile = canonical_exposure_profile
        self.accounts_profile = canonical_accounts_profile

    # Adapted from a solution by Martijn Pieters
    # https://stackoverflow.com/a/23499088
    def _depth(self, d, level=1):
        if not d or not isinstance(d, dict):
            return level
        return max(self._depth(d[k], level + 1) for k in d)

    def test_no_canonical_profiles_or_profile_paths_provided__oasis_exception_is_raised(self):
        with self.assertRaises(OasisException):
            unified_canonical_fm_profile_by_level_and_term_group()

    def test_only_canonical_profiles_provided(self):
        profiles = (self.exposure_profile, self.accounts_profile,)

        cpft = unified_canonical_fm_profile_by_level_and_term_group(profiles=profiles)

        self.assertEqual(self._depth(cpft), 5)

        fm_levels = set(p[l].get('FMLevel') for p in profiles for l in p if p[l].get('FMLevel'))

        self.assertEqual(fm_levels, set(cpft.keys()))

        for l in fm_levels:
            for gid in cpft[l]:
                for tty, v in cpft[l][gid].items():
                    self.assertTrue(set(v.keys()).issuperset(['FMLevel', 'FMLevelName', 'FMTermGroupID', 'FMTermType']))
                    self.assertEqual(l, v['FMLevel'])
                    self.assertEqual(gid, v['FMTermGroupID'])
                    self.assertEqual(tty, v['FMTermType'].lower())

        matching_profile_term = lambda t: (
            [
                cpft[l][gid][tty] for l in cpft for gid in cpft[l] for tty in cpft[l][gid] for _t in cpft[l][gid][tty] if t.lower() == cpft[l][gid][tty]['ProfileElementName'].lower()
            ][0] if [cpft[l][gid][tty] for l in cpft for gid in cpft[l] for tty in cpft[l][gid] for _t in cpft[l][gid][tty] if t.lower() == cpft[l][gid][tty]['ProfileElementName'].lower()]
            else None
        )

        non_fm_terms = set(t for p in profiles for t in p if 'FMLevel' not in p[t])

        for t in (_t for p in profiles for _t in p):
            pt = matching_profile_term(t)
            self.assertIsNotNone(pt) if t not in non_fm_terms else self.assertIsNone(pt)

    def test_only_canonical_profile_paths_provided(self):
        profiles = (self.exposure_profile, self.accounts_profile,)

        with NamedTemporaryFile('w') as exposure_profile_file, NamedTemporaryFile('w') as accounts_profile_file:
            with io.open(exposure_profile_file.name, 'w', encoding='utf-8') as f1, io.open(accounts_profile_file.name, 'w', encoding='utf-8') as f2:
                f1.write(u'{}'.format(json.dumps(self.exposure_profile)))
                f2.write(u'{}'.format(json.dumps(self.accounts_profile)))

            paths = (exposure_profile_file.name, accounts_profile_file.name,)

            cpft = unified_canonical_fm_profile_by_level_and_term_group(profile_paths=paths)

        self.assertEqual(self._depth(cpft), 5)

        fm_levels = set(p[l].get('FMLevel') for p in profiles for l in p if p[l].get('FMLevel'))

        for l in fm_levels:
            for gid in cpft[l]:
                for tty, v in cpft[l][gid].items():
                    self.assertTrue(set(v.keys()).issuperset(['FMLevel', 'FMLevelName', 'FMTermGroupID', 'FMTermType']))
                    self.assertEqual(l, v['FMLevel'])
                    self.assertEqual(gid, v['FMTermGroupID'])
                    self.assertEqual(tty, v['FMTermType'].lower())

        matching_profile_term = lambda t: (
            [
                cpft[l][gid][tty] for l in cpft for gid in cpft[l] for tty in cpft[l][gid] for _t in cpft[l][gid][tty] if t.lower() == cpft[l][gid][tty]['ProfileElementName'].lower()
            ][0] if [cpft[l][gid][tty] for l in cpft for gid in cpft[l] for tty in cpft[l][gid] for _t in cpft[l][gid][tty] if t.lower() == cpft[l][gid][tty]['ProfileElementName'].lower()]
            else None
        )

        self.assertEqual(fm_levels, set(cpft.keys()))

        non_fm_terms = set(t for p in profiles for t in p if 'FMLevel' not in p[t])

        for t in (_t for p in profiles for _t in p):
            pt = matching_profile_term(t)
            self.assertIsNotNone(pt) if t not in non_fm_terms else self.assertIsNone(pt)

    def test_canonical_profiles_and_profile_paths_provided(self):
        profiles = (self.exposure_profile, self.accounts_profile,)

        with NamedTemporaryFile('w') as exposure_profile_file, NamedTemporaryFile('w') as accounts_profile_file:
            with io.open(exposure_profile_file.name, 'w', encoding='utf-8') as f1, io.open(accounts_profile_file.name, 'w', encoding='utf-8') as f2:
                f1.write(u'{}'.format(json.dumps(self.exposure_profile)))
                f2.write(u'{}'.format(json.dumps(self.accounts_profile)))

            paths = (exposure_profile_file.name, accounts_profile_file.name,)

            cpft = unified_canonical_fm_profile_by_level_and_term_group(profiles=profiles, profile_paths=paths)

        self.assertEqual(self._depth(cpft), 5)

        fm_levels = set(p[l].get('FMLevel') for p in profiles for l in p if p[l].get('FMLevel'))

        for l in fm_levels:
            for gid in cpft[l]:
                for tty, v in cpft[l][gid].items():
                    self.assertTrue(set(v.keys()).issuperset(['FMLevel', 'FMLevelName', 'FMTermGroupID', 'FMTermType']))
                    self.assertEqual(l, v['FMLevel'])
                    self.assertEqual(gid, v['FMTermGroupID'])
                    self.assertEqual(tty, v['FMTermType'].lower())

        matching_profile_term = lambda t: (
            [
                cpft[l][gid][tty] for l in cpft for gid in cpft[l] for tty in cpft[l][gid] for _t in cpft[l][gid][tty] if t.lower() == cpft[l][gid][tty]['ProfileElementName'].lower()
            ][0] if [cpft[l][gid][tty] for l in cpft for gid in cpft[l] for tty in cpft[l][gid] for _t in cpft[l][gid][tty] if t.lower() == cpft[l][gid][tty]['ProfileElementName'].lower()]
            else None
        )

        self.assertEqual(fm_levels, set(cpft.keys()))

        non_fm_terms = set(t for p in profiles for t in p if 'FMLevel' not in p[t])

        for t in (_t for p in profiles for _t in p):
            pt = matching_profile_term(t)
            self.assertIsNotNone(pt) if t not in non_fm_terms else self.assertIsNone(pt)


class TestSubLayerCalcruleIDFunc(TestCase):

    @given(
        deductible=just(0.0),
        deductible_code=just(0),
        deductible_min=just(0.0),
        deductible_max=just(0.0),
        limit=just(0.0),
        limit_code=just(0)
    )
    def test_calcrule_id_12(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):

        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 12)

    @given(
        deductible=floats(min_value=1, allow_infinity=False),
        deductible_code=just(0),
        deductible_min=just(0.0),
        deductible_max=just(0.0),
        limit=floats(min_value=1, allow_infinity=False),
        limit_code=just(0)
    )
    def test_calcrule_id_1(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 1)

    @given(
        deductible=floats(min_value=0.001, max_value=0.99),
        deductible_code=just(2),
        deductible_min=just(0.0),
        deductible_max=just(0.0),
        limit=floats(min_value=1, allow_infinity=False),
        limit_code=just(0)
    )
    def test_calcrule_id_4(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 4)

    @given(
        deductible=floats(min_value=0.001, max_value=0.99),
        deductible_code=just(1),
        deductible_min=just(0.0),
        deductible_max=just(0.0),
        limit=floats(min_value=0.001, max_value=0.99),
        limit_code=just(1)
    )
    def test_calcrule_id_5(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 5)

    @given(
        deductible=floats(min_value=0.001, max_value=0.99),
        deductible_code=just(2),
        deductible_min=just(0.0),
        deductible_max=just(0.0),
        limit=just(0.0),
        limit_code=just(0)
    )
    def test_calcrule_id_6(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 6)

    @given(
        deductible=just(0.0),
        deductible_code=just(0),
        deductible_min=just(0.0),
        deductible_max=floats(min_value=1, allow_infinity=False),
        limit=floats(min_value=1, allow_infinity=False),
        limit_code=just(0)
    )
    def test_calcrule_id_7(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 7)

    @given(
        deductible=just(0.0),
        deductible_code=just(0),
        deductible_min=floats(min_value=1, allow_infinity=False),
        deductible_max=just(0.0),
        limit=floats(min_value=1, allow_infinity=False),
        limit_code=just(0)
    )
    def test_calcrule_id_8(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 8)

    @given(
        deductible=just(0.0),
        deductible_code=just(0),
        deductible_min=just(0.0),
        deductible_max=floats(min_value=1, allow_infinity=False),
        limit=just(0.0),
        limit_code=just(0)
    )
    def test_calcrule_id_10(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 10)

    @given(
        deductible=just(0.0),
        deductible_code=just(0),
        deductible_min=floats(min_value=1, allow_infinity=False),
        deductible_max=just(0.0),
        limit=just(0.0),
        limit_code=just(0)
    )
    def test_calcrule_id_11(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 11)

    @given(
        deductible=floats(min_value=1, allow_infinity=False),
        deductible_code=just(0),
        deductible_min=just(0.0),
        deductible_max=just(0.0),
        limit=just(0.0),
        limit_code=just(0)
    )
    def test_calcrule_id_12(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 12)

    @given(
        deductible=just(0.0),
        deductible_code=just(0),
        deductible_min=floats(min_value=1, allow_infinity=False),
        deductible_max=floats(min_value=1, allow_infinity=False),
        limit=just(0.0),
        limit_code=just(0)
    )
    def test_calcrule_id_13(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 13)

    @given(
        deductible=just(0.0),
        deductible_code=just(0),
        deductible_min=just(0.0),
        deductible_max=just(0.0),
        limit=floats(min_value=1, allow_infinity=False),
        limit_code=just(0)
    )
    def test_calcrule_id_14(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 14)

    @given(
        deductible=just(0.0),
        deductible_code=just(0),
        deductible_min=just(0.0),
        deductible_max=just(0.0),
        limit=floats(min_value=0.001, max_value=0.99),
        limit_code=just(1)
    )
    def test_calcrule_id_15(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 15)

    @given(
        deductible=floats(min_value=0.001, max_value=0.99),
        deductible_code=just(1),
        deductible_min=just(0.0),
        deductible_max=just(0.0),
        limit=just(0.0),
        limit_code=just(0)
    )
    def test_calcrule_id_16(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 16)

    @given(
        deductible=floats(min_value=0.001, max_value=0.99),
        deductible_code=just(1),
        deductible_min=floats(min_value=1, allow_infinity=False),
        deductible_max=floats(min_value=1, allow_infinity=False),
        limit=just(0.0),
        limit_code=just(0)
    )
    def test_calcrule_id_19(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 19)

    @given(
        deductible=floats(min_value=0.001, max_value=0.99),
        deductible_code=just(2),
        deductible_min=floats(min_value=1, allow_infinity=False),
        deductible_max=floats(min_value=1, allow_infinity=False),
        limit=just(0.0),
        limit_code=just(0)
    )
    def test_calcrule_id_21(self, deductible, deductible_code, deductible_min, deductible_max, limit, limit_code):
        self.assertEqual(get_sub_layer_calcrule_id(deductible, deductible_min, deductible_max, limit, deductible_code, limit_code), 21)

class TestLayerCalcruleIDFunc(TestCase):

    @given(
        attachment=just(0),
        limit=just(0),
        share=floats(min_value=0.001, max_value=1)
    )
    def test_calcrule_id_2__with_positive_share(self, attachment, limit, share):
        self.assertEqual(get_layer_calcrule_id(attachment, limit, share), 2)

    @given(
        attachment=just(0),
        limit=floats(min_value=1, allow_infinity=False),
        share=just(0)
    )
    def test_calcrule_id_2__with_positive_limit(self, attachment, limit, share):
        self.assertEqual(get_layer_calcrule_id(attachment, limit, share), 2)

    @given(
        attachment=floats(min_value=1, allow_infinity=False),
        limit=floats(min_value=1, allow_infinity=False),
        share=just(0)
    )
    def test_calcrule_id_2__with_positive_attachment(self, attachment, limit, share):
        self.assertEqual(get_layer_calcrule_id(attachment, limit, share), 2)

    @given(
        attachment=just(0),
        limit=floats(min_value=1, allow_infinity=False),
        share=floats(min_value=1, allow_infinity=False)
    )
    def test_calcrule_id_2__with_positive_limit_and_share(self, attachment, limit, share):
        self.assertEqual(get_layer_calcrule_id(attachment, limit, share), 2)

    @given(
        attachment=floats(min_value=0.001, max_value=1),
        limit=just(0),
        share=floats(min_value=1, allow_infinity=False)
    )
    def test_calcrule_id_2__with_positive_attachment_and_share(self, attachment, limit, share):
        self.assertEqual(get_layer_calcrule_id(attachment, limit, share), 2)

    @given(
        attachment=floats(min_value=0.001, max_value=1),
        limit=floats(min_value=1, allow_infinity=False),
        share=just(0)
    )
    def test_calcrule_id_2__with_positive_attachment_and_limit(self, attachment, limit, share):
        self.assertEqual(get_layer_calcrule_id(attachment, limit, share), 2)

class GetFmTermsByLevel(TestCase):

    def setUp(self):
        self.exposure_profile = canonical_exposure_profile
        self.accounts_profile = canonical_accounts_profile
        self.unified_canonical_profile = unified_canonical_fm_profile_by_level_and_term_group(
            profiles=[self.exposure_profile, self.accounts_profile]
        )
        self.fm_agg_profile = oasis_fm_agg_profile

    #@pytest.mark.flaky
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposure=canonical_exposure(
            from_account_nums=just('A1'),
            from_tivs1=just(100),
            from_tivs2=just(0),
            from_tivs3=just(0),
            from_tivs4=just(0),
            from_deductibles1=just(1),
            from_deductibles2=just(0),
            from_deductibles3=just(0),
            from_deductibles4=just(0),
            from_limits1=just(1),
            from_limits2=just(0),
            from_limits3=just(0),
            from_limits4=just(0),
            size=2
        ),
        accounts=canonical_accounts(
            from_account_nums=just('A1'),
            from_policy_nums=just('A1P1'),
            size=1
        ),
        fm_input_items=fm_input_items(
            from_peril_ids=just(OASIS_PERILS['wind']['id']),
            from_coverage_type_ids=just(OASIS_COVERAGE_TYPES['buildings']['id']),
            from_level_ids=just(1),
            from_canacc_ids=just(0),
            from_policy_nums=just('A1P1'),
            from_layer_ids=just(1),
            from_tiv_elements=just('wscv1val'),
            from_tiv_tgids=just(1),
            from_tivs=just(100),
            from_deductible_elements=just('wscv1ded'),
            from_deductibles=just(1),
            from_min_deductible_elements=just(None),
            from_min_deductibles=just(0),
            from_max_deductible_elements=just(None),
            from_max_deductibles=just(0),
            from_attachments=just(0),
            from_limit_elements=just('wscv1limit'),
            from_limits=just(1),
            from_share_elements=just(None),
            from_shares=just(0),
            size=2
        ) 
    )
    def test_exposure_with_one_coverage_type_and_fm_terms_with_one_account_and_one_top_level_layer_per_account_and_model_lookup_supporting_single_peril_and_coverage_type__all_coverage_level_terms_generated(self, exposure, accounts, fm_input_items):
        lufcp = self.unified_canonical_profile[1]
        lfmaggp = self.fm_agg_profile[1]

        for it in exposure:
            it['cond1name'] = 0

        for i, it in enumerate(fm_input_items):
            it['index'] = i
            it['agg_id'] = i + 1

        results = sorted([r for r in get_coverage_level_fm_terms(
            lufcp,
            lfmaggp,
            {i:it for i, it in enumerate(fm_input_items)},
            pd.DataFrame(data=exposure),
            pd.DataFrame(data=accounts),
            oed=False
        )], key=lambda it: it['item_id'])


        self.assertEqual(len(fm_input_items), len(results))

        self.assertEqual(tuple(it['item_id'] for it in fm_input_items), tuple(r['item_id'] for r in results))

        for i, res in enumerate(results):
            it = fm_input_items[i]

            self.assertEqual(it['level_id'], res['level_id'])
            self.assertEqual(it['index'], res['index'])
            self.assertEqual(it['item_id'], res['item_id'])
            self.assertEqual(it['gul_item_id'], res['gul_item_id'])

            self.assertEqual(it['canexp_id'], res['canexp_id'])
            self.assertEqual(it['canacc_id'], res['canacc_id'])

            self.assertEqual(it['peril_id'], res['peril_id'])
            self.assertEqual(it['coverage_type_id'], res['coverage_type_id'])

            self.assertEqual(it['layer_id'], res['layer_id'])
            self.assertEqual(it['agg_id'], res['agg_id'])

            self.assertEqual(it['tiv_elm'], res['tiv_elm'])
            self.assertEqual(it['tiv_tgid'], res['tiv_tgid'])

            self.assertEqual(it['ded_elm'], res['ded_elm'])
            self.assertEqual(it['ded_min_elm'], res['ded_min_elm'])
            self.assertEqual(it['ded_max_elm'], res['ded_max_elm'])

            self.assertEqual(it['lim_elm'], res['lim_elm'])

            self.assertEqual(it['shr_elm'], res['shr_elm'])

            self.assertEqual(it['tiv'], res['tiv'])

            self.assertEqual(it['deductible'], res['deductible'])
            self.assertEqual(it['deductible_min'], res['deductible_min'])
            self.assertEqual(it['deductible_max'], res['deductible_max'])

            self.assertEqual(it['attachment'], res['attachment'])

            self.assertEqual(it['limit'], res['limit'])

            self.assertEqual(it['share'], res['share'])

            calcrule_id = get_sub_layer_calcrule_id(it['deductible'], it['deductible_min'], it['deductible_max'], it['limit'])
            self.assertEqual(calcrule_id, res['calcrule_id'])

    #@pytest.mark.flaky
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposure=canonical_exposure(
            from_account_nums=just('A1'),
            from_tivs1=just(100),
            size=2
        ),
        accounts=canonical_accounts(
            from_account_nums=just('A1'),
            from_policy_nums=just('A1P1'),
            size=1
        ),
        fm_input_items=fm_input_items(
            from_peril_ids=just(OASIS_PERILS['wind']['id']),
            from_coverage_type_ids=just(OASIS_COVERAGE_TYPES['buildings']['id']),
            from_canacc_ids=just(0),
            from_policy_nums=just('A1P1'),
            from_layer_ids=just(1),
            from_tiv_elements=just('wscv1val'),
            from_tiv_tgids=just(1),
            from_tivs=just(100),
            size=2
        )
    )
    def test_exposure_with_one_coverage_type_and_fm_terms_with_one_account_and_one_top_level_layer_per_account_and_model_lookup_supporting_single_peril_and_coverage_type__all_non_coverage_level_terms_generated(self, exposure, accounts, fm_input_items):
        ufcp = self.unified_canonical_profile

        for i, _ in enumerate(exposure):
            exposure[i]['cond1name'] = fm_input_items[i]['cond1name'] = 0

        levels = sorted(ufcp.keys())
        levels.remove(min(levels))

        cacc_it = accounts[0]

        for l in levels:
            lufcp = ufcp[l]

            lfmaggp = self.fm_agg_profile[l]

            ded_fld = lufcp[1].get('deductible')
            ded_elm = ded_fld['ProfileElementName'].lower() if ded_fld else None
            ded_min_fld = lufcp[1].get('deductiblemin')
            ded_min_elm = ded_min_fld['ProfileElementName'].lower() if ded_min_fld else None
            ded_max_fld = lufcp[1].get('deductiblemax')
            ded_max_elm = ded_max_fld['ProfileElementName'].lower() if ded_max_fld else None

            lim_fld = lufcp[1].get('limit')
            lim_elm = lim_fld['ProfileElementName'].lower() if lim_fld else None

            shr_fld = lufcp[1].get('share')
            shr_elm = shr_fld['ProfileElementName'].lower() if shr_fld else None

            for i, it in enumerate(fm_input_items):
                it['level_id'] = l
                it['index'] = i

                it['agg_id'] = i + 1 if l in [1,2,3] else 1

                cexp_it = exposure[it['canexp_id']]

                if ded_fld and ded_fld['ProfileType'].lower() == 'loc':
                    cexp_it[ded_elm] = 1
                elif ded_fld and ded_fld['ProfileType'].lower() == 'acc':
                    cacc_it[ded_elm] = 1

                if ded_min_fld and ded_min_fld['ProfileType'].lower() == 'loc':
                    cexp_it[ded_min_elm] = 0
                elif ded_min_fld and ded_min_fld['ProfileType'].lower() == 'acc':
                    cacc_it[ded_min_elm] = 0

                if ded_max_fld and ded_max_fld['ProfileType'].lower() == 'loc':
                    cexp_it[ded_max_elm] = 0
                elif ded_max_fld and ded_max_fld['ProfileType'].lower() == 'acc':
                    cacc_it[ded_max_elm] = 0

                it['attachment'] = 0 if l < max(levels) else cacc_it[ded_elm]

                if lim_fld and lim_fld['ProfileType'].lower() == 'loc':
                    cexp_it[lim_elm] = 1
                elif lim_fld and lim_fld['ProfileType'].lower() == 'acc':
                    cacc_it[lim_elm] = 1

                if shr_fld and shr_fld['ProfileType'].lower() == 'loc':
                    cexp_it[shr_elm] = 0
                elif shr_fld and shr_fld['ProfileType'].lower() == 'acc':
                    cacc_it[shr_elm] = 0

            level_fm_terms_func = get_sub_layer_non_coverage_level_fm_terms if l < max(levels) else get_layer_level_fm_terms

            results = sorted([r for r in level_fm_terms_func(
                lufcp,
                lfmaggp,
                {i:it for i, it in enumerate(fm_input_items)},
                pd.DataFrame(data=exposure),
                pd.DataFrame(data=accounts),
                oed=False
			)], key=lambda it: it['item_id'])


            self.assertEqual(len(fm_input_items), len(results))

            self.assertEqual(tuple(it['item_id'] for it in fm_input_items), tuple(r['item_id'] for r in results))

            for i, res in enumerate(results):
                it = fm_input_items[i]

                self.assertEqual(it['level_id'], res['level_id'])
                self.assertEqual(it['index'], res['index'])
                self.assertEqual(it['item_id'], res['item_id'])
                self.assertEqual(it['gul_item_id'], res['gul_item_id'])

                self.assertEqual(it['canexp_id'], res['canexp_id'])
                self.assertEqual(it['canacc_id'], res['canacc_id'])

                self.assertEqual(it['peril_id'], res['peril_id'])
                self.assertEqual(it['coverage_type_id'], res['coverage_type_id'])

                self.assertEqual(it['layer_id'], res['layer_id'])
                self.assertEqual(it['agg_id'], res['agg_id'])

                self.assertEqual(it['tiv_elm'], res['tiv_elm'])
                self.assertEqual(it['tiv_tgid'], res['tiv_tgid'])

                self.assertEqual(it['ded_elm'], res['ded_elm'])
                self.assertEqual(it['ded_min_elm'], res['ded_min_elm'])
                self.assertEqual(it['ded_max_elm'], res['ded_max_elm'])

                self.assertEqual(it['lim_elm'], res['lim_elm'])

                self.assertEqual(it['shr_elm'], res['shr_elm'])

                self.assertEqual(it['tiv'], res['tiv'])

                self.assertEqual(it['deductible'], res['deductible'])
                self.assertEqual(it['deductible_min'], res['deductible_min'])
                self.assertEqual(it['deductible_max'], res['deductible_max'])

                self.assertEqual(it['attachment'], (0 if l < max(levels) else it['deductible']))

                self.assertEqual(it['limit'], res['limit'])

                self.assertEqual(it['share'], res['share'])

                calcrule_id = (
                    get_sub_layer_calcrule_id(it['deductible'], it['deductible_min'], it['deductible_max'], it['limit']) if l < max(levels) else
                    get_layer_calcrule_id(it['attachment'], it['limit'], it['share'])
                )
                self.assertEqual(calcrule_id, res['calcrule_id'])


class GetPolicyTcIds(TestCase):
    
    @pytest.mark.flaky
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        fm_input_items=fm_input_items(
        from_level_ids=sampled_from([1,6]),
        from_deductibles=sampled_from([0,100]),
        from_min_deductibles=sampled_from([0, 100]),
        from_max_deductibles=sampled_from([0, 100]),
        from_attachments=just(0),
        from_limits=sampled_from([0,100]),
        from_shares=sampled_from([0,0.99]),
        from_calcrule_ids=just(12),
        size=10
        )
    )
    def test_policytc_ids(self, fm_input_items):

        fm_input_items.sort(key=lambda it: it['level_id'])

        for it in fm_input_items:
            it['attachment'] = 0 if it['level_id'] < 6 else it['deductible']
            it['calcrule_id'] = (
                get_sub_layer_calcrule_id(it['deductible'], it['deductible_min'], it['deductible_max'], it['limit']) if it['level_id'] < 6 else
                get_layer_calcrule_id(it['attachment'], it['limit'], it['share'])
            ) or -1

        term_combs = {}

        ptc_id = 0
        for i, it in enumerate(fm_input_items):
            t = (it['limit'], it['deductible'], it['deductible_min'], it['deductible_max'], it['attachment'], it['share'], it['calcrule_id'],)
            if t not in term_combs.values():
                ptc_id += 1
                term_combs[ptc_id] = t

        fm_input_items_df = pd.DataFrame(data=fm_input_items)

        policytc_ids = get_policytc_ids(fm_input_items_df)

        for policytc_id, policytc_comb in policytc_ids.items():
            t = dict(zip(('limit', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'share', 'calcrule_id',), term_combs[policytc_id]))
            self.assertEqual(t, policytc_comb)
