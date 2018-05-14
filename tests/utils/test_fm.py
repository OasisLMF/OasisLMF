from __future__ import unicode_literals

import io
import json

from unittest import TestCase

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

from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.fm import (
    canonical_profiles_fm_terms_grouped_by_level,
    canonical_profiles_fm_terms_grouped_by_level_and_term_type,
    get_calcrule_id,
)

from tests.data import (
    calcrule_ids,
    canonical_accounts_profile_piwind,
    canonical_exposures_profile_piwind,
    deductible_types,
)

class CanonicalProfilesFmTermsGroupedByLevel(TestCase):

    def setUp(self):
        self.exposures_profile = canonical_exposures_profile_piwind
        self.accounts_profile = canonical_accounts_profile_piwind

    # Adapted from a solution by Martijn Pieters
    # https://stackoverflow.com/a/23499088
    def _depth(self, d, level=1):
        if not d or not isinstance(d, dict):
            return level
        return max(self._depth(d[k], level + 1) for k in d)

    def test_no_canonical_profiles_or_profiles_paths_provided__oasis_exception_is_raised(self):
        with self.assertRaises(OasisException):
            canonical_profiles_fm_terms_grouped_by_level()

    def test_only_canonical_profiles_provided(self):
        profiles = (self.exposures_profile, self.accounts_profile,)

        cpft = canonical_profiles_fm_terms_grouped_by_level(canonical_profiles=profiles)

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

    def test_only_canonical_profiles_paths_provided(self):
        profiles = (self.exposures_profile, self.accounts_profile,)

        with NamedTemporaryFile('w') as exposures_profile_file, NamedTemporaryFile('w') as accounts_profile_file:
            with io.open(exposures_profile_file.name, 'w', encoding='utf-8') as f1, io.open(accounts_profile_file.name, 'w', encoding='utf-8') as f2:
                f1.write(u'{}'.format(json.dumps(self.exposures_profile)))
                f2.write(u'{}'.format(json.dumps(self.accounts_profile)))

            paths = (exposures_profile_file.name, accounts_profile_file.name,)

            cpft = canonical_profiles_fm_terms_grouped_by_level(canonical_profiles_paths=paths)

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

    def test_canonical_profiles_and_profiles_paths_provided(self):
        profiles = (self.exposures_profile, self.accounts_profile,)

        with NamedTemporaryFile('w') as exposures_profile_file, NamedTemporaryFile('w') as accounts_profile_file:
            with io.open(exposures_profile_file.name, 'w', encoding='utf-8') as f1, io.open(accounts_profile_file.name, 'w', encoding='utf-8') as f2:
                f1.write(u'{}'.format(json.dumps(self.exposures_profile)))
                f2.write(u'{}'.format(json.dumps(self.accounts_profile)))

            paths = (exposures_profile_file.name, accounts_profile_file.name,)

            cpft = canonical_profiles_fm_terms_grouped_by_level(canonical_profiles=profiles, canonical_profiles_paths=paths)

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


class CanonicalProfilesFmTermsGroupedByLevelAndTermType(TestCase):

    def setUp(self):
        self.exposures_profile = canonical_exposures_profile_piwind
        self.accounts_profile = canonical_accounts_profile_piwind

    # Adapted from a solution by Martijn Pieters
    # https://stackoverflow.com/a/23499088
    def _depth(self, d, level=1):
        if not d or not isinstance(d, dict):
            return level
        return max(self._depth(d[k], level + 1) for k in d)

    def test_no_canonical_profiles_or_profiles_paths_provided__oasis_exception_is_raised(self):
        with self.assertRaises(OasisException):
            canonical_profiles_fm_terms_grouped_by_level_and_term_type()

    def test_only_canonical_profiles_provided(self):
        profiles = (self.exposures_profile, self.accounts_profile,)

        cpft = canonical_profiles_fm_terms_grouped_by_level_and_term_type(canonical_profiles=profiles)

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

    def test_only_canonical_profiles_paths_provided(self):
        profiles = (self.exposures_profile, self.accounts_profile,)

        with NamedTemporaryFile('w') as exposures_profile_file, NamedTemporaryFile('w') as accounts_profile_file:
            with io.open(exposures_profile_file.name, 'w', encoding='utf-8') as f1, io.open(accounts_profile_file.name, 'w', encoding='utf-8') as f2:
                f1.write(u'{}'.format(json.dumps(self.exposures_profile)))
                f2.write(u'{}'.format(json.dumps(self.accounts_profile)))

            paths = (exposures_profile_file.name, accounts_profile_file.name,)

            cpft = canonical_profiles_fm_terms_grouped_by_level_and_term_type(canonical_profiles_paths=paths)

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

    def test_canonical_profiles_and_profiles_paths_provided(self):
        profiles = (self.exposures_profile, self.accounts_profile,)

        with NamedTemporaryFile('w') as exposures_profile_file, NamedTemporaryFile('w') as accounts_profile_file:
            with io.open(exposures_profile_file.name, 'w', encoding='utf-8') as f1, io.open(accounts_profile_file.name, 'w', encoding='utf-8') as f2:
                f1.write(u'{}'.format(json.dumps(self.exposures_profile)))
                f2.write(u'{}'.format(json.dumps(self.accounts_profile)))

            paths = (exposures_profile_file.name, accounts_profile_file.name,)

            cpft = canonical_profiles_fm_terms_grouped_by_level_and_term_type(canonical_profiles=profiles, canonical_profiles_paths=paths)

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


class GetCalcruleID(TestCase):

    @given(
        deductible_type=just('B'),
        limit=just(0.0),
        share=just(0.0)
    )
    def test_limit_and_share_eq_0_and_deductible_type_eq_B__produces_12(self, deductible_type, limit, share):

        self.assertEqual(get_calcrule_id(limit, share, deductible_type), 12)

    @given(
        deductible_type=just('B'),
        limit=just(0.0),
        share=just(0.001)
    )
    def test_limit_eq_0_and_share_gt_0_and_deductible_type_eq_B__produces_15(self, deductible_type, limit, share):

        self.assertEqual(get_calcrule_id(limit, share, deductible_type), 15)

    @given(
        deductible_type=just('B'),
        limit=just(0.001),
        share=just(0.0)
    )
    def test_limit_gt_0_and_share_eq_0_and_deductible_type_eq_B__produces_1(self, deductible_type, limit, share):

        self.assertEqual(get_calcrule_id(limit, share, deductible_type), 1)

    @given(
        deductible_type=just('MI'),
        limit=floats(min_value=0.0, allow_infinity=False),
        share=floats(min_value=0.0, allow_infinity=False)
    )
    def test_deductible_type_eq_MI__produces_11(self, deductible_type, limit, share):

        self.assertEqual(get_calcrule_id(limit, share, deductible_type), 11)

    @given(
        deductible_type=just('MA'),
        limit=floats(min_value=0.0, allow_infinity=False),
        share=floats(min_value=0.0, allow_infinity=False)
    )
    def test_deductible_type_eq_MA__produces_10(self, deductible_type, limit, share):

        self.assertEqual(get_calcrule_id(limit, share, deductible_type), 10)

    @given(
        deductible_type=just('B'),
        limit=floats(min_value=0.001, allow_infinity=False),
        share=floats(min_value=0.001, allow_infinity=False)
    )
    def test_limit_gt_0_and_share_gt_0_and_deductible_type_eq_B__produces_2(self, deductible_type, limit, share):

        self.assertEqual(get_calcrule_id(limit, share, deductible_type), 2)

    @given(
        deductible_type=just('B'),
        limit=floats(min_value=0.001, allow_infinity=False),
        share=floats(min_value=0.001, allow_infinity=False)
    )
    def test_limit_gt_0_and_share_eq_limit_and_deductible_type_eq_B__produces_2(self, deductible_type, limit, share):

        share = limit
        self.assertEqual(get_calcrule_id(limit, share, deductible_type), 2)


class GetFmTermsByItem(TestCase):
    pass

class GetCoverageLevelTerms(TestCase):
    pass

class GetFmTermsByLevel(TestCase):
    pass

class GetFmTermsByLevelAsList(TestCase):
    pass

class GetPolicyTcId(TestCase):
    pass

class GetPolicyTcIds(TestCase):
    pass