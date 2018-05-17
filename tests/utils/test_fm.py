from __future__ import unicode_literals

import io
import json

from unittest import TestCase

import pandas as pd

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
    canonical_profiles_fm_terms_grouped_by_level,
    canonical_profiles_fm_terms_grouped_by_level_and_term_type,
    get_calcrule_id,
    get_coverage_level_terms,
    get_fm_terms_by_item,
    get_fm_terms_by_level,
)

from tests.data import (
    calcrule_ids,
    canonical_accounts_data,
    canonical_accounts_profile_piwind,
    canonical_exposures_data,
    canonical_exposures_profile_piwind,
    canonical_exposures_profile_piwind_simple,
    deductible_types,
    deductible_types_piwind,
    fm_items_data,
    fm_levels_piwind_simple,
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

    def setUp(self):
        self.exposures_profile = canonical_exposures_profile_piwind_simple
        self.accounts_profile = canonical_accounts_profile_piwind
        self.grouped_profile = canonical_profiles_fm_terms_grouped_by_level_and_term_type(
            canonical_profiles=[self.exposures_profile, self.accounts_profile]
        )

    @given(
        exposures=canonical_exposures_data(
            from_accounts_nums=just(10101),
            from_tivs1=just(100),
            from_limits1=floats(min_value=1, allow_infinity=False),
            from_deductibles1=just(1),
            from_tivs2=just(0),
            from_limits2=just(0),
            from_deductibles2=just(0),
            size=10
        ),
        accounts=canonical_accounts_data(
            from_accounts_nums=just(10101),
            from_attachment_points=floats(min_value=1, allow_infinity=False),
            from_blanket_deductibles=just(0),
            from_blanket_limits=just(0.1),
            from_layer_limits=floats(min_value=1, allow_infinity=False),
            from_policy_types=just(1),
            size=1
        ),
        fm_items=fm_items_data(
            from_level_ids=sampled_from(fm_levels_piwind_simple),
            from_tivs=just(100),
            from_deductible_types=just('B'),
            size=11
        )
    )
    def test_fm_terms_by_item(self, exposures, accounts, fm_items):

        gfmt = self.grouped_profile

        exposures[0]['wscv2val'] = 50

        for i, fm_item in enumerate(fm_items):
            level_id = fm_item['level_id']
            fm_item['canexp_id'] = 0 if i in [0, 1] else fm_item['canexp_id'] - 1
            fm_item['canacc_id'] = 0
            fm_item['index'] = i

            canexp_item = exposures[fm_item['canexp_id']]
            canacc_item = accounts[fm_item['canacc_id']]

            fm_terms = get_fm_terms_by_item(gfmt, canexp_item, canacc_item ,fm_item)

            self.assertEqual(level_id, fm_terms['level_id'])
            self.assertEqual(fm_item['index'], fm_terms['index'])
            self.assertEqual(fm_item['item_id'], fm_terms['item_id'])

            tiv = fm_item['tiv']
            limit = 0
            ded = 0
            ded_type = fm_item['deductible_type']
            share = 0
            calcrule_id = -1

            is_coverage_level = any('tiv' in gfmt[level_id][gid] for gid in gfmt[level_id])

            if is_coverage_level:
                for gid in gfmt[level_id]:
                    tiv_element_name = gfmt[level_id][gid]['tiv']['ProfileElementName'].lower()
                    if canexp_item[tiv_element_name] == tiv:
                        self.assertEqual(tiv, fm_terms['tiv'])

                        limit_element_name = gfmt[level_id][gid]['limit']['ProfileElementName'].lower()
                        limit = canexp_item[limit_element_name]
                        self.assertEqual(limit, fm_terms['limit'])

                        ded_element_name = gfmt[level_id][gid]['deductible']['ProfileElementName'].lower()
                        ded = canexp_item[ded_element_name]
                        self.assertEqual(ded, fm_terms['deductible'])

                        self.assertEqual(ded_type, fm_terms['deductible_type'])
                        break
            else:
                limit_element_name = gfmt[level_id][1].get('limit')['ProfileElementName'].lower() if gfmt[level_id][1].get('limit') else None
                if limit_element_name and limit_element_name in canexp_item:
                    limit = canexp_item[limit_element_name]
                elif limit_element_name and limit_element_name in canacc_item:
                    limit = canacc_item[limit_element_name]
                self.assertEqual(limit, fm_terms['limit'])

                ded_element_name = gfmt[level_id][1].get('deductible')['ProfileElementName'].lower() if gfmt[level_id][1].get('deductible') else None
                if ded_element_name and ded_element_name in canexp_item:
                    ded = canexp_item[ded_element_name]
                elif ded_element_name and ded_element_name in canacc_item:
                    ded = canacc_item[ded_element_name]
                self.assertEqual(ded, fm_terms['deductible'])

                self.assertEqual(ded_type, fm_terms['deductible_type'])

                share_element_name = gfmt[level_id][1].get('share')['ProfileElementName'].lower() if gfmt[level_id][1].get('share') else None
                if share_element_name and share_element_name in canexp_item:
                    share = canexp_item[share_element_name]
                elif share_element_name and share_element_name in canacc_item:
                    share = canacc_item[share_element_name]
                self.assertEqual(share, fm_terms['share'])

            calcrule_id = get_calcrule_id(limit, share, ded_type)

            self.assertEqual(calcrule_id, fm_terms['calcrule_id'])


class GetFmTermsByLevel(TestCase):

    def setUp(self):
        self.exposures_profile = canonical_exposures_profile_piwind_simple
        self.accounts_profile = canonical_accounts_profile_piwind
        self.grouped_profile = canonical_profiles_fm_terms_grouped_by_level_and_term_type(
            canonical_profiles=[self.exposures_profile, self.accounts_profile]
        )

    @given(
        exposures=canonical_exposures_data(
            from_accounts_nums=just(10101),
            from_tivs1=just(100),
            from_limits1=floats(min_value=1, allow_infinity=False),
            from_deductibles1=just(1),
            from_tivs2=just(0),
            from_limits2=just(0),
            from_deductibles2=just(0),
            size=10
        ),
        accounts=canonical_accounts_data(
            from_accounts_nums=just(10101),
            from_attachment_points=floats(min_value=1, allow_infinity=False),
            from_blanket_deductibles=just(0),
            from_blanket_limits=just(0.1),
            from_layer_limits=floats(min_value=1, allow_infinity=False),
            from_policy_types=just(1),
            size=1
        ),
        fm_items=fm_items_data(
            from_level_ids=just(1),
            from_tivs=just(100),
            from_deductible_types=just('B'),
            size=11
        ) 
    )
    def test_coverage_level_terms(self, exposures, accounts, fm_items):
        lgfmt = self.grouped_profile[1]

        exposures[0]['wscv2val'] = 50

        for i, fm_item in enumerate(fm_items):
            fm_item['canexp_id'] = 0 if i in [0, 1] else fm_item['canexp_id'] - 1
            fm_item['canacc_id'] = 0
            fm_item['index'] = i

        results = list(get_coverage_level_terms(
            1,
            lgfmt,
            pd.DataFrame(data=exposures, dtype=object),
            pd.DataFrame(data=accounts, dtype=object),
            pd.DataFrame(data=fm_items, dtype=object)
        ))

        self.assertEqual(len(fm_items), len(results))

        self.assertEqual(tuple(it['item_id'] for it in fm_items), tuple(r['item_id'] for r in results))

        for i, res in enumerate(results):
            it = fm_items[i]

            self.assertEqual(it['level_id'], res['level_id'])
            self.assertEqual(it['index'], res['index'])
            self.assertEqual(it['item_id'], res['item_id'])

            tiv = it['tiv']
            self.assertEqual(tiv, res['tiv'])

            cexp_it = exposures[it['canexp_id']]
            cacc_it = accounts[it['canacc_id']]

            cf = [v for v in lgfmt.values() if cexp_it.get(v['tiv']['ProfileElementName'].lower()) and cexp_it[v['tiv']['ProfileElementName'].lower()] == tiv][0]

            self.assertEqual(tiv, res['tiv'])

            le = cf['limit']['ProfileElementName'].lower() if cf.get('limit') else None
            limit = cexp_it[le] if le and cexp_it.get(le) else (cacc_it[le] if le and cacc_it.get(le) else 0)
            self.assertEqual(limit, res['limit'])

            de = cf['deductible']['ProfileElementName'].lower() if cf.get('deductible') else None
            deductible = cexp_it[de] if de and cexp_it.get(de) else (cacc_it[de] if de and cacc_it.get(de) else 0)
            self.assertEqual(deductible, res['deductible'])

            ded_type = it['deductible_type']

            self.assertEqual(ded_type, res['deductible_type'])

            se = cf['share']['ProfileElementName'].lower() if cf.get('share') else None
            share = cexp_it[se] if se and cexp_it.get(se) else (cacc_it[se] if se and cacc_it.get(se) else 0)
            self.assertEqual(share, res['share'])

            calcrule_id = get_calcrule_id(limit, share, ded_type)
            self.assertEqual(calcrule_id, res['calcrule_id'])

    @given(
        exposures=canonical_exposures_data(
            from_accounts_nums=just(10101),
            from_tivs1=just(100),
            from_limits1=floats(min_value=1, allow_infinity=False),
            from_deductibles1=just(1),
            from_tivs2=just(0),
            from_limits2=just(0),
            from_deductibles2=just(0),
            size=10
        ),
        accounts=canonical_accounts_data(
            from_accounts_nums=just(10101),
            from_attachment_points=floats(min_value=1, allow_infinity=False),
            from_blanket_deductibles=just(0),
            from_blanket_limits=just(0.1),
            from_layer_limits=floats(min_value=1, allow_infinity=False),
            from_policy_types=just(1),
            size=1
        ),
        fm_items=fm_items_data(
            from_tivs=just(100),
            from_deductible_types=just('B'),
            size=11
        ) 
    )
    def test_non_coverage_level_terms(self, exposures, accounts, fm_items):
        gfmt = self.grouped_profile

        exposures[0]['wscv2val'] = 50

        levels = sorted(gfmt.keys())
        levels.remove(1)

        for l in levels:
            lgfmt = gfmt[l]
            for i, it in enumerate(fm_items):
                it['level_id'] = l
                it['canexp_id'] = 0 if i in [0, 1] else it['canexp_id'] - 1
                it['canacc_id'] = 0
                it['index'] = i

            results = list(get_fm_terms_by_level(
                l,
                lgfmt,
                pd.DataFrame(data=exposures, dtype=object),
                pd.DataFrame(data=accounts, dtype=object),
                pd.DataFrame(data=fm_items, dtype=object)
            ))

            self.assertEqual(len(fm_items), len(results))

            self.assertEqual(tuple(it['item_id'] for it in fm_items), tuple(r['item_id'] for r in results))

            for i, res in enumerate(results):
                it = fm_items[i]
                self.assertEqual(it['level_id'], res['level_id'])
                self.assertEqual(it['index'], res['index'])
                self.assertEqual(it['item_id'], res['item_id'])

                tiv = it['tiv']
                self.assertEqual(tiv, res['tiv'])

                cexp_it = exposures[it['canexp_id']]
                cacc_it = accounts[it['canacc_id']]

                self.assertEqual(tiv, res['tiv'])

                le = lgfmt[1].get('limit')['ProfileElementName'].lower() if lgfmt[1].get('limit') else None
                limit = cexp_it[le] if le and cexp_it.get(le) else (cacc_it[le] if le and cacc_it.get(le) else 0)
                self.assertEqual(limit, res['limit'])

                de = lgfmt[1].get('deductible')['ProfileElementName'].lower() if lgfmt[1].get('deductible') else None
                deductible = cexp_it[de] if de and cexp_it.get(de) else (cacc_it[de] if de and cacc_it.get(de) else 0)
                self.assertEqual(deductible, res['deductible'])

                ded_type = it['deductible_type']

                self.assertEqual(ded_type, res['deductible_type'])

                se = lgfmt[1].get('share')['ProfileElementName'].lower() if lgfmt[1].get('share') else None
                share = cexp_it[se] if se and cexp_it.get(se) else (cacc_it[se] if se and cacc_it.get(se) else 0)
                self.assertEqual(share, res['share'])

                calcrule_id = get_calcrule_id(limit, share, ded_type)
                self.assertEqual(calcrule_id, res['calcrule_id'])


class GetPolicyTcId(TestCase):
    pass


class GetPolicyTcIds(TestCase):
    pass