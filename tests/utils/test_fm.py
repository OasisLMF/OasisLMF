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
    canonical_profiles_fm_terms_grouped_by_level,
    canonical_profiles_fm_terms_grouped_by_level_and_term_type,
    get_calcrule_id,
    get_coverage_level_fm_terms,
    get_non_coverage_level_fm_terms,
    get_policytc_ids,
)
from oasislmf.utils.coverage import (
    BUILDING_COVERAGE_CODE,
    CONTENTS_COVERAGE_CODE,
    OTHER_STRUCTURES_COVERAGE_CODE,
    TIME_COVERAGE_CODE,
)
from tests.data import (
    calcrule_ids,
    canonical_accounts_data,
    canonical_accounts_profile_piwind,
    canonical_exposures_data,
    canonical_exposures_profile_piwind,
    fm_agg_profile_piwind,
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


class GetFmTermsByLevel(TestCase):

    def setUp(self):
        self.exposures_profile = canonical_exposures_profile_piwind
        self.accounts_profile = canonical_accounts_profile_piwind
        self.combined_grouped_canonical_profile = canonical_profiles_fm_terms_grouped_by_level_and_term_type(
            canonical_profiles=[self.exposures_profile, self.accounts_profile]
        )
        self.fm_agg_profile = fm_agg_profile_piwind

    @pytest.mark.flaky
    @settings(deadline=300, suppress_health_check=[HealthCheck.too_slow])
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
            from_policy_nums=just('Layer1'),
            from_policy_types=just(1),
            size=1
        ),
        fm_items=fm_items_data(
            from_coverage_type_ids=just(BUILDING_COVERAGE_CODE),
            from_level_ids=just(1),
            from_canacc_ids=just(0),
            from_layer_ids=just(1),
            from_tiv_elements=just('wscv1val'),
            from_tiv_tgids=just(1),
            from_tivs=just(100),
            from_limit_elements=just('wscv1limit'),
            from_deductible_elements=just('wscv1ded'),
            from_deductible_types=just('B'),
            from_shares=just(0),
            size=10
        ) 
    )
    def test_coverage_level_terms(self, exposures, accounts, fm_items):
        lcgcp = self.combined_grouped_canonical_profile[1]
        lfmaggp = self.fm_agg_profile[1]

        for it in exposures:
            it['cond1name'] = 0

        exposures[0]['wscv2val'] = 50

        for i, it in enumerate(fm_items):
            it['index'] = i

        results = list(get_coverage_level_fm_terms(
            lcgcp,
            lfmaggp,
            {i:it for i, it in enumerate(fm_items)},
            pd.DataFrame(data=exposures, dtype=object),
            pd.DataFrame(data=accounts, dtype=object),
        ))

        self.assertEqual(len(fm_items), len(results))

        self.assertEqual(tuple(it['item_id'] for it in fm_items), tuple(r['item_id'] for r in results))

        cacc_it = accounts[0]

        for i, res in enumerate(results):
            it = fm_items[i]

            self.assertEqual(it['level_id'], res['level_id'])
            self.assertEqual(it['index'], res['index'])
            self.assertEqual(it['item_id'], res['item_id'])
            self.assertEqual(it['gul_item_id'], res['gul_item_id'])

            self.assertEqual(it['canexp_id'], res['canexp_id'])
            self.assertEqual(it['canacc_id'], res['canacc_id'])
            self.assertEqual(it['layer_id'], res['layer_id'])

            self.assertEqual(it['tiv_elm'], res['tiv_elm'])
            self.assertEqual(it['tiv_tgid'], res['tiv_tgid'])

            self.assertEqual(it['lim_elm'], res['lim_elm'])
            self.assertEqual(it['ded_elm'], res['ded_elm'])

            tiv = it['tiv']
            self.assertEqual(tiv, res['tiv'])

            cexp_it = exposures[it['canexp_id']]

            cf = lcgcp[it['tiv_tgid']]

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

    @pytest.mark.flaky
    @settings(deadline=300, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(
            from_accounts_nums=just(10101),
            from_tivs1=just(100),
            size=10
        ),
        accounts=canonical_accounts_data(
            from_accounts_nums=just(10101),
            from_attachment_points=just(1),
            from_blanket_deductibles=just(1),
            from_blanket_limits=just(1),
            from_layer_limits=just(1),
            from_policy_nums=just('Layer1'),
            from_policy_types=just(1),
            size=1
        ),
        fm_items=fm_items_data(
            from_coverage_type_ids=just(BUILDING_COVERAGE_CODE),
            from_canacc_ids=just(0),
            from_policy_nums=just('Layer1'),
            from_layer_ids=just(1),
            from_tiv_elements=just('wscv1val'),
            from_tivs=just(100),
            from_tiv_tgids=just(1),
            size=10
        )
    )
    def test_non_coverage_level_terms(self, exposures, accounts, fm_items):
        cgcp = self.combined_grouped_canonical_profile

        for i, _ in enumerate(exposures):
            exposures[i]['cond1name'] = fm_items[i]['cond1name'] = 0

        levels = sorted(cgcp.keys())
        levels.remove(1)

        cacc_it = accounts[0]

        for l in levels:
            lcgcp = cgcp[l]

            lfmaggp = self.fm_agg_profile[l]

            lim_fld = lcgcp[1].get('limit')
            lim_elm = lim_fld['ProfileElementName'].lower() if lim_fld else None
            ded_fld = lcgcp[1].get('deductible')
            ded_elm = ded_fld['ProfileElementName'].lower() if ded_fld else None
            ded_type = ded_fld['DeductibleType'] if ded_fld else 'B'
            shr_fld = lcgcp[1].get('share')
            shr_elm = shr_fld['ProfileElementName'].lower() if shr_fld else None

            for i, it in enumerate(fm_items):
                it['level_id'] = l
                it['index'] = i

                cexp_it = exposures[it['canexp_id']]

                if lim_fld and lim_fld['ProfileType'].lower() == 'loc':
                    cexp_it[lim_elm] = 1
                elif lim_fld and lim_fld['ProfileType'].lower() == 'acc':
                    cacc_it[lim_elm] = 1

                if ded_fld and ded_fld['ProfileType'].lower() == 'loc':
                    cexp_it[ded_elm] = 1
                elif ded_fld and ded_fld['ProfileType'].lower() == 'acc':
                    cacc_it[ded_elm] = 1

                if shr_fld and shr_fld['ProfileType'].lower() == 'loc':
                    cexp_it[shr_elm] = 1
                elif shr_fld and shr_fld['ProfileType'].lower() == 'acc':
                    cacc_it[shr_elm] = 1

            results = list(get_non_coverage_level_fm_terms(
                lcgcp,
                lfmaggp,
                {i:it for i, it in enumerate(fm_items)},
                pd.DataFrame(data=exposures, dtype=object),
                pd.DataFrame(data=accounts, dtype=object),
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

                le = lcgcp[1].get('limit')['ProfileElementName'].lower() if lcgcp[1].get('limit') else None
                limit = cexp_it[le] if le and cexp_it.get(le) else (cacc_it[le] if le and cacc_it.get(le) else 0)
                self.assertEqual(limit, res['limit'])

                de = lcgcp[1].get('deductible')['ProfileElementName'].lower() if lcgcp[1].get('deductible') else None
                deductible = cexp_it[de] if de and cexp_it.get(de) else (cacc_it[de] if de and cacc_it.get(de) else 0)
                self.assertEqual(deductible, res['deductible'])

                ded_type = it['deductible_type']

                self.assertEqual(ded_type, res['deductible_type'])

                se = lcgcp[1].get('share')['ProfileElementName'].lower() if lcgcp[1].get('share') else None
                share = cexp_it[se] if se and cexp_it.get(se) else (cacc_it[se] if se and cacc_it.get(se) else 0)
                self.assertEqual(share, res['share'])

                calcrule_id = get_calcrule_id(limit, share, ded_type)
                self.assertEqual(calcrule_id, res['calcrule_id'])


class GetPolicyTcIds(TestCase):
    
    @pytest.mark.flaky
    @settings(deadline=300, suppress_health_check=[HealthCheck.too_slow])
    @given(
        fm_items=fm_items_data(
            from_limits=sampled_from([50, 100]),
            from_deductibles=sampled_from([25, 50]),
            from_shares=sampled_from([0, 0.5]),
            size=10
        )
    )
    def test_policytc_ids(self, fm_items):

        term_combs = {}

        ptc_id = 0
        for i, it in enumerate(fm_items):
            t = (it['limit'], it['deductible'], it['share'], it['calcrule_id'],)
            if t not in term_combs.values():
                ptc_id += 1
                term_combs[ptc_id] = t

        fm_items_df = pd.DataFrame(data=fm_items, dtype=object)

        policytc_ids = get_policytc_ids(fm_items_df)

        for policytc_id, policytc_comb in policytc_ids.items():
            t = dict(zip(('limit', 'deductible', 'share', 'calcrule_id',), term_combs[policytc_id]))
            self.assertEqual(t, policytc_comb)
