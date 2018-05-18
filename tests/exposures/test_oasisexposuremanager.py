from __future__ import unicode_literals

import copy
import io
import json
import os
import string

from collections import OrderedDict
from unittest import TestCase

import pandas as pd
import pytest
import six

from backports.tempfile import TemporaryDirectory
from tempfile import NamedTemporaryFile

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
    text,
    tuples,
)
from mock import patch, Mock

from oasislmf.exposures.manager import OasisExposuresManager
from oasislmf.exposures.pipeline import OasisFilesPipeline
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.fm import (
    canonical_profiles_fm_terms_grouped_by_level_and_term_type
)
from oasislmf.utils.status import (
    KEYS_STATUS_FAIL,
    KEYS_STATUS_NOMATCH,
    KEYS_STATUS_SUCCESS,
)
from ..models.fakes import fake_model

from tests.data import (
    canonical_accounts_data,
    canonical_accounts_profile_piwind,
    canonical_exposures_data,
    canonical_exposures_profile_piwind_simple,
    fm_items_data,
    gul_items_data,
    keys_data,
    write_canonical_files,
    write_keys_files,
)


class OasisExposureManagerAddModel(TestCase):
    def test_models_is_empty___model_is_added_to_model_dict(self):
        model = fake_model('supplier', 'model', 'version')

        manager = OasisExposuresManager()
        manager.add_model(model)

        self.assertEqual({model.key: model}, manager.models)

    def test_manager_already_contains_a_model_with_the_given_key___model_is_replaced_in_models_dict(self):
        first = fake_model('supplier', 'model', 'version')
        second = fake_model('supplier', 'model', 'version')

        manager = OasisExposuresManager(oasis_models=[first])
        manager.add_model(second)

        self.assertIs(second, manager.models[second.key])

    def test_manager_already_contains_a_diferent_model___model_is_added_to_dict(self):
        first = fake_model('first', 'model', 'version')
        second = fake_model('second', 'model', 'version')

        manager = OasisExposuresManager(oasis_models=[first])
        manager.add_model(second)

        self.assertEqual({
            first.key: first,
            second.key: second,
        }, manager.models)


class OasisExposureManagerDeleteModels(TestCase):
    def test_models_is_not_in_manager___no_model_is_removed(self):
        manager = OasisExposuresManager([
            fake_model('supplier', 'model', 'version'),
            fake_model('supplier2', 'model2', 'version2'),
        ])
        expected = manager.models

        manager.delete_models([fake_model('supplier3', 'model3', 'version3')])

        self.assertEqual(expected, manager.models)

    def test_models_exist_in_manager___models_are_removed(self):
        models = [
            fake_model('supplier', 'model', 'version'),
            fake_model('supplier2', 'model2', 'version2'),
            fake_model('supplier3', 'model3', 'version3'),
        ]

        manager = OasisExposuresManager(models)
        manager.delete_models(models[1:])

        self.assertEqual({models[0].key: models[0]}, manager.models)


class OasisExposureManagerLoadCanonicalExposuresProfile(TestCase):
    def test_model_and_kwargs_are_not_set___result_is_empty_dict(self):
        profile = OasisExposuresManager().load_canonical_exposures_profile()

        self.assertEqual({}, profile)

    @given(expected=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json___models_profile_is_set_to_expected_json(self, expected):
        model = fake_model(resources={'canonical_exposures_profile_json': json.dumps(expected)})

        profile = OasisExposuresManager().load_canonical_exposures_profile(oasis_model=model)

        self.assertEqual(expected, profile)
        self.assertEqual(expected, model.resources['canonical_exposures_profile'])

    @given(model_profile=dictionaries(text(), text()), kwargs_profile=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_and_profile_json_is_passed_through_kwargs___kwargs_profile_is_used(
        self,
        model_profile,
        kwargs_profile
    ):
        model = fake_model(resources={'canonical_exposures_profile_json': json.dumps(model_profile)})

        profile = OasisExposuresManager().load_canonical_exposures_profile(oasis_model=model, canonical_exposures_profile_json=json.dumps(kwargs_profile))

        self.assertEqual(kwargs_profile, profile)
        self.assertEqual(kwargs_profile, model.resources['canonical_exposures_profile'])

    @given(expected=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_path___models_profile_is_set_to_expected_json(self, expected):
        with NamedTemporaryFile('w') as f:
            json.dump(expected, f)
            f.flush()

            model = fake_model(resources={'canonical_exposures_profile_json_path': f.name})

            profile = OasisExposuresManager().load_canonical_exposures_profile(oasis_model=model)

            self.assertEqual(expected, profile)
            self.assertEqual(expected, model.resources['canonical_exposures_profile'])

    @given(model_profile=dictionaries(text(), text()), kwargs_profile=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_path_and_profile_json_path_is_passed_through_kwargs___kwargs_profile_is_used(
        self,
        model_profile,
        kwargs_profile
    ):
        with NamedTemporaryFile('w') as model_file, NamedTemporaryFile('w') as kwargs_file:
            json.dump(model_profile, model_file)
            model_file.flush()
            json.dump(kwargs_profile, kwargs_file)
            kwargs_file.flush()

            model = fake_model(resources={'canonical_exposures_profile_json_path': model_file.name})

            profile = OasisExposuresManager().load_canonical_exposures_profile(oasis_model=model, canonical_exposures_profile_json_path=kwargs_file.name)

            self.assertEqual(kwargs_profile, profile)
            self.assertEqual(kwargs_profile, model.resources['canonical_exposures_profile'])


class OasisExposureManagerLoadCanonicalAccountProfile(TestCase):
    def test_model_and_kwargs_are_not_set___result_is_empty_dict(self):
        profile = OasisExposuresManager().load_canonical_accounts_profile()

        self.assertEqual({}, profile)

    @given(expected=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json___models_profile_is_set_to_expected_json(self, expected):
        model = fake_model(resources={'canonical_accounts_profile_json': json.dumps(expected)})

        profile = OasisExposuresManager().load_canonical_accounts_profile(oasis_model=model)

        self.assertEqual(expected, profile)
        self.assertEqual(expected, model.resources['canonical_accounts_profile'])

    @given(model_profile=dictionaries(text(), text()), kwargs_profile=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_and_profile_json_is_passed_through_kwargs___kwargs_profile_is_used(
        self,
        model_profile,
        kwargs_profile
    ):
        model = fake_model(resources={'canonical_accounts_profile_json': json.dumps(model_profile)})

        profile = OasisExposuresManager().load_canonical_accounts_profile(oasis_model=model, canonical_accounts_profile_json=json.dumps(kwargs_profile))

        self.assertEqual(kwargs_profile, profile)
        self.assertEqual(kwargs_profile, model.resources['canonical_accounts_profile'])

    @given(expected=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_path___models_profile_is_set_to_expected_json(self, expected):
        with NamedTemporaryFile('w') as f:
            json.dump(expected, f)
            f.flush()

            model = fake_model(resources={'canonical_accounts_profile_json_path': f.name})

            profile = OasisExposuresManager().load_canonical_accounts_profile(oasis_model=model)

            self.assertEqual(expected, profile)
            self.assertEqual(expected, model.resources['canonical_accounts_profile'])

    @given(model_profile=dictionaries(text(), text()), kwargs_profile=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_path_and_profile_json_path_is_passed_through_kwargs___kwargs_profile_is_used(
        self,
        model_profile,
        kwargs_profile
    ):
        with NamedTemporaryFile('w') as model_file, NamedTemporaryFile('w') as kwargs_file:
            json.dump(model_profile, model_file)
            model_file.flush()
            json.dump(kwargs_profile, kwargs_file)
            kwargs_file.flush()

            model = fake_model(resources={'canonical_accounts_profile_json_path': model_file.name})

            profile = OasisExposuresManager().load_canonical_accounts_profile(oasis_model=model, canonical_accounts_profile_json_path=kwargs_file.name)

            self.assertEqual(kwargs_profile, profile)
            self.assertEqual(kwargs_profile, model.resources['canonical_accounts_profile'])


class OasisExposureManagerGetKeys(TestCase):
    def create_model(
        self,
        lookup='lookup',
        keys_file_path='key_file_path',
        keys_errors_file_path='keys_error_file_path',
        model_exposures_file_path='model_exposures_file_path'
    ):
        model = fake_model(resources={'lookup': lookup})

        model.resources['oasis_files_pipeline'].keys_file_path = keys_file_path
        model.resources['oasis_files_pipeline'].keys_errors_file_path = keys_errors_file_path
        model.resources['oasis_files_pipeline'].model_exposures_file_path = model_exposures_file_path

        return model

    @given(
        lookup=text(min_size=1, alphabet=string.ascii_letters),
        keys=text(min_size=1, alphabet=string.ascii_letters),
        keys_errors=text(min_size=1, alphabet=string.ascii_letters),
        exposure=text(min_size=1, alphabet=string.ascii_letters)
    )
    def test_model_is_supplied_kwargs_are_not___lookup_keys_files_and_exposures_file_from_model_are_used(
        self,
        lookup,
        keys,
        keys_errors,
        exposure
    ):
        model = self.create_model(lookup=lookup, keys_file_path=keys, keys_errors_file_path=keys_errors, model_exposures_file_path=exposure)

        with patch('oasislmf.exposures.manager.OasisKeysLookupFactory.save_keys', Mock(return_value=(keys, 1, keys_errors, 1))) as oklf_mock:
            res_keys_file_path, res_keys_error_file_path = OasisExposuresManager().get_keys(oasis_model=model)

            oklf_mock.assert_called_once_with(
                lookup=lookup,
                model_exposures_file_path=os.path.abspath(exposure),
                keys_file_path=os.path.abspath(keys),
                keys_errors_file_path=os.path.abspath(keys_errors)
            )

            self.assertEqual(model.resources['oasis_files_pipeline'].keys_file_path, keys)
            self.assertEqual(res_keys_file_path, keys)
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_errors_file_path, keys_errors)
            self.assertEqual(res_keys_error_file_path, keys_errors)

    @given(
        model_lookup=text(min_size=1, alphabet=string.ascii_letters), 
        model_keys=text(min_size=1, alphabet=string.ascii_letters),
        model_keys_errors=text(min_size=1, alphabet=string.ascii_letters),
        model_exposure=text(min_size=1, alphabet=string.ascii_letters),
        lookup=text(min_size=1, alphabet=string.ascii_letters),
        keys=text(min_size=1, alphabet=string.ascii_letters),
        keys_errors=text(min_size=1, alphabet=string.ascii_letters),
        exposure=text(min_size=1, alphabet=string.ascii_letters)
    )
    def test_model_and_kwargs_are_supplied___lookup_keys_files_and_exposures_file_from_kwargs_are_used(
        self,
        model_lookup,
        model_keys,
        model_keys_errors,
        model_exposure,
        lookup,
        keys,
        keys_errors,
        exposure
    ):
        model = self.create_model(lookup=model_lookup, keys_file_path=model_keys, keys_errors_file_path=keys_errors, model_exposures_file_path=model_exposure)

        with patch('oasislmf.exposures.manager.OasisKeysLookupFactory.save_keys', Mock(return_value=(keys, 1, keys_errors, 1))) as oklf_mock:
            res_keys_file_path, res_keys_error_file_path = OasisExposuresManager().get_keys(
                oasis_model=model,
                lookup=lookup,
                model_exposures_file_path=exposure,
                keys_file_path=keys,
                keys_errors_file_path=keys_errors
            )

            oklf_mock.assert_called_once_with(
                lookup=lookup,
                model_exposures_file_path=os.path.abspath(exposure),
                keys_file_path=os.path.abspath(keys),
                keys_errors_file_path=os.path.abspath(keys_errors)
            )

            self.assertEqual(model.resources['oasis_files_pipeline'].keys_file_path, keys)
            self.assertEqual(res_keys_file_path, keys)
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_errors_file_path, keys_errors)
            self.assertEqual(res_keys_error_file_path, keys_errors)


class GulFilesGenerationTestCase(TestCase):

    def check_items_file(self, gul_master_data_frame, items_file_path):
        expected = tuple(
            {
                k:item[k] for k in ('item_id', 'coverage_id', 'areaperil_id', 'vulnerability_id', 'group_id',)
            } for _, item in gul_master_data_frame.iterrows()
        )

        with io.open(items_file_path, 'r', encoding='utf-8') as f:
            result = tuple(pd.read_csv(f).T.to_dict().values())

        self.assertEqual(expected, result)

    def check_coverages_file(self, gul_master_data_frame, coverages_file_path):
        expected = tuple(
            {
                k:item[k] for k in ('coverage_id', 'tiv',)
            } for _, item in gul_master_data_frame.iterrows()
        )

        with io.open(coverages_file_path, 'r', encoding='utf-8') as f:
            result = tuple(pd.read_csv(f).T.to_dict().values())

        self.assertEqual(expected, result)

    def check_gulsummaryxref_file(self, gul_master_data_frame, gulsummaryxref_file_path):
        expected = tuple(
            {
                k:item[k] for k in ('coverage_id', 'summary_id', 'summaryset_id',)
            } for _, item in gul_master_data_frame.iterrows()
        )

        with io.open(gulsummaryxref_file_path, 'r', encoding='utf-8') as f:
            result = tuple(pd.read_csv(f).T.to_dict().values())

        self.assertEqual(expected, result)


class FmFilesGenerationTestCase(TestCase):
    pass


class OasisExposuresManagerWriteGulFiles(GulFilesGenerationTestCase):

    @pytest.mark.flaky
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(
            from_tivs1=just(1.0),
            from_tivs2=just(0.0),
            size=10
        ),
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), size=10),
    )
    def test_paths_are_stored_in_the_model___model_paths_are_used(self, exposures, keys):
        profile = canonical_exposures_profile_piwind_simple

        oasis_model = fake_model(resources={'canonical_exposures_profile': profile})

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            omr = oasis_model.resources
            ofp = omr['oasis_files_pipeline']

            ofp.keys_file_path = keys_file.name
            ofp.canonical_exposures_file_path = exposures_file.name

            ofp.items_file_path = os.path.join(out_dir, 'items.csv')
            ofp.coverages_file_path = os.path.join(out_dir, 'coverages.csv')
            ofp.gulsummaryxref_file_path = os.path.join(out_dir, 'gulsummaryxref.csv')

            gul_files = OasisExposuresManager().write_gul_files(oasis_model=oasis_model)

            gulm_df = omr['gul_master_data_frame']

            self.check_items_file(gulm_df, gul_files['items'])
            self.check_coverages_file(gulm_df, gul_files['coverages'])
            self.check_gulsummaryxref_file(gulm_df, gul_files['gulsummaryxref'])

    @pytest.mark.flaky
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(
            from_tivs1=just(1.0),
            from_tivs2=just(0.0),
            size=10
        ),
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), size=10)
    )
    def test_paths_are_stored_in_the_kwargs___kwarg_paths_are_used(self, exposures, keys):
        profile = canonical_exposures_profile_piwind_simple

        manager = OasisExposuresManager()

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            gulm_df, _ = manager.load_gul_master_data_frame(profile, exposures_file.name, keys_file.name)

            gul_files = manager.write_gul_files(
                canonical_exposures_profile=profile,
                keys_file_path=keys_file.name,
                canonical_exposures_file_path=exposures_file.name,
                items_file_path=os.path.join(out_dir, 'items.csv'),
                coverages_file_path=os.path.join(out_dir, 'coverages.csv'),
                gulsummaryxref_file_path=os.path.join(out_dir, 'gulsummaryxref.csv')
            )

            self.check_items_file(gulm_df, gul_files['items'])
            self.check_coverages_file(gulm_df, gul_files['coverages'])
            self.check_gulsummaryxref_file(gulm_df, gul_files['gulsummaryxref'])


class OasisExposuresManagerWriteFmFiles(FmFilesGenerationTestCase):
    pass


class OasisExposureManagerLoadGulMasterDataframe(TestCase):

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(size=0),
        keys=keys_data(size=10)
    )
    def test_no_canonical_items__oasis_exception_is_raised(
        self,
        exposures,
        keys
    ):
        profile = canonical_exposures_profile_piwind_simple

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            with self.assertRaises(OasisException):
                OasisExposuresManager().load_gul_master_data_frame(profile, exposures_file.name, keys_file.name)

    @pytest.mark.flaky
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(size=10),
        keys=keys_data(size=0)
    )
    def test_no_keys_items__oasis_exception_is_raised(
        self,
        exposures,
        keys
    ):
        profile = canonical_exposures_profile_piwind_simple

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            with self.assertRaises(OasisException):
                OasisExposuresManager().load_gul_master_data_frame(profile, exposures_file.name, keys_file.name)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(size=10),
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), size=10)
    )
    def test_canonical_items_dont_match_any_keys_items__oasis_exception_is_raised(
        self,
        exposures,
        keys
    ):
        profile = canonical_exposures_profile_piwind_simple

        l = len(exposures)
        for key in keys:
            key['id'] += l

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            with self.assertRaises(OasisException):
                OasisExposuresManager().load_gul_master_data_frame(profile, exposures_file.name, keys_file.name)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(size=10),
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), size=10)
    )
    def test_canonical_profile_doesnt_have_any_tiv_fields__oasis_exception_is_raised(
        self,
        exposures,
        keys
    ):
        profile = copy.deepcopy(canonical_exposures_profile_piwind_simple)

        tivs = [profile[e]['ProfileElementName'] for e in profile if profile[e].get('FMTermType') and profile[e]['FMTermType'].lower() == 'tiv']

        for t in tivs:
            profile.pop(t)

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            with self.assertRaises(OasisException):
                OasisExposuresManager().load_gul_master_data_frame(profile, exposures_file.name, keys_file.name)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(
            from_tivs1=just(0.0),
            from_tivs2=just(0.0),
            size=10
        ),
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), size=10)
    )
    def test_canonical_items_dont_have_any_positive_tivs__oasis_exception_is_raised(
        self,
        exposures,
        keys
    ):
        profile = canonical_exposures_profile_piwind_simple 

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            with self.assertRaises(OasisException):
                OasisExposuresManager().load_gul_master_data_frame(profile, exposures_file.name, keys_file.name)


    @pytest.mark.flaky
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(
            from_tivs1=just(1.0),
            from_tivs2=just(0.0),
            size=10
        ),
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), size=5)
    )
    def test_at_least_some_canonical_items_have_matching_keys_items_and_at_least_one_positive_tiv_and_gul_items_are_generated(
        self,
        exposures,
        keys
    ):
        profile = canonical_exposures_profile_piwind_simple 

        for k in keys:
            k['id'] += 5

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            matching_canonical_and_keys_item_ids = set(k['id'] for k in keys).intersection([e['row_id'] for e in exposures])

            gulm_df, canexp_df = OasisExposuresManager().load_gul_master_data_frame(profile, exposures_file.name, keys_file.name)

        get_canonical_item = lambda i: (
            [e for e in exposures if e['row_id'] == i + 1][0] if len([e for e in exposures if e['row_id'] == i + 1]) == 1
            else None
        )

        get_keys_item = lambda i: (
            [k for k in keys if k['id'] == i + 1][0] if len([k for k in keys if k['id'] == i + 1]) == 1
            else None
        )

        tiv_elements = tuple(sorted(
            [v for v in profile.itervalues() if v.get('FieldName') and v['FieldName'] == 'TIV'],
            key=lambda v: v['FMTermGroupID']
        ))

        for i, gul_it in enumerate(gulm_df.T.to_dict().values()):
            can_it = get_canonical_item(int(gul_it['canexp_id']))
            self.assertIsNotNone(can_it)

            keys_it = get_keys_item(int(gul_it['canexp_id']))
            self.assertIsNotNone(keys_it)

            positive_tiv_elements = [t for t in tiv_elements if can_it.get(t['ProfileElementName'].lower()) and can_it[t['ProfileElementName'].lower()] > 0 and t['CoverageTypeID'] == keys_it['coverage']]
            for _, t in enumerate(positive_tiv_elements):
                self.assertEqual(can_it[t['ProfileElementName'].lower()], gul_it['tiv'])

            self.assertEqual(keys_it['area_peril_id'], gul_it['areaperil_id'])
            self.assertEqual(keys_it['vulnerability_id'], gul_it['vulnerability_id'])

            self.assertEqual(i + 1, gul_it['item_id'])

            self.assertEqual(i + 1, gul_it['coverage_id'])

            self.assertEqual(i + 1, gul_it['group_id'])


class OasisExposureManagerLoadFmMasterDataframe(TestCase):

    def setUp(self):
        self.exposures_profile = canonical_exposures_profile_piwind_simple
        self.accounts_profile = canonical_accounts_profile_piwind
        self.grouped_profile = canonical_profiles_fm_terms_grouped_by_level_and_term_type(
            canonical_profiles=[self.exposures_profile, self.accounts_profile]
        )

    @given(
        exposures=canonical_exposures_data(size=10),
        guls=gul_items_data(size=10)

    )
    def test_no_canonical_accounts_items__oasis_exception_is_raised(
        self,
        exposures,
        guls
    ):
        with NamedTemporaryFile('w') as accounts_file:
            write_canonical_files(canonical_accounts=[], canonical_accounts_file_path=accounts_file.name)

            with self.assertRaises(OasisException):
                fm_df, canacc_df = OasisExposuresManager().load_fm_master_data_frame(
                    pd.DataFrame(data=exposures),
                    pd.DataFrame(data=guls),
                    self.exposures_profile,
                    self.accounts_profile,
                    accounts_file.name
                )


class OasisExposuresTransformSourceToCanonical(TestCase):
    @given(
        source_exposures_file_path=text(),
        source_exposures_validation_file_path=text(),
        source_to_canonical_exposures_transformation_file_path=text(),
        canonical_exposures_file_path=text()
    )
    def test_model_is_not_set___parameters_are_taken_from_kwargs(
            self,
            source_exposures_file_path,
            source_exposures_validation_file_path,
            source_to_canonical_exposures_transformation_file_path,
            canonical_exposures_file_path
    ):
        trans_call_mock = Mock()
        with patch('oasislmf.exposures.manager.Translator', Mock(return_value=trans_call_mock)) as trans_mock:
            OasisExposuresManager().transform_source_to_canonical(
                source_exposures_file_path=source_exposures_file_path,
                source_exposures_validation_file_path=source_exposures_validation_file_path,
                source_to_canonical_exposures_transformation_file_path=source_to_canonical_exposures_transformation_file_path,
                canonical_exposures_file_path=canonical_exposures_file_path
            )

            trans_mock.assert_called_once_with(
                os.path.abspath(source_exposures_file_path),
                os.path.abspath(canonical_exposures_file_path),
                os.path.abspath(source_exposures_validation_file_path),
                os.path.abspath(source_to_canonical_exposures_transformation_file_path),
                append_row_nums=True,
            )
            trans_call_mock.assert_called_once_with()

    @given(
        source_exposures_file_path=text(),
        source_exposures_validation_file_path=text(),
        source_to_canonical_exposures_transformation_file_path=text(),
        canonical_exposures_file_path=text()
    )
    def test_model_is_set___parameters_are_taken_from_model(
            self,
            source_exposures_file_path,
            source_exposures_validation_file_path,
            source_to_canonical_exposures_transformation_file_path,
            canonical_exposures_file_path):

        model = fake_model(resources={
            'source_exposures_file_path': source_exposures_file_path,
            'source_exposures_validation_file_path': source_exposures_validation_file_path,
            'source_to_canonical_exposures_transformation_file_path': source_to_canonical_exposures_transformation_file_path,
        })
        model.resources['oasis_files_pipeline'].canonical_exposures_path = canonical_exposures_file_path

        trans_call_mock = Mock()
        with patch('oasislmf.exposures.manager.Translator', Mock(return_value=trans_call_mock)) as trans_mock:
            OasisExposuresManager().transform_source_to_canonical(
                source_exposures_file_path=source_exposures_file_path,
                source_exposures_validation_file_path=source_exposures_validation_file_path,
                source_to_canonical_exposures_transformation_file_path=source_to_canonical_exposures_transformation_file_path,
                canonical_exposures_file_path=canonical_exposures_file_path
            )

            trans_mock.assert_called_once_with(
                os.path.abspath(source_exposures_file_path),
                os.path.abspath(canonical_exposures_file_path),
                os.path.abspath(source_exposures_validation_file_path),
                os.path.abspath(source_to_canonical_exposures_transformation_file_path),
                append_row_nums=True,
            )
            trans_call_mock.assert_called_once_with()


class OasisExposuresTransformCanonicalToModel(TestCase):
    @given(
        canonical_exposures_file_path=text(),
        canonical_exposures_validation_file_path=text(),
        canonical_to_model_exposures_transformation_file_path=text(),
        model_exposures_file_path=text()
    )
    def test_model_is_not_set___parameters_are_taken_from_kwargs(
            self,
            canonical_exposures_file_path,
            canonical_exposures_validation_file_path,
            canonical_to_model_exposures_transformation_file_path,
            model_exposures_file_path):

        trans_call_mock = Mock()
        with patch('oasislmf.exposures.manager.Translator', Mock(return_value=trans_call_mock)) as trans_mock:
            OasisExposuresManager().transform_canonical_to_model(
                canonical_exposures_file_path=canonical_exposures_file_path,
                canonical_exposures_validation_file_path=canonical_exposures_validation_file_path,
                canonical_to_model_exposures_transformation_file_path=canonical_to_model_exposures_transformation_file_path,
                model_exposures_file_path=model_exposures_file_path,
            )

            trans_mock.assert_called_once_with(
                os.path.abspath(canonical_exposures_file_path),
                os.path.abspath(model_exposures_file_path),
                os.path.abspath(canonical_exposures_validation_file_path),
                os.path.abspath(canonical_to_model_exposures_transformation_file_path),
                append_row_nums=False,
            )
            trans_call_mock.assert_called_once_with()

    @given(
        canonical_exposures_file_path=text(),
        canonical_exposures_validation_file_path=text(),
        canonical_to_model_exposures_transformation_file_path=text(),
        model_exposures_file_path=text()
    )
    def test_model_is_set___parameters_are_taken_from_model(
            self,
            canonical_exposures_file_path,
            canonical_exposures_validation_file_path,
            canonical_to_model_exposures_transformation_file_path,
            model_exposures_file_path):

        model = fake_model(resources={
            'canonical_exposures_validation_file_path': canonical_exposures_validation_file_path,
            'canonical_to_model_exposures_transformation_file_path': canonical_to_model_exposures_transformation_file_path,
        })
        model.resources['oasis_files_pipeline'].canonical_exposures_path = canonical_exposures_file_path
        model.resources['oasis_files_pipeline'].model_exposures_file_path = model_exposures_file_path

        trans_call_mock = Mock()
        with patch('oasislmf.exposures.manager.Translator', Mock(return_value=trans_call_mock)) as trans_mock:
            OasisExposuresManager().transform_canonical_to_model(
                canonical_exposures_file_path=canonical_exposures_file_path,
                canonical_exposures_validation_file_path=canonical_exposures_validation_file_path,
                canonical_to_model_exposures_transformation_file_path=canonical_to_model_exposures_transformation_file_path,
                model_exposures_file_path=model_exposures_file_path,
            )

            trans_mock.assert_called_once_with(
                os.path.abspath(canonical_exposures_file_path),
                os.path.abspath(model_exposures_file_path),
                os.path.abspath(canonical_exposures_validation_file_path),
                os.path.abspath(canonical_to_model_exposures_transformation_file_path),
                append_row_nums=False,
            )
            trans_call_mock.assert_called_once_with()


class OasisExposureManagerCreate(TestCase):
    @given(supplier=text(), model_id=text(), version=text())
    def test_supplier_model_and_version_are_supplied___correct_key_is_created(self, supplier, model_id, version):
        model = fake_model(supplier=supplier, model=model_id, version=version)

        self.assertEqual('{}/{}/{}'.format(supplier, model_id, version), model.key)

    def test_oasis_file_path_is_given___path_is_stored_as_absolute_path(self):
        model = fake_model(resources={'oasis_files_path': 'some_path'})

        result = model.resources['oasis_files_path']
        expected = os.path.abspath('some_path')

        self.assertEqual(expected, result)

    def test_oasis_file_path_is_not_given___path_is_abs_path_of_default(self):
        model = fake_model()

        result = model.resources['oasis_files_path']
        expected = os.path.abspath(os.path.join('Files', model.key.replace('/', '-')))

        self.assertEqual(expected, result)

    def test_file_pipeline_is_not_supplied___default_pipeline_is_set(self):
        model = fake_model()

        pipeline = model.resources['oasis_files_pipeline']

        self.assertIsInstance(pipeline, OasisFilesPipeline)
        self.assertEqual(pipeline.model_key, model.key)

    def test_file_pipeline_is_supplied___pipeline_is_unchanged(self):
        pipeline = OasisFilesPipeline()

        model = fake_model(resources={'oasis_files_pipeline': pipeline})

        self.assertIs(pipeline, model.resources['oasis_files_pipeline'])

    def test_pipeline_is_not_a_pipeline_instance___oasis_exception_is_raised(self):
        class FakePipeline(object):
            pass

        pipeline = FakePipeline()

        with self.assertRaises(OasisException):
            fake_model(resources={'oasis_files_pipeline': pipeline})

    def test_canonical_exposures_profile_not_set___canonical_exposures_profile_in_none(self):
        model = fake_model()

        profile = model.resources['canonical_exposures_profile']

        self.assertEqual({}, profile)

    @given(expected=dictionaries(text(), text()))
    def test_canonical_exposures_profile_json_set___canonical_exposures_profile_matches_json(self, expected):
        model = fake_model(resources={'canonical_exposures_profile_json': json.dumps(expected)})

        profile = model.resources['canonical_exposures_profile']

        self.assertEqual(expected, profile)

    @given(expected=dictionaries(text(), text()))
    def test_canonical_exposures_profile_path_set___canonical_exposures_profile_matches_json(self, expected):
        with NamedTemporaryFile('w') as f:
            json.dump(expected, f)
            f.flush()

            model = fake_model(resources={'canonical_exposures_profile_json_path': f.name})

            profile = model.resources['canonical_exposures_profile']

            self.assertEqual(expected, profile)

    @given(expected=dictionaries(text(), text()), new=dictionaries(text(), text()))
    def test_canonical_exposures_profile_set___profile_is_not_updated(self, expected, new):
        model = fake_model(resources={
            'canonical_exposures_profile': expected,
            'canonical_exposures_profile_json': json.dumps(new),
        })

        profile = model.resources['canonical_exposures_profile']

        self.assertEqual(expected, profile)
