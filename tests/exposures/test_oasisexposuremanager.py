from __future__ import unicode_literals

import csv
import io
import json
import os
import string

from collections import OrderedDict
from unittest import TestCase

import pandas as pd

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
    just,
    lists,
    text,
    tuples,
)
from mock import patch, Mock

from oasislmf.exposures.manager import OasisExposuresManager
from oasislmf.exposures.pipeline import OasisFilesPipeline
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.status import (
    KEYS_STATUS_FAIL,
    KEYS_STATUS_NOMATCH,
    KEYS_STATUS_SUCCESS,
)
from ..models.fakes import fake_model

from tests import (
    canonical_exposure_data,
    keys_data,
    write_input_files,
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


class OasisExposureManagerLoadCanonicalProfile(TestCase):
    def test_model_and_kwargs_are_not_set___result_is_empty_dict(self):
        profile = OasisExposuresManager().load_canonical_profile()

        self.assertEqual({}, profile)

    @given(dictionaries(text(), text()))
    def test_model_is_set_with_profile_json___models_profile_is_set_to_expected_json(self, expected):
        model = fake_model(resources={'canonical_exposures_profile_json': json.dumps(expected)})

        profile = OasisExposuresManager().load_canonical_profile(oasis_model=model)

        self.assertEqual(expected, profile)
        self.assertEqual(expected, model.resources['canonical_exposures_profile'])

    @given(dictionaries(text(), text()), dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_and_profile_json_is_passed_through_kwargs___kwargs_profile_is_used(
        self,
        model_profile,
        kwargs_profile
    ):
        model = fake_model(resources={'canonical_exposures_profile_json': json.dumps(model_profile)})

        profile = OasisExposuresManager().load_canonical_profile(oasis_model=model, canonical_exposures_profile_json=json.dumps(kwargs_profile))

        self.assertEqual(kwargs_profile, profile)
        self.assertEqual(kwargs_profile, model.resources['canonical_exposures_profile'])

    @given(dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_path___models_profile_is_set_to_expected_json(self, expected):
        with NamedTemporaryFile('w') as f:
            json.dump(expected, f)
            f.flush()

            model = fake_model(resources={'canonical_exposures_profile_json_path': f.name})

            profile = OasisExposuresManager().load_canonical_profile(oasis_model=model)

            self.assertEqual(expected, profile)
            self.assertEqual(expected, model.resources['canonical_exposures_profile'])

    @given(dictionaries(text(), text()), dictionaries(text(), text()))
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

            profile = OasisExposuresManager().load_canonical_profile(oasis_model=model, canonical_exposures_profile_json_path=kwargs_file.name)

            self.assertEqual(kwargs_profile, profile)
            self.assertEqual(kwargs_profile, model.resources['canonical_exposures_profile'])


class OasisExposureManagerGetKeys(TestCase):
    def create_model(
        self,
        lookup='lookup',
        keys_file_path='key_file_path',
        keys_error_file_path='keys_error_file_path',
        model_exposures_file_path='model_exposures_file_path'
    ):
        model = fake_model(resources={'lookup': lookup})
        model.resources['oasis_files_pipeline'].keys_file_path = keys_file_path
        model.resources['oasis_files_pipeline'].keys_error_file_path = keys_error_file_path
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
        model = self.create_model(lookup=lookup, keys_file_path=keys, keys_error_file_path=keys_errors, model_exposures_file_path=exposure)

        with patch('oasislmf.exposures.manager.OasisKeysLookupFactory.save_keys', Mock(return_value=(keys, 1, keys_errors, 1))) as oklf_mock:
            res_keys_file_path, res_keys_error_file_path = OasisExposuresManager().get_keys(oasis_model=model)

            oklf_mock.assert_called_once_with(
                lookup=lookup,
                model_exposures_file_path=os.path.abspath(exposure),
                keys_file_path=os.path.abspath(keys),
                keys_error_file_path=os.path.abspath(keys_errors)
            )
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_file_path, keys)
            self.assertEqual(res_keys_file_path, keys)
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_error_file_path, keys_errors)
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
        model = self.create_model(lookup=model_lookup, keys_file_path=model_keys, keys_error_file_path=keys_errors, model_exposures_file_path=model_exposure)

        with patch('oasislmf.exposures.manager.OasisKeysLookupFactory.save_keys', Mock(return_value=(keys, 1, keys_errors, 1))) as oklf_mock:
            res_keys_file_path, res_keys_error_file_path = OasisExposuresManager().get_keys(
                oasis_model=model,
                lookup=lookup,
                model_exposures_file_path=exposure,
                keys_file_path=keys,
                keys_error_file_path=keys_errors
            )

            oklf_mock.assert_called_once_with(
                lookup=lookup,
                model_exposures_file_path=os.path.abspath(exposure),
                keys_file_path=os.path.abspath(keys),
                keys_error_file_path=os.path.abspath(keys_errors)
            )
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_file_path, keys)
            self.assertEqual(res_keys_file_path, keys)
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_error_file_path, keys_errors)
            self.assertEqual(res_keys_error_file_path, keys_errors)


class OasisExposureManagerLoadMasterDataframe(TestCase):
    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        profile_element_name=text(alphabet=string.ascii_letters, min_size=1),
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), size=10),
        exposures=canonical_exposure_data(10, min_value=1)
    )
    def test_row_in_keys_data_is_missing_from_exposure_data___oasis_exception_is_raised(
        self,
        profile_element_name,
        keys,
        exposures
    ):
        matching_exposures = [e for e in exposures if e[0] in map(lambda k: k['id'], keys)] 
        exposures.pop(exposures.index(matching_exposures[0]))
        profile = {
            profile_element_name: {'ProfileElementName': profile_element_name, 'FieldName': 'TIV', 'CoverageTypeID': 1}
        }

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file:
            write_input_files(keys, keys_file.name, exposures, exposures_file.name, profile_element_name=profile_element_name)

            with self.assertRaises(OasisException):
                OasisExposuresManager().load_master_data_frame(exposures_file.name, keys_file.name, profile)

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        profile_element_name=text(alphabet=string.ascii_letters, min_size=1),
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), size=10),
        exposures=canonical_exposure_data(num_rows=10, min_value=1)
    )
    def test_each_row_has_a_single_row_per_element_with_each_row_having_a_positive_value_for_the_profile_element___each_row_is_present(
        self,
        profile_element_name,
        keys,
        exposures
    ):
        profile = {
            profile_element_name: {'ProfileElementName': profile_element_name, 'FieldName': 'TIV', 'CoverageTypeID': 1}
        }

        expected = []
        keys_values_tuples = map(lambda li: tuple(filter(lambda v: type(v) == int, li)), [k.values() for k in keys])
        for i, zipped_data in enumerate(zip(keys_values_tuples, exposures)):
            expected.append((
                i + 1,
                zipped_data[0],
                zipped_data[1][1],
            ))

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file:
            write_input_files(keys, keys_file.name, exposures, exposures_file.name, profile_element_name=profile_element_name)

            result = OasisExposuresManager().load_master_data_frame(
                exposures_file.name,
                keys_file.name,
                profile,
            )
        self.assertEqual(len(expected), len(result))

        for i in range(len(result)):
            row = {k:(int(v) if k != 'tiv' else v) for k, v in result.iloc[i].to_dict().items()}
            self.assertEqual(i + 1, row['item_id'])
            self.assertEqual(i + 1, row['coverage_id'])
            self.assertEqual(exposures[i][1], row['tiv'])
            self.assertEqual(keys[i]['area_peril_id'], row['areaperil_id'])
            self.assertEqual(keys[i]['vulnerability_id'], row['vulnerability_id'])
            self.assertEqual(i + 1, row['group_id'])
            self.assertEqual(1, row['summary_id'])
            self.assertEqual(1, row['summaryset_id'])

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        profile_element_name=text(alphabet=string.ascii_letters, min_size=1),
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), size=10),
        exposures=canonical_exposure_data(num_rows=10, min_value=1)
    )
    def test_each_row_has_a_single_row_per_element_with_each_row_having_any_value_for_the_profile_element___rows_with_profile_elements_gt_0_are_present(
        self,
        profile_element_name,
        keys,
        exposures
    ):
        profile = {
            profile_element_name: {'ProfileElementName': profile_element_name, 'FieldName': 'TIV', 'CoverageTypeID': 1}
        }

        expected = []
        keys_values_tuples = map(lambda li: tuple(filter(lambda v: type(v) == int, li)), [k.values() for k in keys])
        row_id = 0
        for zipped_keys, zipped_exposure in zip(keys_values_tuples, exposures):
            if zipped_exposure[1] > 0:
                row_id += 1
                expected.append((
                    row_id,
                    zipped_keys,
                    zipped_exposure[1],
                ))

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file:
            write_input_files(keys, keys_file.name, exposures, exposures_file.name, profile_element_name=profile_element_name)

            result = OasisExposuresManager().load_master_data_frame(
                exposures_file.name,
                keys_file.name,
                profile,
            )

        self.assertEqual(len(expected), len(result))
        for i in range(len(result)):
            row = {k:(int(v) if k != 'tiv' else v) for k, v in result.iloc[i].to_dict().items()}
            self.assertEqual(i + 1, row['item_id'])
            self.assertEqual(i + 1, row['coverage_id'])
            self.assertEqual(exposures[i][1], row['tiv'])
            self.assertEqual(keys[i]['area_peril_id'], row['areaperil_id'])
            self.assertEqual(keys[i]['vulnerability_id'], row['vulnerability_id'])
            self.assertEqual(i + 1, row['group_id'])
            self.assertEqual(1, row['summary_id'])
            self.assertEqual(1, row['summaryset_id'])


class FileGenerationTestCase(TestCase):
    def setUp(self):
        self.items_filename = 'items.csv'
        self.coverages_filename = 'coverages.csv'
        self.gulsummaryxref_filename = 'gulsummaryxref.csv'

    def check_items_file(self, keys, out_dir):
        expected = [
            {
                'item_id': i + 1,
                'coverage_id': i + 1,
                'areaperil_id': key['area_peril_id'],
                'vulnerability_id': key['vulnerability_id'],
                'group_id': i + 1,
            } for i, key in enumerate(keys)
        ]

        with io.open(os.path.join(out_dir, self.items_filename), 'r', encoding='utf-8') as f:
            result = list(pd.read_csv(f).T.to_dict().values())
        
        self.assertEqual(expected, result)

    def check_coverages_file(self, exposures, out_dir):
        expected = [
            {
                'coverage_id': item_id + 1,
                'tiv': item[1],
            } for item_id, item in enumerate(exposures)
        ]

        with io.open(os.path.join(out_dir, self.coverages_filename), 'r', encoding='utf-8') as f:
            result = list(pd.read_csv(f).T.to_dict().values())
        
        self.assertEqual(expected, result)

    def check_gul_file(self, exposures, out_dir):
        expected = [
            {
                'coverage_id': item_id + 1,
                'summary_id': 1,
                'summaryset_id': 1,
            } for item_id in range(len(exposures))
        ]

        with io.open(os.path.join(out_dir, self.gulsummaryxref_filename), 'r', encoding='utf-8') as f:
            result = list(pd.read_csv(f).T.to_dict().values())
        
        self.assertEqual(expected, result)


class OasisExposuresManagerGenerateItemsFile(FileGenerationTestCase):

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), min_size=10),
        exposures=canonical_exposure_data(10, min_value=1)
    )
    def test_paths_are_stored_in_the_model___model_paths_are_used(self, keys, exposures):
        profile = {
            'profile_element': {'ProfileElementName': 'profile_element', 'FieldName': 'TIV', 'CoverageTypeID': 1}
        }

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_input_files(keys, keys_file.name, exposures, exposures_file.name)

            model = fake_model(resources={'canonical_exposures_profile': profile})
            model.resources['oasis_files_pipeline'].keys_file_path = keys_file.name
            model.resources['oasis_files_pipeline'].canonical_exposures_file_path = exposures_file.name
            model.resources['oasis_files_pipeline'].items_file_path = os.path.join(out_dir, self.items_filename)

            OasisExposuresManager().generate_items_file(oasis_model=model)

            self.check_items_file(keys, out_dir)

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), min_size=10),
        exposures=canonical_exposure_data(10, min_value=1)
    )
    def test_paths_are_stored_in_the_kwargs___kwarg_paths_are_used(self, keys, exposures):
        profile = {
            'profile_element': {'ProfileElementName': 'profile_element', 'FieldName': 'TIV', 'CoverageTypeID': 1}
        }

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_input_files(keys, keys_file.name, exposures, exposures_file.name)

            model = fake_model()

            OasisExposuresManager().generate_items_file(
                oasis_model=model,
                canonical_exposures_profile=profile,
                keys_file_path=keys_file.name,
                canonical_exposures_file_path=exposures_file.name,
                items_file_path=os.path.join(out_dir, self.items_filename)
            )

            self.check_items_file(keys, out_dir)


class OasisExposuresManagerGenerateCoveragesFile(FileGenerationTestCase):

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), min_size=10),
        exposures=canonical_exposure_data(10, min_value=1)
    )
    def test_paths_are_stored_in_the_model___model_paths_are_used(self, keys, exposures):
        profile = {
            'profile_element': {'ProfileElementName': 'profile_element', 'FieldName': 'TIV', 'CoverageTypeID': 1}
        }

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_input_files(keys, keys_file.name, exposures, exposures_file.name)

            model = fake_model(resources={'canonical_exposures_profile': profile})
            model.resources['oasis_files_pipeline'].keys_file_path = keys_file.name
            model.resources['oasis_files_pipeline'].canonical_exposures_file_path = exposures_file.name
            model.resources['oasis_files_pipeline'].coverages_file_path = os.path.join(out_dir, self.coverages_filename)

            OasisExposuresManager().generate_coverages_file(oasis_model=model)

            self.check_coverages_file(exposures, out_dir)

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), min_size=10),
        exposures=canonical_exposure_data(10, min_value=1)
    )
    def test_paths_are_stored_in_the_kwargs___kwarg_paths_are_used(self, keys, exposures):
        profile = {
            'profile_element': {'ProfileElementName': 'profile_element', 'FieldName': 'TIV', 'CoverageTypeID': 1}
        }

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_input_files(keys, keys_file.name, exposures, exposures_file.name)

            model = fake_model()

            OasisExposuresManager().generate_coverages_file(
                oasis_model=model,
                canonical_exposures_profile=profile,
                keys_file_path=keys_file.name,
                canonical_exposures_file_path=exposures_file.name,
                coverages_file_path=os.path.join(out_dir, self.coverages_filename)
            )

            self.check_coverages_file(exposures, out_dir)


class OasisExposuresManagerGenerateGulsummaryxrefFile(FileGenerationTestCase):

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), min_size=10),
        exposures=canonical_exposure_data(10, min_value=1)
    )
    def test_paths_are_stored_in_the_model___model_paths_are_used(self, keys, exposures):
        profile = {
            'profile_element': {'ProfileElementName': 'profile_element', 'FieldName': 'TIV', 'CoverageTypeID': 1}
        }

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_input_files(keys, keys_file.name, exposures, exposures_file.name)

            model = fake_model(resources={'canonical_exposures_profile': profile})
            model.resources['oasis_files_pipeline'].keys_file_path = keys_file.name
            model.resources['oasis_files_pipeline'].canonical_exposures_file_path = exposures_file.name
            model.resources['oasis_files_pipeline'].gulsummaryxref_file_path = os.path.join(out_dir, self.gulsummaryxref_filename)

            OasisExposuresManager().generate_gulsummaryxref_file(oasis_model=model)

            self.check_gul_file(exposures, out_dir)

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), min_size=10),
        exposures=canonical_exposure_data(10, min_value=1)
    )
    def test_paths_are_stored_in_the_kwargs___kwarg_paths_are_used(self, keys, exposures):
        profile = {
            'profile_element': {'ProfileElementName': 'profile_element', 'FieldName': 'TIV', 'CoverageTypeID': 1}
        }

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_input_files(keys, keys_file.name, exposures, exposures_file.name)

            model = fake_model()
            OasisExposuresManager().generate_gulsummaryxref_file(
                oasis_model=model,
                canonical_exposures_profile=profile,
                keys_file_path=keys_file.name,
                canonical_exposures_file_path=exposures_file.name,
                gulsummaryxref_file_path=os.path.join(out_dir, self.gulsummaryxref_filename)
            )

            self.check_gul_file(exposures, out_dir)


class OasisExposuresManagerGenerateOasisFiles(FileGenerationTestCase):
    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), min_size=10),
        exposures=canonical_exposure_data(10, min_value=1)
    )
    def test_paths_are_stored_in_the_model___model_paths_are_used(self, keys, exposures):
        profile = {
            'profile_element': {'ProfileElementName': 'profile_element', 'FieldName': 'TIV', 'CoverageTypeID': 1}
        }

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_input_files(keys, keys_file.name, exposures, exposures_file.name)

            model = fake_model(resources={'canonical_exposures_profile': profile})
            model.resources['oasis_files_pipeline'].keys_file_path = keys_file.name
            model.resources['oasis_files_pipeline'].canonical_exposures_file_path = exposures_file.name
            model.resources['oasis_files_pipeline'].items_file_path = os.path.join(out_dir, self.items_filename)
            model.resources['oasis_files_pipeline'].coverages_file_path = os.path.join(out_dir, self.coverages_filename)
            model.resources['oasis_files_pipeline'].gulsummaryxref_file_path = os.path.join(out_dir, self.gulsummaryxref_filename)

            OasisExposuresManager().generate_oasis_files(oasis_model=model)

            self.check_items_file(keys, out_dir)
            self.check_coverages_file(exposures, out_dir)
            self.check_gul_file(exposures, out_dir)

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        keys=keys_data(from_statuses=just(KEYS_STATUS_SUCCESS), min_size=10),
        exposures=canonical_exposure_data(10, min_value=1)
    )
    def test_paths_are_stored_in_the_kwargs___kwarg_paths_are_used(self, keys, exposures):
        profile = {
            'profile_element': {'ProfileElementName': 'profile_element', 'FieldName': 'TIV', 'CoverageTypeID': 1}
        }

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_input_files(keys, keys_file.name, exposures, exposures_file.name)

            model = fake_model()

            OasisExposuresManager().generate_oasis_files(
                oasis_model=model,
                canonical_exposures_profile=profile,
                keys_file_path=keys_file.name,
                canonical_exposures_file_path=exposures_file.name,
                items_file_path=os.path.join(out_dir, self.items_filename),
                coverages_file_path=os.path.join(out_dir, self.coverages_filename),
                gulsummaryxref_file_path=os.path.join(out_dir, self.gulsummaryxref_filename)
            )

            self.check_items_file(keys, out_dir)
            self.check_coverages_file(exposures, out_dir)
            self.check_gul_file(exposures, out_dir)


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
