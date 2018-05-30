from __future__ import unicode_literals

import copy
import io
import itertools
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
from oasislmf.models.model import OasisModel
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
    def test_model_and_kwargs_are_not_set___result_is_null(self):
        profile = OasisExposuresManager().load_canonical_exposures_profile()

        self.assertEqual(None, profile)

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


class OasisExposureManagerCreateModel(TestCase):

    @given(
        supplier_id=text(min_size=1),
        model_id=text(min_size=1),
        version_id=text(min_size=1)
    )
    def test_supplier_and_model_and_version_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)
        
        model = OasisExposuresManager().create_model(supplier_id, model_id, version_id)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        expected_oasis_files_path = os.path.abspath(os.path.join('Files', expected_key.replace('/', '-')))
        self.assertEquals(expected_oasis_files_path, model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], OasisFilesPipeline))

        self.assertIsNone(model.resources.get('canonical_exposures_profile'))

    @given(
        supplier_id=text(min_size=1),
        model_id=text(min_size=1),
        version_id=text(min_size=1),
        oasis_files_path=text(min_size=1)
    )
    def test_supplier_and_model_and_version_and_absolute_oasis_files_path_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        oasis_files_path
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={'oasis_files_path': os.path.abspath(oasis_files_path)}

        model = OasisExposuresManager().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        self.assertEqual(resources, model.resources)

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], OasisFilesPipeline))

        self.assertIsNone(model.resources.get('canonical_exposures_profile'))

    @given(
        supplier_id=text(min_size=1),
        model_id=text(min_size=1),
        version_id=text(min_size=1),
        oasis_files_path=text(min_size=1)
    )
    def test_supplier_and_model_and_version_and_relative_oasis_files_path_only_are_supplied___correct_model_is_returned_with_absolute_oasis_file_path(
        self,
        supplier_id,
        model_id,
        version_id,
        oasis_files_path
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        oasis_files_path = oasis_files_path.lstrip(os.path.sep)

        resources={'oasis_files_path': oasis_files_path}

        model = OasisExposuresManager().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        self.assertTrue(os.path.isabs(model.resources['oasis_files_path']))
        self.assertEqual(os.path.abspath(resources['oasis_files_path']), model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], OasisFilesPipeline))

        self.assertIsNone(model.resources.get('canonical_exposures_profile'))

    @given(
        supplier_id=text(min_size=1),
        model_id=text(min_size=1),
        version_id=text(min_size=1),
        canonical_exposures_profile=dictionaries(text(min_size=1), text(min_size=1))
    )
    def test_supplier_and_model_and_version_and_canonical_exposures_profile_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        canonical_exposures_profile
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={'canonical_exposures_profile': canonical_exposures_profile}

        model = OasisExposuresManager().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        self.assertEqual(resources['canonical_exposures_profile'], model.resources['canonical_exposures_profile'])

        expected_oasis_files_path = os.path.abspath(os.path.join('Files', expected_key.replace('/', '-')))
        self.assertEquals(expected_oasis_files_path, model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], OasisFilesPipeline))

    @given(
        supplier_id=text(min_size=1),
        model_id=text(min_size=1),
        version_id=text(min_size=1),
        oasis_files_path=text(min_size=1),
        canonical_exposures_profile=dictionaries(text(min_size=1), text(min_size=1))
    )
    def test_supplier_and_model_and_version_and_relative_oasis_files_path_and_canonical_exposures_profile_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        oasis_files_path,
        canonical_exposures_profile
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={'oasis_files_path': oasis_files_path, 'canonical_exposures_profile': canonical_exposures_profile}

        model = OasisExposuresManager().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        self.assertTrue(os.path.isabs(model.resources['oasis_files_path']))
        self.assertEqual(os.path.abspath(resources['oasis_files_path']), model.resources['oasis_files_path'])

        self.assertEqual(resources['canonical_exposures_profile'], model.resources['canonical_exposures_profile'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], OasisFilesPipeline))

    @given(
        supplier_id=text(min_size=1),
        model_id=text(min_size=1),
        version_id=text(min_size=1),
        oasis_files_path=text(min_size=1),
        canonical_exposures_profile=dictionaries(text(min_size=1), text(min_size=1))
    )
    def test_supplier_and_model_and_version_and_absolute_oasis_files_path_and_canonical_exposures_profile_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        oasis_files_path,
        canonical_exposures_profile
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={'oasis_files_path': os.path.abspath(oasis_files_path), 'canonical_exposures_profile': canonical_exposures_profile}

        model = OasisExposuresManager().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        self.assertEqual(resources['oasis_files_path'], model.resources['oasis_files_path'])

        self.assertEqual(resources['canonical_exposures_profile'], model.resources['canonical_exposures_profile'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], OasisFilesPipeline))

    @given(
        supplier_id=text(min_size=1),
        model_id=text(min_size=1),
        version_id=text(min_size=1),
        source_accounts_file_path=text(min_size=1)
    )
    def test_supplier_and_model_and_version_and_source_accounts_file_path_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        source_accounts_file_path
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={'source_accounts_file_path': source_accounts_file_path}
        
        model = OasisExposuresManager().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        expected_oasis_files_path = os.path.abspath(os.path.join('Files', expected_key.replace('/', '-')))
        self.assertEquals(expected_oasis_files_path, model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], OasisFilesPipeline))

        self.assertEquals(resources['source_accounts_file_path'], model.resources['source_accounts_file_path'])

        self.assertIsNone(model.resources.get('canonical_exposures_profile'))
        self.assertIsNone(model.resources.get('canonical_accounts_profile'))

    @given(
        supplier_id=text(min_size=1),
        model_id=text(min_size=1),
        version_id=text(min_size=1),
        source_accounts_file_path=text(min_size=1),
        canonical_accounts_profile=dictionaries(text(min_size=1), text(min_size=1))
    )
    def test_supplier_and_model_and_version_and_source_accounts_file_path_and_canonical_accounts_profile_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        source_accounts_file_path,
        canonical_accounts_profile
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={'source_accounts_file_path': source_accounts_file_path, 'canonical_accounts_profile': canonical_accounts_profile}
        
        model = OasisExposuresManager().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        expected_oasis_files_path = os.path.abspath(os.path.join('Files', expected_key.replace('/', '-')))
        self.assertEquals(expected_oasis_files_path, model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], OasisFilesPipeline))

        self.assertIsNone(model.resources.get('canonical_exposures_profile'))

        self.assertEqual(resources['source_accounts_file_path'], model.resources['source_accounts_file_path'])
        
        self.assertEqual(resources['canonical_accounts_profile'], model.resources['canonical_accounts_profile'])

    @given(
        supplier_id=text(min_size=1),
        model_id=text(min_size=1),
        version_id=text(min_size=1),
        canonical_exposures_profile=dictionaries(text(min_size=1), text(min_size=1)),
        source_accounts_file_path=text(min_size=1),
        canonical_accounts_profile=dictionaries(text(min_size=1), text(min_size=1))
    )
    def test_supplier_and_model_and_version_and_canonical_exposures_profile_and_source_accounts_file_path_and_canonical_accounts_profile_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        canonical_exposures_profile,
        source_accounts_file_path,
        canonical_accounts_profile
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={
            'canonical_exposures_profile': canonical_exposures_profile,
            'source_accounts_file_path': source_accounts_file_path,
            'canonical_accounts_profile': canonical_accounts_profile
        }
        
        model = OasisExposuresManager().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        expected_oasis_files_path = os.path.abspath(os.path.join('Files', expected_key.replace('/', '-')))
        self.assertEquals(expected_oasis_files_path, model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], OasisFilesPipeline))

        self.assertEqual(resources['canonical_exposures_profile'], model.resources.get('canonical_exposures_profile'))

        self.assertEqual(resources['source_accounts_file_path'], model.resources['source_accounts_file_path'])
        
        self.assertEqual(resources['canonical_accounts_profile'], model.resources['canonical_accounts_profile'])

    @given(
        supplier_id=text(min_size=1),
        model_id=text(min_size=1),
        version_id=text(min_size=1),
        oasis_files_path=text(min_size=1),
        canonical_exposures_profile=dictionaries(text(min_size=1), text(min_size=1)),
        source_accounts_file_path=text(min_size=1),
        canonical_accounts_profile=dictionaries(text(min_size=1), text(min_size=1))
    )
    def test_supplier_and_model_and_version_and_absolute_oasis_files_path_and_canonical_exposures_profile_and_source_accounts_file_path_and_canonical_accounts_profile_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        oasis_files_path,
        canonical_exposures_profile,
        source_accounts_file_path,
        canonical_accounts_profile
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={
            'oasis_files_path': os.path.abspath(oasis_files_path),
            'canonical_exposures_profile': canonical_exposures_profile,
            'source_accounts_file_path': source_accounts_file_path,
            'canonical_accounts_profile': canonical_accounts_profile
        }
        
        model = OasisExposuresManager().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        self.assertEquals(resources['oasis_files_path'], model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], OasisFilesPipeline))

        self.assertEqual(resources['canonical_exposures_profile'], model.resources.get('canonical_exposures_profile'))

        self.assertEqual(resources['source_accounts_file_path'], model.resources['source_accounts_file_path'])
        
        self.assertEqual(resources['canonical_accounts_profile'], model.resources['canonical_accounts_profile'])


class OasisExposureManagerLoadCanonicalAccountsProfile(TestCase):
    def test_model_and_kwargs_are_not_set___result_is_null(self):
        profile = OasisExposuresManager().load_canonical_accounts_profile()

        self.assertEqual(None, profile)

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


class OasisExposureManagerLoadGulItems(TestCase):

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
                OasisExposuresManager().load_gul_items(profile, exposures_file.name, keys_file.name)

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
                OasisExposuresManager().load_gul_items(profile, exposures_file.name, keys_file.name)

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
                OasisExposuresManager().load_gul_items(profile, exposures_file.name, keys_file.name)

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
                OasisExposuresManager().load_gul_items(profile, exposures_file.name, keys_file.name)

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
                OasisExposuresManager().load_gul_items(profile, exposures_file.name, keys_file.name)


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
        gcep = canonical_profiles_fm_terms_grouped_by_level_and_term_type(canonical_profiles=(profile,))

        for k in keys:
            k['id'] += 5

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            matching_canonical_and_keys_item_ids = set(k['id'] for k in keys).intersection([e['row_id'] for e in exposures])

            gul_items_df, canexp_df = OasisExposuresManager().load_gul_items(profile, exposures_file.name, keys_file.name)

        get_canonical_item = lambda i: (
            [e for e in exposures if e['row_id'] == i + 1][0] if len([e for e in exposures if e['row_id'] == i + 1]) == 1
            else None
        )

        get_keys_item = lambda i: (
            [k for k in keys if k['id'] == i + 1][0] if len([k for k in keys if k['id'] == i + 1]) == 1
            else None
        )

        tiv_elements = tuple(t for t in [gcep[1][gid].get('tiv') for gid in gcep[1]] if t)

        fm_term_elements = {
            tiv_tgid: {
                term_type: (
                    gcep[1][tiv_tgid][term_type]['ProfileElementName'].lower() if gcep[1][tiv_tgid].get(term_type) else None
                ) if term_type != 'deductible_type' else gcep[1][tiv_tgid]['deductible']['DeductibleType']if gcep[1][tiv_tgid].get('deductible') else 'B'
                for term_type in ('limit', 'deductible', 'deductible_type', 'share',)
            } for tiv_tgid in gcep[1]
        }

        for i, gul_it in enumerate(gul_items_df.T.to_dict().values()):
            can_it = get_canonical_item(int(gul_it['canexp_id']))
            self.assertIsNotNone(can_it)

            keys_it = get_keys_item(int(gul_it['canexp_id']))
            self.assertIsNotNone(keys_it)

            positive_tiv_elements = [t for t in tiv_elements if can_it.get(t['ProfileElementName'].lower()) and can_it[t['ProfileElementName'].lower()] > 0 and t['CoverageTypeID'] == keys_it['coverage']]

            for _, t in enumerate(positive_tiv_elements):
                tiv_elm = t['ProfileElementName'].lower()
                self.assertEqual(tiv_elm, gul_it['tiv_elm'])
                
                tiv_tgid = t['FMTermGroupID']
                self.assertEqual(can_it[tiv_elm], gul_it['tiv'])
                
                lim_elm = fm_term_elements[tiv_tgid]['limit']
                self.assertEqual(lim_elm, gul_it['lim_elm'])
                
                ded_elm = fm_term_elements[tiv_tgid]['deductible']
                self.assertEqual(ded_elm, gul_it['ded_elm'])
                
                ded_type = fm_term_elements[tiv_tgid]['deductible_type']
                self.assertEqual(ded_type, gul_it['ded_type'])
                
                shr_elm = fm_term_elements[tiv_tgid]['share']
                self.assertEqual(shr_elm, gul_it['shr_elm'])

            self.assertEqual(keys_it['area_peril_id'], gul_it['areaperil_id'])
            self.assertEqual(keys_it['vulnerability_id'], gul_it['vulnerability_id'])

            self.assertEqual(i + 1, gul_it['item_id'])

            self.assertEqual(i + 1, gul_it['coverage_id'])

            self.assertEqual(i + 1, gul_it['group_id'])


class OasisExposureManagerLoadFmItems(TestCase):

    def setUp(self):
        self.exposures_profile = canonical_exposures_profile_piwind_simple
        self.accounts_profile = canonical_accounts_profile_piwind
        self.combined_grouped_canonical_profile = canonical_profiles_fm_terms_grouped_by_level_and_term_type(
            canonical_profiles=[self.exposures_profile, self.accounts_profile]
        )

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
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
                fm_df, canacc_df = OasisExposuresManager().load_fm_items(
                    pd.DataFrame(data=exposures),
                    pd.DataFrame(data=guls),
                    self.exposures_profile,
                    self.accounts_profile,
                    accounts_file.name
                )

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(
            from_accounts_nums=just(10101),
            from_tivs1=just(100),
            from_limits1=just(1),
            from_deductibles1=just(1),
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
        guls=gul_items_data(
            from_tiv_elements=just('wscv1val'),
            from_tivs=just(100),
            from_tiv_tgids=just(1),
            from_limit_elements=just('wscv1limit'),
            from_deductible_elements=just('wscv1ded'),
            from_share_elements=just(None),
            size=10
        )
    )
    def test_with_one_account_and_one_top_level_layer_load_items_with_preset_data_only(
        self,
        exposures,
        accounts,
        guls
    ):
        cgcp = self.combined_grouped_canonical_profile

        for _, gul in enumerate(guls):
            gul['ded_type'] = cgcp[1][gul['tiv_tgid']]['deductible']['DeductibleType'] if cgcp[1][gul['tiv_tgid']].get('deductible') else 'B'

        canexp_df, gul_items_df = (pd.DataFrame(data=its, dtype=object) for its in [exposures, guls])

        for df in [canexp_df, gul_items_df]:
            df = df.where(df.notnull(), None)
            df.columns = df.columns.str.lower()
        
        canexp_df['index'] = pd.Series(data=list(canexp_df.index), dtype=int)
        gul_items_df['index'] = pd.Series(data=list(gul_items_df.index), dtype=int)

        with NamedTemporaryFile('w') as accounts_file:
            write_canonical_files(canonical_accounts=accounts, canonical_accounts_file_path=accounts_file.name)

            preset_fm_items = OasisExposuresManager().load_fm_items(
                canexp_df,
                gul_items_df,
                self.exposures_profile,
                self.accounts_profile,
                accounts_file.name,
                preset_only=True
            )[0].T.to_dict().values()

        num_top_level_layers = len(set(a['policynum'] for a in accounts))
        bottom_levels = sorted(cgcp.keys())[:-1]

        self.assertEquals(len(preset_fm_items), (len(bottom_levels) + num_top_level_layers) * len(guls))

        get_gul_item = lambda i: guls[i % len(guls)]

        for i, (l, it) in enumerate(itertools.chain((l, it) for l in sorted(cgcp.keys()) for l, it in itertools.product([l],(it for it in preset_fm_items if it['level_id'] == l)))):
            self.assertEquals(it['level_id'], l)

            gul_it = get_gul_item(i)

            self.assertEquals(it['canexp_id'], gul_it['canexp_id'])

            self.assertEquals(it['canacc_id'], 0)

            self.assertEquals(it['layer_id'], 1)

            self.assertEquals(it['gul_item_id'], gul_it['item_id'])

            self.assertEquals(it['tiv_elm'], gul_it['tiv_elm'])
            self.assertEquals(it['tiv_tgid'], gul_it['tiv_tgid'])
            self.assertEquals(it['tiv'], gul_it['tiv'])

            self.assertEquals(it['lim_elm'], gul_it['lim_elm'])
            self.assertEquals(it['ded_elm'], gul_it['ded_elm'])
            self.assertEquals(it['shr_elm'], gul_it['shr_elm'])

            self.assertEquals(it['limit'], 0)
            self.assertEquals(it['deductible'], 0)
            self.assertEquals(it['deductible_type'], 'B')
            self.assertEquals(it['share'], 0)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(
            from_accounts_nums=just(10101),
            from_tivs1=just(100),
            from_tivs2=just(0),
            from_limits1=just(1),
            from_limits2=just(0),
            from_deductibles1=just(1),
            from_deductibles2=just(0),
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
        guls=gul_items_data(
            from_tiv_elements=just('wscv1val'),
            from_tivs=just(100),
            from_tiv_tgids=just(1),
            from_limit_elements=just('wscv1limit'),
            from_deductible_elements=just('wscv1ded'),
            from_share_elements=just(None),
            size=10
        )
    )
    def test_with_one_account_and_one_top_level_layer_load_items_with_all_fm_terms_present(
        self,
        exposures,
        accounts,
        guls
    ):
        cgcp = self.combined_grouped_canonical_profile

        for _, gul in enumerate(guls):
            gul['ded_type'] = cgcp[1][gul['tiv_tgid']]['deductible']['DeductibleType'] if cgcp[1][gul['tiv_tgid']].get('deductible') else 'B'

        canexp_df, gul_items_df = (pd.DataFrame(data=its, dtype=object) for its in [exposures, guls])

        for df in [canexp_df, gul_items_df]:
            df = df.where(df.notnull(), None)
            df.columns = df.columns.str.lower()
        
        canexp_df['index'] = pd.Series(data=list(canexp_df.index), dtype=int)
        gul_items_df['index'] = pd.Series(data=list(gul_items_df.index), dtype=int)

        with NamedTemporaryFile('w') as accounts_file:
            write_canonical_files(canonical_accounts=accounts, canonical_accounts_file_path=accounts_file.name)

            fm_items = OasisExposuresManager().load_fm_items(
                canexp_df,
                gul_items_df,
                self.exposures_profile,
                self.accounts_profile,
                accounts_file.name
            )[0].T.to_dict().values()

        num_top_level_layers = len(set(a['policynum'] for a in accounts))
        bottom_levels = sorted(cgcp.keys())[:-1]

        self.assertEquals(len(fm_items), (len(bottom_levels) + num_top_level_layers) * len(guls))

        get_can_item = lambda i: {
            k:v for k, v in itertools.chain(
                ((k if k != 'row_id' else 'canexp_id', v if k != 'row_id' else v - 1) for k, v in exposures[guls[i % len(guls)]['canexp_id']].items()),
                ((k if k != 'row_id' else 'canacc_id', v if k != 'row_id' else v - 1) for k, v in accounts[0].items())
            )
        }

        get_gul_item = lambda i: guls[i % len(guls)]

        for i, (l, it) in enumerate(itertools.chain((l, it) for l in sorted(cgcp.keys()) for l, it in itertools.product([l],(it for it in fm_items if it['level_id'] == l)))):
            self.assertEquals(it['level_id'], l)

            gul_it = get_gul_item(i)

            self.assertEquals(it['canexp_id'], gul_it['canexp_id'])

            self.assertEquals(it['canacc_id'], 0)

            self.assertEquals(it['layer_id'], 1)

            self.assertEquals(it['gul_item_id'], gul_it['item_id'])

            self.assertEquals(it['tiv_elm'], gul_it['tiv_elm'])
            self.assertEquals(it['tiv_tgid'], gul_it['tiv_tgid'])
            self.assertEquals(it['tiv'], gul_it['tiv'])

            self.assertEquals(it['lim_elm'], gul_it['lim_elm'])
            self.assertEquals(it['ded_elm'], gul_it['ded_elm'])
            self.assertEquals(it['shr_elm'], gul_it['shr_elm'])

            can_it = get_can_item(i)

            lim = can_it.get(gul_it['lim_elm'] if l == 1 else (cgcp[l][1]['limit']['ProfileElementName'].lower() if cgcp[l][1].get('limit') else None)) or 0.0
            self.assertEquals(it['limit'], lim)
            
            ded = can_it.get(gul_it['ded_elm']  if l == 1 else (cgcp[l][1]['deductible']['ProfileElementName'].lower() if cgcp[l][1].get('deductible') else None)) or 0.0
            self.assertEquals(it['deductible'], ded)
            
            ded_type = gul_it['ded_type'] if l == 1 else (cgcp[l][1]['deductible']['DeductibleType'] if cgcp[l][1].get('deductible') else 'B')
            self.assertEquals(it['deductible_type'], ded_type)
            
            shr = can_it.get(gul_it['shr_elm']  if l == 1 else (cgcp[l][1]['share']['ProfileElementName'].lower() if cgcp[l][1].get('share') else None)) or 0.0
            self.assertEquals(it['share'], shr)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(
            from_accounts_nums=just(10101),
            from_tivs1=just(100),
            from_tivs2=just(0),
            from_limits1=just(1),
            from_limits2=just(0),
            from_deductibles1=just(1),
            from_deductibles2=just(0),
            size=10
        ),
        accounts=canonical_accounts_data(
            from_accounts_nums=just(10101),
            from_attachment_points=just(1),
            from_blanket_deductibles=just(1),
            from_blanket_limits=just(1),
            from_layer_limits=just(1),
            from_policy_types=just(1),
            size=2
        ),
        guls=gul_items_data(
            from_tiv_elements=just('wscv1val'),
            from_tivs=just(100),
            from_tiv_tgids=just(1),
            from_limit_elements=just('wscv1limit'),
            from_deductible_elements=just('wscv1ded'),
            from_share_elements=just(None),
            size=10
        )
    )
    def test_with_one_account_and_two_top_level_layers_load_items_with_all_fm_terms_present(
        self,
        exposures,
        accounts,
        guls
    ):
        cgcp = self.combined_grouped_canonical_profile

        for i, acc in enumerate(accounts):
            acc['policynum'] = 'Layer{}'.format(i + 1)

        for _, gul in enumerate(guls):
            gul['ded_type'] = cgcp[1][gul['tiv_tgid']]['deductible']['DeductibleType'] if cgcp[1][gul['tiv_tgid']].get('deductible') else 'B'

        canexp_df, gul_items_df = (pd.DataFrame(data=its, dtype=object) for its in [exposures, guls])

        for df in [canexp_df, gul_items_df]:
            df = df.where(df.notnull(), None)
            df.columns = df.columns.str.lower()
        
        canexp_df['index'] = pd.Series(data=list(canexp_df.index), dtype=int)
        gul_items_df['index'] = pd.Series(data=list(gul_items_df.index), dtype=int)

        with NamedTemporaryFile('w') as accounts_file:
            write_canonical_files(canonical_accounts=accounts, canonical_accounts_file_path=accounts_file.name)

            fm_items = OasisExposuresManager().load_fm_items(
                canexp_df,
                gul_items_df,
                self.exposures_profile,
                self.accounts_profile,
                accounts_file.name
            )[0].T.to_dict().values()

        fm_levels = sorted(cgcp.keys())
        num_top_level_layers = len(set(a['policynum'] for a in accounts))
        bottom_levels = fm_levels[:-1]

        self.assertEquals(len(fm_items), (len(bottom_levels) + num_top_level_layers) * len(guls))

        get_can_item = lambda i, layer_id: {
            k:v for k, v in itertools.chain(
                ((k if k != 'row_id' else 'canexp_id', v if k != 'row_id' else v - 1) for k, v in exposures[guls[i % len(guls)]['canexp_id']].items()),
                ((k if k != 'row_id' else 'canacc_id', v if k != 'row_id' else v - 1) for k, v in (accounts[0].items() if layer_id == 1 else accounts[1].items()))
            )
        }

        get_gul_item = lambda i: guls[i % len(guls)]

        for i, (l, it) in enumerate(itertools.chain((l, it) for l in sorted(cgcp.keys()) for l, it in itertools.product([l],(it for it in fm_items if it['level_id'] == l)))):
            self.assertEquals(it['level_id'], l)

            gul_it = get_gul_item(i)
            
            self.assertEquals(it['canexp_id'], gul_it['canexp_id'])

            layer_id = 1 if i < len(fm_levels) * len(guls) else 2
            self.assertEquals(it['layer_id'], layer_id)

            can_it = get_can_item(i, layer_id)

            self.assertEquals(it['canacc_id'], can_it['canacc_id'])

            self.assertEquals(it['gul_item_id'], gul_it['item_id'])

            self.assertEquals(it['tiv_elm'], gul_it['tiv_elm'])
            self.assertEquals(it['tiv_tgid'], gul_it['tiv_tgid'])
            self.assertEquals(it['tiv'], gul_it['tiv'])

            self.assertEquals(it['lim_elm'], gul_it['lim_elm'])
            self.assertEquals(it['ded_elm'], gul_it['ded_elm'])
            self.assertEquals(it['shr_elm'], gul_it['shr_elm'])

            lim = can_it.get(gul_it['lim_elm'] if l == 1 else (cgcp[l][1]['limit']['ProfileElementName'].lower() if cgcp[l][1].get('limit') else None)) or 0.0
            self.assertEquals(it['limit'], lim)
            
            ded = can_it.get(gul_it['ded_elm']  if l == 1 else (cgcp[l][1]['deductible']['ProfileElementName'].lower() if cgcp[l][1].get('deductible') else None)) or 0.0
            self.assertEquals(it['deductible'], ded)
            
            ded_type = gul_it['ded_type'] if l == 1 else (cgcp[l][1]['deductible']['DeductibleType'] if cgcp[l][1].get('deductible') else 'B')
            self.assertEquals(it['deductible_type'], ded_type)
            
            shr = can_it.get(gul_it['shr_elm']  if l == 1 else (cgcp[l][1]['share']['ProfileElementName'].lower() if cgcp[l][1].get('share') else None)) or 0.0
            self.assertEquals(it['share'], shr)


class GulFilesGenerationTestCase(TestCase):

    def setUp(self):
        self.profile = canonical_exposures_profile_piwind_simple
        self.manager = OasisExposuresManager()

    def check_items_file(self, gul_items_df, items_file_path):
        expected = tuple(
            {
                k:it[k] for k in ('item_id', 'coverage_id', 'areaperil_id', 'vulnerability_id', 'group_id',)
            } for _, it in gul_items_df.iterrows()
        )

        with io.open(items_file_path, 'r', encoding='utf-8') as f:
            result = tuple(pd.read_csv(f).T.to_dict().values())

        self.assertEqual(expected, result)

    def check_coverages_file(self, gul_items_df, coverages_file_path):
        expected = tuple(
            {
                k:it[k] for k in ('coverage_id', 'tiv',)
            } for _, it in gul_items_df.iterrows()
        )

        with io.open(coverages_file_path, 'r', encoding='utf-8') as f:
            result = tuple(pd.read_csv(f).T.to_dict().values())

        self.assertEqual(expected, result)

    def check_gulsummaryxref_file(self, gul_items_df, gulsummaryxref_file_path):
        expected = tuple(
            {
                k:it[k] for k in ('coverage_id', 'summary_id', 'summaryset_id',)
            } for _, it in gul_items_df.iterrows()
        )

        with io.open(gulsummaryxref_file_path, 'r', encoding='utf-8') as f:
            result = tuple(pd.read_csv(f).T.to_dict().values())

        self.assertEqual(expected, result)


class FmFilesGenerationTestCase(TestCase):

    def setUp(self):
        self.exposures_profile = canonical_exposures_profile_piwind_simple
        self.accounts_profile = canonical_accounts_profile_piwind
        self.combined_grouped_canonical_profile = canonical_profiles_fm_terms_grouped_by_level_and_term_type(
            canonical_profiles=(self.exposures_profile, self.accounts_profile,)
        )
        self.manager = OasisExposuresManager()

    def check_fm_policytc_file(self, fm_items_df, fm_policytc_file_path):
        expected = tuple(
            {
                k:it[k] for k in ('layer_id', 'level_id', 'agg_id', 'policytc_id',)
            } for _, it in fm_items_df.iterrows()
        )

        with io.open(fm_policytc_file_path, 'r', encoding='utf-8') as f:
            result = tuple(pd.read_csv(f).T.to_dict().values())

        self.assertEqual(expected, result)


class OasisExposureManagerWriteGulFiles(GulFilesGenerationTestCase):

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
        profile = self.profile

        model = fake_model(resources={'canonical_exposures_profile': profile})

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            omr = model.resources
            ofp = omr['oasis_files_pipeline']

            ofp.keys_file_path = keys_file.name
            ofp.canonical_exposures_file_path = exposures_file.name

            ofp.items_file_path = os.path.join(out_dir, 'items.csv')
            ofp.coverages_file_path = os.path.join(out_dir, 'coverages.csv')
            ofp.gulsummaryxref_file_path = os.path.join(out_dir, 'gulsummaryxref.csv')

            gul_files = self.manager.write_gul_files(oasis_model=model)

            gul_items_df = omr['gul_items_df']

            self.check_items_file(gul_items_df, gul_files['items'])
            self.check_coverages_file(gul_items_df, gul_files['coverages'])
            self.check_gulsummaryxref_file(gul_items_df, gul_files['gulsummaryxref'])

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
        profile = self.profile

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            gul_items_df, _ = self.manager.load_gul_items(profile, exposures_file.name, keys_file.name)

            gul_files = self.manager.write_gul_files(
                canonical_exposures_profile=profile,
                keys_file_path=keys_file.name,
                canonical_exposures_file_path=exposures_file.name,
                items_file_path=os.path.join(out_dir, 'items.csv'),
                coverages_file_path=os.path.join(out_dir, 'coverages.csv'),
                gulsummaryxref_file_path=os.path.join(out_dir, 'gulsummaryxref.csv')
            )

            self.check_items_file(gul_items_df, gul_files['items'])
            self.check_coverages_file(gul_items_df, gul_files['coverages'])
            self.check_gulsummaryxref_file(gul_items_df, gul_files['gulsummaryxref'])


class OasisExposureManagerWriteFmFiles(FmFilesGenerationTestCase):

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(
            from_accounts_nums=just(10101),
            from_tivs1=just(100),
            from_limits1=just(1),
            from_deductibles1=just(1),
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
        guls=gul_items_data(
            from_tiv_elements=just('wscv1val'),
            from_tivs=just(100),
            from_tiv_tgids=just(1),
            from_limit_elements=just('wscv1limit'),
            from_deductible_elements=just('wscv1ded'),
            from_share_elements=just(None),
            size=10
        )
    )
    def test_paths_are_stored_in_the_model___model_paths_are_used(self, exposures, accounts, guls):
        cep = self.exposures_profile
        cap = self.accounts_profile
        cgcp = self.combined_grouped_canonical_profile

        for _, gul in enumerate(guls):
            gul['ded_type'] = cgcp[1][gul['tiv_tgid']]['deductible']['DeductibleType'] if cgcp[1][gul['tiv_tgid']].get('deductible') else 'B'

        canexp_df, gul_items_df = (pd.DataFrame(data=its, dtype=object) for its in [exposures, guls])

        for df in [canexp_df, gul_items_df]:
            df = df.where(df.notnull(), None)
            df.columns = df.columns.str.lower()
        
        canexp_df['index'] = pd.Series(data=list(canexp_df.index), dtype=int)
        gul_items_df['index'] = pd.Series(data=list(gul_items_df.index), dtype=int)

        with NamedTemporaryFile('w') as accounts_file, TemporaryDirectory() as out_dir:
            write_canonical_files(canonical_accounts=accounts, canonical_accounts_file_path=accounts_file.name)

            model = fake_model(resources={
                'canonical_exposures_df': canexp_df,
                'gul_items_df': gul_items_df,
                'canonical_exposures_profile': cep,
                'canonical_accounts_profile': cap
            })
            omr = model.resources
            ofp = omr['oasis_files_pipeline']
            
            ofp.canonical_accounts_file_path = accounts_file.name
            ofp.fm_policytc_file_path = os.path.join(out_dir, 'fm_policytc.csv')

            fm_files = self.manager.write_fm_files(oasis_model=model)

            fm_items_df = omr['fm_items_df']

            self.check_fm_policytc_file(fm_items_df, fm_files['fm_policytc'])

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposures_data(
            from_accounts_nums=just(10101),
            from_tivs1=just(100),
            from_limits1=just(1),
            from_deductibles1=just(1),
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
        guls=gul_items_data(
            from_tiv_elements=just('wscv1val'),
            from_tivs=just(100),
            from_tiv_tgids=just(1),
            from_limit_elements=just('wscv1limit'),
            from_deductible_elements=just('wscv1ded'),
            from_share_elements=just(None),
            size=10
        )
    )
    def test_paths_are_stored_in_the_kwargs___kwarg_paths_are_used(self, exposures, accounts, guls):
        cep = self.exposures_profile
        cap = self.accounts_profile
        cgcp = self.combined_grouped_canonical_profile

        for _, gul in enumerate(guls):
            gul['ded_type'] = cgcp[1][gul['tiv_tgid']]['deductible']['DeductibleType'] if cgcp[1][gul['tiv_tgid']].get('deductible') else 'B'

        canexp_df, gul_items_df = (pd.DataFrame(data=its, dtype=object) for its in [exposures, guls])

        for df in [canexp_df, gul_items_df]:
            df = df.where(df.notnull(), None)
            df.columns = df.columns.str.lower()
        
        canexp_df['index'] = pd.Series(data=list(canexp_df.index), dtype=int)
        gul_items_df['index'] = pd.Series(data=list(gul_items_df.index), dtype=int)

        with NamedTemporaryFile('w') as accounts_file, TemporaryDirectory() as out_dir:
            write_canonical_files(canonical_accounts=accounts, canonical_accounts_file_path=accounts_file.name)

            fm_items_df, canacc_df = self.manager.load_fm_items(
                canexp_df,
                gul_items_df,
                cep,
                cap,
                accounts_file.name
            )
            
            os.path.join(out_dir, 'fm_policytc.csv')

            fm_files = self.manager.write_fm_files(
                canonical_exposures_df=canexp_df,
                gul_items_df=gul_items_df,
                canonical_exposures_profile=cep,
                canonical_accounts_profile=cap,
                canonical_accounts_file_path=accounts_file.name,
                fm_policytc_file_path=os.path.join(out_dir, 'fm_policytc.csv')
            )

            self.check_fm_policytc_file(fm_items_df, fm_files['fm_policytc'])


class OasisExposureManagerStartOasisFilesPipeline(TestCase):

    def setUp(self):
        self.manager = OasisExposuresManager()
        self.exposures_profile = canonical_exposures_profile_piwind_simple
        self.accounts_profile = canonical_accounts_profile_piwind

    def test_start_oasis_files_pipeline_with_model_and_no_oasis_files_path__oasis_exception_is_raised(self):
        mgr = self.manager
        model = fake_model(resources={})

        with self.assertRaises(OasisException):
            mgr.start_oasis_files_pipeline(oasis_model=model)

    def test_start_oasis_files_pipeline_with_model_and_invalid_oasis_files_path__oasis_exception_is_raised(self):
        mgr = self.manager

        oasis_files_path = None
        with TemporaryDirectory() as d:
            oasis_files_path = d

        model = fake_model(resources={'oasis_files_path': oasis_files_path})

        with self.assertRaises(OasisException):
            mgr.start_oasis_files_pipeline(oasis_model=model)

    def test_start_oasis_files_pipeline_with_model_and_no_source_exposures_files_path__oasis_exception_is_raised(self):
        mgr = self.manager

        with TemporaryDirectory() as oasis_files_path:
            model = fake_model(resources={'oasis_files_path': oasis_files_path})

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_model=model)

    def test_start_oasis_files_pipeline_with_model_and_no_canonical_exposures_profile__oasis_exception_is_raised(self):
        mgr = self.manager

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposures_file:
            model = fake_model(resources={
                'oasis_files_path': oasis_files_path,
                'source_exposures_file_path': source_exposures_file.name
            })

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_model=model)

    def test_start_oasis_files_pipeline_with_model_and_fm_and_no_source_accounts_file_path__oasis_exception_is_raised(self):
        mgr = self.manager
        cep = self.exposures_profile

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposures_file:
            model = fake_model(resources={
                'oasis_files_path': oasis_files_path,
                'source_exposures_file_path': source_exposures_file.name,
                'canonical_exposures_profile': cep,
                'fm': True
            })

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_model=model)

    def test_start_oasis_files_pipeline_with_model_and_fm_set_in_kwargs_and_no_source_accounts_file_path__oasis_exception_is_raised(self):
        mgr = self.manager
        cep = self.exposures_profile

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposures_file:
            model = fake_model(resources={
                'oasis_files_path': oasis_files_path,
                'source_exposures_file_path': source_exposures_file.name,
                'canonical_exposures_profile': cep
            })

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_model=model, fm=True)

    def test_start_oasis_files_pipeline_with_model_and_fm_and_no_canonical_accounts_profile__oasis_exception_is_raised(self):
        mgr = self.manager
        cep = self.exposures_profile

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposures_file:
            model = fake_model(resources={
                'oasis_files_path': oasis_files_path,
                'source_exposures_file_path': source_exposures_file.name,
                'canonical_exposures_profile': cep,
                'fm': True
            })

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_model=model)

    def test_start_oasis_files_pipeline_with_model_and_fm_set_in_kwargs_and_no_canonical_accounts_profile__oasis_exception_is_raised(self):
        mgr = self.manager
        cep = self.exposures_profile

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposures_file:
            model = fake_model(resources={
                'oasis_files_path': oasis_files_path,
                'source_exposures_file_path': source_exposures_file.name,
                'canonical_exposures_profile': cep
            })

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_model=model, fm=True)

    def test_start_oasis_files_pipeline_with_kwargs_and_no_oasis_files_path__oasis_exception_is_raised(self):
        mgr = self.manager

        with self.assertRaises(OasisException):
            mgr.start_oasis_files_pipeline()

    def test_start_oasis_files_pipeline_with_kwargs_and_invalid_oasis_files_path__oasis_exception_is_raised(self):
        mgr = self.manager

        oasis_files_path = None
        with TemporaryDirectory() as d:
            oasis_files_path = d

        with self.assertRaises(OasisException):
            mgr.start_oasis_files_pipeline(oasis_files_path=oasis_files_path)

    def test_start_oasis_files_pipeline_with_kwargs_and_no_source_exposures_files_path__oasis_exception_is_raised(self):
        mgr = self.manager

        with TemporaryDirectory() as oasis_files_path:

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_files_path=oasis_files_path)

    def test_start_oasis_files_pipeline_with_kwargs_and_no_canonical_exposures_profile__oasis_exception_is_raised(self):
        mgr = self.manager

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposures_file:

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(
                    oasis_files_path=oasis_files_path,
                    source_exposures_file_path=source_exposures_file.name
                )

    def test_start_oasis_files_pipeline_with_kwargs_and_fm_and_no_source_accounts_file_path__oasis_exception_is_raised(self):
        mgr = self.manager
        cep = self.exposures_profile

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposures_file:

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(
                    oasis_files_path=oasis_files_path,
                    source_exposures_file_path=source_exposures_file.name,
                    canonical_exposures_profile=cep,
                    fm=True
                )

    def test_start_oasis_files_pipeline_with_kwargs_and_fm_and_no_canonical_accounts_profile__oasis_exception_is_raised(self):
        mgr = self.manager
        cep = self.exposures_profile

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposures_file:

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(
                    oasis_files_path=oasis_files_path,
                    source_exposures_file_path=source_exposures_file.name,
                    canonical_exposures_profile=cep,
                    fm=True
                )