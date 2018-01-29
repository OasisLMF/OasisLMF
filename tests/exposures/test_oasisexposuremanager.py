import json
import string
from tempfile import NamedTemporaryFile
from unittest import TestCase

import os
from hypothesis import given
from hypothesis.strategies import text, dictionaries
from mock import patch, Mock

from oasislmf.exposures.manager import OasisExposuresManager
from ..models.fakes import fake_model


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
        profile = OasisExposuresManager.load_canonical_profile()

        self.assertEqual({}, profile)

    @given(dictionaries(text(), text()))
    def test_model_is_set_with_profile_json___models_profile_is_set_to_expected_json(self, expected):
        model = fake_model(resources={'canonical_exposures_profile_json': json.dumps(expected)})

        profile = OasisExposuresManager.load_canonical_profile(oasis_model=model)

        self.assertEqual(expected, profile)
        self.assertEqual(expected, model.resources['canonical_exposures_profile'])

    @given(dictionaries(text(), text()), dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_and_profile_json_is_passed_through_kwargs___kwargs_profile_is_used(self, model_profile, kwargs_profile):
        model = fake_model(resources={'canonical_exposures_profile_json': json.dumps(model_profile)})

        profile = OasisExposuresManager.load_canonical_profile(oasis_model=model, canonical_exposures_profile_json=json.dumps(kwargs_profile))

        self.assertEqual(kwargs_profile, profile)
        self.assertEqual(kwargs_profile, model.resources['canonical_exposures_profile'])

    @given(dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_path___models_profile_is_set_to_expected_json(self, expected):
        with NamedTemporaryFile('w') as f:
            json.dump(expected, f)
            f.flush()

            model = fake_model(resources={'canonical_exposures_profile_json_path': f.name})

            profile = OasisExposuresManager.load_canonical_profile(oasis_model=model)

            self.assertEqual(expected, profile)
            self.assertEqual(expected, model.resources['canonical_exposures_profile'])

    @given(dictionaries(text(), text()), dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_path_and_profile_json_path_is_passed_through_kwargs___kwargs_profile_is_used(self, model_profile, kwargs_profile):
        with NamedTemporaryFile('w') as model_file, NamedTemporaryFile('w') as kwargs_file:
            json.dump(model_profile, model_file)
            model_file.flush()
            json.dump(kwargs_profile, kwargs_file)
            kwargs_file.flush()

            model = fake_model(resources={'canonical_exposures_profile_json_path': model_file.name})

            profile = OasisExposuresManager.load_canonical_profile(oasis_model=model, canonical_exposures_profile_json_path=kwargs_file.name)

            self.assertEqual(kwargs_profile, profile)
            self.assertEqual(kwargs_profile, model.resources['canonical_exposures_profile'])


class OasisExposureManagerGetKeys(TestCase):
    def create_model(self, lookup='lookup', keys_file_path='key_file_path', exposures_file_path='exposures_file_path'):
        model = fake_model(resources={'lookup': lookup})
        model.files_pipeline.keys_file_path = keys_file_path
        model.files_pipeline.model_exposures_path = exposures_file_path
        return model

    @given(text(min_size=1, alphabet=string.ascii_letters), text(min_size=1, alphabet=string.ascii_letters), text(min_size=1, alphabet=string.ascii_letters))
    def test_model_is_supplied_kwargs_are_not___lookup_keys_file_and_exposures_file_from_model_are_used(self, lookup, keys, exposure):
        model = self.create_model(lookup=lookup, keys_file_path=keys, exposures_file_path=exposure)

        with patch('oasislmf.exposures.manager.OasisKeysLookupFactory.save_keys', Mock(return_value=(keys, 1))) as oklf_mock:
            res = OasisExposuresManager.get_keys(oasis_model=model)

            oklf_mock.assert_called_once_with(
                lookup=lookup,
                model_exposures_file_path=os.path.abspath(exposure),
                output_file_path=os.path.abspath(keys),
            )
            self.assertEqual(model.files_pipeline.keys_file_path, keys)
            self.assertEqual(res, keys)
