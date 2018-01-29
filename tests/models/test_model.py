from __future__ import unicode_literals

import json
from tempfile import NamedTemporaryFile
from unittest import TestCase

import os
from hypothesis import given
from hypothesis.strategies import text, dictionaries

from oasislmf.exposures.pipeline import OasisFilesPipeline
from oasislmf.models import OasisModel
from oasislmf.utils.exceptions import OasisException


def create_model(supplier=None, model=None, version=None, resources=None):
    supplier = supplier if supplier is not None else text().example()
    model = model if model is not None else text().example()
    version = version if version is not None else text().example()

    return OasisModel(supplier, model, version, resources=resources)


class ModelInit(TestCase):
    @given(text(), text(), text())
    def test_supplier_model_and_version_are_supplied___correct_key_is_created(self, supplier, model_id, version):
        model = create_model(supplier=supplier, model=model_id, version=version)

        self.assertEqual('{}/{}/{}'.format(supplier, model_id, version), model.key)

    def test_oasis_file_path_is_given___path_is_stored_as_absolute_path(self):
        model = create_model(resources={'oasis_files_path': 'some_path'})

        result = model.resources['oasis_files_path']
        expected = os.path.abspath('some_path')

        self.assertEqual(expected, result)

    def test_oasis_file_path_is_not_given___path_is_abs_path_of_default(self):
        model = create_model()

        result = model.resources['oasis_files_path']
        expected = os.path.abspath(os.path.join('Files', model.key.replace('/', '-')))

        self.assertEqual(expected, result)

    def test_file_pipeline_is_not_supplied___default_pipeline_is_set(self):
        model = create_model()

        pipeline = model.resources['oasis_files_pipeline']

        self.assertIsInstance(pipeline, OasisFilesPipeline)
        self.assertEqual(pipeline.model_key, model.key)

    def test_file_pipeline_is_supplied___pipeline_is_unchanged(self):
        pipeline = OasisFilesPipeline()

        model = create_model(resources={'oasis_files_pipeline': pipeline})

        self.assertIs(pipeline, model.resources['oasis_files_pipeline'])

    def test_pipeline_is_not_a_pipeline_instance___oasis_exception_is_raised(self):
        class FakePipeline(object):
            pass

        pipeline = FakePipeline()

        with self.assertRaises(OasisException):
            create_model(resources={'oasis_files_pipeline': pipeline})

    def test_exposure_file_path_is_not_supplied___source_exposure_file_is_not_set_on_pipeline(self):
        model = create_model()

        pipeline = model.resources['oasis_files_pipeline']

        self.assertIsNone(pipeline.source_exposures_file)

    def test_exposure_file_path_is_supplied___source_exposure_file_is_set_on_pipeline(self):
        with NamedTemporaryFile() as f:
            model = create_model(resources={'source_exposures_file_path': f.name})

            pipeline = model.resources['oasis_files_pipeline']

            self.assertEqual(f.name, pipeline.source_exposures_file.name)

    def test_canonical_exposures_profile_not_set___canonical_exposures_profile_in_none(self):
        model = create_model()

        profile = model.resources['canonical_exposures_profile']

        self.assertEqual({}, profile)

    @given(dictionaries(text(), text()))
    def test_canonical_exposures_profile_json_set___canonical_exposures_profile_matches_json(self, expected):
        model = create_model(resources={'canonical_exposures_profile_json': json.dumps(expected)})

        profile = model.resources['canonical_exposures_profile']

        self.assertEqual(expected, profile)

    @given(dictionaries(text(), text()))
    def test_canonical_exposures_profile_path_set___canonical_exposures_profile_matches_json(self, expected):
        with NamedTemporaryFile('w') as f:
            json.dump(expected, f)
            f.flush()

            model = create_model(resources={'canonical_exposures_profile_json_path': f.name})

            profile = model.resources['canonical_exposures_profile']

            self.assertEqual(expected, profile)


class OasisModelLoadCanonicalProfile(TestCase):
    def test_model_and_kwargs_are_not_set___result_is_empty_dict(self):
        profile = OasisModel.load_canonical_profile()

        self.assertEqual({}, profile)

    @given(dictionaries(text(), text()))
    def test_model_is_set_with_profile_json___models_profile_is_set_to_expected_json(self, expected):
        model = create_model(resources={'canonical_exposures_profile_json': json.dumps(expected)})

        profile = OasisModel.load_canonical_profile(oasis_model=model)

        self.assertEqual(expected, profile)
        self.assertEqual(expected, model.resources['canonical_exposures_profile'])

    @given(dictionaries(text(), text()), dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_and_profile_json_is_passed_through_kwargs___kwargs_profile_is_used(self, model_profile, kwargs_profile):
        model = create_model(resources={'canonical_exposures_profile_json': json.dumps(model_profile)})

        profile = OasisModel.load_canonical_profile(oasis_model=model, canonical_exposures_profile_json=json.dumps(kwargs_profile))

        self.assertEqual(kwargs_profile, profile)
        self.assertEqual(kwargs_profile, model.resources['canonical_exposures_profile'])

    @given(dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_path___models_profile_is_set_to_expected_json(self, expected):
        with NamedTemporaryFile('w') as f:
            json.dump(expected, f)
            f.flush()

            model = create_model(resources={'canonical_exposures_profile_json_path': f.name})

            profile = OasisModel.load_canonical_profile(oasis_model=model)

            self.assertEqual(expected, profile)
            self.assertEqual(expected, model.resources['canonical_exposures_profile'])

    @given(dictionaries(text(), text()), dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_path_and_profile_json_path_is_passed_through_kwargs___kwargs_profile_is_used(self, model_profile, kwargs_profile):
        with NamedTemporaryFile('w') as model_file, NamedTemporaryFile('w') as kwargs_file:
            json.dump(model_profile, model_file)
            model_file.flush()
            json.dump(kwargs_profile, kwargs_file)
            kwargs_file.flush()

            model = create_model(resources={'canonical_exposures_profile_json_path': model_file.name})

            profile = OasisModel.load_canonical_profile(oasis_model=model, canonical_exposures_profile_json_path=kwargs_file.name)

            self.assertEqual(kwargs_profile, profile)
            self.assertEqual(kwargs_profile, model.resources['canonical_exposures_profile'])
