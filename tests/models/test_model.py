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


class ModelInit(TestCase):
    def create_model(self, supplier=None, model=None, version=None, resources=None):
        supplier = supplier if supplier is not None else text().example()
        model = model if model is not None else text().example()
        version = version if version is not None else text().example()

        return OasisModel(supplier, model, version, resources=resources)

    @given(text(), text(), text())
    def test_supplier_model_and_version_are_supplied___correct_key_is_created(self, supplier, model_id, version):
        model = self.create_model(supplier=supplier, model=model_id, version=version)

        self.assertEqual('{}/{}/{}'.format(supplier, model_id, version), model.key)

    def test_oasis_file_path_is_given___path_is_stored_as_absolute_path(self):
        model = self.create_model(resources={'oasis_files_path': 'some_path'})

        result = model.resources['oasis_files_path']
        expected = os.path.abspath('some_path')

        self.assertEqual(expected, result)

    def test_oasis_file_path_is_not_given___path_is_abs_path_of_default(self):
        model = self.create_model()

        result = model.resources['oasis_files_path']
        expected = os.path.abspath(os.path.join('Files', model.key.replace('/', '-')))

        self.assertEqual(expected, result)

    def test_file_pipeline_is_not_supplied___default_pipeline_is_set(self):
        model = self.create_model()

        pipeline = model.resources['oasis_files_pipeline']

        self.assertIsInstance(pipeline, OasisFilesPipeline)
        self.assertEqual(pipeline.model_key, model.key)

    def test_file_pipeline_is_supplied___pipeline_is_unchanged(self):
        pipeline = OasisFilesPipeline()

        model = self.create_model(resources={'oasis_files_pipeline': pipeline})

        self.assertIs(pipeline, model.resources['oasis_files_pipeline'])

    def test_pipeline_is_not_a_pipeline_instance___oasis_exception_is_raised(self):
        class FakePipeline(object):
            pass

        pipeline = FakePipeline()

        with self.assertRaises(OasisException):
            self.create_model(resources={'oasis_files_pipeline': pipeline})

    def test_exposure_file_path_is_not_supplied___source_exposure_file_is_not_set_on_pipeline(self):
        model = self.create_model()

        pipeline = model.resources['oasis_files_pipeline']

        self.assertIsNone(pipeline.source_exposures_file)

    def test_exposure_file_path_is_supplied___source_exposure_file_is_set_on_pipeline(self):
        with NamedTemporaryFile() as f:
            model = self.create_model(resources={'source_exposures_file_path': f.name})

            pipeline = model.resources['oasis_files_pipeline']

            self.assertEqual(f.name, pipeline.source_exposures_file.name)

    def test_canonical_exposures_profile_not_set___canonical_exposures_profile_in_none(self):
        model = self.create_model()

        profile = model.resources['canonical_exposures_profile']

        self.assertIsNone(profile)

    @given(dictionaries(text(), text()))
    def test_canonical_exposures_profile_json_set___canonical_exposures_profile_matches_json(self, expected):
        model = self.create_model(resources={'canonical_exposures_profile_json': json.dumps(expected)})

        profile = model.resources['canonical_exposures_profile']

        self.assertEqual(expected, profile)

    @given(dictionaries(text(), text()))
    def test_canonical_exposures_profile_path_set___canonical_exposures_profile_matches_json(self, expected):
        with NamedTemporaryFile('w') as f:
            json.dump(expected, f)
            f.flush()

            model = self.create_model(resources={'canonical_exposures_profile_json_path': f.name})

            profile = model.resources['canonical_exposures_profile']

            self.assertEqual(expected, profile)
