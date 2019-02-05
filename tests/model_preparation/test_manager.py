from __future__ import unicode_literals

import copy
import io
import itertools
import json
import os
import shutil
import string
import sys

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

from oasislmf.model_preparation.manager import OasisManager as om
from oasislmf.model_preparation.pipeline import OasisFilesPipeline as ofp
from oasislmf.models.model import OasisModel

from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.fm import (
    unified_canonical_fm_profile_by_level_and_term_group,
)
from oasislmf.utils.metadata import (
    OASIS_COVERAGE_TYPES,
    OASIS_FM_LEVELS,
    OASIS_KEYS_STATUS,
    OASIS_PERILS,
    OED_COVERAGE_TYPES,
    OED_PERILS,
)

from ..models.fakes import fake_model

from ..data import (
    canonical_accounts,
    canonical_accounts_profile,
    canonical_exposure,
    canonical_exposure_profile,
    canonical_oed_accounts,
    canonical_oed_accounts_profile,
    canonical_oed_exposure,
    canonical_oed_exposure_profile,
    fm_input_items,
    gul_input_items,
    keys,
    oasis_fm_agg_profile,
    oed_fm_agg_profile,
    write_canonical_files,
    write_canonical_oed_files,
    write_keys_files,
)


class AddModel(TestCase):

    def test_models_is_empty___model_is_added_to_model_dict(self):
        model = fake_model('supplier', 'model', 'version')

        manager = om()
        manager.add_model(model)

        self.assertEqual({model.key: model}, manager.models)

    def test_manager_already_contains_a_model_with_the_given_key___model_is_replaced_in_models_dict(self):
        first = fake_model('supplier', 'model', 'version')
        second = fake_model('supplier', 'model', 'version')

        manager = om(oasis_models=[first])
        manager.add_model(second)

        self.assertIs(second, manager.models[second.key])

    def test_manager_already_contains_a_diferent_model___model_is_added_to_dict(self):
        first = fake_model('first', 'model', 'version')
        second = fake_model('second', 'model', 'version')

        manager = om(oasis_models=[first])
        manager.add_model(second)

        self.assertEqual({
            first.key: first,
            second.key: second,
        }, manager.models)


class DeleteModels(TestCase):
    def test_models_is_not_in_manager___no_model_is_removed(self):
        manager = om([
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

        manager = om(models)
        manager.delete_models(models[1:])

        self.assertEqual({models[0].key: models[0]}, manager.models)


class GetCanonicalExposureProfile(TestCase):
    def test_model_and_kwargs_are_not_set___result_is_null(self):
        profile = om().get_canonical_exposure_profile()

        self.assertEqual(None, profile)

    @given(expected=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json___models_profile_is_set_to_expected_json(self, expected):
        model = fake_model(resources={'canonical_exposure_profile_json': json.dumps(expected)})

        profile = om().get_canonical_exposure_profile(oasis_model=model)

        self.assertEqual(expected, profile)
        self.assertEqual(expected, model.resources['canonical_exposure_profile'])

    @given(model_profile=dictionaries(text(), text()), kwargs_profile=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_and_profile_json_is_passed_through_kwargs___kwargs_profile_is_used(
        self,
        model_profile,
        kwargs_profile
    ):
        model = fake_model(resources={'canonical_exposure_profile_json': json.dumps(model_profile)})

        profile = om().get_canonical_exposure_profile(oasis_model=model, canonical_exposure_profile_json=json.dumps(kwargs_profile))

        self.assertEqual(kwargs_profile, profile)
        self.assertEqual(kwargs_profile, model.resources['canonical_exposure_profile'])

    @given(expected=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_path___models_profile_is_set_to_expected_json(self, expected):
        with NamedTemporaryFile('w') as f:
            json.dump(expected, f)
            f.flush()

            model = fake_model(resources={'canonical_exposure_profile_path': f.name})

            profile = om().get_canonical_exposure_profile(oasis_model=model)

            self.assertEqual(expected, profile)
            self.assertEqual(expected, model.resources['canonical_exposure_profile'])

    @given(model_profile=dictionaries(text(), text()), kwargs_profile=dictionaries(text(), text()))
    @settings(suppress_health_check=[HealthCheck.too_slow])
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

            model = fake_model(resources={'canonical_exposure_profile_path': model_file.name})

            profile = om().get_canonical_exposure_profile(oasis_model=model, canonical_exposure_profile_path=kwargs_file.name)

            self.assertEqual(kwargs_profile, profile)
            self.assertEqual(kwargs_profile, model.resources['canonical_exposure_profile'])


class CreateModel(TestCase):

    def create_model(
        self,
        lookup='lookup',
        keys_file_path='key_file_path',
        keys_errors_file_path='keys_error_file_path',
        model_exposure_file_path='model_exposure_file_path'
    ):
        model = fake_model(resources={'lookup': lookup})

        model.resources['oasis_files_pipeline'].keys_file_path = keys_file_path
        model.resources['oasis_files_pipeline'].keys_errors_file_path = keys_errors_file_path
        model.resources['oasis_files_pipeline'].model_exposure_file_path = model_exposure_file_path

        return model

    @given(
        lookup=text(min_size=1, alphabet=string.ascii_letters),
        keys_file_path=text(min_size=1, alphabet=string.ascii_letters),
        keys_errors_file_path=text(min_size=1, alphabet=string.ascii_letters),
        exposure_file_path=text(min_size=1, alphabet=string.ascii_letters)
    )
    def test_supplier_and_model_and_version_only_are_supplied___correct_model_is_returned(
        self,
        lookup,
        keys_file_path,
        keys_errors_file_path,
        exposure_file_path
    ):
        model = self.create_model(lookup=lookup, keys_file_path=keys_file_path, keys_errors_file_path=keys_errors_file_path, model_exposure_file_path=exposure_file_path)

        with patch('oasislmf.model_preparation.lookup.OasisLookupFactory.save_results', Mock(return_value=(keys_file_path, 1, keys_errors_file_path, 1))) as oklf_mock:
            res_keys_file_path, res_keys_errors_file_path = om().get_keys(oasis_model=model)

            oklf_mock.assert_called_once_with(
                lookup,
                keys_file_path,
                errors_fp=keys_errors_file_path,
                model_exposure_fp=exposure_file_path
            )
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_file_path, keys_file_path)
            self.assertEqual(res_keys_file_path, keys_file_path)
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_errors_file_path, keys_errors_file_path)
            self.assertEqual(res_keys_errors_file_path, keys_errors_file_path)

    @given(
        supplier_id=text(alphabet=string.ascii_letters, min_size=1),
        model_id=text(alphabet=string.ascii_letters, min_size=1),
        version=text(alphabet=string.ascii_letters, min_size=1),
        model_lookup=text(min_size=1, alphabet=string.ascii_letters), 
        model_keys_fp=text(alphabet=string.ascii_letters, min_size=1),
        model_keys_errors_fp=text(alphabet=string.ascii_letters, min_size=1),
        model_exposure_fp=text(alphabet=string.ascii_letters, min_size=1),
        lookup=text(min_size=1, alphabet=string.ascii_letters),
        keys_fp=text(alphabet=string.ascii_letters, min_size=1),
        keys_errors_fp=text(alphabet=string.ascii_letters, min_size=1),
        exposure_fp=text(alphabet=string.ascii_letters, min_size=1)
    )
    def test_supplier_and_model_and_version_and_absolute_oasis_files_path_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version,
        model_lookup,
        model_keys_fp,
        model_keys_errors_fp,
        model_exposure_fp,
        lookup,
        keys_fp,
        keys_errors_fp,
        exposure_fp
    ):
        resources={
            'lookup': model_lookup,
            'keys_file_path': model_keys_fp,
            'keys_errors_file_path': model_keys_errors_fp,
            'model_exposure_file_path': model_exposure_fp
        }
        model = om().create_model(supplier_id, model_id, version, resources=resources)

        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version)

        with patch('oasislmf.model_preparation.lookup.OasisLookupFactory.save_results', Mock(return_value=(keys_fp, 1, keys_errors_fp, 1))) as oklf_mock:
            res_keys_file_path, res_keys_errors_file_path = om().get_keys(
                oasis_model=model,
                lookup=lookup,
                model_exposure_file_path=exposure_fp,
                keys_file_path=keys_fp,
                keys_errors_file_path=keys_errors_fp
            )

            oklf_mock.assert_called_once_with(
                lookup,
                keys_fp,
                errors_fp=keys_errors_fp,
                model_exposure_fp=exposure_fp
            )
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_file_path, keys_fp)
            self.assertEqual(res_keys_file_path, keys_fp)
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_errors_file_path, keys_errors_fp)
            self.assertEqual(res_keys_errors_file_path, keys_errors_fp)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        self.assertEqual(resources, model.resources)

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], ofp))

        self.assertIsNone(model.resources.get('canonical_exposure_profile'))

    @given(
        supplier_id=text(alphabet=string.ascii_letters, min_size=1),
        model_id=text(alphabet=string.ascii_letters, min_size=1),
        version_id=text(alphabet=string.ascii_letters, min_size=1),
        oasis_files_path=text(alphabet=string.ascii_letters, min_size=1)
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

        model = om().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        self.assertTrue(os.path.isabs(model.resources['oasis_files_path']))
        self.assertEqual(os.path.abspath(resources['oasis_files_path']), model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], ofp))

        self.assertIsNone(model.resources.get('canonical_exposure_profile'))

    @given(
        supplier_id=text(alphabet=string.ascii_letters, min_size=1),
        model_id=text(alphabet=string.ascii_letters, min_size=1),
        version_id=text(alphabet=string.ascii_letters, min_size=1),
        canonical_exposure_profile=dictionaries(text(min_size=1), text(min_size=1))
    )
    def test_supplier_and_model_and_version_and_canonical_exposure_profile_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        canonical_exposure_profile
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={'canonical_exposure_profile': canonical_exposure_profile}

        model = om().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        self.assertEqual(resources['canonical_exposure_profile'], model.resources['canonical_exposure_profile'])

        expected_oasis_files_path = os.path.abspath(os.path.join('Files', expected_key.replace('/', '-')))
        self.assertEqual(expected_oasis_files_path, model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], ofp))

    @given(
        supplier_id=text(alphabet=string.ascii_letters, min_size=1),
        model_id=text(alphabet=string.ascii_letters, min_size=1),
        version_id=text(alphabet=string.ascii_letters, min_size=1),
        oasis_files_path=text(alphabet=string.ascii_letters, min_size=1),
        canonical_exposure_profile=dictionaries(text(min_size=1), text(min_size=1))
    )
    def test_supplier_and_model_and_version_and_relative_oasis_files_path_and_canonical_exposure_profile_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        oasis_files_path,
        canonical_exposure_profile
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={'oasis_files_path': oasis_files_path, 'canonical_exposure_profile': canonical_exposure_profile}

        model = om().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        self.assertTrue(os.path.isabs(model.resources['oasis_files_path']))
        self.assertEqual(os.path.abspath(resources['oasis_files_path']), model.resources['oasis_files_path'])

        self.assertEqual(resources['canonical_exposure_profile'], model.resources['canonical_exposure_profile'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], ofp))

    @given(
        supplier_id=text(alphabet=string.ascii_letters, min_size=1),
        model_id=text(alphabet=string.ascii_letters, min_size=1),
        version_id=text(alphabet=string.ascii_letters, min_size=1),
        oasis_files_path=text(alphabet=string.ascii_letters, min_size=1),
        canonical_exposure_profile=dictionaries(text(min_size=1), text(min_size=1))
    )
    def test_supplier_and_model_and_version_and_absolute_oasis_files_path_and_canonical_exposure_profile_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        oasis_files_path,
        canonical_exposure_profile
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={'oasis_files_path': os.path.abspath(oasis_files_path), 'canonical_exposure_profile': canonical_exposure_profile}

        model = om().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        self.assertEqual(resources['oasis_files_path'], model.resources['oasis_files_path'])

        self.assertEqual(resources['canonical_exposure_profile'], model.resources['canonical_exposure_profile'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], ofp))

    @given(
        supplier_id=text(alphabet=string.ascii_letters, min_size=1),
        model_id=text(alphabet=string.ascii_letters, min_size=1),
        version_id=text(alphabet=string.ascii_letters, min_size=1),
        source_accounts_file_path=text(alphabet=string.ascii_letters, min_size=1)
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
        
        model = om().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        expected_oasis_files_path = os.path.abspath(os.path.join('Files', expected_key.replace('/', '-')))
        self.assertEqual(expected_oasis_files_path, model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], ofp))

        self.assertEqual(resources['source_accounts_file_path'], model.resources['source_accounts_file_path'])

        self.assertIsNone(model.resources.get('canonical_exposure_profile'))
        self.assertIsNone(model.resources.get('canonical_accounts_profile'))

    @given(
        supplier_id=text(alphabet=string.ascii_letters, min_size=1),
        model_id=text(alphabet=string.ascii_letters, min_size=1),
        version_id=text(alphabet=string.ascii_letters, min_size=1),
        source_accounts_file_path=text(alphabet=string.ascii_letters, min_size=1),
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
        
        model = om().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        expected_oasis_files_path = os.path.abspath(os.path.join('Files', expected_key.replace('/', '-')))
        self.assertEqual(expected_oasis_files_path, model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], ofp))

        self.assertIsNone(model.resources.get('canonical_exposure_profile'))

        self.assertEqual(resources['source_accounts_file_path'], model.resources['source_accounts_file_path'])
        
        self.assertEqual(resources['canonical_accounts_profile'], model.resources['canonical_accounts_profile'])

    @given(
        supplier_id=text(alphabet=string.ascii_letters, min_size=1),
        model_id=text(alphabet=string.ascii_letters, min_size=1),
        version_id=text(alphabet=string.ascii_letters, min_size=1),
        canonical_exposure_profile=dictionaries(text(min_size=1), text(min_size=1)),
        source_accounts_file_path=text(alphabet=string.ascii_letters, min_size=1),
        canonical_accounts_profile=dictionaries(text(min_size=1), text(min_size=1))
    )
    def test_supplier_and_model_and_version_and_canonical_exposure_profile_and_source_accounts_file_path_and_canonical_accounts_profile_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        canonical_exposure_profile,
        source_accounts_file_path,
        canonical_accounts_profile
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={
            'canonical_exposure_profile': canonical_exposure_profile,
            'source_accounts_file_path': source_accounts_file_path,
            'canonical_accounts_profile': canonical_accounts_profile
        }
        
        model = om().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        expected_oasis_files_path = os.path.abspath(os.path.join('Files', expected_key.replace('/', '-')))
        self.assertEqual(expected_oasis_files_path, model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], ofp))

        self.assertEqual(resources['canonical_exposure_profile'], model.resources.get('canonical_exposure_profile'))

        self.assertEqual(resources['source_accounts_file_path'], model.resources['source_accounts_file_path'])
        
        self.assertEqual(resources['canonical_accounts_profile'], model.resources['canonical_accounts_profile'])

    @given(
        supplier_id=text(alphabet=string.ascii_letters, min_size=1),
        model_id=text(alphabet=string.ascii_letters, min_size=1),
        version_id=text(alphabet=string.ascii_letters, min_size=1),
        oasis_files_path=text(alphabet=string.ascii_letters, min_size=1),
        canonical_exposure_profile=dictionaries(text(min_size=1), text(min_size=1)),
        source_accounts_file_path=text(alphabet=string.ascii_letters, min_size=1),
        canonical_accounts_profile=dictionaries(text(min_size=1), text(min_size=1))
    )
    def test_supplier_and_model_and_version_and_absolute_oasis_files_path_and_canonical_exposure_profile_and_source_accounts_file_path_and_canonical_accounts_profile_only_are_supplied___correct_model_is_returned(
        self,
        supplier_id,
        model_id,
        version_id,
        oasis_files_path,
        canonical_exposure_profile,
        source_accounts_file_path,
        canonical_accounts_profile
    ):
        expected_key = '{}/{}/{}'.format(supplier_id, model_id, version_id)

        resources={
            'oasis_files_path': os.path.abspath(oasis_files_path),
            'canonical_exposure_profile': canonical_exposure_profile,
            'source_accounts_file_path': source_accounts_file_path,
            'canonical_accounts_profile': canonical_accounts_profile
        }
        
        model = om().create_model(supplier_id, model_id, version_id, resources=resources)

        self.assertTrue(isinstance(model, OasisModel))

        self.assertEqual(expected_key, model.key)

        self.assertEqual(resources['oasis_files_path'], model.resources['oasis_files_path'])

        self.assertTrue(isinstance(model.resources['oasis_files_pipeline'], ofp))

        self.assertEqual(resources['canonical_exposure_profile'], model.resources.get('canonical_exposure_profile'))

        self.assertEqual(resources['source_accounts_file_path'], model.resources['source_accounts_file_path'])
        
        self.assertEqual(resources['canonical_accounts_profile'], model.resources['canonical_accounts_profile'])


class LoadCanonicalAccountsProfile(TestCase):
    def test_model_and_kwargs_are_not_set___result_is_null(self):
        profile = om().get_canonical_accounts_profile()

        self.assertEqual(None, profile)

    @given(expected=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json___models_profile_is_set_to_expected_json(self, expected):
        model = fake_model(resources={'canonical_accounts_profile_json': json.dumps(expected)})

        profile = om().get_canonical_accounts_profile(oasis_model=model)

        self.assertEqual(expected, profile)
        self.assertEqual(expected, model.resources['canonical_accounts_profile'])

    @given(model_profile=dictionaries(text(), text()), kwargs_profile=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_and_profile_json_is_passed_through_kwargs___kwargs_profile_is_used(
        self,
        model_profile,
        kwargs_profile
    ):
        model = fake_model(resources={'canonical_accounts_profile_json': json.dumps(model_profile)})

        profile = om().get_canonical_accounts_profile(oasis_model=model, canonical_accounts_profile_json=json.dumps(kwargs_profile))

        self.assertEqual(kwargs_profile, profile)
        self.assertEqual(kwargs_profile, model.resources['canonical_accounts_profile'])

    @given(expected=dictionaries(text(), text()))
    def test_model_is_set_with_profile_json_path___models_profile_is_set_to_expected_json(self, expected):
        with NamedTemporaryFile('w') as f:
            json.dump(expected, f)
            f.flush()

            model = fake_model(resources={'canonical_accounts_profile_path': f.name})

            profile = om().get_canonical_accounts_profile(oasis_model=model)

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

            model = fake_model(resources={'canonical_accounts_profile_path': model_file.name})

            profile = om().get_canonical_accounts_profile(oasis_model=model, canonical_accounts_profile_path=kwargs_file.name)

            self.assertEqual(kwargs_profile, profile)
            self.assertEqual(kwargs_profile, model.resources['canonical_accounts_profile'])


class GetFmAggregationProfile(TestCase):
    def setUp(self):
        self.profile = oasis_fm_agg_profile

    def test_model_and_kwargs_are_not_set___result_is_null(self):
        profile = om().get_fm_aggregation_profile()

        self.assertEqual(None, profile)

    def test_model_is_set_with_profile_json___models_profile_is_set_to_expected_json(self):
        expected = self.profile
        profile_json = json.dumps(self.profile)
        model = fake_model(resources={'fm_agg_profile_json': profile_json})

        profile = om().get_fm_aggregation_profile(oasis_model=model)


        self.assertEqual(expected, profile)
        self.assertEqual(expected, model.resources['fm_agg_profile'])

    def test_model_is_set_with_profile_json_and_profile_json_is_passed_through_kwargs___kwargs_profile_is_used(self):
        model = fake_model(resources={'fm_agg_profile_json': json.dumps(self.profile)})

        profile = om().get_fm_aggregation_profile(oasis_model=model, fm_agg_profile_json=json.dumps(self.profile))

        self.assertEqual(self.profile, profile)
        self.assertEqual(self.profile, model.resources['fm_agg_profile'])

    def test_model_is_set_with_profile_path___models_profile_is_set_to_expected_json(self):
        expected = self.profile

        with NamedTemporaryFile('w') as f:
            json.dump(expected, f)
            f.flush()

            model = fake_model(resources={'fm_agg_profile_path': f.name})

            profile = om().get_fm_aggregation_profile(oasis_model=model)

            self.assertEqual(expected, profile)
            self.assertEqual(expected, model.resources['fm_agg_profile'])

    def test_model_is_set_with_profile_path_and_profile_path_is_passed_through_kwargs___kwargs_profile_is_used(
        self
    ):
        with NamedTemporaryFile('w') as model_file, NamedTemporaryFile('w') as kwargs_file:
            json.dump(self.profile, model_file)
            model_file.flush()
            json.dump(self.profile, kwargs_file)
            kwargs_file.flush()

            model = fake_model(resources={'fm_agg_profile_path': model_file.name})

            profile = om().get_fm_aggregation_profile(oasis_model=model, fm_agg_profile_path=kwargs_file.name)

            self.assertEqual(self.profile, profile)
            self.assertEqual(self.profile, model.resources['fm_agg_profile'])


@pytest.mark.skipif(True, reason="CSV file transformations to be removed")
class TransformSourceToCanonical(TestCase):

    @given(
        source_exposure_file_path=text(min_size=1, alphabet=string.ascii_letters),
        source_to_canonical_exposure_transformation_file_path=text(min_size=1, alphabet=string.ascii_letters),
        source_exposure_validation_file_path=text(alphabet=string.ascii_letters),
        canonical_exposure_file_path=text(min_size=1, alphabet=string.ascii_letters)
    )
    def test_model_is_not_set___parameters_are_taken_from_kwargs(
            self,
            source_exposure_file_path,
            source_to_canonical_exposure_transformation_file_path,
            source_exposure_validation_file_path,
            canonical_exposure_file_path
    ):
        trans_call_mock = Mock()
        with patch('oasislmf.model_preparation.csv_trans.Translator', Mock(return_value=trans_call_mock)) as trans_mock:
            om().transform_source_to_canonical(
                source_exposure_file_path=source_exposure_file_path,
                source_to_canonical_exposure_transformation_file_path=source_to_canonical_exposure_transformation_file_path,
                canonical_exposure_file_path=canonical_exposure_file_path
            )

            trans_mock.assert_called_once_with(
                os.path.abspath(source_exposure_file_path),
                os.path.abspath(canonical_exposure_file_path),
                os.path.abspath(source_to_canonical_exposure_transformation_file_path),
                xsd_path=None,
                append_row_nums=True,
            )
            trans_call_mock.assert_called_once_with()

    @given(
        source_exposure_file_path=text(min_size=1, alphabet=string.ascii_letters),
        source_to_canonical_exposure_transformation_file_path=text(min_size=1, alphabet=string.ascii_letters),
        source_exposure_validation_file_path=text(alphabet=string.ascii_letters),
        canonical_exposure_file_path=text(min_size=1, alphabet=string.ascii_letters)
    )
    def test_model_is_set___parameters_are_taken_from_model(
            self,
            source_exposure_file_path,
            source_to_canonical_exposure_transformation_file_path,
            source_exposure_validation_file_path,
            canonical_exposure_file_path):

        model = fake_model(resources={
            'source_exposure_file_path': source_exposure_file_path,
            'source_exposure_validation_file_path': source_exposure_validation_file_path,
            'source_to_canonical_exposure_transformation_file_path': source_to_canonical_exposure_transformation_file_path,
        })
        model.resources['oasis_files_pipeline'].canonical_exposure_path = canonical_exposure_file_path

        trans_call_mock = Mock()
        with patch('oasislmf.model_preparation.csv_trans.Translator', Mock(return_value=trans_call_mock)) as trans_mock:
            #import ipdb; ipdb.set_trace()
            om().transform_source_to_canonical(
                source_exposure_file_path=source_exposure_file_path,
                source_to_canonical_exposure_transformation_file_path=source_to_canonical_exposure_transformation_file_path,
                canonical_exposure_file_path=canonical_exposure_file_path
            )

            trans_mock.assert_called_once_with(
                os.path.abspath(source_exposure_file_path),
                os.path.abspath(canonical_exposure_file_path),
                os.path.abspath(source_to_canonical_exposure_transformation_file_path),
                xsd_path=None,
                append_row_nums=True
            )
            trans_call_mock.assert_called_once_with()


@pytest.mark.skipif(True, reason="CSV file transformations to be removed")
class TransformCanonicalToModel(TestCase):
    @given(
        canonical_exposure_file_path=text(min_size=1, alphabet=string.ascii_letters),
        canonical_to_model_exposure_transformation_file_path=text(min_size=1, alphabet=string.ascii_letters),
        canonical_exposure_validation_file_path=text(alphabet=string.ascii_letters),
        model_exposure_file_path=text(min_size=1, alphabet=string.ascii_letters)
    )
    def test_model_is_not_set___parameters_are_taken_from_kwargs(
            self,
            canonical_exposure_file_path,
            canonical_to_model_exposure_transformation_file_path,
            canonical_exposure_validation_file_path,
            model_exposure_file_path):

        trans_call_mock = Mock()
        with patch('oasislmf.model_preparation.csv_trans.Translator', Mock(return_value=trans_call_mock)) as trans_mock:
            om().transform_canonical_to_model(
                canonical_exposure_file_path=canonical_exposure_file_path,
                canonical_to_model_exposure_transformation_file_path=canonical_to_model_exposure_transformation_file_path,
                model_exposure_file_path=model_exposure_file_path,
            )

            trans_mock.assert_called_once_with(
                os.path.abspath(canonical_exposure_file_path),
                os.path.abspath(model_exposure_file_path),
                os.path.abspath(canonical_to_model_exposure_transformation_file_path),
                xsd_path=None,
                append_row_nums=False
            )
            trans_call_mock.assert_called_once_with()

    @given(
        canonical_exposure_file_path=text(min_size=1, alphabet=string.ascii_letters),
        canonical_to_model_exposure_transformation_file_path=text(min_size=1, alphabet=string.ascii_letters),
        canonical_exposure_validation_file_path=text(alphabet=string.ascii_letters),
        model_exposure_file_path=text(min_size=1, alphabet=string.ascii_letters)
    )
    def test_model_is_set___parameters_are_taken_from_model(
            self,
            canonical_exposure_file_path,
            canonical_to_model_exposure_transformation_file_path,
            canonical_exposure_validation_file_path,
            model_exposure_file_path):

        model = fake_model(resources={
            'canonical_exposure_validation_file_path': canonical_exposure_validation_file_path,
            'canonical_to_model_exposure_transformation_file_path': canonical_to_model_exposure_transformation_file_path,
        })
        model.resources['oasis_files_pipeline'].canonical_exposure_path = canonical_exposure_file_path
        model.resources['oasis_files_pipeline'].model_exposure_file_path = model_exposure_file_path

        trans_call_mock = Mock()
        with patch('oasislmf.model_preparation.csv_trans.Translator', Mock(return_value=trans_call_mock)) as trans_mock:
            om().transform_canonical_to_model(
                canonical_exposure_file_path=canonical_exposure_file_path,
                canonical_to_model_exposure_transformation_file_path=canonical_to_model_exposure_transformation_file_path,
                model_exposure_file_path=model_exposure_file_path,
            )

            trans_mock.assert_called_once_with(
                os.path.abspath(canonical_exposure_file_path),
                os.path.abspath(model_exposure_file_path),
                os.path.abspath(canonical_to_model_exposure_transformation_file_path),
                xsd_path=None,
                append_row_nums=False
            )
            trans_call_mock.assert_called_once_with()


class GetKeys(TestCase):
    def create_model(
        self,
        lookup='lookup',
        keys_file_path='key_file_path',
        keys_errors_file_path='keys_errors_file_path',
        model_exposure_file_path='model_exposure_file_path'
    ):
        model = fake_model(resources={'lookup': lookup})
        model.resources['oasis_files_pipeline'].keys_file_path = keys_file_path
        model.resources['oasis_files_pipeline'].keys_errors_file_path = keys_errors_file_path
        model.resources['oasis_files_pipeline'].model_exposure_file_path = model_exposure_file_path
        return model

    @given(
        lookup=text(min_size=1, alphabet=string.ascii_letters),
        keys_file_path=text(min_size=1, alphabet=string.ascii_letters),
        keys_errors_file_path=text(min_size=1, alphabet=string.ascii_letters),
        exposure_file_path=text(min_size=1, alphabet=string.ascii_letters)
    )
    def test_model_is_supplied_kwargs_are_not___lookup_keys_files_and_exposures_file_from_model_are_used(
        self,
        lookup,
        keys_file_path,
        keys_errors_file_path,
        exposure_file_path
    ):
        model = self.create_model(lookup=lookup, keys_file_path=keys_file_path, keys_errors_file_path=keys_errors_file_path, model_exposure_file_path=exposure_file_path)

        with patch('oasislmf.model_preparation.lookup.OasisLookupFactory.save_results', Mock(return_value=(keys_file_path, 1, keys_errors_file_path, 1))) as oklf_mock:
            res_keys_file_path, res_keys_errors_file_path = om().get_keys(oasis_model=model)

            oklf_mock.assert_called_once_with(
                lookup,
                keys_file_path,
                errors_fp=keys_errors_file_path,
                model_exposure_fp=exposure_file_path
            )
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_file_path, keys_file_path)
            self.assertEqual(res_keys_file_path, keys_file_path)
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_errors_file_path, keys_errors_file_path)
            self.assertEqual(res_keys_errors_file_path, keys_errors_file_path)

    @given(
        model_lookup=text(min_size=1, alphabet=string.ascii_letters), 
        model_keys_fp=text(min_size=1, alphabet=string.ascii_letters),
        model_keys_errors_fp=text(min_size=1, alphabet=string.ascii_letters),
        model_exposure_fp=text(min_size=1, alphabet=string.ascii_letters),
        lookup=text(min_size=1, alphabet=string.ascii_letters),
        keys_fp=text(min_size=1, alphabet=string.ascii_letters),
        keys_errors_fp=text(min_size=1, alphabet=string.ascii_letters),
        exposures_fp=text(min_size=1, alphabet=string.ascii_letters)
    )
    def test_model_and_kwargs_are_supplied___lookup_keys_files_and_exposures_file_from_kwargs_are_used(
        self,
        model_lookup,
        model_keys_fp,
        model_keys_errors_fp,
        model_exposure_fp,
        lookup,
        keys_fp,
        keys_errors_fp,
        exposures_fp
    ):
        model = self.create_model(lookup=model_lookup, keys_file_path=model_keys_fp, keys_errors_file_path=model_keys_errors_fp, model_exposure_file_path=model_exposure_fp)

        with patch('oasislmf.model_preparation.lookup.OasisLookupFactory.save_results', Mock(return_value=(keys_fp, 1, keys_errors_fp, 1))) as oklf_mock:
            res_keys_file_path, res_keys_errors_file_path = om().get_keys(
                oasis_model=model,
                lookup=lookup,
                model_exposure_file_path=exposures_fp,
                keys_file_path=keys_fp,
                keys_errors_file_path=keys_errors_fp
            )

            oklf_mock.assert_called_once_with(
                lookup,
                keys_fp,
                errors_fp=keys_errors_fp,
                model_exposure_fp=exposures_fp
            )
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_file_path, keys_fp)
            self.assertEqual(res_keys_file_path, keys_fp)
            self.assertEqual(model.resources['oasis_files_pipeline'].keys_errors_file_path, keys_errors_fp)
            self.assertEqual(res_keys_errors_file_path, keys_errors_fp)


class GetGulInputItems(TestCase):

    def setUp(self):
        self.profile = copy.deepcopy(canonical_exposure_profile)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(size=0),
        keys=keys(size=2)
    )
    def test_no_fm_terms_in_canonical_profile__oasis_exception_is_raised(
        self,
        exposures,
        keys
    ):
        profile = copy.deepcopy(self.profile)
        _p =copy.deepcopy(profile)

        for _k, _v in six.iteritems(_p):
            for __k, __v in six.iteritems(_v):
                if 'FM' in __k:
                    profile[_k].pop(__k)

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            with self.assertRaises(OasisException):
                om().get_gul_input_items(profile, exposures_file.name, keys_file.name)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(size=0),
        keys=keys(size=2)
    )
    def test_no_canonical_items__oasis_exception_is_raised(
        self,
        exposures,
        keys
    ):
        profile = copy.deepcopy(self.profile)

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            with self.assertRaises(OasisException):
                om().get_gul_input_items(profile, exposures_file.name, keys_file.name)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(size=2),
        keys=keys(size=0)
    )
    def test_no_keys_items__oasis_exception_is_raised(
        self,
        exposures,
        keys
    ):
        profile = copy.deepcopy(self.profile)

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            with self.assertRaises(OasisException):
                om().get_gul_input_items(profile, exposures_file.name, keys_file.name)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(size=2),
        keys=keys(from_statuses=just(OASIS_KEYS_STATUS['success']['id']), size=2)
    )
    def test_canonical_items_dont_match_any_keys_items__oasis_exception_is_raised(
        self,
        exposures,
        keys
    ):
        profile = copy.deepcopy(self.profile)

        l = len(exposures)
        for key in keys:
            key['id'] += l

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            with self.assertRaises(OasisException):
                om().get_gul_input_items(profile, exposures_file.name, keys_file.name)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(size=2),
        keys=keys(from_statuses=just(OASIS_KEYS_STATUS['success']['id']), size=2)
    )
    def test_canonical_profile_doesnt_have_any_tiv_fields__oasis_exception_is_raised(
        self,
        exposures,
        keys
    ):
        profile = copy.deepcopy(self.profile)

        tivs = [profile[e]['ProfileElementName'] for e in profile if profile[e].get('FMTermType') and profile[e]['FMTermType'].lower() == 'tiv']

        for t in tivs:
            profile.pop(t)

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            with self.assertRaises(OasisException):
                om().get_gul_input_items(profile, exposures_file.name, keys_file.name)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(
            from_tivs1=just(0.0),
            size=2
        ),
        keys=keys(from_statuses=just(OASIS_KEYS_STATUS['success']['id']), size=2)
    )
    def test_canonical_items_dont_have_any_positive_tivs__oasis_exception_is_raised(
        self,
        exposures,
        keys
    ):
        profile = copy.deepcopy(self.profile)

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            with self.assertRaises(OasisException):
                om().get_gul_input_items(profile, exposures_file.name, keys_file.name)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(
            from_tivs1=just(1.0),
            size=2
        ),
        keys=keys(
            from_coverage_type_ids=just(OASIS_COVERAGE_TYPES['buildings']['id']),
            from_statuses=just(OASIS_KEYS_STATUS['success']['id']),
            size=2
        )
    )
    def test_only_buildings_coverage_type_in_exposure_and_model_lookup_supporting_single_peril_and_buildings_coverage_type__gul_items_are_generated(
        self,
        exposures,
        keys
    ):
        profile = copy.deepcopy(self.profile)
        ufcp = unified_canonical_fm_profile_by_level_and_term_group(profiles=(profile,))

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            matching_canonical_and_keys_item_ids = set(k['id'] for k in keys).intersection([e['row_id'] for e in exposures])

            gul_items_df, canexp_df = om().get_gul_input_items(profile, exposures_file.name, keys_file.name)

        get_canonical_item = lambda i: (
            [e for e in exposures if e['row_id'] == i + 1][0] if len([e for e in exposures if e['row_id'] == i + 1]) == 1
            else None
        )

        get_keys_item = lambda i: (
            [k for k in keys if k['id'] == i + 1][0] if len([k for k in keys if k['id'] == i + 1]) == 1
            else None
        )

        tiv_elements = (ufcp[1][1]['tiv'],)

        fm_terms = {
            1: {
                'deductible': 'wscv1ded',
                'deductible_min': None,
                'deductible_max': None,
                'limit': 'wscv1limit',
                'share': None
            }
        }

        for i, gul_it in enumerate(gul_items_df.T.to_dict().values()):
            can_it = get_canonical_item(int(gul_it['canexp_id']))
            self.assertIsNotNone(can_it)

            keys_it = get_keys_item(int(gul_it['canexp_id']))
            self.assertIsNotNone(keys_it)

            positive_tiv_elements = [
                t for t in tiv_elements if can_it.get(t['ProfileElementName'].lower()) and can_it[t['ProfileElementName'].lower()] > 0 and t['CoverageTypeID'] == keys_it['coverage_type']
            ]

            for _, t in enumerate(positive_tiv_elements):
                tiv_elm = t['ProfileElementName'].lower()
                self.assertEqual(tiv_elm, gul_it['tiv_elm'])
                
                tiv_tgid = t['FMTermGroupID']
                self.assertEqual(can_it[tiv_elm], gul_it['tiv'])
                                
                ded_elm = fm_terms[tiv_tgid].get('deductible')
                self.assertEqual(ded_elm, gul_it['ded_elm'])
                
                ded_min_elm = fm_terms[tiv_tgid].get('deductible_min')
                self.assertEqual(ded_min_elm, gul_it['ded_min_elm'])

                ded_max_elm = fm_terms[tiv_tgid].get('deductible_max')
                self.assertEqual(ded_max_elm, gul_it['ded_max_elm'])

                lim_elm = fm_terms[tiv_tgid].get('limit')
                self.assertEqual(lim_elm, gul_it['lim_elm'])

                shr_elm = fm_terms[tiv_tgid].get('share')
                self.assertEqual(shr_elm, gul_it['shr_elm'])

            self.assertEqual(keys_it['area_peril_id'], gul_it['areaperil_id'])
            self.assertEqual(keys_it['vulnerability_id'], gul_it['vulnerability_id'])

            self.assertEqual(i + 1, gul_it['item_id'])

            self.assertEqual(i + 1, gul_it['coverage_id'])

            self.assertEqual(can_it['row_id'], gul_it['group_id'])

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(
            from_tivs1=floats(min_value=1.0, allow_infinity=False),
            from_tivs2=floats(min_value=2.0, allow_infinity=False),
            from_tivs3=floats(min_value=3.0, allow_infinity=False),
            from_tivs4=floats(min_value=4.0, allow_infinity=False),
            size=2
        ),
        keys=keys(
            from_peril_ids=just(OASIS_PERILS['wind']['id']),
            from_coverage_type_ids=just(OASIS_COVERAGE_TYPES['buildings']['id']),
            from_statuses=just(OASIS_KEYS_STATUS['success']['id']),
            size=8
        )
    )
    def test_all_coverage_types_in_exposure_and_model_lookup_supporting_multiple_perils_but_only_buildings_and_other_structures_coverage_types__gul_items_are_generated(
        self,
        exposures,
        keys
    ):
        profile = copy.deepcopy(self.profile)
        ufcp = unified_canonical_fm_profile_by_level_and_term_group(profiles=(profile,))

        exposures[1]['wscv2val'] = exposures[1]['wscv3val'] = exposures[1]['wscv4val'] = 0.0

        keys[1]['id'] = keys[2]['id'] = keys[3]['id'] = 1
        keys[2]['peril_id'] = keys[3]['peril_id'] = OASIS_PERILS['quake']['id']
        keys[1]['coverage_type'] = keys[3]['coverage_type'] = OASIS_COVERAGE_TYPES['other']['id']

        keys[4]['id'] = keys[5]['id'] = keys[6]['id'] = keys[7]['id'] = 2
        keys[6]['peril_id'] = keys[7]['peril_id'] = OASIS_PERILS['quake']['id']
        keys[5]['coverage_type'] = keys[7]['coverage_type'] = OASIS_COVERAGE_TYPES['other']['id']

        with NamedTemporaryFile('w') as exposures_file, NamedTemporaryFile('w') as keys_file:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            matching_canonical_and_keys_item_ids = set(k['id'] for k in keys).intersection([e['row_id'] for e in exposures])

            gul_items_df, canexp_df = om().get_gul_input_items(profile, exposures_file.name, keys_file.name)

        self.assertEqual(len(gul_items_df), 6)
        self.assertEqual(len(canexp_df), 2)

        tiv_elements = (ufcp[1][1]['tiv'], ufcp[1][2]['tiv'])

        fm_terms = {
            1: {
                'deductible': 'wscv1ded',
                'deductible_min': None,
                'deductible_max': None,
                'limit': 'wscv1limit',
                'share': None
            },
            2: {
                'deductible': 'wscv2ded',
                'deductible_min': None,
                'deductible_max': None,
                'limit': 'wscv2limit',
                'share': None

            }
        }

        for i, gul_it in enumerate(gul_items_df.T.to_dict().values()):
            can_it = canexp_df.iloc[gul_it['canexp_id']].to_dict()

            keys_it = [k for k in keys if k['id'] == gul_it['canexp_id'] + 1 and k['peril_id'] == gul_it['peril_id'] and k['coverage_type'] == gul_it['coverage_type_id']][0]

            positive_tiv_term = [t for t in tiv_elements if can_it.get(t['ProfileElementName'].lower()) and can_it[t['ProfileElementName'].lower()] > 0 and t['CoverageTypeID'] == keys_it['coverage_type']][0]

            tiv_elm = positive_tiv_term['ProfileElementName'].lower()
            self.assertEqual(tiv_elm, gul_it['tiv_elm'])
            
            tiv_tgid = positive_tiv_term['FMTermGroupID']
            self.assertEqual(can_it[tiv_elm], gul_it['tiv'])
                            
            ded_elm = fm_terms[tiv_tgid].get('deductible')
            self.assertEqual(ded_elm, gul_it['ded_elm'])
            
            ded_min_elm = fm_terms[tiv_tgid].get('deductible_min')
            self.assertEqual(ded_min_elm, gul_it['ded_min_elm'])

            ded_max_elm = fm_terms[tiv_tgid].get('deductible_max')
            self.assertEqual(ded_max_elm, gul_it['ded_max_elm'])

            lim_elm = fm_terms[tiv_tgid].get('limit')
            self.assertEqual(lim_elm, gul_it['lim_elm'])

            shr_elm = fm_terms[tiv_tgid].get('share')
            self.assertEqual(shr_elm, gul_it['shr_elm'])

            self.assertEqual(keys_it['area_peril_id'], gul_it['areaperil_id'])
            self.assertEqual(keys_it['vulnerability_id'], gul_it['vulnerability_id'])

            self.assertEqual(i + 1, gul_it['item_id'])

            self.assertEqual(i + 1, gul_it['coverage_id'])

            self.assertEqual(can_it['row_id'], gul_it['group_id'])


class GetFmInputItems(TestCase):

    def setUp(self):
        self.exposures_profile = copy.deepcopy(canonical_exposure_profile)
        self.accounts_profile = copy.deepcopy(canonical_accounts_profile)
        self.unified_canonical_profile = unified_canonical_fm_profile_by_level_and_term_group(
            profiles=[self.exposures_profile, self.accounts_profile]
        )
        self.fm_agg_profile = copy.deepcopy(oasis_fm_agg_profile)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(size=2),
        accounts=canonical_accounts(size=1),
        guls=gul_input_items(size=2)
    )
    def test_no_fm_terms_in_canonical_profiles__oasis_exception_is_raised(
        self,
        exposures,
        accounts,
        guls
    ):
        cep = copy.deepcopy(self.exposures_profile)
        cap = copy.deepcopy(self.accounts_profile)

        _cep =copy.deepcopy(cep)
        _cap =copy.deepcopy(cap)

        for _k, _v in six.iteritems(_cep):
            for __k, __v in six.iteritems(_v):
                if 'FM' in __k:
                    cep[_k].pop(__k)

        for _k, _v in six.iteritems(_cap):
            for __k, __v in six.iteritems(_v):
                if 'FM' in __k:
                    cap[_k].pop(__k)

        with NamedTemporaryFile('w') as accounts_file:
            write_canonical_files(accounts, accounts_file.name)

            with self.assertRaises(OasisException):
                fm_df, canacc_df = om().get_fm_input_items(
                    pd.DataFrame(data=exposures),
                    pd.DataFrame(data=guls),
                    cep,
                    cap,
                    accounts_file.name,
                    self.fm_agg_profile
                )

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(size=2),
        guls=gul_input_items(size=2)
    )
    def test_no_aggregation_profile__oasis_exception_is_raised(
        self,
        exposures,
        guls
    ):
        cep = copy.deepcopy(self.exposures_profile)
        cap = copy.deepcopy(self.accounts_profile)
        fmap = {}

        with NamedTemporaryFile('w') as accounts_file:
            write_canonical_files(canonical_accounts=[], canonical_accounts_file_path=accounts_file.name)

            with self.assertRaises(OasisException):
                fm_df, canacc_df = om().get_fm_input_items(
                    pd.DataFrame(data=exposures),
                    pd.DataFrame(data=guls),
                    cep,
                    cap,
                    accounts_file.name,
                    fmap
                )

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(size=2),
        guls=gul_input_items(size=2)
    )
    def test_no_canonical_accounts_items__oasis_exception_is_raised(
        self,
        exposures,
        guls
    ):
        cep = copy.deepcopy(self.exposures_profile)
        cap = copy.deepcopy(self.accounts_profile)
        fmap = copy.deepcopy(self.fm_agg_profile)

        with NamedTemporaryFile('w') as accounts_file:
            write_canonical_files(canonical_accounts=[], canonical_accounts_file_path=accounts_file.name)

            with self.assertRaises(OasisException):
                fm_df, canacc_df = om().get_fm_input_items(
                    pd.DataFrame(data=exposures),
                    pd.DataFrame(data=guls),
                    cep,
                    cap,
                    accounts_file.name,
                    fmap
                )


class GulInputFilesGenerationTestCase(TestCase):

    def setUp(self):
        self.profile = canonical_exposure_profile
        self.manager = om()

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


class FmInputFilesGenerationTestCase(TestCase):

    def setUp(self):
        self.exposures_profile = canonical_exposure_profile
        self.accounts_profile = canonical_accounts_profile
        self.unified_canonical_profile = unified_canonical_fm_profile_by_level_and_term_group(
            profiles=(self.exposures_profile, self.accounts_profile,)
        )
        self.fm_agg_profile = oasis_fm_agg_profile
        self.manager = om()

    def check_fm_policytc_file(self, fm_items_df, fm_policytc_file_path):
        fm_policytc_df = pd.DataFrame(
            columns=['layer_id', 'level_id', 'agg_id', 'policytc_id'],
            data=[key[:4] for key, _ in fm_items_df.groupby(['layer_id', 'level_id', 'agg_id', 'policytc_id', 'limit', 'deductible', 'share'])],
            dtype=object
        )
        expected = tuple(
            {
                k:it[k] for k in ('layer_id', 'level_id', 'agg_id', 'policytc_id',)
            } for _, it in fm_policytc_df.iterrows()
        )

        with io.open(fm_policytc_file_path, 'r', encoding='utf-8') as f:
            result = tuple(pd.read_csv(f).T.to_dict().values())

        self.assertEqual(expected, result)

    def check_fm_profile_file(self, fm_items_df, fm_profile_file_path):
            cols = ['policytc_id', 'calcrule_id', 'limit', 'deductible', 'deductible_min', 'deductible_max', 'attachment', 'share']

            fm_profile_df = fm_items_df[cols]

            fm_profile_df = pd.DataFrame(
                columns=cols,
                data=[key for key, _ in fm_profile_df.groupby(cols)]
            )

            col_repl = [
                {'deductible': 'deductible1'},
                {'deductible_min': 'deductible2'},
                {'deductible_max': 'deductible3'},
                {'attachment': 'attachment1'},
                {'limit': 'limit1'},
                {'share': 'share1'}
            ]
            for repl in col_repl:
                fm_profile_df.rename(columns=repl, inplace=True)

            n = len(fm_profile_df)

            fm_profile_df['index'] = range(n)

            fm_profile_df['share2'] = fm_profile_df['share3'] = [0]*n

            expected = tuple(
                {
                    k:it[k] for k in ('policytc_id','calcrule_id','deductible1', 'deductible2', 'deductible3', 'attachment1', 'limit1', 'share1', 'share2', 'share3',)
                } for _, it in fm_profile_df.iterrows()
            )

            with io.open(fm_profile_file_path, 'r', encoding='utf-8') as f:
                result = tuple(pd.read_csv(f).T.to_dict().values())

            self.assertEqual(expected, result)

    def check_fm_programme_file(self, fm_items_df, fm_programme_file_path):
            fm_programme_df = pd.DataFrame(
                pd.concat([fm_items_df[fm_items_df['level_id']==OASIS_FM_LEVELS['coverage']['id']], fm_items_df])[['level_id', 'agg_id']],
                dtype=int
            ).reset_index(drop=True)

            num_cov_items = len(fm_items_df[fm_items_df['level_id']==OASIS_FM_LEVELS['coverage']['id']])

            for i in range(num_cov_items):
                fm_programme_df.at[i, 'level_id'] = 0

            def from_agg_id_to_agg_id(from_level_id, to_level_id):
                iterator = (
                    (from_level_it, to_level_it)
                    for (_,from_level_it), (_, to_level_it) in zip(
                        fm_programme_df[fm_programme_df['level_id']==from_level_id].iterrows(),
                        fm_programme_df[fm_programme_df['level_id']==to_level_id].iterrows()
                    )
                )
                for from_level_it, to_level_it in iterator:
                    yield from_level_it['agg_id'], to_level_id, to_level_it['agg_id']

            levels = list(set(fm_programme_df['level_id']))

            data = [
                (from_agg_id, level_id, to_agg_id) for from_level_id, to_level_id in zip(levels, levels[1:]) for from_agg_id, level_id, to_agg_id in from_agg_id_to_agg_id(from_level_id, to_level_id)
            ]

            fm_programme_df = pd.DataFrame(columns=['from_agg_id', 'level_id', 'to_agg_id'], data=data, dtype=int).drop_duplicates()

            expected = tuple(
                {
                    k:it[k] for k in ('from_agg_id', 'level_id', 'to_agg_id',)
                } for _, it in fm_programme_df.iterrows()
            )

            with io.open(fm_programme_file_path, 'r', encoding='utf-8') as f:
                result = tuple(pd.read_csv(f).T.to_dict().values())

            self.assertEqual(expected, result)

    def check_fm_xref_file(self, fm_items_df, fm_xref_file_path):
            data = [
                (i + 1, agg_id, layer_id) for i, (agg_id, layer_id) in enumerate(itertools.product(set(fm_items_df['agg_id']), set(fm_items_df['layer_id'])))
            ]

            fm_xref_df = pd.DataFrame(columns=['output', 'agg_id', 'layer_id'], data=data, dtype=int)

            expected = tuple(
                {
                    k:it[k] for k in ('output', 'agg_id', 'layer_id',)
                } for _, it in fm_xref_df.iterrows()
            )

            with io.open(fm_xref_file_path, 'r', encoding='utf-8') as f:
                result = tuple(pd.read_csv(f).T.to_dict().values())

            self.assertEqual(expected, result)

    def check_fmsummaryxref_file(self, fm_items_df, fmsummaryxref_file_path):
            data = [
                (i + 1, 1, 1) for i, _ in enumerate(itertools.product(set(fm_items_df['agg_id']), set(fm_items_df['layer_id'])))
            ]

            fmsummaryxref_df = pd.DataFrame(columns=['output', 'summary_id', 'summaryset_id'], data=data, dtype=int)

            expected = tuple(
                {
                    k:it[k] for k in ('output', 'summary_id', 'summaryset_id',)
                } for _, it in fmsummaryxref_df.iterrows()
            )

            with io.open(fmsummaryxref_file_path, 'r', encoding='utf-8') as f:
                result = tuple(pd.read_csv(f).T.to_dict().values())

            self.assertEqual(expected, result)

class WriteGulInputFiles(GulInputFilesGenerationTestCase):

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(
            from_tivs1=just(1.0),
            from_tivs2=just(0.0),
            size=2
        ),
        keys=keys(from_statuses=just(OASIS_KEYS_STATUS['success']['id']), size=2),
    )
    def test_paths_are_stored_in_the_model___model_paths_are_used(self, exposures, keys):
        profile = self.profile

        model = fake_model(resources={'canonical_exposure_profile': profile})

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            omr = model.resources
            ofp = omr['oasis_files_pipeline']

            ofp.keys_file_path = keys_file.name
            ofp.canonical_exposure_file_path = exposures_file.name

            ofp.items_file_path = os.path.join(out_dir, 'items.csv')
            ofp.coverages_file_path = os.path.join(out_dir, 'coverages.csv')
            ofp.gulsummaryxref_file_path = os.path.join(out_dir, 'gulsummaryxref.csv')

            gul_input_files = self.manager.write_gul_input_files(oasis_model=model)

            gul_items_df = omr['gul_items_df']

            self.check_items_file(gul_items_df, gul_input_files['items'])
            self.check_coverages_file(gul_items_df, gul_input_files['coverages'])
            self.check_gulsummaryxref_file(gul_items_df, gul_input_files['gulsummaryxref'])

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(
            from_tivs1=just(1.0),
            from_tivs2=just(0.0),
            size=2
        ),
        keys=keys(from_statuses=just(OASIS_KEYS_STATUS['success']['id']), size=2)
    )
    def test_paths_are_stored_in_the_kwargs___kwarg_paths_are_used(self, exposures, keys):
        profile = self.profile

        with NamedTemporaryFile('w') as keys_file, NamedTemporaryFile('w') as exposures_file, TemporaryDirectory() as out_dir:
            write_canonical_files(exposures, exposures_file.name)
            write_keys_files(keys, keys_file.name)

            gul_items_df, _ = self.manager.get_gul_input_items(profile, exposures_file.name, keys_file.name)

            gul_input_files = self.manager.write_gul_input_files(
                canonical_exposure_profile=profile,
                keys_file_path=keys_file.name,
                canonical_exposure_file_path=exposures_file.name,
                items_file_path=os.path.join(out_dir, 'items.csv'),
                coverages_file_path=os.path.join(out_dir, 'coverages.csv'),
                gulsummaryxref_file_path=os.path.join(out_dir, 'gulsummaryxref.csv')
            )

            self.check_items_file(gul_items_df, gul_input_files['items'])
            self.check_coverages_file(gul_items_df, gul_input_files['coverages'])
            self.check_gulsummaryxref_file(gul_items_df, gul_input_files['gulsummaryxref'])


class WriteFmInputFiles(FmInputFilesGenerationTestCase):

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(
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
            from_policy_types=just(1),
            from_account_deductibles=just(0),
            from_account_min_deductibles=just(0),
            from_account_max_deductibles=just(0),
            from_account_limits=just(0.1),
            from_layer_deductibles=just(1),
            from_layer_limits=just(1),
            size=1
        ),
        guls=gul_input_items(
            from_peril_ids=just(OASIS_PERILS['wind']['id']),
            from_coverage_type_ids=just(OASIS_COVERAGE_TYPES['buildings']['id']),
            from_tiv_elements=just('wscv1val'),
            from_tivs=just(100),
            from_tiv_tgids=just(1),
            from_deductible_elements=just('wscv1ded'),
            from_min_deductible_elements=just(None),
            from_max_deductible_elements=just(None),
            from_limit_elements=just('wscv1limit'),
            from_share_elements=just(None),
            size=2
        )
    )
    def test_exposure_with_one_coverage_type_and_fm_terms_with_one_account_and_one_top_level_layer_per_account_and_model_lookup_supporting_single_peril_and_coverage_type_paths_are_stored_in_the_model___model_paths_are_used_and_all_fm_files_are_generated(self, exposures, accounts, guls):
        cep = self.exposures_profile
        cap = self.accounts_profile
        ufcp = self.unified_canonical_profile
        fmap = self.fm_agg_profile

        for it in exposures:
            it['cond1name'] = 0

        canexp_df, gul_items_df = (pd.DataFrame(data=its, dtype=object) for its in [exposures, guls])

        for df in [canexp_df, gul_items_df]:
            df = df.where(df.notnull(), None)
            df.columns = df.columns.str.lower()

        with NamedTemporaryFile('w') as accounts_file, TemporaryDirectory() as out_dir:
            write_canonical_files(canonical_accounts=accounts, canonical_accounts_file_path=accounts_file.name)

            model = fake_model(resources={
                'canonical_exposure_df': canexp_df,
                'gul_items_df': gul_items_df,
                'canonical_exposure_profile': cep,
                'canonical_accounts_profile': cap,
                'fm_agg_profile': fmap
            })
            omr = model.resources
            ofp = omr['oasis_files_pipeline']
            
            ofp.canonical_accounts_file_path = accounts_file.name
            ofp.fm_policytc_file_path = os.path.join(out_dir, 'fm_policytc.csv')
            ofp.fm_profile_file_path = os.path.join(out_dir, 'fm_profile.csv')
            ofp.fm_programme_file_path = os.path.join(out_dir, 'fm_programme.csv')
            ofp.fm_xref_file_path = os.path.join(out_dir, 'fm_xref.csv')
            ofp.fmsummaryxref_file_path = os.path.join(out_dir, 'fmsummaryxref.csv')

            fm_input_files = self.manager.write_fm_input_files(oasis_model=model)

            fm_items_df = omr['fm_items_df']

            self.check_fm_policytc_file(fm_items_df, fm_input_files['fm_policytc'])
            self.check_fm_profile_file(fm_items_df, fm_input_files['fm_profile'])
            self.check_fm_programme_file(fm_items_df, fm_input_files['fm_programme'])
            self.check_fm_xref_file(fm_items_df, fm_input_files['fm_xref'])
            self.check_fmsummaryxref_file(fm_items_df, fm_input_files['fmsummaryxref'])

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        exposures=canonical_exposure(
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
            from_policy_types=just(1),
            from_account_deductibles=just(0),
            from_account_min_deductibles=just(0),
            from_account_max_deductibles=just(0),
            from_account_limits=just(0.1),
            from_layer_deductibles=just(1),
            from_layer_limits=just(1),
            size=1
        ),
        guls=gul_input_items(
            from_peril_ids=just(OASIS_PERILS['wind']['id']),
            from_coverage_type_ids=just(OASIS_COVERAGE_TYPES['buildings']['id']),
            from_tiv_elements=just('wscv1val'),
            from_tivs=just(100),
            from_tiv_tgids=just(1),
            from_deductible_elements=just('wscv1ded'),
            from_min_deductible_elements=just(None),
            from_max_deductible_elements=just(None),
            from_limit_elements=just('wscv1limit'),
            from_share_elements=just(None),
            size=2
        )
    )
    def test_exposure_with_one_coverage_type_and_fm_terms_with_one_account_and_one_top_level_layer_per_account_and_model_lookup_supporting_single_peril_and_coverage_type_paths_are_stored_in_the_kwargs___kwarg_paths_are_used_and_all_fm_files_are_generated(self, exposures, accounts, guls):
        cep = self.exposures_profile
        cap = self.accounts_profile
        ufcp = self.unified_canonical_profile
        fmap = self.fm_agg_profile

        for it in exposures:
            it['cond1name'] = 0

        canexp_df, gul_items_df = (pd.DataFrame(data=its, dtype=object) for its in [exposures, guls])

        for df in [canexp_df, gul_items_df]:
            df = df.where(df.notnull(), None)
            df.columns = df.columns.str.lower()
        
        with NamedTemporaryFile('w') as accounts_file, TemporaryDirectory() as out_dir:
            write_canonical_files(canonical_accounts=accounts, canonical_accounts_file_path=accounts_file.name)

            fm_items_df, canacc_df = self.manager.get_fm_input_items(
                canexp_df,
                gul_items_df,
                cep,
                cap,
                accounts_file.name,
                fmap
            )
            
            os.path.join(out_dir, 'fm_policytc.csv')

            fm_input_files = self.manager.write_fm_input_files(
                canonical_exposure_df=canexp_df,
                gul_items_df=gul_items_df,
                canonical_exposure_profile=cep,
                canonical_accounts_profile=cap,
                canonical_accounts_file_path=accounts_file.name,
                fm_agg_profile=fmap,
                fm_policytc_file_path=os.path.join(out_dir, 'fm_policytc.csv'),
                fm_profile_file_path=os.path.join(out_dir, 'fm_profile.csv'),
                fm_programme_file_path=os.path.join(out_dir, 'fm_programme.csv'),
                fm_xref_file_path=os.path.join(out_dir, 'fm_xref.csv'),
                fmsummaryxref_file_path=os.path.join(out_dir, 'fmsummaryxref.csv')
            )

            self.check_fm_policytc_file(fm_items_df, fm_input_files['fm_policytc'])
            self.check_fm_profile_file(fm_items_df, fm_input_files['fm_profile'])
            self.check_fm_programme_file(fm_items_df, fm_input_files['fm_programme'])
            self.check_fm_xref_file(fm_items_df, fm_input_files['fm_xref'])
            self.check_fmsummaryxref_file(fm_items_df, fm_input_files['fmsummaryxref'])

class Startofp(TestCase):

    def setUp(self):
        self.manager = om()
        self.exposures_profile = canonical_exposure_profile
        self.accounts_profile = canonical_accounts_profile

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

    def test_start_oasis_files_pipeline_with_model_and_no_source_exposure_files_path__oasis_exception_is_raised(self):
        mgr = self.manager

        with TemporaryDirectory() as oasis_files_path:
            model = fake_model(resources={'oasis_files_path': oasis_files_path})

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_model=model)

    def test_start_oasis_files_pipeline_with_model_and_no_canonical_exposure_profile__oasis_exception_is_raised(self):
        mgr = self.manager

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposure_file:
            model = fake_model(resources={
                'oasis_files_path': oasis_files_path,
                'source_exposure_file_path': source_exposure_file.name
            })

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_model=model)

    def test_start_oasis_files_pipeline_with_model_and_fm_and_no_source_accounts_file_path__oasis_exception_is_raised(self):
        mgr = self.manager
        cep = self.exposures_profile

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposure_file:
            model = fake_model(resources={
                'oasis_files_path': oasis_files_path,
                'source_exposure_file_path': source_exposure_file.name,
                'canonical_exposure_profile': cep,
                'fm': True
            })

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_model=model)

    def test_start_oasis_files_pipeline_with_model_and_fm_set_in_kwargs_and_no_source_accounts_file_path__oasis_exception_is_raised(self):
        mgr = self.manager
        cep = self.exposures_profile

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposure_file:
            model = fake_model(resources={
                'oasis_files_path': oasis_files_path,
                'source_exposure_file_path': source_exposure_file.name,
                'canonical_exposure_profile': cep
            })

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_model=model, fm=True)

    def test_start_oasis_files_pipeline_with_model_and_fm_and_no_canonical_accounts_profile__oasis_exception_is_raised(self):
        mgr = self.manager
        cep = self.exposures_profile

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposure_file:
            model = fake_model(resources={
                'oasis_files_path': oasis_files_path,
                'source_exposure_file_path': source_exposure_file.name,
                'canonical_exposure_profile': cep,
                'fm': True
            })

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_model=model)

    def test_start_oasis_files_pipeline_with_model_and_fm_set_in_kwargs_and_no_canonical_accounts_profile__oasis_exception_is_raised(self):
        mgr = self.manager
        cep = self.exposures_profile

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposure_file:
            model = fake_model(resources={
                'oasis_files_path': oasis_files_path,
                'source_exposure_file_path': source_exposure_file.name,
                'canonical_exposure_profile': cep
            })

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_model=model, fm=True)

    def test_start_oasis_files_pipeline_with_model_gul_only_all_resources_provided__all_gul_files_generated(self):
        pass

    def test_start_oasis_files_pipeline_with_model_fm_all_resources_provided__all_gul_and_fm_files_generated(self):
        pass

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

    def test_start_oasis_files_pipeline_with_kwargs_and_no_source_exposure_files_path__oasis_exception_is_raised(self):
        mgr = self.manager

        with TemporaryDirectory() as oasis_files_path:

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(oasis_files_path=oasis_files_path)

    def test_start_oasis_files_pipeline_with_kwargs_and_no_canonical_exposure_profile__oasis_exception_is_raised(self):
        mgr = self.manager

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposure_file:

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(
                    oasis_files_path=oasis_files_path,
                    source_exposure_file_path=source_exposure_file.name
                )

    def test_start_oasis_files_pipeline_with_kwargs_and_fm_and_no_source_accounts_file_path__oasis_exception_is_raised(self):
        mgr = self.manager
        cep = self.exposures_profile

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposure_file:

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(
                    oasis_files_path=oasis_files_path,
                    source_exposure_file_path=source_exposure_file.name,
                    canonical_exposure_profile=cep,
                    fm=True
                )

    def test_start_oasis_files_pipeline_with_kwargs_and_fm_and_no_canonical_accounts_profile__oasis_exception_is_raised(self):
        mgr = self.manager
        cep = self.exposures_profile

        with TemporaryDirectory() as oasis_files_path, NamedTemporaryFile('r') as source_exposure_file:

            with self.assertRaises(OasisException):
                mgr.start_oasis_files_pipeline(
                    oasis_files_path=oasis_files_path,
                    source_exposure_file_path=source_exposure_file.name,
                    canonical_exposure_profile=cep,
                    fm=True
                )

    def test_start_oasis_files_pipeline_with_kwargs_gul_only_all_resources_provided__all_gul_files_generated(self):
        pass

    def test_start_oasis_files_pipeline_with_kwargs_fm_all_resources_provided__all_gul_and_fm_files_generated(self):
        pass
