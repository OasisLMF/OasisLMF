from __future__ import unicode_literals, print_function

import csv
import io
import json
import os
import string

from collections import OrderedDict
from unittest import TestCase

import pandas as pd

from backports.tempfile import TemporaryDirectory
from hypothesis import given
from hypothesis.strategies import (
    booleans,
    fixed_dictionaries,
    integers,
    just,
    lists,
    sampled_from,
    text,
    tuples,
)
from mock import Mock, patch
from six import StringIO
from tempfile import NamedTemporaryFile

from oasislmf.keys.lookup import OasisKeysLookupFactory
from oasislmf.utils.coverage import (
    BUILDING_COVERAGE_CODE,
    CONTENTS_COVERAGE_CODE,
    OTHER_STRUCTURES_COVERAGE_CODE,
    TIME_COVERAGE_CODE,
)
from oasislmf.utils.peril import (
    PERIL_ID_FLOOD,
    PERIL_ID_QUAKE,
    PERIL_ID_SURGE,
    PERIL_ID_WIND,
)
from oasislmf.utils.status import (
    KEYS_STATUS_FAIL,
    KEYS_STATUS_NOMATCH,
    KEYS_STATUS_SUCCESS,
)
from oasislmf.utils.exceptions import OasisException

from tests import keys_data


class OasisKeysLookupFactoryCreate(TestCase):
    def write_version_file(self, supplier, model, version, path):
        with io.open(path, 'w', encoding='utf-8') as f:
            f.write('{},{},{}'.format(supplier, model, version))

    def write_py_module(self, model, path):
        with io.open(path, 'w', encoding='utf-8') as f:
            f.writelines([
                'from oasislmf.keys.lookup import OasisBaseKeysLookup\n',
                'class {}KeysLookup(OasisBaseKeysLookup):\n'.format(model),
                '    pass\n'
            ])

    @given(
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
    )
    def test_keys_path_is_not_supplied___correct_instance_is_created_with_correct_model_info(self, supplier, model, version):
        with TemporaryDirectory() as d:
            version_path = os.path.join(d, 'version.csv')
            self.write_version_file(supplier, model, version, version_path)

            module_path = os.path.join(d, '{}_lookup.py'.format(model))
            self.write_py_module(model, module_path)

            _, instance = OasisKeysLookupFactory.create(
                model_version_file_path=version_path,
                lookup_package_path=module_path,
            )

            self.assertEqual(type(instance).__name__, '{}KeysLookup'.format(model))
            self.assertEqual(instance.supplier, supplier)
            self.assertEqual(instance.model_name, model)
            self.assertEqual(instance.model_version, version)
            self.assertEqual(instance.keys_data_directory, os.path.join(os.sep, 'var', 'oasis', 'keys_data'))

    @given(
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
    )
    def test_keys_path_is_supplied___correct_instance_is_created_with_correct_model_info_and_keys_path(self, supplier, model, version):
        with TemporaryDirectory() as d:
            keys_path = os.path.join(d, 'keys')
            version_path = os.path.join(d, 'version.csv')
            self.write_version_file(supplier, model, version, version_path)

            module_path = os.path.join(d, '{}_lookup.py'.format(model))
            self.write_py_module(model, module_path)

            _, instance = OasisKeysLookupFactory.create(
                model_keys_data_path=keys_path,
                model_version_file_path=version_path,
                lookup_package_path=module_path,
            )

            self.assertEqual(type(instance).__name__, '{}KeysLookup'.format(model))
            self.assertEqual(instance.supplier, supplier)
            self.assertEqual(instance.model_name, model)
            self.assertEqual(instance.model_version, version)
            self.assertEqual(instance.keys_data_directory, keys_path)


class OasisKeysLookupFactoryGetModelExposures(TestCase):
    def test_no_file_or_exposures_are_provided___oasis_exception_is_raised(self):
        with self.assertRaises(OasisException):
            OasisKeysLookupFactory.get_model_exposures()

    @given(lists(tuples(integers(min_value=0, max_value=100), integers(min_value=0, max_value=100))))
    def test_file_is_provided___file_content_is_loaded(self, data):
        data = [('first', 'second')] + data

        with NamedTemporaryFile('w') as f:
            csv.writer(f).writerows(data)
            f.flush()

            res = OasisKeysLookupFactory.get_model_exposures(model_exposures_file_path=f.name)
            res = [tuple(res)] + [tuple(res.iloc[i]) for i in range(len(res))]

            self.assertEqual(res, data)

    @given(lists(tuples(integers(min_value=0, max_value=100), integers(min_value=0, max_value=100))))
    def test_exposures_string_is_provided___file_content_is_loaded(self, data):
        stream = StringIO()
        data = [('first', 'second')] + data

        csv.writer(stream).writerows(data)

        res = OasisKeysLookupFactory.get_model_exposures(model_exposures=stream.getvalue())
        res = [tuple(res)] + [tuple(res.iloc[i]) for i in range(len(res))]

        self.assertEqual(res, data)


class OasisKeysLookupFactoryWriteOasisKeysFiles(TestCase):

    @given(
        successes=keys_data(status=KEYS_STATUS_SUCCESS, min_size=5, max_size=5),
        nonsuccesses=keys_data(status='unsuccessful', min_size=5, max_size=5)
    )
    def test_records_are_given___records_are_written_to_oasis_keys_files_correctly(self, successes, nonsuccesses):

        oasis_keys_file_to_record_metadict = {
            'LocID': 'id',
            'PerilID': 'peril_id',
            'CoverageID': 'coverage',
            'AreaPerilID': 'area_peril_id',
            'VulnerabilityID': 'vulnerability_id'
        }
        oasis_keys_error_file_to_record_metadict = {
            'LocID': 'id',
            'PerilID': 'peril_id',
            'CoverageID': 'coverage',
            'Message': 'message'
        }

        with TemporaryDirectory() as d:
            keys_file_path = os.path.join(d, 'keys.csv')
            keys_error_file_path = os.path.join(d, 'keys-errors.csv')

            _, successes_count = OasisKeysLookupFactory.write_oasis_keys_file(successes, keys_file_path)
            _, nonsuccesses_count = OasisKeysLookupFactory.write_oasis_keys_error_file(nonsuccesses, keys_error_file_path)

            with io.open(keys_file_path, 'r', encoding='utf-8') as f1, io.open(keys_error_file_path, 'r', encoding='utf-8') as f2:
                written_successes = [dict((oasis_keys_file_to_record_metadict[k], r[k]) for k in r) for r in pd.read_csv(f1).T.to_dict().values()]
                written_nonsuccesses = [dict((oasis_keys_error_file_to_record_metadict[k], r[k]) for k in r) for r in pd.read_csv(f2).T.to_dict().values()]

            success_matches = filter(lambda r: (r['id'] == ws['id'] for ws in written_successes), successes)
            nonsuccess_matches = filter(lambda r: (r['id'] == ws['id'] for ws in written_nonsuccesses), nonsuccesses)

            self.assertEqual(successes_count, len(successes))
            self.assertEqual(success_matches, successes)

            self.assertEqual(nonsuccesses_count, len(nonsuccesses))
            self.assertEqual(nonsuccess_matches, nonsuccesses)


class OasisKeysLookupFactoryWriteJsonFiles(TestCase):
    @given(
        successes=keys_data(status=KEYS_STATUS_SUCCESS),
        nonsuccesses=keys_data(status='unsuccessful')
    )
    def test_records_are_given___records_are_written_to_json_keys_files_correctly(self, successes, nonsuccesses):

        with TemporaryDirectory() as d:
            keys_file_path = os.path.join(d, 'keys.json')
            keys_error_file_path = os.path.join(d, 'keys-errors.json')

            _, successes_count = OasisKeysLookupFactory.write_json_keys_file(successes, keys_file_path)
            _, nonsuccesses_count = OasisKeysLookupFactory.write_json_keys_file(nonsuccesses, keys_error_file_path)

            with io.open(keys_file_path, 'r', encoding='utf-8') as f1, io.open(keys_error_file_path, 'r', encoding='utf-8') as f2:
                written_successes = json.load(f1)
                written_nonsuccesses = json.load(f2)

            self.assertEqual(successes_count, len(successes))
            self.assertEqual(written_successes, successes)

            self.assertEqual(nonsuccesses_count, len(nonsuccesses))
            self.assertEqual(written_nonsuccesses, nonsuccesses)


class OasisKeysLookupFactoryGetKeys(TestCase):
    def create_fake_lookup(self, return_value=None):
        self.lookup_instance = Mock()
        self.lookup_instance.process_locations = Mock(return_value=return_value or [])
        return self.lookup_instance

    def test_no_model_exposures_are_provided___oasis_exception_is_raised(self):
        with self.assertRaises(OasisException):
            list(OasisKeysLookupFactory.get_keys(self.create_fake_lookup()))

    @given(text(min_size=1, max_size=10, alphabet=string.ascii_letters), text(min_size=1, max_size=10, alphabet=string.ascii_letters))
    def test_model_exposures_path_is_provided___path_is_passed_to_get_model_exposures_result_is_passed_to_lokkup_process_locations(self, path, result):
        with patch('oasislmf.keys.lookup.OasisKeysLookupFactory.get_model_exposures', Mock(return_value=result)):
            list(OasisKeysLookupFactory.get_keys(self.create_fake_lookup(), model_exposures_file_path=path))

            OasisKeysLookupFactory.get_model_exposures.assert_called_once_with(model_exposures_file_path=path, model_exposures=None)
            self.lookup_instance.process_locations.assert_called_once_with(result)

    @given(text(min_size=1, max_size=10, alphabet=string.ascii_letters), text(min_size=1, max_size=10, alphabet=string.ascii_letters))
    def test_model_exposures_are_provided___exposures_are_passed_to_get_model_exposures_result_is_passed_to_lookup_process_locations(self, exposures, result):
        with patch('oasislmf.keys.lookup.OasisKeysLookupFactory.get_model_exposures', Mock(return_value=result)):
            list(OasisKeysLookupFactory.get_keys(self.create_fake_lookup(), model_exposures=exposures))

            OasisKeysLookupFactory.get_model_exposures.assert_called_once_with(model_exposures=exposures, model_exposures_file_path=None)
            self.lookup_instance.process_locations.assert_called_once_with(result)

    @given(lists(fixed_dictionaries({
        'id': integers(),
        'status': sampled_from(['success', 'failure'])
    })))
    def test_entries_are_dictionaries_success_only_is_true___only_successes_are_included(self, data):
        with patch('oasislmf.keys.lookup.OasisKeysLookupFactory.get_model_exposures'):
            self.create_fake_lookup(return_value=data)

            res = list(OasisKeysLookupFactory.get_keys(lookup=self.lookup_instance, model_exposures_file_path='path'))

            self.assertEqual(res, [d for d in data if d['status'] == 'success'])

    @given(lists(fixed_dictionaries({
        'id': integers(),
        'status': sampled_from(['success', 'failure'])
    })))
    def test_entries_are_dictionaries_success_only_is_false___all_entries_are_included(self, data):
        with patch('oasislmf.keys.lookup.OasisKeysLookupFactory.get_model_exposures'):
            self.create_fake_lookup(return_value=data)

            res = list(OasisKeysLookupFactory.get_keys(lookup=self.lookup_instance, model_exposures_file_path='path', success_only=False))

            self.assertEqual(res, data)


class OasisKeysLookupFactoryWriteKeys(TestCase):
    def create_fake_lookup(self):
        self.lookup_instance = Mock()
        return self.lookup_instance

    def test_no_model_exposures_are_provided___oasis_exception_is_raised(self):
        with self.assertRaises(OasisException):
            list(OasisKeysLookupFactory.get_keys(self.create_fake_lookup()))

    @given(
        data=keys_data(status=KEYS_STATUS_SUCCESS, min_size=10, max_size=10)
    )
    def test_produced_keys_are_passed_to_write_oasis_keys_file(self, data):
        with TemporaryDirectory() as d,\
             patch('oasislmf.keys.lookup.OasisKeysLookupFactory.get_keys', Mock(return_value=(r for r in data))) as get_keys_mock,\
             patch('oasislmf.keys.lookup.OasisKeysLookupFactory.write_oasis_keys_file') as write_oasis_keys_file_mock:

            keys_file_path = os.path.join(d, 'piwind-keys.csv')
            OasisKeysLookupFactory.save_keys(
                lookup=self.create_fake_lookup(),
                keys_file_path=keys_file_path,
                model_exposures=json.dumps(data).decode()
            )

            get_keys_mock.assert_called_once_with(
                lookup=self.lookup_instance,
                model_exposures=json.dumps(data).decode(),
                model_exposures_file_path=None,
                success_only=True
            )
            write_oasis_keys_file_mock.assert_called_once_with(data, keys_file_path)

