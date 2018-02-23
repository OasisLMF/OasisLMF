from __future__ import unicode_literals

import csv
import json
import string
from tempfile import NamedTemporaryFile
from unittest import TestCase

import os
import io

from six import StringIO
from backports.tempfile import TemporaryDirectory
from hypothesis import given
from hypothesis.strategies import text, integers, tuples, lists, fixed_dictionaries, sampled_from, booleans
from mock import Mock, patch

from oasislmf.keys.lookup import OasisKeysLookupFactory
from oasislmf.utils.exceptions import OasisException


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


class OasisKeysLookupFactoryWriteOasisKeyFile(TestCase):
    @given(lists(fixed_dictionaries({
        'id': integers(),
        'peril_id': integers(),
        'coverage': integers(),
        'area_peril_id': integers(),
        'vulnerability_id': integers(),
    })))
    def test_records_are_given___records_are_written_to_csv_correctly(self, data):
        expected_heading = {
            'id': 'LocID',
            'peril_id': 'PerilID',
            'coverage': 'CoverageID',
            'area_peril_id': 'AreaPerilID',
            'vulnerability_id': 'VulnerabilityID',
        }

        with TemporaryDirectory() as d:
            output_file = os.path.join(d, 'output.csv')

            res_path, res_count = OasisKeysLookupFactory.write_oasis_keys_file(data, output_file)

            expected_data = [
                {k: str(v) for k, v in row.items()} for row in data
            ]

            with io.open(output_file, encoding='utf-8') as f:
                res_data = list(csv.DictReader(f, fieldnames=['id', 'peril_id', 'coverage', 'area_peril_id', 'vulnerability_id']))

            self.assertEqual(res_count, len(data))
            self.assertEqual(res_path, output_file)
            self.assertEqual(res_data[0], expected_heading)
            self.assertEqual(res_data[1:], expected_data)


class OasisKeysLookupFactoryWriteListKeysFiles(TestCase):
    @given(lists(fixed_dictionaries({
        'id': integers(),
        'peril_id': integers(),
        'coverage': integers(),
        'area_peril_id': integers(),
        'vulnerability_id': integers(),
    })))
    def test_records_are_given___records_are_written_to_file_correctly(self, data):
        with TemporaryDirectory() as d:
            output_file = os.path.join(d, 'output')

            res_path, res_count = OasisKeysLookupFactory.write_list_keys_file(data, output_file)

            with io.open(output_file, encoding='utf-8') as f:
                result_data = json.loads('[{}]'.format(f.read().strip()[:-1]))

            self.assertEqual(res_count, len(data))
            self.assertEqual(res_path, output_file)
            self.assertEqual(result_data, data)


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
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        booleans(),
    )
    def test_produced_keys_are_passed_to_write_oasis_keys_file(self, exposures_path, exposures, output, success_only):
        with patch('oasislmf.keys.lookup.OasisKeysLookupFactory.get_keys', Mock(return_value=['got keys'])) as get_keys_mock, \
                patch('oasislmf.keys.lookup.OasisKeysLookupFactory.write_oasis_keys_file') as write_oasis_keys_file_mock:

            OasisKeysLookupFactory.save_keys(
                self.create_fake_lookup(),
                output,
                model_exposures_file_path=exposures_path,
                model_exposures=exposures,
                success_only=success_only,
            )

            get_keys_mock.assert_called_once_with(
                lookup=self.lookup_instance,
                model_exposures=exposures,
                model_exposures_file_path=exposures_path,
                success_only=success_only,
            )
            write_oasis_keys_file_mock.assert_called_once_with(['got keys'], os.path.abspath(output))
