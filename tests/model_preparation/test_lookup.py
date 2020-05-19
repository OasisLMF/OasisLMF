import io
import json
import os
import string

from unittest import TestCase

import pandas as pd
import pytest

from backports.tempfile import TemporaryDirectory
from hypothesis import (
    given,
    HealthCheck,
    settings,
)
from hypothesis.strategies import (
    fixed_dictionaries,
    integers,
    just,
    lists,
    sampled_from,
    text,
    tuples,
)
from mock import Mock, patch
from tempfile import NamedTemporaryFile

from oasislmf.model_preparation.lookup import OasisLookupFactory as olf
from oasislmf.utils.data import get_dtypes_and_required_cols, get_location_df 
from oasislmf.utils.defaults import get_loc_dtypes
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.status import OASIS_KEYS_STATUS

from tests.data import keys

# Determine number and names of required columns in loc file
_, loc_required_cols = get_dtypes_and_required_cols(get_loc_dtypes)
loc_required_cols = [name.lower() for name in loc_required_cols]
loc_required_cols.append('loc_id')
loc_data_cols = [
    integers(min_value=0, max_value=100)
    for _ in range(len(loc_required_cols))
]


class OasisLookupFactoryCreate(TestCase):

    @staticmethod
    def write_version_file(supplier, model, version, path):
        with io.open(path, 'w', encoding='utf-8') as f:
            f.write('{},{},{}'.format(supplier, model, version))

    @staticmethod
    def write_py_module(model, path):
        with io.open(path, 'w', encoding='utf-8') as f:
            f.writelines([
                'from oasislmf.model_preparation.lookup import OasisBaseKeysLookup\n',
                'class {}KeysLookup(OasisBaseKeysLookup):\n'.format(model),
                '    pass\n'
            ])

    @staticmethod
    def write_complex_config_file(data, path):
        with io.open(path, "w", encoding='utf-8') as f:
            f.write(data)

    @given(
        supplier=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        model=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        version=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
    )
    def test_keys_path_is_supplied___correct_instance_is_created_with_correct_model_info_and_keys_path(self, supplier,
                                                                                                       model, version):
        with TemporaryDirectory() as d:
            keys_path = os.path.join(d, 'keys')
            os.mkdir(keys_path)

            version_path = os.path.join(d, 'version.csv')
            self.write_version_file(supplier, model, version, version_path)

            module_path = os.path.join(d, '{}_lookup.py'.format(model))
            self.write_py_module(model, module_path)

            _, instance = olf.create(
                model_keys_data_path=keys_path,
                model_version_file_path=version_path,
                lookup_module_path=module_path
            )

            self.assertEqual(type(instance).__name__, '{}KeysLookup'.format(model))
            self.assertEqual(instance.supplier, supplier)
            self.assertEqual(instance.model_name, model)
            self.assertEqual(instance.model_version, version)
            self.assertEqual(instance.keys_data_directory, keys_path)
            self.assertEqual(instance.complex_lookup_config_fp, None)

    @given(
        supplier=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        model=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        version=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
    )
    def test_keys_path_not_supplied___correct_instance_is_created_with_correct_model_info_and_keys_path(self, supplier,
                                                                                                        model, version):
        with TemporaryDirectory() as d:
            version_path = os.path.join(d, 'version.csv')
            self.write_version_file(supplier, model, version, version_path)

            module_path = os.path.join(d, '{}_lookup.py'.format(model))
            self.write_py_module(model, module_path)

            _, instance = olf.create(
                model_version_file_path=version_path,
                lookup_module_path=module_path
            )

            self.assertEqual(type(instance).__name__, '{}KeysLookup'.format(model))
            self.assertEqual(instance.supplier, supplier)
            self.assertEqual(instance.model_name, model)
            self.assertEqual(instance.model_version, version)
            self.assertEqual(instance.keys_data_directory, os.path.join(os.sep, 'var', 'oasis', 'keys_data'))
            self.assertEqual(instance.complex_lookup_config_fp, None)

    @given(
        supplier=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        model=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        version=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        complex_lookup_data=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
    )
    def test_complex_lookup_config_is_supplied___correct_instance_is_created_with_correct_config_path(self, supplier,
                                                                                                      model, version,
                                                                                                      complex_lookup_data):
        with TemporaryDirectory() as d:
            keys_path = os.path.join(d, 'keys')
            os.mkdir(keys_path)

            version_path = os.path.join(d, 'version.csv')
            self.write_version_file(supplier, model, version, version_path)

            module_path = os.path.join(d, '{}_lookup.py'.format(model))
            self.write_py_module(model, module_path)

            complex_lookup_config_path = os.path.join(d, 'lookup_config.json'.format(model))
            self.write_complex_config_file(complex_lookup_data, complex_lookup_config_path)

            output_directory = os.path.join(d, 'output')
            os.mkdir(output_directory)

            _, instance = olf.create(
                model_keys_data_path=keys_path,
                model_version_file_path=version_path,
                lookup_module_path=module_path,
                complex_lookup_config_fp=complex_lookup_config_path,
                output_directory=output_directory
            )

            self.assertEqual(type(instance).__name__, '{}KeysLookup'.format(model))
            self.assertEqual(instance.supplier, supplier)
            self.assertEqual(instance.model_name, model)
            self.assertEqual(instance.model_version, version)
            self.assertEqual(instance.keys_data_directory, keys_path)
            self.assertEqual(instance.complex_lookup_config_fp, complex_lookup_config_path)
            self.assertEqual(instance.output_directory, output_directory)

    @given(
        model=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
    )
    def test_version_file_missing___correct_exception_raised(self, model):
        with TemporaryDirectory() as d:
            keys_path = os.path.join(d, 'keys')
            os.mkdir(keys_path)

            version_path = os.path.join(d, 'version.csv')

            module_path = os.path.join(d, '{}_lookup.py'.format(model))
            self.write_py_module(model, module_path)

            with self.assertRaisesRegex(OasisException,
                                        r"The path .*/version.csv \(model_version_file_path\) is indicated as preexisting"
                                        r" but does not exist"):
                _, instance = olf.create(
                    model_keys_data_path=keys_path,
                    model_version_file_path=version_path,
                    lookup_module_path=module_path
                )

    @given(
        supplier=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        model=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        version=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
    )
    def test_lookup_module_path_missing___correct_exception_raised(self, supplier, model, version):
        with TemporaryDirectory() as d:
            keys_path = os.path.join(d, 'keys')
            os.mkdir(keys_path)

            version_path = os.path.join(d, 'version.csv')
            self.write_version_file(supplier, model, version, version_path)

            module_path = os.path.join(d, '{}_lookup.py'.format(model))

            with self.assertRaisesRegex(OasisException,
                                        r"The path .*_lookup\.py \(lookup_module_path\) is indicated as preexisting"
                                        r" but does not exist"):
                _, instance = olf.create(
                    model_keys_data_path=keys_path,
                    model_version_file_path=version_path,
                    lookup_module_path=module_path
                )

    @given(
        supplier=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        model=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        version=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
    )
    def test_model_keys_data_path_missing___correct_exception_raised(self, supplier, model, version):
        with TemporaryDirectory() as d:
            keys_path = os.path.join(d, 'keys')

            version_path = os.path.join(d, 'version.csv')
            self.write_version_file(supplier, model, version, version_path)

            module_path = os.path.join(d, '{}_lookup.py'.format(model))
            self.write_py_module(model, module_path)

            with self.assertRaisesRegex(OasisException,
                                        r"The path .*/keys \(model_keys_data_path\) is indicated as preexisting"
                                        r" but does not exist"):
                _, instance = olf.create(
                    model_version_file_path=version_path,
                    lookup_module_path=module_path,
                    model_keys_data_path=keys_path
                )

    @given(
        supplier=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        model=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        version=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
    )
    def test_lookup_module_path_not_supplied___correct_exception_raised(self, supplier, model, version):
        with TemporaryDirectory() as d:
            keys_path = os.path.join(d, 'keys')
            os.mkdir(keys_path)

            version_path = os.path.join(d, 'version.csv')
            self.write_version_file(supplier, model, version, version_path)

            with self.assertRaisesRegex(OasisException,
                                        r"The path None \(lookup_module_path\) is indicated as preexisting"
                                        r" but is not a valid path"):
                _, instance = olf.create(
                    model_version_file_path=version_path,
                    model_keys_data_path = keys_path
                )

    @given(
        model=text(min_size=1, max_size=10, alphabet=string.ascii_letters),
    )
    def test_model_version_file_path_not_supplied___correct_exception_raised(self, model):
        with TemporaryDirectory() as d:
            keys_path = os.path.join(d, 'keys')
            os.mkdir(keys_path)

            module_path = os.path.join(d, '{}_lookup.py'.format(model))
            self.write_py_module(model, module_path)

            with self.assertRaisesRegex(OasisException,
                                        r"The path None \(model_version_file_path\) is indicated as preexisting"
                                        r" but is not a valid path"):
                _, instance = olf.create(
                    model_keys_data_path=keys_path,
                    lookup_module_path=module_path
                )


class OasisLookupFactoryGetSourceExposure(TestCase):

    def test_no_file_or_exposure_are_provided___oasis_exception_is_raised(self):
        with self.assertRaises(OasisException):
            get_location_df(exposure_fp=None)

class OasisLookupFactoryWriteOasisKeysFiles(TestCase):

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        successes=keys(from_statuses=just(OASIS_KEYS_STATUS['success']['id']), size=5),
        nonsuccesses=keys(
            from_statuses=sampled_from([OASIS_KEYS_STATUS['fail']['id'], OASIS_KEYS_STATUS['nomatch']['id']]), size=5)
    )
    def test_records_are_given___records_are_written_to_oasis_keys_files_correctly(self, successes, nonsuccesses):
        oasis_keys_file_to_record_metadict = {
            'LocID': 'id',
            'PerilID': 'peril_id',
            'CoverageTypeID': 'coverage_type',
            'AreaPerilID': 'area_peril_id',
            'VulnerabilityID': 'vulnerability_id'
        }
        oasis_keys_errors_file_to_record_metadict = {
            'LocID': 'id',
            'PerilID': 'peril_id',
            'CoverageTypeID': 'coverage_type',
            'Status': 'status',
            'Message': 'message'
        }

        with TemporaryDirectory() as d:
            keys_file_path = os.path.join(d, 'keys.csv')
            keys_errors_file_path = os.path.join(d, 'keys-errors.csv')

            _, successes_count = olf.write_oasis_keys_file(successes, keys_file_path)
            _, nonsuccesses_count = olf.write_oasis_keys_errors_file(nonsuccesses, keys_errors_file_path)

            with io.open(keys_file_path, 'r', encoding='utf-8') as f1, io.open(keys_errors_file_path, 'r',
                                                                               encoding='utf-8') as f2:
                written_successes = [dict((oasis_keys_file_to_record_metadict[k], r[k]) for k in r) for r in
                                     pd.read_csv(f1).T.to_dict().values()]
                written_nonsuccesses = [dict((oasis_keys_errors_file_to_record_metadict[k], r[k]) for k in r) for r in
                                        pd.read_csv(f2).T.to_dict().values()]

            success_matches = list(filter(lambda r: (r['id'] == ws['id'] for ws in written_successes), successes))
            nonsuccess_matches = list(
                filter(lambda r: (r['id'] == ws['id'] for ws in written_nonsuccesses), nonsuccesses))

            self.assertEqual(successes_count, len(successes))
            self.assertEqual(success_matches, successes)

            self.assertEqual(nonsuccesses_count, len(nonsuccesses))
            self.assertEqual(nonsuccess_matches, nonsuccesses)


class OasisLookupFactoryWriteJsonFiles(TestCase):

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        successes=keys(from_statuses=just(OASIS_KEYS_STATUS['success']['id']), size=5),
        nonsuccesses=keys(
            from_statuses=sampled_from([OASIS_KEYS_STATUS['fail']['id'], OASIS_KEYS_STATUS['nomatch']['id']]), size=5)
    )
    def test_records_are_given___records_are_written_to_json_keys_files_correctly(self, successes, nonsuccesses):
        with TemporaryDirectory() as d:
            keys_file_path = os.path.join(d, 'keys.json')
            keys_errors_file_path = os.path.join(d, 'keys-errors.json')

            _, successes_count = olf.write_json_keys_file(successes, keys_file_path)
            _, nonsuccesses_count = olf.write_json_keys_file(nonsuccesses, keys_errors_file_path)

            with io.open(keys_file_path, 'r', encoding='utf-8') as f1, io.open(keys_errors_file_path, 'r',
                                                                               encoding='utf-8') as f2:
                written_successes = json.load(f1)
                written_nonsuccesses = json.load(f2)

            self.assertEqual(successes_count, len(successes))
            self.assertEqual(written_successes, successes)

            self.assertEqual(nonsuccesses_count, len(nonsuccesses))
            self.assertEqual(written_nonsuccesses, nonsuccesses)


class OasisLookupFactoryGetKeys(TestCase):

    def create_fake_lookup(self, return_value=None):
        self.lookup_instance = Mock()
        self.lookup_instance.process_locations = Mock(return_value=return_value or [])
        return self.lookup_instance

    @given(lists(fixed_dictionaries({
        'id': integers(),
        'status': sampled_from(['success', 'failure'])
    })))
    def test_entries_are_dictionaries_success_only_is_true___only_successes_are_included(self, data):
        self.create_fake_lookup(return_value=data)
        mock_df = pd.DataFrame.from_dict(data)
        res = list(olf.get_keys_base(lookup=self.lookup_instance, loc_df=mock_df, success_only=True))
        self.assertEqual(res, [d for d in data if d['status'] == 'success'])

    @given(lists(fixed_dictionaries({
        'id': integers(),
        'status': sampled_from(['success', 'failure'])
    })))
    def test_entries_are_dictionaries_success_only_is_false___all_entries_are_included(self, data):
        self.create_fake_lookup(return_value=data)
        mock_df = pd.DataFrame.from_dict(data)
        res = list(olf.get_keys_base(lookup=self.lookup_instance, loc_df=mock_df, success_only=False))
        self.assertEqual(res, data)


class OasisLookupFactoryWriteKeys(TestCase):

    def create_fake_lookup(self):
        self.lookup_instance = Mock()
        return self.lookup_instance

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        data=keys(from_statuses=just(OASIS_KEYS_STATUS['success']['id']), size=10)
    )
    def test_produced_keys_are_passed_to_write_oasis_keys_file(self, data):
        write_oasis_keys_file_path = 'oasislmf.model_preparation.lookup.OasisLookupFactory.write_oasis_keys_file'
        with TemporaryDirectory() as d, patch(write_oasis_keys_file_path) as write_oasis_keys_file_mock:
            keys_file_path = os.path.join(d, 'piwind-keys.csv')

            olf.save_keys(
                keys_data=data,
                keys_file_path=keys_file_path,
            )
            write_oasis_keys_file_mock.assert_called_once_with(data, keys_file_path, False)
