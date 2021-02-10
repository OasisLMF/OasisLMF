import io
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
)
from mock import Mock, patch

from oasislmf.lookup.factory import OasisLookupFactory as olf
from oasislmf.lookup.keys_output import CSVKeysOutputStrategy, JSONKeysOutputStrategy
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

            complex_lookup_config_path = os.path.join(d, 'lookup_config.json')
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
                    model_keys_data_path=keys_path
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


@pytest.mark.skip()
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


class TestKeysOutput(TestCase):
    KEY_SUCCESS_1 = {
        'loc_id': 1,
        'peril_id': 1,
        'coverage_type': 1,
        'area_peril_id': 1,
        'vulnerability_id': 1,
        'status': 'success',
        'message': 'ok',
    }

    KEY_SUCCESS_2 = {
        'loc_id': 2,
        'peril_id': 3,
        'coverage_type': 4,
        'area_peril_id': 123456,
        'vulnerability_id': 987654,
        'status': 'success',
        'message': 'ok',
    }

    KEY_CUSTOM_MODEL_SUCCESS_1 = {
        'loc_id': 1,
        'peril_id': 1,
        'coverage_type': 1,
        'model_data': 'stringdata',
        'status': 'success',
        'message': 'ok',
    }

    KEY_FAIL_1 = {
        'loc_id': 3,
        'peril_id': 1,
        'coverage_type': 1,
        'status': 'fail',
        'message': 'bad',
    }

    KEY_FAIL_2 = {
        'loc_id': 4,
        'peril_id': 2,
        'coverage_type': 3,
        'status': 'fail',
        'message': 'verybad',
    }

    def test_single_successful_key__csv(self):
        keys = [self.KEY_SUCCESS_1]

        keys_stringio = io.StringIO()

        keys_output = CSVKeysOutputStrategy(keys_stringio)
        actual_n_success, actual_n_nonsuccess = keys_output.write(keys)
        keys_stringio.seek(0)

        expected_n_success = 1
        expected_n_nonsuccess = 0
        expected_keys_df = pd.DataFrame({
            'LocID': [1],
            'PerilID': [1],
            'CoverageTypeID': [1],
            'AreaPerilID': [1],
            'VulnerabilityID': [1],
        })

        actual_keys_df = pd.read_csv(keys_stringio)

        self.assertEqual(expected_n_success, actual_n_success)
        self.assertEqual(expected_n_nonsuccess, actual_n_nonsuccess)
        pd.testing.assert_frame_equal(expected_keys_df, actual_keys_df)

    def test_single_successful_key__csv__write_success_msg(self):
        keys = [self.KEY_SUCCESS_1]

        keys_stringio = io.StringIO()

        keys_output = CSVKeysOutputStrategy(keys_stringio, write_success_msg=True)
        actual_n_success, actual_n_nonsuccess = keys_output.write(keys)
        keys_stringio.seek(0)

        expected_n_success = 1
        expected_n_nonsuccess = 0
        expected_keys_df = pd.DataFrame({
            'LocID': [1],
            'PerilID': [1],
            'CoverageTypeID': [1],
            'AreaPerilID': [1],
            'VulnerabilityID': [1],
            'Message': ['ok'],
        })

        actual_keys_df = pd.read_csv(keys_stringio)

        self.assertEqual(expected_n_success, actual_n_success)
        self.assertEqual(expected_n_nonsuccess, actual_n_nonsuccess)
        pd.testing.assert_frame_equal(expected_keys_df, actual_keys_df)

    def test_single_successful_key__custom_model__csv(self):
        keys = [self.KEY_CUSTOM_MODEL_SUCCESS_1]

        keys_stringio = io.StringIO()

        keys_output = CSVKeysOutputStrategy(keys_stringio)
        actual_n_success, actual_n_nonsuccess = keys_output.write(keys)
        keys_stringio.seek(0)

        expected_n_success = 1
        expected_n_nonsuccess = 0
        expected_keys_df = pd.DataFrame({
            'LocID': [1],
            'PerilID': [1],
            'CoverageTypeID': [1],
            'ModelData': ['stringdata'],
        })

        actual_keys_df = pd.read_csv(keys_stringio)

        self.assertEqual(expected_n_success, actual_n_success)
        self.assertEqual(expected_n_nonsuccess, actual_n_nonsuccess)
        pd.testing.assert_frame_equal(expected_keys_df, actual_keys_df)

    def test_single_nonsuccessful_key__csv(self):
        keys = [self.KEY_FAIL_1]

        keys_stringio = io.StringIO()
        keys_errors_stringio = io.StringIO()

        keys_output = CSVKeysOutputStrategy(keys_stringio, keys_errors_file=keys_errors_stringio)
        actual_n_success, actual_n_nonsuccess = keys_output.write(keys)
        keys_stringio.seek(0)
        keys_errors_stringio.seek(0)

        expected_n_success = 0
        expected_n_nonsuccess = 1
        expected_keys_df = pd.DataFrame({
            'LocID': [],
            'PerilID': [],
            'CoverageTypeID': [],
            'AreaPerilID': [],
            'VulnerabilityID': [],
        }, index=pd.Index([]), dtype="object")
        expected_keys_errors_df = pd.DataFrame({
            'LocID': [3],
            'PerilID': [1],
            'CoverageTypeID': [1],
            'Status': 'fail',
            'Message': 'bad',
        })

        actual_keys_df = pd.read_csv(keys_stringio)
        actual_keys_errors_df = pd.read_csv(keys_errors_stringio)

        self.assertEqual(expected_n_success, actual_n_success)
        self.assertEqual(expected_n_nonsuccess, actual_n_nonsuccess)
        pd.testing.assert_frame_equal(expected_keys_df, actual_keys_df)
        pd.testing.assert_frame_equal(expected_keys_errors_df, actual_keys_errors_df)

    def test_single_nonsuccessful_key__no_keys_errors_path__csv(self):
        keys = [self.KEY_FAIL_1]

        keys_stringio = io.StringIO()

        keys_output = CSVKeysOutputStrategy(keys_stringio)
        actual_n_success, actual_n_nonsuccess = keys_output.write(keys)
        keys_stringio.seek(0)

        expected_n_success = 0
        expected_n_nonsuccess = 1
        expected_keys_df = pd.DataFrame({
            'LocID': [],
            'PerilID': [],
            'CoverageTypeID': [],
            'AreaPerilID': [],
            'VulnerabilityID': [],
        }, index=pd.Index([]), dtype="object")

        actual_keys_df = pd.read_csv(keys_stringio)

        self.assertEqual(expected_n_success, actual_n_success)
        self.assertEqual(expected_n_nonsuccess, actual_n_nonsuccess)
        pd.testing.assert_frame_equal(expected_keys_df, actual_keys_df)

    def test_mixed_keys__csv(self):
        keys = [self.KEY_SUCCESS_1, self.KEY_SUCCESS_2, self.KEY_FAIL_1, self.KEY_FAIL_2]

        keys_stringio = io.StringIO()
        keys_errors_stringio = io.StringIO()

        keys_output = CSVKeysOutputStrategy(keys_stringio, keys_errors_file=keys_errors_stringio)
        actual_n_success, actual_n_nonsuccess = keys_output.write(keys)
        keys_stringio.seek(0)
        keys_errors_stringio.seek(0)

        expected_n_success = 2
        expected_n_nonsuccess = 2
        expected_keys_df = pd.DataFrame({
            'LocID': [1, 2],
            'PerilID': [1, 3],
            'CoverageTypeID': [1, 4],
            'AreaPerilID': [1, 123456],
            'VulnerabilityID': [1, 987654],
        })
        expected_keys_errors_df = pd.DataFrame({
            'LocID': [3, 4],
            'PerilID': [1, 2],
            'CoverageTypeID': [1, 3],
            'Status': ['fail', 'fail'],
            'Message': ['bad', 'verybad'],
        })

        actual_keys_df = pd.read_csv(keys_stringio)
        actual_keys_errors_df = pd.read_csv(keys_errors_stringio)

        self.assertEqual(expected_n_success, actual_n_success)
        self.assertEqual(expected_n_nonsuccess, actual_n_nonsuccess)
        pd.testing.assert_frame_equal(expected_keys_df, actual_keys_df)
        pd.testing.assert_frame_equal(expected_keys_errors_df, actual_keys_errors_df)

    def test_single_successful_key__json(self):
        keys = [self.KEY_SUCCESS_1]

        keys_stringio = io.StringIO()

        keys_output = JSONKeysOutputStrategy(keys_stringio)
        actual_n_success, actual_n_nonsuccess = keys_output.write(keys)
        keys_stringio.seek(0)

        expected_n_success = 1
        expected_n_nonsuccess = 0
        expected_keys_df = pd.DataFrame(self.KEY_SUCCESS_1, index=[0])

        actual_keys_df = pd.read_json(keys_stringio)

        self.assertEqual(expected_n_success, actual_n_success)
        self.assertEqual(expected_n_nonsuccess, actual_n_nonsuccess)
        pd.testing.assert_frame_equal(expected_keys_df, actual_keys_df, check_like=True)

    def test_single_successful_key__custom_model__json(self):
        keys = [self.KEY_CUSTOM_MODEL_SUCCESS_1]

        keys_stringio = io.StringIO()

        keys_output = JSONKeysOutputStrategy(keys_stringio)
        actual_n_success, actual_n_nonsuccess = keys_output.write(keys)
        keys_stringio.seek(0)

        expected_n_success = 1
        expected_n_nonsuccess = 0
        expected_keys_df = pd.DataFrame(self.KEY_CUSTOM_MODEL_SUCCESS_1, index=[0])

        actual_keys_df = pd.read_json(keys_stringio)

        self.assertEqual(expected_n_success, actual_n_success)
        self.assertEqual(expected_n_nonsuccess, actual_n_nonsuccess)
        pd.testing.assert_frame_equal(expected_keys_df, actual_keys_df, check_like=True)

    def test_single_nonsuccessful_key__json(self):
        keys = [self.KEY_FAIL_1]

        keys_stringio = io.StringIO()
        keys_errors_stringio = io.StringIO()

        keys_output = JSONKeysOutputStrategy(keys_stringio, keys_errors_file=keys_errors_stringio)
        actual_n_success, actual_n_nonsuccess = keys_output.write(keys)
        keys_stringio.seek(0)
        keys_errors_stringio.seek(0)

        expected_n_success = 0
        expected_n_nonsuccess = 1
        expected_keys_errors_df = pd.DataFrame(self.KEY_FAIL_1, index=[0])
        actual_keys_df = pd.read_json(keys_stringio)
        actual_keys_errors_df = pd.read_json(keys_errors_stringio)

        self.assertEqual(expected_n_success, actual_n_success)
        self.assertEqual(expected_n_nonsuccess, actual_n_nonsuccess)
        self.assertTrue(actual_keys_df.empty)
        pd.testing.assert_frame_equal(expected_keys_errors_df, actual_keys_errors_df, check_like=True)

    def test_single_nonsuccessful_key__no_keys_errors_path__json(self):
        keys = [self.KEY_FAIL_1]

        keys_stringio = io.StringIO()

        keys_output = JSONKeysOutputStrategy(keys_stringio)
        actual_n_success, actual_n_nonsuccess = keys_output.write(keys)
        keys_stringio.seek(0)

        expected_n_success = 0
        expected_n_nonsuccess = 1

        actual_keys_df = pd.read_json(keys_stringio)

        self.assertEqual(expected_n_success, actual_n_success)
        self.assertEqual(expected_n_nonsuccess, actual_n_nonsuccess)
        self.assertTrue(actual_keys_df.empty)

    def test_mixed_keys__json(self):
        keys = [self.KEY_SUCCESS_1, self.KEY_SUCCESS_2, self.KEY_FAIL_1, self.KEY_FAIL_2]

        keys_stringio = io.StringIO()
        keys_errors_stringio = io.StringIO()

        keys_output = JSONKeysOutputStrategy(keys_stringio, keys_errors_file=keys_errors_stringio)
        actual_n_success, actual_n_nonsuccess = keys_output.write(keys)
        keys_stringio.seek(0)
        keys_errors_stringio.seek(0)

        expected_n_success = 2
        expected_n_nonsuccess = 2
        expected_keys_df = pd.concat([
            pd.DataFrame(self.KEY_SUCCESS_1, index=[0]),
            pd.DataFrame(self.KEY_SUCCESS_2, index=[1]),
        ])
        expected_keys_errors_df = pd.concat([
            pd.DataFrame(self.KEY_FAIL_1, index=[0]),
            pd.DataFrame(self.KEY_FAIL_2, index=[1]),
        ])
        actual_keys_df = pd.read_json(keys_stringio)
        actual_keys_errors_df = pd.read_json(keys_errors_stringio)

        self.assertEqual(expected_n_success, actual_n_success)
        self.assertEqual(expected_n_nonsuccess, actual_n_nonsuccess)
        pd.testing.assert_frame_equal(expected_keys_df, actual_keys_df, check_like=True)
        pd.testing.assert_frame_equal(expected_keys_errors_df, actual_keys_errors_df, check_like=True)
