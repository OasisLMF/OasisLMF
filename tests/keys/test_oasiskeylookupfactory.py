import csv
import string
from tempfile import NamedTemporaryFile
from unittest import TestCase

import os
import io
from backports.tempfile import TemporaryDirectory
from hypothesis import given
from hypothesis.strategies import text, integers, tuples, lists

from oasislmf.keys.lookup import OasisKeysLookupFactory
from oasislmf.utils.exceptions import OasisException


class OasisKeysLookupFactoryCreate(TestCase):
    def write_version_file(self, supplier, model, version, path):
        with open(path, 'w') as f:
            f.write('{},{},{}'.format(supplier, model, version))

    def write_py_module(self, model, path):
        with open(path, 'w') as f:
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
        stream = io.StringIO()
        data = [('first', 'second')] + data

        csv.writer(stream).writerows(data)

        res = OasisKeysLookupFactory.get_model_exposures(model_exposures=stream.getvalue())
        res = [tuple(res)] + [tuple(res.iloc[i]) for i in range(len(res))]

        self.assertEqual(res, data)
