import string
from unittest import TestCase

import os
from backports.tempfile import TemporaryDirectory
from hypothesis import given
from hypothesis.strategies import text

from oasislmf.keys.lookup import OasisKeysLookupFactory


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
