import io
import json
import filecmp
import os
import pathlib
import string
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import just, sampled_from, text

from oasislmf.lookup.factory import BasicKeyServer, KeyServerFactory
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.status import OASIS_KEYS_STATUS

from oasislmf.computation.generate.keys import GenerateKeys

from tests.data import keys


META_DATA_PATH = pathlib.Path(os.path.realpath(__file__)).parent.joinpath('meta_data')
settings.register_profile("no_deadline", deadline=None)
settings.load_profile("no_deadline")


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
            f.write(f'{{"data": "{data}"}}')

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

            _, key_server = KeyServerFactory.create(
                model_keys_data_path=keys_path,
                model_version_file_path=version_path,
                lookup_module_path=module_path
            )
            instance = key_server.create_lookup(key_server.lookup_cls,
                                                key_server.config,
                                                key_server.config_dir,
                                                key_server.user_data_dir,
                                                key_server.output_dir,
                                                lookup_id=None)
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

            _, key_server = KeyServerFactory.create(
                model_version_file_path=version_path,
                lookup_module_path=module_path
            )
            instance = key_server.create_lookup(key_server.lookup_cls,
                                                key_server.config,
                                                key_server.config_dir,
                                                key_server.user_data_dir,
                                                key_server.output_dir,
                                                lookup_id=None)

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

            _, key_server = KeyServerFactory.create(
                model_keys_data_path=keys_path,
                model_version_file_path=version_path,
                lookup_module_path=module_path,
                complex_lookup_config_fp=complex_lookup_config_path,
                output_directory=output_directory
            )
            instance = key_server.create_lookup(key_server.lookup_cls,
                                                key_server.config,
                                                key_server.complex_lookup_config_fp,
                                                key_server.user_data_dir,
                                                key_server.output_dir,
                                                lookup_id=None)

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
                _, instance = KeyServerFactory.create(
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
                _, instance = KeyServerFactory.create(
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
                _, instance = KeyServerFactory.create(
                    model_version_file_path=version_path,
                    lookup_module_path=module_path,
                    model_keys_data_path=keys_path
                )


class OasisLookupFactoryWriteOasisKeysFiles(TestCase):

    @settings(deadline=None, suppress_health_check=HealthCheck.all())
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

            result = pd.DataFrame(successes + nonsuccesses)
            result.rename(columns={'coverage_type_id': 'coverage_type'}, inplace=True)
            key_server = BasicKeyServer({'builtin_lookup_type': 'peril_covered_deterministic'})
            _, successes_count, _, nonsuccesses_count = key_server.write_keys_file([result],
                                                                                   successes_fp=keys_file_path,
                                                                                   errors_fp=keys_errors_file_path,
                                                                                   output_format='oasis',
                                                                                   keys_success_msg=False)

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
    @settings(deadline=None, suppress_health_check=HealthCheck.all())
    @given(
        successes=keys(from_statuses=just(OASIS_KEYS_STATUS['success']['id']), size=5),
        nonsuccesses=keys(
            from_statuses=sampled_from([OASIS_KEYS_STATUS['fail']['id'], OASIS_KEYS_STATUS['nomatch']['id']]), size=5)
    )
    def test_records_are_given___records_are_written_to_json_keys_files_correctly(self, successes, nonsuccesses):
        with TemporaryDirectory() as d:
            keys_file_path = os.path.join(d, 'keys.json')
            keys_errors_file_path = os.path.join(d, 'keys-errors.json')
            result = pd.DataFrame(successes + nonsuccesses)

            key_server = BasicKeyServer({'builtin_lookup_type': 'peril_covered_deterministic'})
            _, successes_count, _, nonsuccesses_count = key_server.write_keys_file([result],
                                                                                   successes_fp=keys_file_path,
                                                                                   errors_fp=keys_errors_file_path,
                                                                                   output_format='json',
                                                                                   keys_success_msg=False)

            with io.open(keys_file_path, 'r', encoding='utf-8') as f1, io.open(keys_errors_file_path, 'r',
                                                                               encoding='utf-8') as f2:
                written_successes = json.load(f1)
                written_nonsuccesses = json.load(f2)

            self.assertEqual(successes_count, len(successes))
            self.assertEqual(written_successes, successes)

            self.assertEqual(nonsuccesses_count, len(nonsuccesses))
            self.assertEqual(written_nonsuccesses, nonsuccesses)


class GenerateKeysWithBuiltinLookup(TestCase):
    def test_built_in_lookup_step(self):
        with TemporaryDirectory() as d:
            generate_keys = GenerateKeys(
                oed_location_csv=pathlib.Path(META_DATA_PATH, 'location.csv'),
                lookup_config_json=pathlib.Path(META_DATA_PATH, 'lookup_config.json'),
                keys_data_csv=str(pathlib.Path(d, 'keys.csv'))
            )
            generate_keys.run()
            assert filecmp.cmp(pathlib.Path(d, 'keys.csv'), pathlib.Path(META_DATA_PATH, 'keys.csv'), shallow=False)
            assert filecmp.cmp(pathlib.Path(d, 'keys-errors.csv'), pathlib.Path(META_DATA_PATH, 'keys-errors.csv'), shallow=False)

    def test_builtin_geo_grid_peril(self):
        with TemporaryDirectory() as d:
            generate_keys = GenerateKeys(
                oed_location_csv=pathlib.Path(META_DATA_PATH, 'location-geo_grid.csv'),
                lookup_config_json=pathlib.Path(META_DATA_PATH, 'lookup_config-geo_grid.json'),
                keys_data_csv=str(pathlib.Path(d, 'keys-geo_grid.csv'))
            )
            generate_keys.run()
            assert filecmp.cmp(pathlib.Path(d, 'keys-geo_grid.csv'), pathlib.Path(META_DATA_PATH, 'keys-geo_grid.csv'), shallow=False)
