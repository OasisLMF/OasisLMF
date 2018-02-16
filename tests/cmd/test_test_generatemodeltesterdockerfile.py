from __future__ import unicode_literals

import string
from unittest import TestCase

import os
import io

import six
from backports.tempfile import TemporaryDirectory
from hypothesis import given
from hypothesis.strategies import text

from oasislmf.cmd import RootCmd
from oasislmf import __version__


class GenerateModelTesterDockerfileRun(TestCase):
    def get_command(self, api_server_url='http://localhost:8001', extras=None):
        kwargs_str = ' '.join('--{} {}'.format(k, v) for k, v in six.iteritems(extras or {}))

        return RootCmd(argv='test gen-model-tester-dockerfile {} {}'.format(kwargs_str, api_server_url).split())

    def test_api_server_is_not_supplied___parsing_args_raises_an_error(self):
        cmd = self.get_command(api_server_url='')

        with self.assertRaises(SystemExit):
            cmd.parse_args()

    def test_model_data_directory_is_not_supplied___default_is_abs_path_to_model_data_directory(self):
        cmd = self.get_command()

        args = cmd.parse_args()

        self.assertEqual(args.model_data_directory, os.path.abspath('./model_data'))

    def test_model_data_directory_is_supplied___model_data_directory_abs_path_to_supplied_path(self):
        cmd = self.get_command(extras={'model-data-directory': './other_model_data'})

        args = cmd.parse_args()

        self.assertEqual(args.model_data_directory, os.path.abspath('./other_model_data'))

    def test_model_version_is_not_supplied___default_value_is_none(self):
        cmd = self.get_command()

        args = cmd.parse_args()

        self.assertIsNone(args.model_version_file)

    def test_model_version_paths_supplied___version_path_is_abs_path_to_version_file(self):
        cmd = self.get_command(extras={'model-version-file': './specific_version_file.csv'})

        args = cmd.parse_args()

        self.assertEqual(args.model_version_file, os.path.abspath('./specific_version_file.csv'))

    @given(
        text(alphabet=string.ascii_letters, min_size=1),
        text(alphabet=string.ascii_letters, min_size=1),
        text(alphabet=string.ascii_letters, min_size=1),
        text(alphabet=string.ascii_letters, min_size=1),
    )
    def test_version_file_is_not_specified___version_file_from_data_directory_is_used(self, server_url, org, model, model_version):
        with TemporaryDirectory() as d:
            with io.open(os.path.join(d, 'ModelVersion.csv'), 'w', encoding='utf-8') as version_file:
                version_file.write('{},{},{}'.format(org, model, model_version))

            with io.open(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'oasislmf', '_data', 'Dockerfile.model_api_tester'), encoding='utf-8') as tpl_file:
                tpl_content = tpl_file.read()
                expected = tpl_content.replace(
                    '%CLI_VERSION%', __version__,
                ).replace(
                    '%OASIS_API_SERVER_URL%', server_url,
                ).replace(
                    '%SUPPLIER_ID%', org,
                ).replace(
                    '%MODEL_ID%', model,
                ).replace(
                    '%MODEL_VERSION%', model_version,
                ).replace(
                    '%LOCAL_MODEL_DATA_PATH%', d,
                )

            cmd = self.get_command(api_server_url=server_url, extras={'model-data-directory': d})
            res = cmd.run()

            with io.open(os.path.join(d, 'Dockerfile.{}_{}_model_api_tester'.format(org.lower(), model.lower())), encoding='utf-8') as docker_file:
                docker_content = docker_file.read()

            self.assertEqual(0, res)
            self.assertEqual(expected, docker_content)

    @given(
        text(alphabet=string.ascii_letters, min_size=1),
        text(alphabet=string.ascii_letters, min_size=1),
        text(alphabet=string.ascii_letters, min_size=1),
        text(alphabet=string.ascii_letters, min_size=1),
    )
    def test_version_file_is_specified___specified_version_is_used(self, server_url, org, model, model_version):
        with TemporaryDirectory() as d:
            with io.open(os.path.join(d, 'other_version_file.csv'), 'w', encoding='utf-8') as version_file:
                version_file.write('{},{},{}'.format(org, model, model_version))

            with io.open(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'oasislmf', '_data', 'Dockerfile.model_api_tester'), encoding='utf-8') as tpl_file:
                tpl_content = tpl_file.read()
                expected = tpl_content.replace(
                    '%CLI_VERSION%', __version__,
                ).replace(
                    '%OASIS_API_SERVER_URL%', server_url,
                ).replace(
                    '%SUPPLIER_ID%', org,
                ).replace(
                    '%MODEL_ID%', model,
                ).replace(
                    '%MODEL_VERSION%', model_version,
                ).replace(
                    '%LOCAL_MODEL_DATA_PATH%', d,
                )

            cmd = self.get_command(api_server_url=server_url, extras={
                'model-data-directory': d,
                'model-version-file': os.path.join(d, 'other_version_file.csv')
            })
            res = cmd.run()

            with io.open(os.path.join(d, 'Dockerfile.{}_{}_model_api_tester'.format(org.lower(), model.lower())), encoding='utf-8') as docker_file:
                docker_content = docker_file.read()

            self.assertEqual(0, res)
            self.assertEqual(expected, docker_content)
