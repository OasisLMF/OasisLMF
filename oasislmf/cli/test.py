import os

from argparsetree import BaseCommand

from .. import __version__
from ..model_preparation.lookup import OasisLookupFactory as olf
from ..model_testing.validation import csv_validity_test
from ..utils.conf import replace_in_file

from ..utils.path import (
    as_path,
    PathCleaner,
)
from .base import (
    InputValues,
    OasisBaseCommand,
)


class GenerateModelTesterDockerFileCmd(OasisBaseCommand):
    """
    Generates a new a model testing dockerfile from the supplied template
    """

    #: The names of the variables to replace in the docker file
    var_names = [
        'CLI_VERSION',
        'OASIS_API_SERVER_URL',
        'SUPPLIER_ID',
        'MODEL_ID',
        'MODEL_VERSION',
        'LOCAL_MODEL_DATA_PATH'
    ]

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(GenerateModelTesterDockerFileCmd, self).add_args(parser)

        parser.add_argument(
            'api_server_url', type=str,
            help='Oasis API server URL (including protocol and port), e.g. http://localhost:8001'
        )

        parser.add_argument(
            '--model-data-directory', type=PathCleaner('Model data path', preexists=False), default='./model_data',
            help='Model data path (relative or absolute path of model version file)'
        )

        parser.add_argument(
            '--model-version-file', type=PathCleaner('Model version file', preexists=False), default=None,
            help='Model version file path (relative or absolute path of model version file), by default <model-data-path>/ModelVersion.csv is used'
        )

    def action(self, args):
        """
        Generates a new a model testing dockerfile from the supplied template

        :param args: The arguments from the command line
        :type args: Namespace
        """
        dockerfile_src = os.path.join(os.path.dirname(__file__), os.path.pardir, '_data', 'Dockerfile.model_api_tester')

        version_file = args.model_version_file or os.path.join(args.model_data_directory, 'ModelVersion.csv')
        version_info = olf.get_model_info(version_file)

        dockerfile_dst = os.path.join(
            args.model_data_directory,
            'Dockerfile.{}_{}_model_api_tester'.format(version_info['supplier_id'].lower(), version_info['model_id'].lower()),
        )

        replace_in_file(
            dockerfile_src,
            dockerfile_dst,
            ['%{}%'.format(s) for s in self.var_names],
            [
                __version__,
                args.api_server_url,
                version_info['supplier_id'],
                version_info['model_id'],
                version_info['model_version'],
                args.model_data_directory,
            ]
        )

        self.logger.info('File created at {}.'.format(dockerfile_dst))
        return 0


class ModelValidationCmd(OasisBaseCommand):
    """
    Checks model data for validity.
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """

        super(self.__class__, self).add_args(parser)

        parser.add_argument('-d', '--model-data-path', default=None, help='Directory containing additional user-supplied model data files')

    def action(self, args):
        """
        Performs validity checks on model data csv files using ktools
        executables.

        :param args: The arguments from the command line
        :type args: Namespace
        """

        inputs = InputValues(args)

        model_data_fp = as_path(inputs.get('model_data_path', required=True, is_path=True), 'Model data path', is_dir=True)

        csv_validity_test(model_data_fp)


class TestCmd(BaseCommand):
    """
    Test models and keys servers
    """

    sub_commands = {
        'gen-model-tester-dockerfile': GenerateModelTesterDockerFileCmd,
        'model-validation': ModelValidationCmd,
    }
