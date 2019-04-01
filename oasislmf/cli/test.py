# -*- coding: utf-8 -*-
import io
import json
import os
import time

from collections import Counter
from future.utils import string_types
from multiprocessing.pool import ThreadPool

from argparsetree import BaseCommand
from backports.tempfile import TemporaryDirectory

from .. import __version__
from ..model_preparation.lookup import OasisLookupFactory as olf
from ..utils.conf import replace_in_file
from ..utils.exceptions import OasisException
from ..utils.path import PathCleaner
from .base import OasisBaseCommand



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


class TestCmd(BaseCommand):
    """
    Test models and keys servers
    """

    sub_commands = {
        'gen-model-tester-dockerfile': GenerateModelTesterDockerFileCmd,
    }
