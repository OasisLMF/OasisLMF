import json
import os
import io

from collections import Counter
from multiprocessing.pool import ThreadPool

import six
from argparsetree import BaseCommand
from backports.tempfile import TemporaryDirectory

from oasislmf.keys.lookup import OasisKeysLookupFactory
from .. import __version__
from ..utils.exceptions import OasisException
from ..utils.conf import replace_in_file
from ..api_client.client import OasisAPIClient
from .base import OasisBaseCommand
from .cleaners import PathCleaner


class TestModelApiCmd(OasisBaseCommand):
    """
    Tests a model api server
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(TestModelApiCmd, self).add_args(parser)

        parser.add_argument(
            'api_server_url', type=str,
            help='Oasis API server URL (including protocol and port), e.g. http://localhost:8001',
        )

        parser.add_argument(
            '-a', '--analysis-settings-file', type=PathCleaner('Analysis settings'), default='./analysis_settings.json',
            help="Analysis settings JSON file (absolute or relative file path)"
        )

        parser.add_argument(
            '-i', '--input-directory', type=PathCleaner('Input directory'), default='./input',
            help="Input data directory (absolute or relative file path)"
        )

        parser.add_argument(
            '-o', '--output-directory', type=PathCleaner('Output directory', preexists=False), default='./output',
            help="Output data directory (absolute or relative file path)"
        )

        parser.add_argument(
            '-n', '--num-analyses', metavar='N', type=int, default='1',
            help='The number of analyses to run.'
        )

        parser.add_argument(
            '-c', '--health-check-attempts', type=int, default=1,
            help='The maximum number of health check attempts.'
        )

    def load_analysis_settings_json(self, analysis_settings_file):
        """
        Loads the analysis settings JSON file into a dict, also creates a separate
        boolean ``do_il`` for doing insured loss calculations.

        :param analysis_settings_file: Path to the analysis settings file to load
        :type analysis_settings_file: str

        :return: a 2 tuple containing the analysis settings dictionary and a bool
            signifying whether il processing is enabled
        """

        with io.open(analysis_settings_file, encoding='utf-8') as f:
            analysis_settings = json.load(f)

        if isinstance(analysis_settings['analysis_settings']['il_output'], six.string_types):
            do_il = analysis_settings['analysis_settings']["il_output"].lower() == 'true'
        else:
            do_il = bool(analysis_settings['analysis_settings']["il_output"])

        return analysis_settings, do_il

    def run_analysis(self, client, input_directory, output_directory, analysis_settings, do_il, counter):
        """
        Invokes model analysis in the client - is used as a worker function for
        threads.

        :param client: The api client to use for the tests
        :type client: OasisAPIClient

        :param input_directory: The directory to gather the input files from
        :type input_directory: str

        :param output_directory: The directory to store output files in
        :type output_directory: str

        :param analysis_settings: The analysis settings dictionary
        :type analysis_settings: dict

        :param do_il: Flag whether to perform il processing
        :type do_il: bool

        :param counter: A counter object that will record the number of success and fails
        :type counter: Counter
        """
        try:
            with TemporaryDirectory() as upload_directory:
                input_location = client.upload_inputs_from_directory(input_directory, bin_directory=upload_directory, do_il=do_il, do_build=True)
                client.run_analysis_and_poll(analysis_settings, input_location, output_directory)
                counter['completed'] += 1

        except Exception as e:
            client._logger.exception("Model API test failed: {}".format(str(e)))
            counter['failed'] += 1

    def action(self, args):
        """
        Runs the api checks for the model

        :param args: The arguments from the command line
        :type args: Namespace
        """

        # get client
        client = OasisAPIClient(args.api_server_url, self.logger)

        # Do a server healthcheck
        if not client.health_check(args.health_check_attempts):
            raise OasisException('Health check failed for {}.'.format(args.api_server_url))

        # Make output directory if it does not exist
        if not os.path.exists(args.output_directory):
            os.mkdir(args.output_directory)

        # Load analysis settings JSON file and set up boolean for doing insured
        # loss calculations
        self.logger.info('Loading analysis settings JSON file:')
        analysis_settings, do_il = self.load_analysis_settings_json(args.analysis_settings_file)
        self.logger.info('  OK: analysis_settings={}, do_il={}'.format(analysis_settings, do_il))

        # Prepare and run analyses
        self.logger.info('Running {} analyses'.format(args.num_analyses))
        counter = Counter()

        threads = ThreadPool(processes=args.num_analyses)
        threads.map(
            self.run_analysis,
            ((client, args.input_directory, args.output_directory, analysis_settings, do_il, counter) for i in range(args.num_analyses))
        )
        threads.close()
        threads.join()

        # Summary of run results
        self.logger.info("Finished: {} completed, {} failed".format(counter['completed'], counter['failed']))
        return 0


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
        version_info = OasisKeysLookupFactory.get_model_info(version_file)

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
                version_info['model_version_id'],
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
        'model-api': TestModelApiCmd,
        'gen-model-tester-dockerfile': GenerateModelTesterDockerFileCmd,
    }
