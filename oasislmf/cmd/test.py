import csv
import json
import os
import uuid

import shutil
from collections import Counter
from multiprocessing.pool import ThreadPool

from argparsetree import BaseCommand

from ..utils.conf import replace_in_file
from ..api_client.client import OasisAPIClient
from .base import OasisBaseCommand


class TestModelApi(OasisBaseCommand):
    description = 'Tests a model api server'

    def add_args(self, parser):
        parser.add_argument(
            '-s', '--api_server_url', type=str, required=True,
            help='Oasis API server URL (including protocol and port), e.g. http://localhost:8001'
        )

        parser.add_argument(
            '-a', '--analysis_settings_json_file', type=str, required=True,
            help="Analysis settings JSON file (absolute or relative file path)"
        )

        parser.add_argument(
            '-i', '--input_data_directory', type=str, required=True,
            help="Input data directory (absolute or relative file path)"
        )

        parser.add_argument(
            '-o', '--output_data_directory', type=str, required=True,
            help="Output data directory (absolute or relative file path)"
        )

        parser.add_argument(
            '-n', '--num_analyses', metavar='N', type=int, default='1',
            help='The number of analyses to run.'
        )

        parser.add_argument(
            '-c', '--health_check_attempts', type=int, default=1,
            help='The maximum number of health check attempts.'
        )

    def check_input(self, input_data_directory, analysis_settings_json_file):
        """
        Checks that the paths for the input data directory and analysis settings
        JSON file all exist. Raises exceptions which cause the script to exit if
        one of these is missing.
        """

        if not os.path.exists(input_data_directory):
            raise Exception("Input data directory {} does not exist".format(input_data_directory), 'input_data_directory')

        if not os.path.exists(analysis_settings_json_file):
            raise Exception("Analysis settings JSON file path {} does not exist".format(analysis_settings_json_file), 'analysis_settings_json_file')

        return True

    def load_analysis_settings_json(self, analysis_settings_json_file):
        """
        Loads the analysis settings JSON file into a dict, also creates a separate
        boolean ``do_il`` for doing insured loss calculations.
        """

        with open(analysis_settings_json_file) as f:
            analysis_settings = json.load(f)

        do_il = bool(analysis_settings['analysis_settings']["il_output"])

        return analysis_settings, do_il

    def run_analysis(self, client, input_data_directory, output_data_directory, analysis_settings, do_il, counter):
        """
        Invokes model analysis in the client - is used as a worker function for
        threads.
        """
        try:

            upload_directory = os.path.join(os.getcwd(), 'upload', str(uuid.uuid1()))

            shutil.copytree(
                os.path.join(input_data_directory, 'csv'),
                upload_directory
            )

            input_location = client.upload_inputs_from_directory(
                upload_directory, do_il, do_validation=False
            )
            client.run_analysis_and_poll(
                analysis_settings, input_location, output_data_directory
            )
            counter['completed'] += 1

        except Exception as e:
            client._logger.exception("Model API test failed: {}".format(str(e)))
            counter['failed'] += 1

    def action(self, args):
        # get client
        client = OasisAPIClient(args['api_server_url'], self.logger)

        # Do a server healthcheck
        if not client.health_check(args['health_check_attempts']):
            return 1

        # Validate the input data directory path and analysis settings JSON
        # file path.
        input_data_directory = args['input_data_directory']
        analysis_settings_json_file = args['analysis_settings_json_file']
        output_data_directory = args['output_data_directory']

        try:
            self.logger.info('Checking input data exist.')
            self.check_input(input_data_directory, analysis_settings_json_file)
            self.logger.info('OK')
        except Exception as e:
            self.logger.info(str(e))
            self.logger.info('Check input data directory and analysis settings JSON file exist, and try again.')
            return 1

        # Make output directory if it does not exist
        if not os.path.exists(output_data_directory):
            self.logger.info('Output data directory {} does not exist, creating one in working directory'.format(
                output_data_directory))
            output_data_directory = 'output'
            if not os.path.exists(output_data_directory):
                os.mkdir(output_data_directory)

        # Load analysis settings JSON file and set up boolean for doing insured
        # loss calculations
        self.logger.info('Loading analysis settings JSON file: ', end='')
        analysis_settings, do_il = self.load_analysis_settings_json(args['analysis_settings_json_file'])
        self.logger.info('OK: analysis_settings={}, do_il={}'.format(analysis_settings, do_il))

        # Prepare and run analyses
        self.logger.info('Running {} analyses'.format(args['num_analyses']))
        counter = Counter()

        threads = ThreadPool(processes=args['num_analyses'])
        threads.map(
            self.run_analysis,
            ((client, input_data_directory, output_data_directory, analysis_settings, do_il, counter) for i in range(args['num_analyses']))
        )
        threads.close()
        threads.join()

        # Summary of run results
        self.logger.info("Finished: {} completed, {} failed".format(counter['completed'], counter['failed']))
        return 0


class GenerateModelTesterDockerFileCmd(OasisBaseCommand):
    description = 'Generates a new a model testing dockerfile from the supplied template'

    var_names = [
        'OASIS_API_SERVER_URL',
        'MODEL_KEYS_SERVER_URL',
        'SUPPLIER_ID',
        'MODEL_ID',
        'MODEL_VERSION',
        'LOCAL_MODEL_DATA_PATH'
    ]

    def add_args(self, parser):
        parser.add_argument(
            '-s', '--api_server_url', type=str, required=True,
            help='Oasis API server URL (including protocol and port), e.g. http://localhost:8001'
        )

        parser.add_argument(
            '-k', '--key_server_url', type=str, required=True,
            help='Key server URL (including protocol and port), e.g. http://localhost:8001'
        )

        parser.add_argument(
            '--model_vesion_file', type=str, required=True,
            help='Model version file path (relative or absolute path of model version file)'
        )

        parser.add_argument(
            '-d', '--model_data_path', type=str, required=True,
            help='Model data path (relative or absolute path of model version file)'
        )

        parser.add_argument(
            '-o', '--output_path', type=str, required=True, default='.',
            help='Path to the output directory (relative or absolute path of model version file)'
        )

    def action(self, args):
        model_version_file_path = os.path.abspath(args['model_version_file_path'])
        with open(model_version_file_path) as f:
            supplier_id, model_id, model_version = map(lambda s: s.strip(), next(csv.reader(f)))

        dockerfile = 'Dockerfile.{}_{}_model_api_tester'.format(supplier_id.lower(), model_id.lower())
        if args['target_dockerfile_dir'] != '.':
            dockerfile = os.path.join(os.path.abspath(args['target_dockerfile_dir']), dockerfile)

        if os.path.exists(dockerfile):
            os.remove(dockerfile)

        oasisapi_server_url = args['oasisapi_server_url'].strip('/')
        model_keys_server_url = args['model_keys_server_url'].strip('/')

        local_model_data_path = os.path.abspath(args['local_model_data_path'])
        dest_data_folder = os.path.abspath(os.path.join('tests', 'data'))

        if local_model_data_path != dest_data_folder:
            if os.path.exists(dest_data_folder):
                shutil.rmtree(dest_data_folder)
            shutil.copytree(local_model_data_path, dest_data_folder)

        try:
            replace_in_file(
                'Dockerfile.model_api_tester',
                dockerfile,
                map(lambda s: '%{}%'.format(s), self.var_names),
                [
                    oasisapi_server_url,
                    model_keys_server_url,
                    supplier_id,
                    model_id,
                    model_version,
                    local_model_data_path
                ]
            )
        except Exception as e:
            print(str(e))
            return 1
        else:
            print('\nFile "{}" created in working directory.\n'.format(dockerfile))
            return 0


class Test(BaseCommand):
    description = 'Test models and keys servers'
    sub_commands = {
        'model-api': TestModelApi,
        'gen-model-tester-dockerfile': GenerateModelTesterDockerFileCmd,
    }
