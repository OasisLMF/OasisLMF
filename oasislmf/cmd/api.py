
import getpass
import io
import json
import sys

import six

from argparse import RawDescriptionHelpFormatter
from pathlib2 import Path

from ..utils.exceptions import OasisException

from .cleaners import as_path, PathCleaner
from .base import OasisBaseCommand, InputValues

from ..api_client.client_manager import APIClient


def load_credentials(login_arg, logger=None):
    """
    Load credentials from JSON file

    Options:
        1. '--api-server-login ./APIcredentials.json'

        2. Load credentials from default config file
        '-C oasislmf.json'

        3. Prompt for username / password
    """
    if isinstance(login_arg, six.string_types):
        with io.open(login_arg, encoding='utf-8') as f:
            return json.load(f)

    elif isinstance(login_arg, dict):
        if {'password','username'} <= {k for k in login_arg.keys()}:
            return login_arg

    else:
        logger.info('No Login provided - Fallback to prompt')

    try:
        api_login = {}
        api_login['username'] = six.moves.input('Username: ')
        api_login['password'] = getpass.getpass('Password: ')
        return api_login
    except KeyboardInterrupt as e:
        logger.error('\nFailed to get API login details:')
        sys.exit(1)


def open_api_connection(input_args, logger):
    if not input_args.get('api_server_url'):
        logger.error('Error: argument `--api-server-url` not set')
        sys.exit(1)
    try:
        credentials = load_credentials(input_args.get('api_server_login'), logger=logger)
        return APIClient(api_url=input_args.get('api_server_url'),
                api_ver='V1',
                username=credentials['username'],
                password=credentials['password'],
                logger=logger)
    except OasisException as e:
        logger.error('API Connection error:')
        logger.error(e)
        sys.exit(1)




class GetApiCmd(OasisBaseCommand):
    """
    Issue API GET requests via the command line
    """
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        super(self.__class__, self).add_args(parser)
        parser.add_argument(
            '-u','--api-server-url', type=str,
            default=None,
            help='Oasis API server URL (including protocol and port), e.g. http://localhost:8001',
        )
        parser.add_argument(
            '-l', '--api-server-login', type=PathCleaner('credentials file', preexists=False), default=None,
            help='Json file with {"username":"<USER>", "password":"<PASS>"}', required=False,
        )

        parser.add_argument(
            '-m', '--models', type=bool, const=True, default=False, nargs='?', required=False,
            help='Fetch the list of stored models',
        )
        parser.add_argument(
            '-p', '--portfolios',  type=bool, const=True, default=False, nargs='?', required=False,
            help='Fetch the list of stored portfolios',
        )
        parser.add_argument(
            '-a', '--analyses',  type=bool, const=True, default=False, nargs='?', required=False,
            help='Fetch the list of stored analyses',
        )


    def action(self, args):
        inputs = InputValues(args)
        api = open_api_connection(inputs, self.logger)

        if args.models:
            resp = api.models.get()
            self.logger.info(json.dumps(resp.json(), indent=4, sort_keys=True))
        if args.portfolios:
            resp = api.portfolios.get()
            self.logger.info(json.dumps(resp.json(), indent=4, sort_keys=True))
        if args.analyses:
            resp = api.analyses.get()
            self.logger.info(json.dumps(resp.json(), indent=4, sort_keys=True))



class DelApiCmd(OasisBaseCommand):
    formatter_class = RawDescriptionHelpFormatter


    def add_args(self, parser):
        super(self.__class__, self).add_args(parser)
        parser.add_argument(
            '-u','--api-server-url', type=str,
            default=None,
            help='Oasis API server URL (including protocol and port), e.g. http://localhost:8001',
        )
        parser.add_argument(
            '-l', '--api-server-login', type=PathCleaner('credentials file', preexists=False), default=None,
            help='Json file with {"username":"<USER>", "password":"<PASS>"}', required=False,
        )
        parser.add_argument(
            '-y', '--api-no-confirm', type=bool, default=False, const=True, nargs='?', required=False,
            help='Skip confirmation prompt before altering the API resources',
        )

        parser.add_argument(
            '-m', '--model-id', type=int, default=None, required=False,
            help='Model ID to delete',
        )
        parser.add_argument(
            '-p', '--portfolio-id', type=int, default=None, required=False,
            help='Portfolio ID to delete',
        )
        parser.add_argument(
            '-a', '--analysis-id', type=int, default=None, required=False,
            help='Analysis ID to delete',
        )

    def action(self, args):
        inputs = InputValues(args)
        api = open_api_connection(inputs, self.logger)

        try:
            if args.model_id:
                id_ref = inputs.get('model_id')
                r = api.models.get(id_ref)
                r.raise_for_status()
                self.logger.info(json.dumps(r.json(), indent=4, sort_keys=True))
                if self.confirm_action(args.api_no_confirm):
                    r = api.models.delete(id_ref)
                    r.raise_for_status()
                    self.logger.info('Record deleted')

            if args.portfolio_id:
                id_ref = inputs.get('portfolio_id')
                r = api.portfolios.get(id_ref)
                r.raise_for_status()
                self.logger.info(json.dumps(r.json(), indent=4, sort_keys=True))
                if self.confirm_action(args.api_no_confirm):
                    r = api.portfolios.delete(id_ref)
                    r.raise_for_status()
                    self.logger.info('Record deleted')

            if args.analysis_id:
                id_ref = inputs.get('analysis_id')
                r = api.analyses.get(id_ref)
                r.raise_for_status()
                self.logger.info(json.dumps(r.json(), indent=4, sort_keys=True))
                if self.confirm_action(args.api_no_confirm):
                    r = api.analyses.delete(id_ref)
                    r.raise_for_status()
                    self.logger.info('Record deleted')

        except Exception as e:    
            self.logger.error("Error on delete ref({}):".format(id_ref))
            self.logger.error(r.text)

    def confirm_action(self, override=False, question_str='Delete this record from the API?'):
        self.logger.debug('Prompt user for confirmation')
        if override:
            self.logger.debug('Defaulting to YES')
            return True

        try:
            check = str(six.moves.input("%s (Y/N): " % question_str)).lower().strip()
            if check[0] == 'y':
                return True
            elif check[0] == 'n':
                return False
            else:
                print('Invalid Input')
                return self.confirm_action(question_str)
        except KeyboardInterrupt as e:
            self.logger.error('\nKeyboard Interrupt, exiting.')



class PutApiModelCmd(OasisBaseCommand):
    formatter_class = RawDescriptionHelpFormatter
    def add_args(self, parser):
        super(self.__class__, self).add_args(parser)
        parser.add_argument(
            '-u','--api-server-url', type=str,
            default=None,
            help='Oasis API server URL (including protocol and port), e.g. http://localhost:8001',
        )
        parser.add_argument(
            '-l', '--api-server-login',
            type=PathCleaner('credentials file', preexists=False), default=None,
            help='Json file with {"username":"<USER>", "password":"<PASS>"}', required=False,
        )


        # Required
        parser.add_argument('--supplier-id', type=str, default=None,
            required=True,
            help='The supplier ID for the model.'
        )
        parser.add_argument('--model-id', type=str, default=None,
            required=True,
            help='The model ID for the model.'
        )
        parser.add_argument('--version-id', type=str, default=None,
            required=True,
            help='The version ID for the model.'
        )

    def action(self, args):
        inputs = InputValues(args)
        api = open_api_connection(inputs, self.logger)

        api.models.create(
            supplier_id=inputs.get('supplier_id'),
            model_id=inputs.get('model_id'),
            version_id=inputs.get('version_id'),
        )



class RunApiCmd(OasisBaseCommand):
    formatter_class = RawDescriptionHelpFormatter
    def add_args(self, parser):
        super(self.__class__, self).add_args(parser)
        # API Connection
        parser.add_argument(
            '-u','--api-server-url', type=str,
            default=None,
            help='Oasis API server URL (including protocol and port), e.g. http://localhost:8001',
        )
        parser.add_argument(
            '-l', '--api-server-login',
            type=PathCleaner('credentials file', preexists=False), default=None,
            help='Json file with {"username":"<USER>", "password":"<PASS>"}', required=False,
        )


        # Required
        parser.add_argument('-m', '--model-id', type=int, default=None,
            required=True,
            help='API `id` of a model to run the analysis with'
        )
        parser.add_argument('-j', '--analysis-settings-json-file-path',
            type=PathCleaner('analysis settings file'), default=None,
            help='Analysis settings JSON file path'
        )
        parser.add_argument('-x', '--source-exposures-file-path',
            type=PathCleaner('Source exposures file'), default=None,
            help='OED Source exposures file path'
        )


        # Optional
        parser.add_argument('-y', '--source-accounts-file-path',
            type=PathCleaner('Source accounts file', preexists=False),
            default=None, required=False,
            help='OED Source accounts file path'
        )
        parser.add_argument('-i', '--ri-info-file-path',
            type=PathCleaner('Reinsurances Info file', preexists=False),
            default=None, required=False,
            help='OED Reinsurances Info file path'
        )
        parser.add_argument('-r', '--ri-scope-file-path',
            type=PathCleaner('Reinsurance Scope file', preexists=False),
            default=None, required=False,
            help='OED Reinsurance Scope file path'
        )
        parser.add_argument(
            '-o', '--output-directory',
            type=PathCleaner('Output directory', preexists=False), default='./',
            help="Output data directory (absolute or relative file path)"
        )


    def action(self, args):
        inputs = InputValues(args)
        api = open_api_connection(inputs, self.logger)

        # Upload files
        path_location = inputs.get('source_exposures_file_path')
        path_account = inputs.get('source_accounts_file_path')
        path_info = inputs.get('ri_info_file_path')
        path_scope = inputs.get('ri_scope_file_path')

        portfolio = api.upload_inputs(
            portfolio_id=None,
            location_fp=path_location,
            accounts_fp=path_account,
            ri_info_fp=path_info,
            ri_scope_fp=path_scope,
        )

        # Create new analysis
        path_settings = inputs.get('analysis_settings_json_file_path')
        if not path_settings:
            self.logger.error('analysis settings: Not found')
            return False
        analysis = api.create_analysis(
            portfolio_id=portfolio['id'],
            model_id=inputs.get('model_id'),
            analysis_settings_fp=path_settings,
        )
        self.logger.info('Loaded analysis settings:')
        self.logger.info(json.dumps(analysis, indent=4))

        # run and poll
        api.run_generate(analysis['id'], poll_interval=3)
        api.run_analysis(analysis['id'], poll_interval=3)

        # Download Outputs
        api.download_output(
            analysis_id=analysis['id'],
            download_path=inputs.get('output_directory'),
            overwrite=True,
            clean_up=True
        )

        # Clean up
        api.portfolios.delete(portfolio['id'])



class ApiCmd(OasisBaseCommand):
    sub_commands = {
        'list': GetApiCmd,
        'add-model': PutApiModelCmd,
        'delete': DelApiCmd,
        'run': RunApiCmd
    }
