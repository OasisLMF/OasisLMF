import getpass
import io
import json
import sys

from argparse import RawDescriptionHelpFormatter

from ..api.client import APIClient
from ..utils.exceptions import OasisException
from ..utils.path import PathCleaner
from ..utils.defaults import API_EXAMPLE_AUTH

from .base import OasisBaseCommand
from .inputs import InputValues


def load_credentials(login_arg, logger=None):
    """
    Load credentials from JSON file

    Options:
        1. '--api-server-login ./APIcredentials.json'

        2. Load credentials from default config file
        '-C oasislmf.json'

        3. Prompt for username / password
    """
    if isinstance(login_arg, str):
        with io.open(login_arg, encoding='utf-8') as f:
            return json.load(f)
    elif isinstance(login_arg, dict):
        if {'password', 'username'} <= {k for k in login_arg.keys()}:
            return login_arg
    else:
        logger.info('API Login:')

    try:
        api_login = {}
        api_login['username'] = input('Username: ')
        api_login['password'] = getpass.getpass('Password: ')
        return api_login
    except KeyboardInterrupt as e:
        logger.error('\nFailed to get API login details:')
        logger.error(e)
        sys.exit(1)


def open_api_connection(input_args, logger):
    try:
        ## If no password given try the reference example
        return APIClient(
            api_url=input_args.get('api_server_url'),
            api_ver='V1',
            username=API_EXAMPLE_AUTH['user'],
            password=API_EXAMPLE_AUTH['pass'],
            logger=logger
        )
    except OasisException:
        ## Prompt for password and try to re-autehnticate
        try:
            credentials = load_credentials(input_args.get('api_server_login'), logger=logger)
            logger.info('Connecting to - {}'.format(input_args.get('api_server_url')))
            return APIClient(
                api_url=input_args.get('api_server_url'),
                api_ver='V1',
                username=credentials['username'],
                password=credentials['password'],
                logger=logger
            )
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
            '-u', '--api-server-url', type=str,
            default="http://localhost:8000",
            help='Oasis API server URL (including protocol and port), e.g. http://localhost:8000',
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
            '-p', '--portfolios', type=bool, const=True, default=False, nargs='?', required=False,
            help='Fetch the list of stored portfolios',
        )
        parser.add_argument(
            '-a', '--analyses', type=bool, const=True, default=False, nargs='?', required=False,
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
            '-u', '--api-server-url', type=str,
            default="http://localhost:8000",
            help='Oasis API server URL (including protocol and port), e.g. http://localhost:8000',
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
        warn_msg = 'Delete this record from the API?'

        try:
            if args.model_id:
                id_ref = inputs.get('model_id')
                r = api.models.get(id_ref)
                r.raise_for_status()
                self.logger.info(json.dumps(r.json(), indent=4, sort_keys=True))
                if inputs.confirm_action(warn_msg, args.api_no_confirm):
                    r = api.models.delete(id_ref)
                    r.raise_for_status()
                    self.logger.info('Record deleted')

            if args.portfolio_id:
                id_ref = inputs.get('portfolio_id')
                r = api.portfolios.get(id_ref)
                r.raise_for_status()
                self.logger.info(json.dumps(r.json(), indent=4, sort_keys=True))
                if inputs.confirm_action(warn_msg, args.api_no_confirm):
                    r = api.portfolios.delete(id_ref)
                    r.raise_for_status()
                    self.logger.info('Record deleted')

            if args.analysis_id:
                id_ref = inputs.get('analysis_id')
                r = api.analyses.get(id_ref)
                r.raise_for_status()
                self.logger.info(json.dumps(r.json(), indent=4, sort_keys=True))
                if inputs.confirm_action(warn_msg, args.api_no_confirm):
                    r = api.analyses.delete(id_ref)
                    r.raise_for_status()
                    self.logger.info('Record deleted')

        except Exception as e:
            self.logger.error(e)
            self.logger.error("Error on delete ref({}):".format(id_ref))
            self.logger.error(r.text)


class PutApiModelCmd(OasisBaseCommand):
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        super(self.__class__, self).add_args(parser)
        parser.add_argument(
            '-u', '--api-server-url', type=str,
            default="http://localhost:8000",
            help='Oasis API server URL (including protocol and port), e.g. http://localhost:8000',
        )
        parser.add_argument(
            '-l', '--api-server-login',
            type=PathCleaner('credentials file', preexists=False), default=None,
            help='Json file with {"username":"<USER>", "password":"<PASS>"}', required=False,
        )
        # Required
        parser.add_argument(
            '--supplier-id', type=str, default=None,
            required=True, help='The supplier ID for the model.'
        )
        parser.add_argument(
            '--model-id', type=str, default=None,
            required=True, help='The model ID for the model.'
        )
        parser.add_argument(
            '--version-id', type=str, default=None,
            required=True, help='The version ID for the model.'
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
            '-u', '--api-server-url', type=str,
            default="http://localhost:8000",
            help='Oasis API server URL (including protocol and port), e.g. http://localhost:8000',
        )
        parser.add_argument(
            '-l', '--api-server-login',
            type=PathCleaner('credentials file', preexists=False), default=None,
            help='Json file with {"username":"<USER>", "password":"<PASS>"}', required=False,
        )
        # Required
        parser.add_argument(
            '-m', '--model-id', type=int, default=None,
            required=False,
            help='API `id` of a model to run the analysis with'
        )
        parser.add_argument(
            '-a', '--analysis-settings-json',
            type=PathCleaner('analysis settings file'), default=None,
            help='Analysis settings JSON file path'
        )

        parser.add_argument('-x', '--oed-location-csv', type=PathCleaner('OED location file'), default=None, help='Source exposure CSV file path')
        parser.add_argument('-y', '--oed-accounts-csv', type=PathCleaner('OED accounts file'), default=None, help='Source accounts CSV file path')
        parser.add_argument('-i', '--oed-info-csv', type=PathCleaner('OED Reinsurances info file'), default=None, help='Reinsurance info. CSV file path')
        parser.add_argument('-s', '--oed-scope-csv', type=PathCleaner('OED Reinsurances scope file'), default=None, help='Reinsurance scope CSV file path')
        parser.add_argument('-o', '--output-dir', type=PathCleaner('Output directory', preexists=False), default='./', help="Output data directory (absolute or relative file path)")

    def _select_model(self, avalible_models):
        # list options
        for i in range(len(avalible_models)):
            self.logger.info('{} \t {}-{}-{}'.format(
                i,
                avalible_models[i]['supplier_id'],
                avalible_models[i]['model_id'],
                avalible_models[i]['version_id'],
            ))

        # Fetch user choice
        while True:
            try:
                value = int(input('Select model: '))
            except ValueError:
                self.logger.info('Invalid Response: {}'.format(value))
                continue
            except KeyboardInterrupt:
                exit(1)

            if (value < 0) or (value >= len(avalible_models)):
                self.logger.info('Invalid Response: {}'.format(value))
                continue
            else:
                break
        return avalible_models[value]

    def action(self, args):
        inputs = InputValues(args)
        api = open_api_connection(inputs, self.logger)

        # Upload files
        path_location = inputs.get('oed_location_csv')
        path_account = inputs.get('oed_accounts_csv')
        path_info = inputs.get('oed_info_csv')
        path_scope = inputs.get('oed_scope_csv')

        portfolio = api.upload_inputs(
            portfolio_id=None,
            location_fp=path_location,
            accounts_fp=path_account,
            ri_info_fp=path_info,
            ri_scope_fp=path_scope,
        )

        model_id = inputs.get('model_id')
        if not model_id:
            avalible_models = api.models.get().json()

            if len(avalible_models) > 1:
                selected_model = self._select_model(avalible_models)
            elif len(avalible_models) == 1:
                selected_model = avalible_models[0]
            else:
                raise OasisException(
                    'No models found in API: {}'.format(inputs.get('api_server_url'))
                )

            model_id = selected_model['id']
            self.logger.info('Running model:')
            self.logger.info(json.dumps(selected_model, indent=4))

        # Create new analysis
        path_settings = inputs.get('analysis_settings_json')
        if not path_settings:
            self.logger.error('analysis settings: Not found')
            return False
        analysis = api.create_analysis(
            portfolio_id=portfolio['id'],
            model_id=model_id,
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
            download_path=inputs.get('output_dir'),
            overwrite=True,
            clean_up=False
        )


class ApiCmd(OasisBaseCommand):
    sub_commands = {
        'list': GetApiCmd,
        'add-model': PutApiModelCmd,
        'delete': DelApiCmd,
        'run': RunApiCmd
    }
