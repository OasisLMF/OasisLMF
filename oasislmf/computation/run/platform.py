__all__ = [
    'PlatformBase',
    'PlatformList',
    'PlatformGet',
    'PlatformRun',
    'PlatformDelete',
]


import getpass
import io
import os
import json

from tabulate import tabulate
from mimetypes import guess_extension

from requests.exceptions import HTTPError

from ...platform_api.client import APIClient
from ...utils.exceptions import OasisException
from ...utils.defaults import API_EXAMPLE_AUTH

from ..base import ComputationStep


class PlatformBase(ComputationStep):
    """
    Base platform class to handle opening a client connection
    """
    step_params = [
        {'name': 'server_login_json', 'is_path': True, 'pre_exist': False, 'help': 'Source location CSV file path'},
        {'name': 'server_url', 'default': 'http://localhost:8000', 'help': 'URL to Oasis Platform server, default is localhost'},
        {'name': 'server_version', 'default': 'v2', 'help': "Version prefix for OasisPlatform server, 'v1' = single server run, 'v2' = distributed on cluster"},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.server = self.open_connection()

    def load_credentials(self, login_arg):
        """
        Load credentials from JSON file or Prompt for username / password

        Options:
            1.'--server-login ./APIcredentials.json'
            2. Load credentials from default config file '-C oasislmf.json'
        """
        if isinstance(login_arg, str):
            with io.open(login_arg, encoding='utf-8') as f:
                return json.load(f)

        self.logger.info('API Login:')
        api_login = {}
        api_login['username'] = input('Username: ')
        api_login['password'] = getpass.getpass('Password: ')
        return api_login

    def open_connection(self):
        try:
            # If no password given try the reference example
            return APIClient(
                api_url=self.server_url,
                api_ver=self.server_version,
                username=API_EXAMPLE_AUTH['user'],
                password=API_EXAMPLE_AUTH['pass'],
            )
        except OasisException as e:
            if isinstance(e.original_exception, HTTPError) and (e.original_exception.response.status_code == 401):
                # Prompt for password and try to re-autehnticate if Unauthorized 401
                self.logger.info("-- Authentication Required --")
                credentials = self.load_credentials(self.server_login_json)
                self.logger.info(f'Connecting to - {self.server_url}')
                return APIClient(
                    api_url=self.server_url,
                    api_ver=self.server_version,
                    username=credentials['username'],
                    password=credentials['password'],
                )
            raise e

    def tabulate_json(self, json_data, items):
        table_data = dict()

        for i in items:
            table_data[i] = list()

        for m in json_data:
            for k in table_data:
                # will have link+data if dict returned
                if isinstance(m[k], dict):
                    table_data[k].append('Yes')

                # If none then no data
                elif m[k] is None:
                    table_data[k].append('-')

                # If URL then something linked to field
                elif isinstance(m[k], str):
                    if any(v in m[k] for v in ['http://', 'https://']):
                        table_data[k].append('Linked')
                    else:
                        table_data[k].append(m[k])

                # Fallback - add value as string
                else:
                    table_data[k].append(str(m[k]))
        return table_data

    def print_endpoint(self, attr, items):
        endpoint_obj = getattr(self.server, attr)
        self.logger.info(f'\nAvailable {attr}:')
        data = self.tabulate_json(endpoint_obj.get().json(), items)
        self.logger.info(tabulate(data, headers=items, tablefmt='psql'))
        return data

    def select_id(self, msg, valid_ids):
        while True:
            try:
                value = str(input(f'Select {msg} ID: '))
            except ValueError:
                self.logger.info('Invalid Response: {}'.format(value))
                continue
            except KeyboardInterrupt:
                return -1

            if value not in valid_ids:
                self.logger.info(f'id {value} not among the valid ids: {valid_ids} - ctrl-c to exit')
                continue
            else:
                break
        return int(value)


class PlatformList(PlatformBase):
    """ Return status and details from an Oasis Platform API server
    """
    step_params = PlatformBase.step_params + [
        {'name': 'models', 'flag': '-m', 'type': int, 'nargs': '+', 'help': 'List of model ids to print in detail'},
        {'name': 'portfolios', 'flag': '-p', 'type': int, 'nargs': '+', 'help': 'List of portfolio ids to print in detail'},
        {'name': 'analyses', 'flag': '-a', 'type': int, 'nargs': '+', 'help': 'List of analyses ids to print in detail'},
    ]

    def run(self):
        # Default to printing summary of API status
        if not any([self.models, self.portfolios, self.analyses]):
            self.print_endpoint('models', ['id', 'supplier_id', 'model_id', 'version_id'])
            self.print_endpoint('portfolios', ['id', 'name', 'location_file', 'accounts_file', 'reinsurance_info_file', 'reinsurance_scope_file'])
            self.print_endpoint('analyses', ['id', 'name', 'model', 'portfolio', 'status', 'input_file', 'output_file', 'run_log_file'])

        if self.models:
            for Id in self.models:
                msg = f'Model (id={Id}): \n'
                try:
                    rsp = self.server.models.get(Id)
                    self.logger.info(msg + json.dumps(rsp.json(), indent=4, sort_keys=True))
                except HTTPError as e:
                    self.logger.info(msg + e.response.text)

        if self.portfolios:
            for Id in self.portfolios:
                msg = f'Portfolio (id={Id}): \n'
                try:
                    rsp = self.server.portfolios.get(Id)
                    self.logger.info(msg + json.dumps(rsp.json(), indent=4, sort_keys=True))
                except HTTPError as e:
                    self.logger.info(msg + e.response.text)

        if self.analyses:
            for Id in self.analyses:
                msg = f'Analysis (id={Id}): \n'
                try:
                    rsp = self.server.analyses.get(Id)
                    self.logger.info(msg + json.dumps(rsp.json(), indent=4, sort_keys=True))
                except HTTPError as e:
                    self.logger.info(msg + e.response.text)


class PlatformRunInputs(PlatformBase):
    """ run generate inputs via the Oasis Platoform API
    """
    step_params = PlatformBase.step_params + [
        {'name': 'model_id', 'type': int, 'help': 'API `id` of a model to run an analysis with'},
        {'name': 'portfolio_id', 'type': int, 'help': 'API `id` of a portfolio to run an analysis with'},
        {'name': 'analysis_id', 'type': int, 'help': 'API `id` of an analysis to run'},

        {'name': 'analysis_settings_json', 'flag': '-a', 'is_path': True, 'pre_exist': True, 'help': 'Analysis settings JSON file path'},
        {'name': 'oed_location_csv', 'flag': '-x', 'is_path': True, 'pre_exist': True, 'help': 'Source location CSV file path'},
        {'name': 'oed_accounts_csv', 'flag': '-y', 'is_path': True, 'pre_exist': True, 'help': 'Source accounts CSV file path'},
        {'name': 'oed_info_csv', 'flag': '-i', 'is_path': True, 'pre_exist': True, 'help': 'Reinsurance info. CSV file path'},
        {'name': 'oed_scope_csv', 'flag': '-s', 'is_path': True, 'pre_exist': True, 'help': 'Reinsurance scope CSV file path'},
    ]

    def run(self):
        # Run Input geneneration from ID
        if self.analysis_id:
            try:
                status = self.server.analyses.status(self.analysis_id)
                if status in ['RUN_QUEUED', 'RUN_STARTED']:
                    self.server.cancel_analysis(self.analysis_id)
                elif status in ['INPUTS_GENERATION_QUEUED', 'INPUTS_GENERATION_STARTED']:
                    self.server.cancel_generate(self.analysis_id)
                self.server.run_generate(self.analysis_id)
                return self.analysis_id
            except HTTPError as e:
                raise OasisException(f'Error running analysis ({self.analysis_id}) - {e}')

        # Create Portfolio and Ananlysis, then run
        if not (self.portfolio_id or self.oed_location_csv or self.oed_accounts_csv):
            raise OasisException('Error: At least one of the following inputs is required [portfolio_id, oed_location_csv, oed_accounts_csv]')

        # when no model is selected prompt user for choice
        if not self.model_id:
            models = self.server.models.get().json()
            model_count = len(models)

            if model_count < 1:
                raise OasisException(f'No models found in API: {self.server_url}')
            if model_count == 1:
                self.model_id = models[0]['id']
            if model_count > 1:
                models_table = self.print_endpoint('models', ['id', 'supplier_id', 'model_id', 'version_id'])
                self.model_id = self.select_id('models', models_table['id'])
            if self.model_id < 0:
                raise OasisException(' Model selection cancelled')

        # Select or create a portfilo
        if self.portfolio_id:
            portfolios = self.server.portfolios.get().json()
            if self.portfolio_id not in [p['id'] for p in portfolios]:
                raise OasisException(f'Portfolio "{self.portfolio_id}" not found in API: {self.server_url}')
        else:
            portfolio = self.server.upload_inputs(
                portfolio_id=None,
                location_fp=self.oed_location_csv,
                accounts_fp=self.oed_accounts_csv,
                ri_info_fp=self.oed_info_csv,
                ri_scope_fp=self.oed_scope_csv,
            )
            self.portfolio_id = portfolio['id']
        analysis = self.server.create_analysis(
            portfolio_id=self.portfolio_id,
            model_id=self.model_id,
            analysis_settings_fp=self.analysis_settings_json,
        )
        self.analysis_id = analysis['id']

        # Execure run
        self.server.run_generate(self.analysis_id)
        return self.analysis_id


class PlatformRunLosses(PlatformBase):
    """ run generate losses via the Oasis Platoform API
    """
    step_params = PlatformBase.step_params + [
        {'name': 'analysis_id', 'type': int, 'required': True, 'help': 'API `id` of an analysis to run'},
        {'name': 'output_dir', 'flag': '-o', 'is_path': True, 'pre_exist': True,
            'help': 'Output data directory for results data (absolute or relative file path)', 'default': './'},
        {'name': 'analysis_settings_json', 'flag': '-a', 'is_path': True, 'pre_exist': True, 'help': 'Analysis settings JSON file path'},
    ]

    def run(self):
        self.server.run_analysis(self.analysis_id, self.analysis_settings_json)
        self.server.download_output(self.analysis_id, self.output_dir)


class PlatformRun(PlatformBase):
    """ End to End - run model via the Oasis Platoform API
    """
    chained_commands = [PlatformRunInputs, PlatformRunLosses]

    def run(self):
        self.kwargs['analysis_id'] = PlatformRunInputs(**self.kwargs).run()
        PlatformRunLosses(**self.kwargs).run()


class PlatformDelete(PlatformBase):
    """ Delete either a 'model', 'portfolio' or an 'analysis' from the API's Database
    """
    step_params = PlatformBase.step_params + [
        {'name': 'models', 'flag': '-m', 'type': int, 'nargs': '+', 'help': 'List of model ids to Delete.'},
        {'name': 'portfolios', 'flag': '-p', 'type': int, 'nargs': '+', 'help': 'List of Portfolio ids to Delete'},
        {'name': 'analyses', 'flag': '-a', 'type': int, 'nargs': '+', 'help': 'List of Analyses ids to Detele'},
    ]

    def delete_list(self, attr, id_list):
        if not all(isinstance(ID, int) for ID in id_list):
            raise OasisException(f"Invalid input, '{attr}', must be a list of type Int, not {id_list}")

        api_endpoint = getattr(self.server, attr)
        for Id in id_list:
            try:
                if api_endpoint.delete(Id).ok:
                    self.logger.info(f'Deleted {attr}_id={Id}')
            except HTTPError as e:
                self.logger.error('Delete error {}_id={} - {}'.format(attr, Id, e))
                continue

    def run(self):

        if not any([self.models, self.portfolios, self.analyses]):
            raise OasisException("""Select item(s) to delete, list of either:
                --models MODEL_ID [MODEL_ID ... n],
                --portfolios PORTFOLIOS_ID [PORTFOLIO_ID ... n]
                --analyses ANALYSES_ID [ANALYSES_ID ... n]
                """)

        if self.models:
            self.delete_list('models', self.models)
        if self.portfolios:
            self.delete_list('portfolios', self.portfolios)
        if self.analyses:
            self.delete_list('analyses', self.analyses)


class PlatformGet(PlatformBase):
    """ Download file(s) from the api
    """
    step_params = PlatformBase.step_params + [
        {'name': 'output_dir', 'flag': '-o', 'is_path': True, 'pre_exist': True,
            'help': 'Output data directory for results data (absolute or relative file path)', 'default': './'},
        # Files for models object
        {'name': 'model_settings', 'type': int, 'nargs': '+', 'help': 'Model ids to download settings file.'},
        {'name': 'model_versions', 'type': int, 'nargs': '+', 'help': 'Model ids to download versions file'},
        # Files from portfolio
        {'name': 'portfolio_location_file', 'type': int, 'nargs': '+', 'help': 'Portfolio ids to download Location file'},
        {'name': 'portfolio_accounts_file', 'type': int, 'nargs': '+', 'help': 'Portfolio ids to download Accounts file'},
        {'name': 'portfolio_reinsurance_scope_file', 'type': int, 'nargs': '+', 'help': 'Portfolio ids to download RI scope file.'},
        {'name': 'portfolio_reinsurance_info_file', 'type': int, 'nargs': '+', 'help': 'Portfolio ids to download RI info file.'},
        # Files from an analyses
        {'name': 'analyses_settings_file', 'type': int, 'nargs': '+', 'help': 'Analyses ids to download settings file'},
        {'name': 'analyses_run_traceback_file', 'type': int, 'nargs': '+', 'help': 'Analyses ids to download traceback logs'},
        {'name': 'analyses_run_log_file', 'type': int, 'nargs': '+', 'help': 'Analyses ids to download Ktools run logs'},
        {'name': 'analyses_input_generation_traceback_file', 'type': int, 'nargs': '+',
            'help': 'Analyses ids to download input_generation traceback logs'},
        {'name': 'analyses_input_file', 'type': int, 'nargs': '+', 'help': 'Analyses ids to download Generated inputs tar'},
        {'name': 'analyses_output_file', 'type': int, 'nargs': '+', 'help': 'Analyses ids to download Output losses tar'},
        {'name': 'analyses_summary_levels_file', 'type': int, 'nargs': '+', 'help': 'Analyses ids to download summary levels file'},
        {'name': 'analyses_lookup_validation_file', 'type': int, 'nargs': '+', 'help': 'Analyses ids to download exposure summary'},
        {'name': 'analyses_lookup_success_file', 'type': int, 'nargs': '+', 'help': 'Analyses ids to download successful lookups'},
        {'name': 'analyses_lookup_errors_file', 'type': int, 'nargs': '+', 'help': 'Analyses ids to download summary of failed lookups'},
    ]

    def extract_args(self, param_suffix):
        return {k.replace(param_suffix, ''): v for k, v in self.kwargs.items() if v and param_suffix in k}

    def download(self, collection, req_files, chuck_size=1024):
        collection_obj = getattr(self.server, collection)
        for File in req_files:
            resource = getattr(collection_obj, File)
            for ID in req_files[File]:
                try:
                    r = resource.get(ID)

                    ext = guess_extension(r.headers['content-type'].partition(';')[0].strip())
                    filename = os.path.join(self.output_dir, f'{ID}_{collection}_{File}{ext}')

                    with io.open(filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=chuck_size):
                            f.write(chunk)
                    self.logger.info(f'Downloaded: {File} from {collection}_id={ID} "{filename}"')
                except HTTPError as e:
                    self.logger.error('Download failed: - {}'.format(e))

    def run(self):
        model_files = self.extract_args('model_')
        portfolio_files = self.extract_args('portfolio_')
        analyses_files = self.extract_args('analyses_')

        # Check that at least one option is given
        if not any([model_files, portfolio_files, analyses_files]):
            raise OasisException('Select file for download e.g. "--analyses_output <id_1> .. <id_n>"')

        if model_files:
            self.download('models', model_files)
        if portfolio_files:
            self.download('portfolios', portfolio_files)
        if analyses_files:
            self.download('analyses', analyses_files)
