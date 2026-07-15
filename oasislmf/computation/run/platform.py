__all__ = [
    'PlatformBase',
    'PlatformList',
    'PlatformGet',
    'PlatformPost',
    'PlatformRun',
    'PlatformDelete',
    'PlatformValidate',
    'PlatformExposureRun',
    'PlatformExposureTransform',
    'PlatformCombine',
    'PlatformCancel',
    'PlatformSubTasks',
    'PlatformPlot',
    'PlatformServerInfo',
]


import getpass
import io
import os
import json
import time

from datetime import datetime

from tabulate import tabulate
from mimetypes import guess_extension

from requests.exceptions import HTTPError

from ...platform_api.client import APIClient
from ...utils.exceptions import OasisException
from ...utils.defaults import API_EXAMPLE_AUTH
from ...utils.inputs import str2bool

from ..base import ComputationStep


class PlatformBase(ComputationStep):
    """
    Base platform class to handle opening a client connection
    """
    step_params = [
        {'name': 'server_login_json', 'required': False, 'default': None, 'is_path': True,
            'pre_exist': False, 'help': 'Server login credentials json string'},
        {'name': 'server_url', 'default': 'http://localhost:8000', 'help': 'URL to Oasis Platform server, default is localhost'},
        {'name': 'server_version', 'default': 'v2', 'help': "Version prefix for OasisPlatform server, 'v1' = single server run, 'v2' = distributed on cluster"},
        {'name': 'auth_type', 'required': False, 'default': None,
            'choices': ['simple', 'oidc', 'm2m'],
            'help': 'Authentication type: simple (username/password JWT), oidc (client credentials via platform), m2m (client credentials direct to IdP)'},
        {'name': 'oidc_token_url', 'required': False, 'default': None,
            'help': 'Token endpoint URL for m2m client_credentials grant (e.g. https://idp.example.com/oauth2/token)'},
        {'name': 'oidc_scope', 'required': False, 'default': None,
            'help': 'OAuth2 scope to request when fetching an m2m token (e.g. oasis/m2m)'},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.server = self.open_connection()

    def load_credentials(self, login_arg, auth_type=None):
        """
        Load credentials from JSON file or prompt interactively.

        Options:
            1.'--server-login ./APIcredentials.json'
            2. Load credentials from default config file '-C oasislmf.json'
            3. Interactive prompt (menu skipped when auth_type is already known)
        """
        if isinstance(login_arg, str):
            with io.open(login_arg, encoding='utf-8') as f:
                return json.load(f)

        if auth_type is None:
            while True:
                user_response = input("Auth type — simple JWT [1], OIDC via platform [2], M2M direct to IdP [3]: ").strip()
                if user_response == "1":
                    auth_type = "simple"
                    break
                elif user_response == "2":
                    auth_type = "oidc"
                    break
                elif user_response == "3":
                    auth_type = "m2m"
                    break

        self.logger.info('API Login:')
        api_login = {}
        if auth_type == "simple":
            api_login['username'] = input('Username: ')
            api_login['password'] = getpass.getpass('Password: ')
        elif auth_type == "oidc":
            api_login['client_id'] = input('Client ID: ')
            api_login['client_secret'] = getpass.getpass('Client Secret: ')
        elif auth_type == "m2m":
            api_login['client_id'] = input('Client ID: ')
            api_login['client_secret'] = getpass.getpass('Client Secret: ')
            api_login['token_url'] = input('Token URL: ')
            scope = input('Scope (leave blank to omit): ').strip()
            if scope:
                api_login['scope'] = scope

        return api_login

    def try_connection(self, fail_safe=True, **kwargs):
        """Helper to safely try connecting and return None if unauthorized."""
        try:
            return APIClient(
                api_url=self.server_url,
                api_ver=self.server_version,
                **kwargs
            )
        except OasisException as e:
            if not fail_safe:
                raise e
            orig_excep = e.original_exception
            if (isinstance(orig_excep, HTTPError) and orig_excep.response.status_code == 401):
                self.logger.debug("Login attempt failed, reason: Unauthorized (401) for credentials: %s", list(kwargs.keys()))
                return None
            elif isinstance(orig_excep, HTTPError) and orig_excep.response.status_code == 400 and "flow is disabled" in orig_excep.response.text.lower():
                self.logger.debug(
                    "Login attempt failed, reason: Validation Error (400) for credentials, invalid flow used, must be one of username/password or client_id/client_secret: %s", list(
                        kwargs.keys())
                )
                return None
            else:
                raise e  # Some other error – propagate

    def _oidc_m2m_kwargs(self):
        """Return token_url/scope kwargs when the step params are set."""
        kwargs = {}
        if getattr(self, 'oidc_token_url', None):
            kwargs['token_url'] = self.oidc_token_url
        if getattr(self, 'oidc_scope', None):
            kwargs['scope'] = self.oidc_scope
        return kwargs

    def open_connection(self):
        """
        Attempts connection in this order:
        1. API_EXAMPLE_AUTH username/password  (skipped when auth_type is oidc or m2m)
        2. API_EXAMPLE_AUTH client_id/client_secret  (skipped when auth_type is simple)
        3. Prompt or load credentials
        """
        auth_type = getattr(self, 'auth_type', None)

        if not isinstance(self.server_login_json, str):
            # 1. Try example username/password
            if auth_type in (None, "simple"):
                if 'username' in API_EXAMPLE_AUTH and 'password' in API_EXAMPLE_AUTH:
                    conn = self.try_connection(
                        auth_type="simple",
                        username=API_EXAMPLE_AUTH['username'],
                        password=API_EXAMPLE_AUTH['password']
                    )
                    if conn:
                        return conn

            # 2. Try example client_id/client_secret
            if auth_type in (None, "oidc", "m2m"):
                if 'client_id' in API_EXAMPLE_AUTH and 'client_secret' in API_EXAMPLE_AUTH:
                    example_auth_type = auth_type if auth_type in ("oidc", "m2m") else "oidc"
                    conn = self.try_connection(
                        auth_type=example_auth_type,
                        client_id=API_EXAMPLE_AUTH['client_id'],
                        client_secret=API_EXAMPLE_AUTH['client_secret'],
                        **self._oidc_m2m_kwargs()
                    )
                    if conn:
                        return conn

        # 3. Load credentials (file or prompt)
        self.logger.info("-- Authentication Required --")
        credentials = self.load_credentials(self.server_login_json, auth_type=auth_type)
        self.logger.info(f'Connecting to - {self.server_url}')

        if 'username' in credentials and 'password' in credentials:
            return self.try_connection(
                fail_safe=False,
                auth_type="simple",
                username=credentials['username'],
                password=credentials['password']
            )

        if 'client_id' in credentials and 'client_secret' in credentials:
            m2m_kwargs = {k: credentials[k] for k in ('token_url', 'scope') if k in credentials}
            m2m_kwargs.update(self._oidc_m2m_kwargs())
            resolved_type = auth_type if auth_type in ("oidc", "m2m") else (
                "m2m" if m2m_kwargs.get('token_url') else "oidc"
            )
            return self.try_connection(
                fail_safe=False,
                auth_type=resolved_type,
                client_id=credentials['client_id'],
                client_secret=credentials['client_secret'],
                **m2m_kwargs
            )

        raise OasisException(
            f"Error: No valid credentials provided for platform, current credential keys [{list(credentials.keys())}], "
            "must be one of username/password or client_id/client_secret"
        )

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

    # Portfolio.{validation,exposure,exposure_transform}_status_choices on the
    # server: NONE, INSUFFICIENT_DATA, STARTED, ERROR, RUN_COMPLETED.
    # STARTED is the only in-flight state; NONE means it has never been run.
    pending_states = ['NONE', 'STARTED']

    def poll_portfolio_field(self, portfolio_id, status_field, poll_interval, action_name):
        """
        Poll a portfolio's async status field (e.g. `validation_status`,
        `exposure_status`) until it settles on a terminal value.

        A re-triggered run can end up back at the same terminal status it
        started at (e.g. RUN_COMPLETED -> STARTED -> RUN_COMPLETED), so
        completion can't be detected by a status change alone - it must have
        passed through a pending state at least once since this run was
        triggered.
        """
        seen_pending = False
        logged_pending = False
        while True:
            portfolio = self.server.portfolios.get(portfolio_id).json()
            status = portfolio[status_field]
            if status in self.pending_states:
                seen_pending = True
            elif seen_pending:
                break
            if not logged_pending:
                logged_pending = True
                self.logger.info('{}: Pending (id={})'.format(action_name, portfolio_id))
            time.sleep(poll_interval)

        self.logger.info('{}: Complete (id={}, status={})'.format(action_name, portfolio_id, status))
        return status


class PlatformServerInfo(PlatformBase):
    """ Print version/info details of the connected Oasis Platform API server
    """

    def run(self):
        rsp = self.server.server_info()
        data = rsp.json()
        self.logger.info(json.dumps(data, indent=4, sort_keys=True))
        return data


class PlatformList(PlatformBase):
    """ Return status and details from an Oasis Platform API server
    """
    step_params = PlatformBase.step_params + [
        {'name': 'models', 'flag': '-m', 'type': int, 'nargs': '+', 'help': 'List of model ids to print in detail'},
        {'name': 'portfolios', 'flag': '-p', 'type': int, 'nargs': '+', 'help': 'List of portfolio ids to print in detail'},
        {'name': 'analyses', 'flag': '-a', 'type': int, 'nargs': '+', 'help': 'List of analyses ids to print in detail'},
        {'name': 'subtask', 'flag': '-t', 'type': int, 'nargs': '+', 'help': 'List of task status ids to print in detail'},
    ]

    def run(self):
        # Default to printing summary of API status
        if not any([self.models, self.portfolios, self.analyses, self.subtask]):
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

        if self.subtask:
            for Id in self.subtask:
                msg = f'Task status (id={Id}): \n'
                try:
                    rsp = self.server.task_status.get(Id)
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
        {'name': 'currency_conversion_json', 'is_path': True, 'pre_exist': True, 'help': 'settings to perform currency conversion of oed files'},
        {'name': 'reporting_currency', 'type': str, 'help': 'currency to use in the results reported'},
        {'name': 'lookup_chunks', 'type': int, 'help': 'Set the number of lookup chunks in a V2 run'},
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
                currency_conversion_fp=self.currency_conversion_json,
                reporting_currency=self.reporting_currency
            )
            self.portfolio_id = portfolio['id']
        analysis = self.server.create_analysis(
            portfolio_id=self.portfolio_id,
            model_id=self.model_id,
            analysis_settings_fp=self.analysis_settings_json,
        )

        self.analysis_id = analysis['id']
        if self.lookup_chunks:
            self.server.analyses.chunking_configuration.post(self.analysis_id, {
                "lookup_strategy": "FIXED_CHUNKS",
                "fixed_lookup_chunks": self.lookup_chunks
            })

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
        {'name': 'analysis_chunks', 'type': int, 'help': 'Set the number of analysis chunks in a V2 run'},
    ]

    def run(self):
        if self.analysis_chunks:
            self.server.analyses.chunking_configuration.post(self.analysis_id, {
                "loss_strategy": "FIXED_CHUNKS",
                "fixed_analysis_chunks": self.analysis_chunks
            })
        self.server.run_analysis(self.analysis_id, self.analysis_settings_json)
        self.server.download_output(self.analysis_id, self.output_dir)


class PlatformRun(PlatformBase):
    """ End to End - run model via the Oasis Platoform API
    """
    chained_commands = [PlatformRunInputs, PlatformRunLosses]

    def run(self):
        self.kwargs['analysis_id'] = PlatformRunInputs(**self.kwargs).run()
        PlatformRunLosses(**self.kwargs).run()


class PlatformReconnect(PlatformBase):
    """ Reconnect to an in-progress (or finished) analysis and resume polling for status,
    without re-triggering input generation or the run itself.
    """
    step_params = PlatformBase.step_params + [
        {'name': 'analysis_id', 'type': int, 'required': True, 'help': 'API `id` of an analysis to reconnect to'},
        {'name': 'output_dir', 'flag': '-o', 'is_path': True, 'pre_exist': True,
            'help': 'Output data directory for results data (absolute or relative file path)', 'default': './'},
    ]

    def run(self):
        self.server.reconnect(self.analysis_id, self.output_dir)


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
        # Files from a task status (analysis sub-task)
        {'name': 'subtask_output_log', 'type': int, 'nargs': '+', 'help': 'Task status ids to download output_log for'},
        {'name': 'subtask_error_log', 'type': int, 'nargs': '+', 'help': 'Task status ids to download error_log for'},
        {'name': 'subtask_retry_log', 'type': int, 'nargs': '+', 'help': 'Task status ids to download retry_log for'},
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
        subtask_files = self.extract_args('subtask_')

        # Check that at least one option is given
        if not any([model_files, portfolio_files, analyses_files, subtask_files]):
            raise OasisException('Select file for download e.g. "--analyses_output <id_1> .. <id_n>"')

        if model_files:
            self.download('models', model_files)
        if portfolio_files:
            self.download('portfolios', portfolio_files)
        if analyses_files:
            self.download('analyses', analyses_files)
        if subtask_files:
            self.download('task_status', subtask_files)


class PlatformPost(PlatformBase):
    """ Upload file(s) to the api

    Portfolio files are uploaded to a single portfolio per invocation - give
    `--portfolio-id` to update an existing portfolio, or omit it (optionally
    with `--portfolio-name`) to create a new one.
    """
    step_params = PlatformBase.step_params + [
        # Portfolio
        {'name': 'portfolio_id', 'type': int, 'help': 'API `id` of an existing portfolio to update. Omit to create a new portfolio'},
        {'name': 'portfolio_name', 'help': 'Name for a newly created portfolio (ignored if --portfolio-id is given)'},
        {'name': 'oed_location_csv', 'flag': '-x', 'is_path': True, 'pre_exist': True, 'help': 'Source location CSV file path'},
        {'name': 'oed_accounts_csv', 'flag': '-y', 'is_path': True, 'pre_exist': True, 'help': 'Source accounts CSV file path'},
        {'name': 'oed_info_csv', 'flag': '-i', 'is_path': True, 'pre_exist': True, 'help': 'Reinsurance info. CSV file path'},
        {'name': 'oed_scope_csv', 'flag': '-s', 'is_path': True, 'pre_exist': True, 'help': 'Reinsurance scope CSV file path'},
        {'name': 'currency_conversion_json', 'is_path': True, 'pre_exist': True, 'help': 'settings to perform currency conversion of oed files'},
        {'name': 'reporting_currency', 'type': str, 'help': 'currency to use in the results reported'},
        # Analyses
        {'name': 'analyses_id', 'type': int, 'help': 'API `id` of an analysis to upload settings to'},
        {'name': 'analyses_settings_json', 'flag': '-a', 'is_path': True, 'pre_exist': True,
            'help': 'Analyses settings JSON file path to upload (requires --analyses-id)'},
        # Models
        {'name': 'model_id', 'type': int, 'help': 'API `id` of a model to upload settings to'},
        {'name': 'model_settings_json', 'is_path': True, 'pre_exist': True,
            'help': 'Model settings JSON file path to upload (requires --model-id)'},
    ]

    def run(self):
        result = {}

        portfolio_files_given = any([
            self.oed_location_csv, self.oed_accounts_csv, self.oed_info_csv,
            self.oed_scope_csv, self.currency_conversion_json, self.reporting_currency,
        ])
        if portfolio_files_given and (self.portfolio_name or self.portfolio_id):
            portfolio = self.server.upload_inputs(
                portfolio_name=self.portfolio_name,
                portfolio_id=self.portfolio_id,
                location_fp=self.oed_location_csv,
                accounts_fp=self.oed_accounts_csv,
                ri_info_fp=self.oed_info_csv,
                ri_scope_fp=self.oed_scope_csv,
                currency_conversion_fp=self.currency_conversion_json,
                reporting_currency=self.reporting_currency,
            )
            self.logger.info('Portfolio uploaded (id={})'.format(portfolio['id']))
            result['portfolio'] = portfolio

        if self.analyses_settings_json and self.analyses_id:
            self.server.upload_settings(self.analyses_id, self.analyses_settings_json)
            self.logger.info('Analyses settings uploaded (id={})'.format(self.analyses_id))
            result['analyses_id'] = self.analyses_id

        if self.model_settings_json and self.model_id:
            with io.open(self.model_settings_json, encoding='utf-8') as f:
                settings = json.load(f)
            self.server.models.settings.post(self.model_id, settings)
            self.logger.info('Model settings uploaded (id={})'.format(self.model_id))
            result['model_id'] = self.model_id

        if not result:
            raise OasisException(
                'Select at least one file to upload e.g. "--oed-location-csv <path>", '
                '"--analyses-settings-json <path> --analyses-id <id>", or '
                '"--model-settings-json <path> --model-id <id>"'
            )
        return result


class PlatformValidate(PlatformBase):
    """ Validate a portfolio's OED exposure files via the Oasis Platform API
    """
    step_params = PlatformBase.step_params + [
        {'name': 'portfolio_id', 'type': int, 'required': True, 'help': 'API `id` of a portfolio to validate'},
        {'name': 'get_status', 'action': 'store_true',
            'help': 'Fetch the current validation status instead of triggering a new OED validation run'},
        {'name': 'poll_interval', 'type': int, 'default': 5, 'help': 'Polling interval in seconds while waiting for validation to complete'},
    ]

    def run(self):
        if self.get_status:
            rsp = self.server.portfolios.validate.get(self.portfolio_id)
            data = rsp.json()
            self.logger.info(json.dumps(data, indent=4, sort_keys=True))
            return data

        self.server.portfolios.validate.post(self.portfolio_id, {})
        self.logger.info('Portfolio validation: Starting (id={})'.format(self.portfolio_id))
        self.poll_portfolio_field(self.portfolio_id, 'validation_status', self.poll_interval, 'Portfolio validation')

        data = self.server.portfolios.validate.get(self.portfolio_id).json()
        self.logger.info(json.dumps(data, indent=4, sort_keys=True))
        return data


class PlatformExposureRun(PlatformBase):
    """ Run `oasislmf exposure run` on the server against a portfolio's exposure files,
    or fetch the result of the last run
    """
    step_params = PlatformBase.step_params + [
        {'name': 'portfolio_id', 'type': int, 'required': True, 'help': 'API `id` of a portfolio to run exposure calculations against'},
        {'name': 'output_dir', 'flag': '-o', 'is_path': True, 'pre_exist': True, 'default': './',
            'help': 'Output data directory to download the resulting report to'},
        {'name': 'fetch', 'action': 'store_true', 'help': 'Fetch the result of the last exposure run instead of starting a new one'},
        {'name': 'kernel_alloc_rule_il', 'flag': '-a', 'type': int, 'default': 2,
            'help': 'Set the fmcalc allocation rule used in direct insured loss'},
        {'name': 'kernel_alloc_rule_ri', 'flag': '-A', 'type': int, 'default': 3,
            'help': 'Set the fmcalc allocation rule used in reinsurance'},
        {'name': 'model_perils_covered', 'nargs': '+', 'default': ['AA1'], 'help': 'List of perils covered by the model'},
        {'name': 'loss_factor', 'flag': '-l', 'type': float, 'nargs': '+', 'default': [1.0], 'help': 'Loss factor(s) to apply'},
        {'name': 'supported_oed_coverage_types', 'type': int, 'nargs': '+', 'default': [0],
            'help': '1-15 for coverage types to support: [0] gives None'},
        {'name': 'extra_summary_cols', 'nargs': '+', 'default': [], 'help': 'Extra columns to include in the summary'},
        {'name': 'fmpy_low_memory', 'type': str2bool, 'const': True, 'nargs': '?', 'default': False,
            'help': 'use memory map instead of RAM to store loss array (may decrease performance but reduce RAM usage drastically)'},
        {'name': 'fmpy_sort_output', 'type': str2bool, 'const': True, 'nargs': '?', 'default': True, 'help': 'order fmpy output by item_id'},
        {'name': 'check_oed', 'type': str2bool, 'const': True, 'nargs': '?', 'default': True, 'help': 'if True check input oed files'},
        {'name': 'do_disaggregation', 'type': str2bool, 'const': True, 'nargs': '?', 'default': True,
            'help': 'if True run the oasis disaggregation'},
        {'name': 'poll_interval', 'type': int, 'default': 5, 'help': 'Polling interval in seconds while waiting for the exposure run to complete'},
    ]

    def run(self):
        params = {
            'kernel_alloc_rule_il': self.kernel_alloc_rule_il,
            'kernel_alloc_rule_ri': self.kernel_alloc_rule_ri,
            'model_perils_covered': self.model_perils_covered,
            'loss_factor': self.loss_factor,
            'supported_oed_coverage_types': self.supported_oed_coverage_types,
            'extra_summary_cols': self.extra_summary_cols,
            'fmpy_low_memory': self.fmpy_low_memory,
            'fmpy_sort_output': self.fmpy_sort_output,
            'check_oed': self.check_oed,
            'do_disaggregation': self.do_disaggregation,
        }

        if self.fetch:
            rsp = self.server.portfolios.exposure_run.get(self.portfolio_id)
        else:
            self.server.portfolios.exposure_run.post(self.portfolio_id, params)
            self.logger.info('Exposure run: Starting (id={})'.format(self.portfolio_id))
            self.poll_portfolio_field(self.portfolio_id, 'exposure_status', self.poll_interval, 'Exposure run')
            rsp = self.server.portfolios.exposure_run.get(self.portfolio_id)

        content_type = rsp.headers.get('content-type', '').partition(';')[0].strip()
        ext = guess_extension(content_type) or '.bin'
        filename = os.path.join(self.output_dir, 'portfolio_{}_exposure_run{}'.format(self.portfolio_id, ext))
        with io.open(filename, 'wb') as f:
            f.write(rsp.content)
        self.logger.info('Exposure run result saved to: {}'.format(filename))
        return filename


class PlatformExposureTransform(PlatformBase):
    """ Convert a portfolio's exposure data between OED and AIR via the Oasis Platform API
    """
    step_params = PlatformBase.step_params + [
        {'name': 'portfolio_id', 'type': int, 'required': True, 'help': 'API `id` of a portfolio to transform'},
        {'name': 'file_type', 'choices': ['location', 'accounts', 'ri_info', 'ri_scope'], 'required': True,
            'help': 'OED file type to transform'},
        {'name': 'mapping_file', 'is_path': True, 'pre_exist': True, 'required': True, 'help': 'Path to the mapping file'},
        {'name': 'transform_file', 'is_path': True, 'pre_exist': True, 'required': True, 'help': 'Path to the transform file'},
        {'name': 'poll_interval', 'type': int, 'default': 5, 'help': 'Polling interval in seconds while waiting for the transform to complete'},
    ]

    def run(self):
        self.server.portfolios.exposure_transform.post(
            self.portfolio_id, self.file_type, self.mapping_file, self.transform_file)
        self.logger.info('Exposure transform: Starting (id={})'.format(self.portfolio_id))
        self.poll_portfolio_field(self.portfolio_id, 'exposure_transform_status', self.poll_interval, 'Exposure transform')

        data = self.server.portfolios.get(self.portfolio_id).json()
        self.logger.info(json.dumps(data, indent=4, sort_keys=True))
        return data


class PlatformCombine(PlatformBase):
    """ Combine the ORD output of multiple RUN_COMPLETED analyses via the Oasis Platform API
    """
    step_params = PlatformBase.step_params + [
        {'name': 'analysis_ids', 'type': int, 'nargs': '+', 'required': True, 'help': 'List of RUN_COMPLETED analyses ids to combine'},
        {'name': 'combine_settings_json', 'is_path': True, 'pre_exist': True, 'required': True,
            'help': 'Path to a JSON file containing the combine ORD config (e.g. group_number_of_periods)'},
        {'name': 'combine_name', 'default': 'combine-analysis', 'help': 'Name for the combined analysis result'},
    ]

    def run(self):
        with io.open(self.combine_settings_json, encoding='utf-8') as f:
            config = json.load(f)
        rsp = self.server.analyses.combine(self.analysis_ids, config, self.combine_name)
        data = rsp.json()
        self.logger.info(json.dumps(data, indent=4, sort_keys=True))
        return data


class PlatformCancel(PlatformBase):
    """ Cancel a running analysis (input generation or execution) via the Oasis Platform API
    """
    step_params = PlatformBase.step_params + [
        {'name': 'analysis_id', 'type': int, 'required': True, 'help': 'API `id` of an analysis to cancel'},
    ]

    def run(self):
        self.server.analyses.cancel(self.analysis_id)
        self.logger.info('Cancelled analysis (id={})'.format(self.analysis_id))


class PlatformSubTasks(PlatformBase):
    """ List the sub-tasks of an analysis run via the Oasis Platform API
    """
    step_params = PlatformBase.step_params + [
        {'name': 'analysis_id', 'type': int, 'required': True, 'help': 'API `id` of an analysis to list sub-tasks for'},
    ]

    def run(self):
        rsp = self.server.analyses.sub_task_list(self.analysis_id)
        data = rsp.json()
        self.logger.info(json.dumps(data, indent=4, sort_keys=True))
        return data


class PlatformPlot(PlatformBase):
    """ Plot a Gantt chart of an analysis's sub-tasks, with a status summary, via the Oasis Platform API
    """
    step_params = PlatformBase.step_params + [
        {'name': 'analysis_id', 'type': int, 'required': True, 'help': 'API `id` of an analysis to plot'},
        {'name': 'output_file', 'flag': '-o', 'is_path': True, 'pre_exist': False,
            'help': 'Path to write the plot PNG to, default is "./analysis-<id>-gantt.png"'},
    ]

    STATUS_COLORS = {
        "COMPLETED": "#1baf7a",
        "STARTED": "#eda100",
        "QUEUED": "#898781",
        "PENDING": "#c3c2b7",
        "ERROR": "#e34948",
        "CANCELLED": "#4a4a48",
    }
    STATUS_ORDER = ['COMPLETED', 'STARTED', 'QUEUED', 'PENDING', 'ERROR', 'CANCELLED']

    LEGACY_OUTPUT_LABELS = {
        'summarycalc': 'Summary calc',
        'eltcalc': 'ELT',
        'aalcalc': 'AAL',
        'aalcalcmeanonly': 'AAL (mean only)',
        'pltcalc': 'PLT',
        'lec_output': 'LEC',
    }

    ORD_OUTPUT_LABELS = {
        'elt_sample': 'SELT',
        'elt_quantile': 'QELT',
        'elt_moment': 'MELT',
        'plt_sample': 'SPLT',
        'plt_quantile': 'QPLT',
        'plt_moment': 'MPLT',
        'alt_period': 'PALT',
        'alt_meanonly': 'ALT (mean only)',
        'alct_convergence': 'ALCT',
        'ept_full_uncertainty_aep': 'EPT full unc. AEP',
        'ept_full_uncertainty_oep': 'EPT full unc. OEP',
        'ept_mean_sample_aep': 'EPT mean sample AEP',
        'ept_mean_sample_oep': 'EPT mean sample OEP',
        'ept_per_sample_mean_aep': 'EPT per-sample mean AEP',
        'ept_per_sample_mean_oep': 'EPT per-sample mean OEP',
        'psept_aep': 'PSEPT AEP',
        'psept_oep': 'PSEPT OEP',
        'return_period_file': 'RP file',
        'parquet_format': 'Parquet',
    }

    def run(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.patches import Patch
        except ImportError:
            raise OasisException(
                "matplotlib is required for 'oasislmf api plot' but is not installed. "
                "Install with: pip install oasislmf[extra]"
            )

        analysis = self.server.analyses.get(self.analysis_id).json()
        sub_tasks = self.server.analyses.sub_task_list(self.analysis_id).json()
        try:
            settings = self.server.analyses.settings.get(self.analysis_id).json()
        except HTTPError:
            settings = None
        output_file = self.output_file or f'analysis-{self.analysis_id}-gantt.png'

        self._plot(sub_tasks, analysis, settings, output_file, plt, mdates, Patch)
        self.logger.info(f'Wrote {output_file}')
        return output_file

    def _parse_ts(self, s):
        if not s:
            return None
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")

    def _fmt_duration(self, td):
        total_s = int(td.total_seconds())
        h, rem = divmod(total_s, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"

    def _queued_duration_line(self, analysis, sub_tasks):
        task_started = self._parse_ts(analysis.get('task_started'))
        pending_times = [self._parse_ts(t.get('pending_time')) for t in sub_tasks]
        pending_times = [t for t in pending_times if t is not None]
        if not pending_times or not task_started:
            return "Queued: n/a"
        return f"Queued before execution: {self._fmt_duration(task_started - min(pending_times))}"

    def _enabled_outputs(self, summary):
        labels = [label for key, label in self.LEGACY_OUTPUT_LABELS.items() if summary.get(key)]
        ord_output = summary.get('ord_output') or {}
        labels += [label for key, label in self.ORD_OUTPUT_LABELS.items() if ord_output.get(key)]
        return labels

    def _describe_summaries(self, summaries):
        parts = []
        for s in summaries:
            fields = s.get('oed_fields') or []
            group = ', '.join(fields) if fields else 'all'
            part = f"#{s.get('id')} (by: {group})"
            outputs = self._enabled_outputs(s)
            part += f" [{', '.join(outputs)}]" if outputs else " [no outputs selected]"
            parts.append(part)
        return '; '.join(parts) or '—'

    def _settings_rows(self, settings):
        if not settings:
            return []
        rows = [('Samples', str(settings.get('number_of_samples', '—')))]

        model_settings = settings.get('model_settings') or {}
        rows.append(('Event set', str(model_settings.get('event_set', '—'))))
        rows.append(('Event occurrence set', str(model_settings.get('event_occurrence_id', '—'))))

        for perspective, label in [('gul', 'GUL'), ('il', 'IL'), ('ri', 'RI'), ('rl', 'RL')]:
            summaries = settings.get(f'{perspective}_summaries') or []
            if settings.get(f'{perspective}_output') and summaries:
                rows.append((f'{label} summaries', self._describe_summaries(summaries)))
            else:
                rows.append((f'{label} summaries', 'Disabled'))

        return rows

    def _plot(self, sub_tasks, analysis, settings, output_file, plt, mdates, Patch):
        sub_tasks = sorted(sub_tasks, key=lambda t: t["id"])

        starts, ends, statuses, names = [], [], [], []
        now = max(
            (self._parse_ts(t["end_time"]) for t in sub_tasks if t.get("end_time")),
            default=datetime.utcnow(),
        )
        for t in sub_tasks:
            s = self._parse_ts(t["start_time"]) or self._parse_ts(t["queue_time"]) or self._parse_ts(t["pending_time"])
            e = self._parse_ts(t["end_time"]) or (now if t["status"] in ("STARTED",) else s)
            starts.append(s)
            ends.append(e)
            statuses.append(t["status"])
            names.append(f"{t['id']}: {t['name']}")

        has_summary = analysis is not None
        settings_rows = self._settings_rows(settings)
        has_settings = bool(settings_rows)

        summary_h = 1.7
        settings_h = 0.38 * (len(settings_rows) + 1) + 0.25 if has_settings else 0
        gantt_h = 0.32 * len(sub_tasks) + 1.0

        if has_summary:
            height_ratios = [summary_h] + ([settings_h] if has_settings else []) + [gantt_h]
            fig, axes = plt.subplots(
                len(height_ratios), 1,
                figsize=(13, sum(height_ratios)),
                gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
            )
            ax_summary = axes[0]
            ax_settings = axes[1] if has_settings else None
            ax_gantt = axes[-1]
        else:
            fig, ax_gantt = plt.subplots(1, 1, figsize=(13, 0.35 * len(sub_tasks) + 1.5))
            ax_summary = None
            ax_settings = None

        y = range(len(sub_tasks))
        for i, (s, e, status) in enumerate(zip(starts, ends, statuses)):
            color = self.STATUS_COLORS.get(status, "#000000")
            width = e - s
            ax_gantt.barh(i, width, left=s, height=0.6, color=color, edgecolor="none")
            dur_s = width.total_seconds()
            label = status if dur_s == 0 else f"{int(dur_s // 60)}m {int(dur_s % 60)}s"
            ax_gantt.text(e, i, f"  {label}", va="center", ha="left", fontsize=8, color="#52514e")

        ax_gantt.set_yticks(list(y))
        ax_gantt.set_yticklabels(names, fontsize=8)
        ax_gantt.invert_yaxis()
        ax_gantt.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax_gantt.set_xlabel("Time (UTC)")
        ax_gantt.set_title("Sub-task timeline")
        ax_gantt.grid(True, axis="x", alpha=0.3)

        legend_handles = [
            Patch(color=c, label=s) for s, c in self.STATUS_COLORS.items()
            if s in statuses
        ]
        ax_gantt.legend(handles=legend_handles, loc="lower right", fontsize=8)

        if has_summary:
            ax_summary.axis("off")
            sc = analysis.get("status_count", {})
            order = self.STATUS_ORDER
            counts = [sc.get(k, 0) for k in order if sc.get(k, 0) or k in statuses]
            labels = [k for k in order if sc.get(k, 0) or k in statuses]
            colors = [self.STATUS_COLORS[k] for k in labels]

            left = 0.0
            total = sum(counts) or 1
            bar_ax = ax_summary.inset_axes([0.0, 0.55, 1.0, 0.3])
            for label, count, color in zip(labels, counts, colors):
                frac = count / total
                bar_ax.barh(0, frac, left=left, color=color, height=1.0)
                if frac > 0.03:
                    bar_ax.text(left + frac / 2, 0, str(count), ha="center", va="center",
                                fontsize=9, color="white", fontweight="bold")
                left += frac
            bar_ax.set_xlim(0, 1)
            bar_ax.axis("off")

            title = (
                f"Analysis {analysis.get('id')} — {analysis.get('name', '')}   "
                f"status: {analysis.get('status')}"
            )
            events = (
                f"Events: {analysis.get('num_events_complete', 0):,} / "
                f"{analysis.get('num_events_total', 0):,}"
            )
            queued_line = self._queued_duration_line(analysis, sub_tasks)
            sub_task_line = "  ".join(f"{lb}: {c}" for lb, c in zip(labels, counts))
            generated_line = f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
            ax_summary.text(0.0, 0.95, title, fontsize=12, fontweight="bold", va="top")
            ax_summary.text(1.0, 0.95, generated_line, fontsize=8, va="top", ha="right", color="#898781")
            ax_summary.text(0.0, 0.4, events, fontsize=9, va="top", color="#52514e")
            ax_summary.text(0.0, 0.25, queued_line, fontsize=9, va="top", color="#52514e")
            ax_summary.text(0.0, 0.1, sub_task_line, fontsize=9, va="top", color="#52514e")

        if has_settings:
            ax_settings.axis("off")
            table = ax_settings.table(
                cellText=settings_rows,
                colLabels=["Run setting", "Value"],
                cellLoc="left",
                colLoc="left",
                loc="upper left",
                colWidths=[0.25, 0.7],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.4)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor("#e1e0d9")
                if row == 0:
                    cell.set_facecolor("#f0efec")
                    cell.set_text_props(fontweight="bold")
                elif row > 0 and settings_rows[row - 1][1] == 'Disabled':
                    cell.set_text_props(color="#898781", style="italic")

        max_label_chars = max((len(n) for n in names), default=0)
        left_margin = min(0.35, 0.03 + max_label_chars * 0.0047)
        fig.subplots_adjust(left=left_margin, right=0.97, top=0.98, bottom=0.05, hspace=0.12)
        fig.savefig(output_file, dpi=150)
        plt.close(fig)
