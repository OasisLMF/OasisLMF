__all__ = [
    '_'
]


import getpass
import io
import json
import sys
from tabulate import tabulate

from requests.exceptions import HTTPError, ConnectionError

from ...platform.client import APIClient
from ...utils.exceptions import OasisException
from ...utils.defaults import API_EXAMPLE_AUTH

from ..base import ComputationStep


class PlatformBase(ComputationStep):
    """
    - desc here -   
    """
    step_params = [
        {'name': 'server_login_json', 'is_path': True, 'pre_exist': False, 'help': 'Source location CSV file path'},
        {'name': 'server_url', 'default': 'http://localhost:8000', 'help':''},
        {'name': 'server_version', 'default': 'V1'},
    ]

    def __init__(self, **kwargs):
        # Run init & start connection with server 
        super().__init__(**kwargs)
        self.server = self.open_connection()


    def load_credentials(self, login_arg):
        """
        Load credentials from JSON file or Prompt for username / password

        Options:
            1. '--server-login ./APIcredentials.json'

            2. Load credentials from default config file
            '-C oasislmf.json'
        """
        if isinstance(login_arg, str):
            with io.open(login_arg, encoding='utf-8') as f:
                return json.load(f)
        elif isinstance(login_arg, dict):
            if {'password', 'username'} <= {k for k in login_arg.keys()}:
                return login_arg
        else:
            self.logger.info('API Login:')

        try:
            api_login = {}
            api_login['username'] = input('Username: ')
            api_login['password'] = getpass.getpass('Password: ')
            return api_login
        except KeyboardInterrupt as e:
            self.logger.error('\nFailed to get API login details:')
            self.logger.error(e)
            sys.exit(1)


    def open_connection(self):
        try:
            ## If no password given try the reference example
            return APIClient(
                api_url=self.server_url,
                api_ver=self.server_version,
                username=API_EXAMPLE_AUTH['user'],
                password=API_EXAMPLE_AUTH['pass'],
            )
        except OasisException as e:
            if isinstance(e.original_exception, HTTPError):
                ## Prompt for password and try to re-autehnticate
                self.logger.info(f"-- Authentication Required --")
                credentials = self.load_credentials(self.server_login_json)
                self.logger.info(f'Connecting to - {self.server_url}')
                return APIClient(
                    api_url=self.server_url,
                    api_ver=self.server_version,
                    username=credentials['username'],
                    password=credentials['password'],
                )
            elif isinstance(e.original_exception, ConnectionError):
                self.logger.info('API Connection error to "{}"'.format(self.server_url))
                self.logger.debug(e)
                sys.exit(1)
        else:
            self.logger.error('Unhandled error:')
            self.logger.debug(e)
            sys.exit(1)


    def print_endpoint(self, attr, items):
        endpoint_obj = getattr(self.server, attr)
        table_data = dict()

        for i in items:
            table_data[i] = list()

        self.logger.info(f'\nList of available {attr}:')
        for m in endpoint_obj.get().json():

            for k in table_data:
                # will have link+data if dict returned
                if isinstance(m[k], dict):
                    table_data[k].append('Yes')

                # If none then no data     
                elif m[k] == None:    
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
        self.logger.info(tabulate(table_data, headers=items, tablefmt='psql'))


    def run(self):
        """ Add the logic for platform API interaction here 
        """
        raise NotImplemented(f'Methode run must be implemented, this class handles base server connection')




class PlatformGetDetails(PlatformBase):
    """ Return status and details from an Oasis Platform API server 
    """
    step_params = PlatformBase.step_params + [
        {'name': 'models', 'flag': '-m', 'type' :int, 'nargs':'+', 'help': '', 'default': None},
        {'name': 'portfolios', 'flag': '-p', 'type' :int, 'nargs':'+', 'help': '', 'default': None},
        {'name': 'analyses', 'flag': '-a', 'type' :int, 'nargs':'+', 'help': '', 'default': None},
    ]


    def run(self):

        # Default to printing summary of API status 
        if not any([self.models, self.portfolios, self.analyses]):
            self.print_endpoint('models', ['id', 'supplier_id', 'model_id', 'version_id'])
            self.print_endpoint('portfolios', ['id', 'name', 'location_file', 'accounts_file', 'reinsurance_info_file', 'reinsurance_scope_file'])
            self.print_endpoint('analyses', ['id', 'name', 'model', 'portfolio', 'status', 'output_file'])

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





class PlatformRun(PlatformBase):
    """ End to End - run model via the Oasis Platoform API
    """
    step_params = PlatformBase.step_params + [
        {'name': 'model_id',     'type' :int, 'help': 'API `id` of a model to run an analysis with'},
        {'name': 'portfolio_id', 'type' :int, 'help': 'API `id` of a portfolio to run an analysis with'},
        {'name': 'analysis_id',  'type' :int, 'help': 'API `id` of an analysis to run'},

        {'name': 'output_dir',        'flag':'-o', 'is_path': True, 'pre_exist': True, 'help': 'Output data directory for results data (absolute or relative file path)', 'default':'./'},
        {'name': 'analysis_settings_json', 'flag':'-a', 'is_path': True, 'pre_exist': True,  'help': 'Analysis settings JSON file path'},
        {'name': 'oed_location_csv',       'flag':'-x', 'is_path': True, 'pre_exist': True,  'help': 'Source location CSV file path'},
        {'name': 'oed_accounts_csv',       'flag':'-y', 'is_path': True, 'pre_exist': True,  'help': 'Source accounts CSV file path'},
        {'name': 'oed_info_csv',           'flag':'-i', 'is_path': True, 'pre_exist': True,  'help': 'Reinsurance info. CSV file path'},
        {'name': 'oed_scope_csv',          'flag':'-s', 'is_path': True, 'pre_exist': True,  'help': 'Reinsurance scope CSV file path'},
    ]


    def select_id(self, attr):
        while True:
            try:
                value = int(input(f'Select {attr}: '))
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

    


    def run(self):
        
        # if given run that analysis, with new settings file (If given) 
        if self.analysis_id:
            try:
                status = self.server.analyses.status(self.analysis_id)
                if status in ['RUN_QUEUED','RUN_STARTED']:
                    self.server.cancel_analysis(self.analysis_id)
                elif status in ['INPUTS_GENERATION_QUEUED', 'INPUTS_GENERATION_STARTED']
                    self.server.cancel_generate(self.analysis_id)

                self.server.run_generate(self.analysis_id)
                self.server.run_analysis(self.analysis_id, self.analysis_settings_json)
                self.server.download_output(self.analysis_id, output_dir)
            except HTTPError as e:
                self.logger.error('Error running analysis ({}) - {}'.format(
                    self.analysis_id, e))
                sys.exit(1)


         # If not given an analsis then we need to select a model + portfolio 
         # when no model is selected prompt user for choice of installed models
         if not self.model_id:

         if self.portfolio_id


#        # Upload files
#        path_location = inputs.get('oed_location_csv')
#        path_account = inputs.get('oed_accounts_csv')
#        path_info = inputs.get('oed_info_csv')
#        path_scope = inputs.get('oed_scope_csv')
#
#        portfolio = api.upload_inputs(
#            portfolio_id=None,
#            location_fp=path_location,
#            accounts_fp=path_account,
#            ri_info_fp=path_info,
#            ri_scope_fp=path_scope,
#        )
#
#        model_id = inputs.get('model_id')
#        if not model_id:
#            avalible_models = api.models.get().json()
#
#            if len(avalible_models) > 1:
#                selected_model = self._select_model(avalible_models)
#            elif len(avalible_models) == 1:
#                selected_model = avalible_models[0]
#            else:
#                raise OasisException(
#                    'No models found in API: {}'.format(inputs.get('api_server_url'))
#                )
#
#            model_id = selected_model['id']
#            self.logger.info('Running model:')
#            self.logger.info(json.dumps(selected_model, indent=4))
#
#        # Create new analysis
#        path_settings = inputs.get('analysis_settings_json')
#        if not path_settings:
#            self.logger.error('analysis settings: Not found')
#            return False
#        analysis = api.create_analysis(
#            portfolio_id=portfolio['id'],
#            model_id=model_id,
#            analysis_settings_fp=path_settings,
#        )
#        self.logger.info('Loaded analysis settings:')
#        self.logger.info(json.dumps(analysis, indent=4))
#
#        # run and poll
#        api.run_generate(analysis['id'], poll_interval=3)
#        api.run_analysis(analysis['id'], poll_interval=3)
#
#        # Download Outputs
#        api.download_output(
#            analysis_id=analysis['id'],
#            download_path=inputs.get('output_dir'),
#            overwrite=True,
#            clean_up=False
#        )


#class PlatformGet(platformbase):
    """ todo - download file(s) from the api
    """

#class PlatformPut(platformbase):
    """ todo - create obj or Upload files
    """



