__all__ = [
    'APIClient',
    'ApiEndpoint',
    'API_analyses',
    'API_models',
    'API_portfolios',
    'FileEndpoint',
]

import io
import json
import logging
import os
import sys
import tarfile
import time

import pandas as pd

from tqdm import tqdm
from requests_toolbelt import MultipartEncoder
from requests.exceptions import (
    HTTPError,
)
from .session import APISession


class ApiEndpoint(object):
    """
    Used to Implement the default requests common to all Oasis API
    End points.
    """
    def __init__(self, session, url_endpoint, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.session = session
        self.url_endpoint = url_endpoint

    def create(self, data):
        return self.session.post(self.url_endpoint, json=data)

    def get(self, ID=None):
        if ID:
            return self.session.get('{}{}/'.format(self.url_endpoint, ID))
        return self.session.get(self.url_endpoint)

    def delete(self, ID):
        return self.session.delete('{}{}/'.format(self.url_endpoint, ID))

    def search(self, metadata={}):
        search_string = ""
        for key in metadata:
            if not search_string:
                search_string = '?{}={}'.format(key, metadata[key])
            else:
                search_string += '&{}={}'.format(key, metadata[key])
        return self.session.get('{}{}'.format(self.url_endpoint, search_string))


class JsonEndpoint(object):
    """
    Used for JSON data End points.
    """
    def __init__(self, session, url_endpoint, url_resource, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.session = session
        self.url_endpoint = url_endpoint
        self.url_resource = url_resource

    def _build_url(self, ID):
        return '{}{}/{}'.format(
            self.url_endpoint,
            ID,
            self.url_resource
        )

    def get(self, ID):
        return self.session.get(self._build_url(ID))

    def post(self, ID, data):
        return self.session.post(self._build_url(ID), json=data)

    def delete(self, ID):
        return self.session.delete(self._build_url(ID))

    def download(self, ID, file_path, overwrite=True):
        abs_fp = os.path.realpath(os.path.expanduser(file_path))

        # Check and create base dir
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        # Check if file exists
        if os.path.exists(abs_fp) and not overwrite:
            if overwrite:
                os.remove(abs_fp)
            else:
                error_message = 'Local file alreday exists: {}'.format(abs_fp)
                raise IOError(error_message)

        with io.open(abs_fp, 'w', encoding='utf-8') as f:
            r = self.get(ID)
            f.write(json.dumps(r.json(), ensure_ascii=False, indent=4))
        return r

class FileEndpoint(object):
    """
    File Resources Endpoint for Upload / Downloading
    """
    def __init__(self, session, url_endpoint, url_resource, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        self.session = session
        self.url_endpoint = url_endpoint
        self.url_resource = url_resource

    def _build_url(self, ID):
        return '{}{}/{}'.format(
            self.url_endpoint,
            ID,
            self.url_resource
        )

    def upload(self, ID, file_path, content_type='text/csv'):
        try:
            r = self.session.upload(self._build_url(ID), file_path, content_type)
            return r
        except HTTPError as e:
            err_msg = 'File upload Failed: file: {},  url: {}:'.format(file_path, self._build_url(ID))
            self.session.unrecoverable_error(e, err_msg)
            sys.exit(1)

    def download(self, ID, file_path, overwrite=True, chuck_size=1024):
        abs_fp = os.path.realpath(os.path.expanduser(file_path))

        # Check and create base dir
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        # Check if file exists
        if os.path.exists(abs_fp) and not overwrite:
            if overwrite:
                os.remove(abs_fp)
            else:
                error_message = 'Local file alreday exists: {}'.format(abs_fp)
                raise IOError(error_message)

        with io.open(abs_fp, 'wb') as f:
            r = self.session.get(self._build_url(ID), stream=True)
            for chunk in r.iter_content(chunk_size=chuck_size):
                f.write(chunk)
            return r

    def get(self, ID):
        return self.session.get(self._build_url(ID))

    def get_dataframe(self, ID):
        '''
        Return file endpoint as dict of pandas Dataframes:

        either 'application/gzip': search and extract all csv
        or 'text/csv': return as dataframe
        '''
        r = self.get(ID)
        file_type = r.headers['Content-Type']
        if file_type not in ['text/csv', 'application/gzip']:
            self.logger.info(f'Unsupported filetype for Dataframe conversion: {file_type}')
        else:
            dataframes_list = {}
            if file_type == 'text/csv':
                dataframes_list[self.url_resource.strip('/')] = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
            if file_type == 'application/gzip':
                tar = tarfile.open(fileobj=io.BytesIO(r.content))
                csv_files = [f for f in tar.getmembers() if '.csv' in f.name]
                for member in csv_files:
                    csv = tar.extractfile(member)
                    dataframes_list[os.path.basename(member.name)] = pd.read_csv(csv)
            return dataframes_list

    def post(self, ID, data_object, content_type='application/json'):
        m = MultipartEncoder(fields={'file': ('data', data_object, content_type)})
        r = self.session.post(
            self._build_url(ID),
            data=m,
            headers={'Content-Type': m.content_type}
        )
        if not r.ok:
            err_msg = 'Data_Object upload Failed'
            self.logger.error(err_msg)
        return r

    def post_dataframe(self, ID, data_frame):
        csv_buffer = io.StringIO()
        data_frame.to_csv(csv_buffer)
        return self.post(ID, data_object=csv_buffer, content_type='text/csv')

    def delete(self, ID):
        return self.session.delete(self._build_url(ID))


class API_models(ApiEndpoint):
    def __init__(self, session, url_endpoint):
        super(API_models, self).__init__(session, url_endpoint)
        self.resource_file = FileEndpoint(self.session, self.url_endpoint, 'resource_file/')
        self.settings = JsonEndpoint(self.session, self.url_endpoint, 'settings/')
        self.versions = JsonEndpoint(self.session, self.url_endpoint, 'versions/')

        # Platform 2.0 only (Check might be needed here)
        self.chunking_configuration = JsonEndpoint(self.session, self.url_endpoint, 'chunking_configuration/')
        self.scaling_configuration = JsonEndpoint(self.session, self.url_endpoint, 'scaling_configuration/')

    def data_files(self, ID):
        return self.session.get('{}{}/data_files'.format(self.url_endpoint, ID))

    def create(self, supplier_id, model_id, version_id, data_files=[]):
        if isinstance(data_files, list):
            df_ids = data_files
        elif isinstance(data_files, (int, str)):
            df_ids = [data_files]
        else:
            self.logger.warn('data_files, must be of type list(), int() or str()')

        data = {"supplier_id": supplier_id,
                "model_id": model_id,
                "version_id": version_id,
                "data_files": df_ids}
        return self.session.post(self.url_endpoint, json=data)

    def update(self, ID, supplier_id, model_id, version_id, data_files=[]):
        if isinstance(data_files, list):
            df_ids = data_files
        elif isinstance(data_files, (int, str)):
            df_ids = [data_files]
        else:
            self.logger.warn('data_files, must be of type list(), int() or str()')

        data = {"supplier_id": supplier_id,
                "model_id": model_id,
                "version_id": version_id,
                "data_files": df_ids}
        return self.session.put('{}{}/'.format(self.url_endpoint, ID), json=data)


class API_portfolios(ApiEndpoint):

    def __init__(self, session, url_endpoint):
        super(API_portfolios, self).__init__(session, url_endpoint)
        self.accounts_file = FileEndpoint(self.session, self.url_endpoint, 'accounts_file/')
        self.location_file = FileEndpoint(self.session, self.url_endpoint, 'location_file/')
        self.reinsurance_info_file = FileEndpoint(self.session, self.url_endpoint, 'reinsurance_info_file/')
        self.reinsurance_scope_file = FileEndpoint(self.session, self.url_endpoint, 'reinsurance_scope_file/')
        self.storage_links = JsonEndpoint(self.session, self.url_endpoint, 'storage_links/')

    def create(self, name):
        data = {"name": name}
        return self.session.post(self.url_endpoint, json=data)

    def update(self, ID, name):
        data = {"name": name}
        return self.session.put('{}{}/'.format(self.url_endpoint, ID), json=data)

    def create_analyses(self, ID, name, model_id):
        """ Create new analyses from Exisiting portfolio
        """
        data = {"name": name,
                "model": model_id}
        return self.session.post('{}{}/create_analysis/'.format(self.url_endpoint, ID), json=data)


class API_datafiles(ApiEndpoint):

    def __init__(self, session, url_endpoint):
        super(API_datafiles, self).__init__(session, url_endpoint)
        self.content = FileEndpoint(self.session, self.url_endpoint, 'content/')

    def create(self, file_description, linked_models=[]):
        data = {"file_description": file_description}
        return self.session.post(self.url_endpoint, json=data)

    def update(self, ID, file_description, linked_models=[]):
        data = {"file_description": file_description}
        return self.session.put('{}{}/'.format(self.url_endpoint, ID), json=data)


class API_analyses(ApiEndpoint):

    def __init__(self, session, url_endpoint):
        super(API_analyses, self).__init__(session, url_endpoint)
        self.lookup_errors_file = FileEndpoint(self.session, self.url_endpoint, 'lookup_errors_file/')
        self.lookup_success_file = FileEndpoint(self.session, self.url_endpoint, 'lookup_success_file/')
        self.lookup_validation_file = FileEndpoint(self.session, self.url_endpoint, 'lookup_validation_file/')
        self.summary_levels_file = FileEndpoint(self.session, self.url_endpoint, 'summary_levels_file/')
        self.input_file = FileEndpoint(self.session, self.url_endpoint, 'input_file/')
        self.input_generation_traceback_file = FileEndpoint(self.session, self.url_endpoint, 'input_generation_traceback_file/')
        self.output_file = FileEndpoint(self.session, self.url_endpoint, 'output_file/')
        self.run_traceback_file = FileEndpoint(self.session, self.url_endpoint, 'run_traceback_file/')
        self.run_log_file = FileEndpoint(self.session, self.url_endpoint, 'run_log_file/')
        self.settings_file = FileEndpoint(self.session, self.url_endpoint, 'settings_file/')
        self.settings = JsonEndpoint(self.session, self.url_endpoint, 'settings/')

    def create(self, name, portfolio_id, model_id, data_files=[]):
        if isinstance(data_files, list):
            df_ids = data_files
        elif isinstance(data_files, (int, str)):
            df_ids = [data_files]
        else:
            self.logger.warn('data_files, must be of type list(), int() or str()')

        data = {"name": name,
                "portfolio": portfolio_id,
                "model": model_id,
                "complex_model_data_files": df_ids}
        return self.session.post(self.url_endpoint, json=data)

    def update(self, ID, name, portfolio_id, model_id, data_files=[]):
        if isinstance(data_files, list):
            df_ids = data_files
        elif isinstance(data_files, (int, str)):
            df_ids = [data_files]
        else:
            self.logger.warn('data_files, must be of type list(), int() or str()')

        data = {"name": name,
                "portfolio": portfolio_id,
                "model": model_id,
                "complex_model_data_files": df_ids}
        return self.session.put('{}{}/'.format(self.url_endpoint, ID), json=data)

    def status(self, ID):
        return self.get(ID).json()['status']

    def generate(self, ID):
        return self.session.post('{}{}/generate_inputs/'.format(self.url_endpoint, ID), json={})

    def run(self, ID):
        return self.session.post('{}{}/run/'.format(self.url_endpoint, ID), json={})

    def cancel_analysis_run(self, ID):
        return self.session.post('{}{}/cancel_analysis_run/'.format(self.url_endpoint, ID), json={})

    def cancel_generate_inputs(self, ID):
        return self.session.post('{}{}/cancel_generate_inputs/'.format(self.url_endpoint, ID), json={})

    def cancel(self, ID):
        return self.session.post('{}{}/cancel/'.format(self.url_endpoint, ID), json={})

    def copy(self, ID):
        return self.session.post('{}{}/copy/'.format(self.url_endpoint, ID), json={})

    def data_files(self, ID):
        return self.session.get('{}{}/data_files'.format(self.url_endpoint, ID))

    def storage_links(self, ID):
        return self.session.get('{}{}/storage_links'.format(self.url_endpoint, ID))

# --- API Main Client ------------------------------------------------------- #


class APIClient(object):
    def __init__(self, api_url='http://localhost:8000', api_ver='V1', username='admin', password='password', timeout=25, logger=None, **kwargs):
        self.logger = logger or logging.getLogger(__name__)

        self.api = APISession(api_url, username, password, timeout, **kwargs)
        self.models = API_models(self.api, '{}{}/models/'.format(self.api.url_base, api_ver))
        self.portfolios = API_portfolios(self.api, '{}{}/portfolios/'.format(self.api.url_base, api_ver))
        self.analyses = API_analyses(self.api, '{}{}/analyses/'.format(self.api.url_base, api_ver))
        self.data_files = API_datafiles(self.api, '{}{}/data_files/'.format(self.api.url_base, api_ver))

    def oed_peril_codes(self):
        return self.api.get('{}oed_peril_codes/'.format(self.api.url_base))

    def server_info(self):
        return self.api.get('{}server_info/'.format(self.api.url_base))

    def healthcheck(self):
        return self.api.get('{}healthcheck/'.format(self.api.url_base))

    def upload_inputs(self, portfolio_name=None, portfolio_id=None,
                      location_fp=None, accounts_fp=None, ri_info_fp=None, ri_scope_fp=None):

        if not portfolio_name:
            portfolio_name = time.strftime("Portfolio_%d%m%Y-%H%M%S")

        try:
            if portfolio_id:
                self.logger.info('Updating exisiting portfolio')
                portfolio = self.portfolios.update(portfolio_id, portfolio_name)
            else:
                self.logger.info('Creating portfolio')
                portfolio = self.portfolios.create(portfolio_name)
                portfolio_id = portfolio.json()['id']

            # Check or create portfolio
            if not portfolio.ok:
                err_msg = "Failed to find matching `portfolio_id = {}`".format(portfolio_id)
                self.logger.error(err_msg)

            # Upload exposure
            if location_fp:
                self.portfolios.location_file.upload(portfolio_id, location_fp)
                self.logger.info("File uploaded: {}".format(location_fp))
            if accounts_fp:
                self.portfolios.accounts_file.upload(portfolio_id, accounts_fp)
                self.logger.info("File uploaded: {}".format(accounts_fp))
            if ri_info_fp:
                self.portfolios.reinsurance_info_file.upload(portfolio_id, ri_info_fp)
                self.logger.info("File uploaded: {}".format(ri_info_fp))
            if ri_scope_fp:
                self.portfolios.reinsurance_scope_file.upload(portfolio_id, ri_scope_fp)
                self.logger.info("File uploaded: {}".format(ri_scope_fp))
            return portfolio.json()
        except HTTPError as e:
            self.api.unrecoverable_error(e, 'upload_inputs: failed')
            sys.exit(1)

    def upload_settings(self, analyses_id, settings):
        """
        Upload an analyses run settings to an API

        Method to post JSON data or upload a settings file containing JSON data

        Parameters
        ----------
        :param analyses_id: Analyses settings {id} from, `v1/analyses/{id}/settings`
        :type analyses_id: int

        :param settings: Either a valid filepath or dictionary holding the settings
        :type settings: [str, dict]

        :return:
        :rtype None
        """
        if isinstance(settings, dict):
            self.analyses.settings.post(analyses_id, settings)
            self.logger.info("Settings JSON uploaded: {}".format(settings))

        elif os.path.isfile(str(settings)):
            with io.open(settings, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.analyses.settings.post(analyses_id, data)
            self.logger.info("Settings JSON uploaded: {}".format(settings))
        else:
            raise TypeError("'settings': not a valid filepath or dictionary")

    def create_analysis(self, portfolio_id, model_id, analysis_name=None, analysis_settings_fp=None):
        try:
            if not analysis_name:
                analysis_name = time.strftime("Analysis_%d%m%Y-%H%M%S")

            analyses = self.analyses.create(analysis_name, portfolio_id, model_id).json()
            if analysis_settings_fp:
                self.upload_settings(analyses['id'], analysis_settings_fp)

            return analyses
        except HTTPError as e:
            self.api.unrecoverable_error(e, 'create_analysis: failed')
            sys.exit(1)

    def run_generate(self, analysis_id, poll_interval=5):
        """
        Generates the inputs for the analysis based on the portfolio.
        The analysis must have one of the following statuses, `NEW`, `INPUTS_GENERATION_ERROR`,
        `INPUTS_GENERATION_CANCELLED`, `READY`, `RUN_COMPLETED`, `RUN_CANCELLED` or
        `RUN_ERROR`.
        """

        try:
            r = self.analyses.generate(analysis_id)
            analysis = r.json()
            self.logger.info('Inputs Generation: Starting (id={})'.format(analysis_id))
            logged_queued = None
            logged_running = None

            while True:
                if analysis['status'] in ['READY']:
                    self.logger.info('Inputs Generation: Complete (id={})'.format(analysis_id))
                    return True

                elif analysis['status'] in ['INPUTS_GENERATION_CANCELLED']:
                    self.logger.info('Input Generation: Cancelled (id={})'.format(analysis_id))
                    return False

                elif analysis['status'] in ['INPUTS_GENERATION_ERROR']:
                    self.logger.info('Input Generation: Failed (id={})'.format(analysis_id))
                    error_trace = self.analyses.input_generation_traceback_file.get(analysis_id).text
                    self.logger.error(error_trace)
                    return False

                elif analysis['status'] in ['INPUTS_GENERATION_QUEUED']:
                    if not logged_queued:
                        logged_queued = True
                        self.logger.info('Input Generation: Queued (id={})'.format(analysis_id))

                    time.sleep(poll_interval)
                    r = self.analyses.get(analysis_id)
                    analysis = r.json()
                    continue

                elif analysis['status'] in ['INPUTS_GENERATION_STARTED']:
                    if not logged_running:
                        logged_running = True
                        self.logger.info('Input Generation: Executing (id={})'.format(analysis_id))

                    if 'sub_task_statuses' in analysis:
                        with tqdm(total=len(analysis['sub_task_statuses']),
                                  unit=' sub_task',
                                  desc='Input Generation') as pbar:

                            completed = []
                            while len(completed) < len(analysis['sub_task_statuses']):
                                analysis = self.analyses.get(analysis_id).json()
                                completed = [tsk for tsk in analysis['sub_task_statuses'] if tsk['status'] == 'COMPLETED']
                                pbar.update(len(completed) - pbar.n)

                                # Exit conditions
                                if ('_CANCELLED' in analysis['status']) or ('_ERROR' in analysis['status']):
                                    break
                                elif 'READY' in analysis['status']:
                                    pbar.update(pbar.total - pbar.n)
                                    break

                                time.sleep(poll_interval)
                    else:
                        time.sleep(poll_interval)
                        analysis = self.analyses.get(analysis_id).json()

                    continue

                else:
                    err_msg = "Input Generation: Unknown State'{}'".format(analysis['status'])
                    self.logger.error(err_msg)
                    sys.exit(1)
        except HTTPError as e:
            self.api.unrecoverable_error(e, 'run_generate: failed')
            sys.exit(1)

    def run_analysis(self, analysis_id, analysis_settings_fp=None, poll_interval=5):
        """
        Runs all the analysis. The analysis must have one of the following
        statuses, `NEW`, `RUN_COMPLETED`, `RUN_CANCELLED` or
        `RUN_ERROR`
        """

        try:
            if analysis_settings_fp:
                self.upload_settings(analysis_id, analysis_settings_fp)

            analysis = self.analyses.run(analysis_id).json()
            self.logger.info('Analysis Run: Starting (id={})'.format(analysis_id))
            logged_queued = None
            logged_running = None

            while True:
                if analysis['status'] in ['RUN_COMPLETED']:
                    self.logger.info('Analysis Run: Complete (id={})'.format(analysis_id))
                    return True

                elif analysis['status'] in ['RUN_CANCELLED']:
                    self.logger.info('Analysis Run: Cancelled (id={})'.format(analysis_id))
                    return False

                elif analysis['status'] in ['RUN_ERROR']:
                    self.logger.error('Analysis Run: Failed (id={})'.format(analysis_id))
                    error_trace = self.analyses.run_traceback_file.get(analysis_id).text
                    self.logger.error(error_trace)
                    return False

                elif analysis['status'] in ['RUN_QUEUED']:
                    if not logged_queued:
                        logged_queued = True
                        self.logger.info('Analysis Run: Queued (id={})'.format(analysis_id))

                    time.sleep(poll_interval)
                    r = self.analyses.get(analysis_id)
                    analysis = r.json()
                    continue

                elif analysis['status'] in ['RUN_STARTED']:
                    if not logged_running:
                        logged_running = True
                        self.logger.info('Analysis Run: Executing (id={})'.format(analysis_id))

                    if 'sub_task_statuses' in analysis:
                        with tqdm(total=len(analysis['sub_task_statuses']),
                                  unit=' sub_task',
                                  desc='Analysis Run') as pbar:

                            completed = []
                            while len(completed) < len(analysis['sub_task_statuses']):
                                analysis = self.analyses.get(analysis_id).json()
                                completed = [tsk for tsk in analysis['sub_task_statuses'] if tsk['status'] == 'COMPLETED']
                                pbar.update(len(completed) - pbar.n)

                                # Exit conditions
                                if ('_CANCELLED' in analysis['status']) or ('_ERROR' in analysis['status']):
                                    break
                                elif 'COMPLETED' in analysis['status']:
                                    pbar.update(pbar.total - pbar.n)
                                    break
                                time.sleep(poll_interval)
                    else:
                        time.sleep(poll_interval)
                        analysis = self.analyses.get(analysis_id).json()
                    continue

                else:
                    err_msg = "Execution status in Unknown State: '{}'".format(analysis['status'])
                    self.logger.error(err_msg)
                    sys.exit(1)
        except HTTPError as e:
            self.api.unrecoverable_error(e, 'run_analysis: failed')
            sys.exit(1)

    def download_output(self, analysis_id, download_path, filename=None, clean_up=False, overwrite=True):
        if not filename:
            filename = 'analysis_{}_output'.format(analysis_id)
        try:
            output_file = os.path.join(download_path, filename + '.tar')
            self.analyses.output_file.download(ID=analysis_id, file_path=output_file, overwrite=overwrite)
            self.logger.info('Analysis Download output: filename={}, (id={})'.format(output_file, analysis_id))
            if clean_up:
                self.analyses.delete(analysis_id)
                self.analyses.output_file.delete(analysis_id)
                self.analyses.input_file.delete(analysis_id)
        except HTTPError as e:
            err_msg = 'Analysis Download output: Failed (id={})'.format(analysis_id)
            self.api.unrecoverable_error(e, err_msg)
            sys.exit(1)

    def cancel_generate(self, analysis_id):
        """
        Cancels a currently inputs generation. The analysis status must be `GENERATING_INPUTS`
        """
        try:
            self.analyses.cancel_generate_inputs(analysis_id)
            self.logger.info('Cancelled Input generation: (Id={})'.format(analysis_id))
            return True
        except HTTPError as e:
            self.api.unrecoverable_error(e, 'cancel_generate: Failed')
            return False

    def cancel_analysis(self, analysis_id):
        """
        Cancels a currently running analysis. The analysis must have one of the following
        statuses, `PENDING` or `STARTED`
        """
        try:
            self.analyses.cancel_analysis_run(analysis_id)
            self.logger.info('Cancelled analysis run: (Id={})'.format(analysis_id))
            return True
        except HTTPError as e:
            self.api.unrecoverable_error(e, 'cancel_analysis: Failed')
            return False
