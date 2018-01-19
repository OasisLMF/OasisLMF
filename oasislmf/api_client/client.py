"""
    Provides a simple class to build Oasis API clients.
"""
import logging
from itertools import chain

import six
from six.moves import urllib
from requests import RequestException

from oasislmf.utils.exceptions import OasisException

__all__ = [
    'OasisAPIClient'
]


# Python 2 standard imports
import csv
import inspect
import os
import subprocess
import tarfile
import time

# Python 3rd party imports
import jsonpickle
import requests
import shutilwhich

from requests_toolbelt.multipart.encoder import MultipartEncoder

# Oasis imports
from ..utils.log import oasis_log
from ..utils.status import STATUS_PENDING, STATUS_SUCCESS, STATUS_FAILURE


class OasisAPIClient(object):

    GUL_INPUTS_FILES = [
        'coverages',
        'gulsummaryxref',
        'items'
    ]

    IL_INPUTS_FILES = [
        'fm_policytc',
        'fm_profile',
        'fm_programme',
        'fm_xref',
        'fmsummaryxref'
    ]

    OPTIONAL_INPUTS_FILES = [
        'events'
    ]

    CONVERSION_TOOLS = {
        'coverages': 'coveragetobin',
        'events': 'evetobin',
        'fm_policytc': 'fmpolicytctobin',
        'fm_profile': 'fmprofiletobin',
        'fm_programme': 'fmprogrammetobin',
        'fm_xref': 'fmxreftobin',
        'fmsummaryxref': 'fmsummaryxreftobin',
        'gulsummaryxref': 'gulsummaryxreftobin',
        'items': "itemtobin"
    }

    TAR_FILE = "inputs.tar.gz"

    # Analysis settings files
    GENERAL_SETTINGS_FILE = "general_settings.csv"
    MODEL_SETTINGS_FILE = "model_settings.csv"
    GUL_SUMMARIES_FILE = "gul_summaries.csv"
    IL_SUMMARIES_FILE = "il_summaries.csv"

    DOWNLOAD_CHUCK_SIZE_IN_BYTES = 1024

    @oasis_log
    def __init__(self, oasis_api_url, logger=None):
        """
        Construct the client.
        Args:
            ``oasis_api_url``: the URL for the API.
            ``logger``: the logger.
        """

        self._oasis_api_url = oasis_api_url
        self._logger = logger or logging.getLogger()

    def build_uri(self, path):
        return urllib.parse.urljoin(self._oasis_api_url, path)

    def check_conversion_tools(self):
        # Check that the conversion tools are available
        for tool in six.itervalues(self.CONVERSION_TOOLS):
            self._logger.debug(shutilwhich.which(tool))
            if shutilwhich.which(str(tool)) is None:
                error_message = "Failed to find conversion tool: {}".format(tool)
                self._logger.error(error_message)
                raise OasisException(error_message)

        return True

    @oasis_log
    def upload_inputs_from_directory(self, directory, do_il=True, do_validation=False, do_clean=True):
        """
        Upload the CSV files from a specified directory.
        Args:
            ``directory`` (string): the directory containting the CSVs.
            ``do_il``: if True, require files for insured loss (IL) calculation.
            ``do_validation`` (bool): if True, validate the data intrgrity
            ``do_clean`` (bool): if True, remove the tar and bin files
        Returns:
            The location of the uploaded inputs.
        """
        try:
            self.check_inputs_directory(directory, do_il)
            if do_validation:
                self._validate_inputs(directory)
            self.create_binary_files(directory, do_il)
            self._create_tar_file(directory)

            self._logger.debug("Uploading inputs")
            tar_file = 'inputs.tar.gz'
            inputs_tar_to_upload = os.path.join(directory, tar_file)

            with open(inputs_tar_to_upload, 'rb') as f:
                inputs_multipart_data = MultipartEncoder(
                    fields={
                        'file': (tar_file, f, 'text/plain')
                    }
                )

                response = requests.post(
                    self.build_uri('/exposure'),
                    data=inputs_multipart_data,
                    headers={'Content-Type': inputs_multipart_data.content_type}
                )

            if not response.ok:
                self._logger.error(
                    "POST {} failed: {}, {}".format(response.request.url, str(response.status_code), str(response.json))
                )
                raise Exception("Failed to save exposure.")

            exposure_location = response.json()['exposures'][0]['location']
            self._logger.debug("Uploaded exposure. Location: " + exposure_location)

            # Return the location of the uploaded inputs
            return exposure_location
        finally:
            # Tidy up
            if do_clean:
                self._clean_directory(directory)

    @oasis_log
    def run_analysis(self, analysis_settings_json, input_location):
        """
        Starts an analysis running. Calling code will need to poll for
        completion or failure, and handle results.
        Args:
            ``analysis_setting_json`` (string): The analysis settings, as JSON.
            ``input_location``: The location of the inputs resource.
        Returns:
            The location of analysis status, to poll.
        """
        response = requests.post(
            self.build_uri("/analysis/" + input_location),
            json=analysis_settings_json,
        )

        if not response.ok:
            self._logger.error("POST {} failed: {}".format(response.request.url, str(response.status_code)))
            raise OasisException("Failed to start analysis")

        analysis_status_location = response.json()['location']
        self._logger.info("Analysis started")

        return analysis_status_location

    def get_analysis_status(self, analysis_status_location):
        response = requests.get(self.build_uri('/analysis_status/' + analysis_status_location))
        if response.status_code != 200:
            raise OasisException("GET analysis status failed: {}".format(response.status_code))

        self._logger.debug("Response: {}".format(response.json()))
        status = response.json()['status']
        self._logger.debug("Analysis status: " + status)

        if status == STATUS_FAILURE:
            error_message = "Analysis failed: {}".format(response.json()['message'])
            self._logger.error(error_message)
            raise OasisException(error_message)
        elif status == STATUS_SUCCESS:
            return status, response.json()['outputs_location']
        else:
            return status, ''

    @oasis_log
    def run_analysis_and_poll(self, analysis_settings_json, input_location, outputs_directory, analysis_poll_interval=5):
        """
        Run an analysis to completion or failure.
        Args:
            analysis_setting_json (string): The analysis settings, as JSON.
            input_location: The location of the inputs resource.
            outputs_directory: The local directory to save the outputs.
        Returns:
            The location of the uploaded inputs.
        """
        analysis_status_location = self.run_analysis(analysis_settings_json, input_location)

        self._logger.info("Analysis started")

        status, outputs_location = self.get_analysis_status(analysis_status_location)
        while status == STATUS_PENDING:
            time.sleep(analysis_poll_interval)
            status, outputs_location = self.get_analysis_status(analysis_status_location)

        self._logger.debug("Analysis completed")

        self._logger.debug("Downloading outputs")
        outputs_file = os.path.join(outputs_directory, outputs_location + ".tar.gz")
        self.download_outputs(outputs_location, outputs_file)
        self._logger.debug("Downloaded outputs")

        # cleanup
        self.delete_exposure(input_location)
        self.delete_outputs(outputs_location)

    def delete_exposure(self, input_location):
        self._logger.debug("Deleting exposure")
        response = requests.delete(self.build_uri('/exposure/' + input_location))
        if response.status_code != 200:
            # Do not fail if tidy up fails
            self._logger.warning("DELETE /exposure failed: {}".format(str(response.status_code)))
        else:
            self._logger.debug("Deleted exposure")

    def delete_outputs(self, outputs_location):
        self._logger.info("Deleting outputs")
        response = requests.delete(self._oasis_api_url + "/outputs/" + outputs_location)
        if response.status_code != 200:
            # Do not fail if tidy up fails
            self._logger.warning("DELETE /outputs failed: {}".format(str(response.status_code)))
        else:
            self._logger.info("Deleted outputs")

    @oasis_log
    def get_output_files(self, outputs_location, outputs_directory, input_location):
        frame = inspect.currentframe()
        func_name = inspect.getframeinfo(frame)[2]
        self._logger.info("STARTED: {}".format(func_name))
        args, _, _, values = inspect.getargvalues(frame)
        for i in args:
            if i == 'self':
                continue
            self._logger.info("{}={}".format(i, values[i]))
        start = time.time()

        self._logger.debug("Downloading outputs")
        outputs_file = os.path.join(
            outputs_directory, outputs_location + ".tar.gz")
        download_outputs_status = self.download_outputs(outputs_location, outputs_file)
        self._logger.debug("Downloaded outputs")

        self._logger.debug("Deleting exposure")
        response = requests.delete(
            self._oasis_api_url + "/exposure/" + input_location)
        if not response.ok:
            # Do not fail if tidy up fails
            self._logger.warn(
                "DELETE /exposure failed: {}".format(
                    str(response.status_code)))
        self._logger.debug("Deleted exposure")

        self._logger.info("Deleting outputs")
        response = requests.delete(
            self._oasis_api_url + "/outputs/" + outputs_location)
        if not response.ok:
            # Do not fail if tidy up fails
            self._logger.warn(
                "DELETE /outputs failed: {}".format(str(response.status_code)))
        self._logger.info("Deleted outputs")

        end = time.time()
        self._logger.debug(
            "COMPLETED: OasisApiClient.run_analysis in {}s".format(
                round(end - start, 2)))
        return download_outputs_status

    @oasis_log
    def download_exposure(self, exposure_location, localfile):
        '''
        Download exposure data to a specified local file.
        Args:
            ``exposure_location`` (string): The location of the exposure resource.
            ``localfile` (string): The localfile to download to.
        '''
        frame = inspect.currentframe()
        func_name = inspect.getframeinfo(frame)[2]
        self._logger.info("STARTED: {}".format(func_name))
        args, _, _, values = inspect.getargvalues(frame)
        for i in args:
            if i == 'self':
                continue
            self._logger.info("{}={}".format(i, values[i]))
        start = time.time()

        response = requests.get(
            self._oasis_api_url + "/exposure/" + exposure_location,
            stream=True)
        if not response.ok:
            exception_message = "GET /exposure failed: {}".format(
                str(response.status_code))
            self._logger.error(exception_message)
            raise Exception(exception_message)

        with open(localfile, 'wb') as f:
            for chunk in response.iter_content(
                    chunk_size=self.DOWNLOAD_CHUCK_SIZE_IN_BYTES):
                if chunk:
                    f.write(chunk)

        end = time.time()
        self._logger.info(
            "COMPLETED: {} in {}s".format(
                func_name, round(end - start, 2)))

    @oasis_log
    def download_outputs(self, outputs_location, localfile):
        '''
        Download outputs data to a specified local file.
        Args:
            ``outputs_location`` (string): The location of the outputs resource.
            ``localfile`` (string): The localfile to download to.
        '''
        frame = inspect.currentframe()
        func_name = inspect.getframeinfo(frame)[2]
        self._logger.info("STARTED: {}".format(func_name))
        args, _, _, values = inspect.getargvalues(frame)
        for i in args:
            if i == 'self':
                continue
            self._logger.info("{}={}".format(i, values[i]))
        start = time.time()

        if os.path.exists(localfile):
            error_message = 'Local file alreday exists: {}'.format(localfile)
            self._logger.error(error_message)
            raise Exception(error_message)

        response = requests.get(
            self._oasis_api_url + "/outputs/" + outputs_location,
            stream=True)
        if not response.ok:
            exception_message = "GET /outputs failed: {}".format(
                str(response.status_code))
            self._logger.error(exception_message)
            raise Exception(exception_message)

        with open(localfile, 'wb') as f:
            for chunk in response.iter_content(
                    chunk_size=self.DOWNLOAD_CHUCK_SIZE_IN_BYTES):
                if chunk:
                    f.write(chunk)

        end = time.time()
        self._logger.info(
            "COMPLETED: {} in {}s".format(
                func_name, round(end - start, 2)))

    @oasis_log
    def create_analysis_settings_json(self, directory):
        '''
        Generate an analysis settings JSON from a set of
        CSV files in a specified directory.
        Args:
            ``directory`` (string): the directory containing the CSV files.
        Returns:
            The analysis settings JSON.
        '''
        frame = inspect.currentframe()
        func_name = inspect.getframeinfo(frame)[2]
        self._logger.info("STARTED: {}".format(func_name))
        args, _, _, values = inspect.getargvalues(frame)
        for i in args:
            if i == 'self':
                continue
            self._logger.info("{}={}".format(i, values[i]))
        start = time.time()

        if not os.path.exists(directory):
            error_message = "Directory does not exist: {}".format(directory)
            self._logger.error(error_message)
            raise Exception(error_message)

        general_settings_file = os.path.join(
            directory, self.GENERAL_SETTINGS_FILE)
        model_settings_file = os.path.join(
            directory, self.MODEL_SETTINGS_FILE)
        gul_summaries_file = os.path.join(
            directory, self.GUL_SUMMARIES_FILE)
        il_summaries_file = os.path.join(
            directory, self.IL_SUMMARIES_FILE)

        for file in [general_settings_file, model_settings_file, gul_summaries_file, il_summaries_file]:
            if not os.path.exists(directory):
                error_message = "File does not exist: {}".format(directory)
                self._logger.error(error_message)
                raise Exception(error_message)

        general_settings = dict()
        with open(general_settings_file, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                general_settings[row[0]] = \
                    eval("{}('{}')".format(row[2], row[1]))

        model_settings = dict()
        with open(model_settings_file, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                model_settings[row[0]] = \
                    eval("{}('{}')".format(row[2], row[1]))

        gul_summaries = self._get_summaries(gul_summaries_file)
        il_summaries = self._get_summaries(il_summaries_file)

        analysis_settings = general_settings
        analysis_settings['model_settings'] = model_settings
        analysis_settings['gul_summaries'] = gul_summaries
        analysis_settings['il_summaries'] = il_summaries
        json = jsonpickle.encode(analysis_settings)
        self._logger.info("Analysis settings json: {}".format(json))

        end = time.time()
        self._logger.info(
            "COMPLETED: {} in {}s".format(func_name, round(end - start, 2)))

        return json

    @oasis_log
    def create_binary_files(self, directory, do_il):
        '''
        Create the binary files.
        Args:
            ``directory`` (string): the directory containing the CSV files.
            ``do_il`` (bool): do insured loss. If True, FM file must be present.
        Returns:
            None
        '''
        if do_il:
            input_files = chain(self.GUL_INPUTS_FILES, self.IL_INPUTS_FILES, self.OPTIONAL_INPUTS_FILES)
        else:
            input_files = chain(self.GUL_INPUTS_FILES, self.OPTIONAL_INPUTS_FILES)

        for input_file in input_files:
            conversion_tool = self.CONVERSION_TOOLS[input_file]
            input_file_path = os.path.join(directory, input_file + ".csv")
            if not os.path.exists(input_file_path):
                continue

            output_file_path = os.path.join(directory, input_file + ".bin")
            command = "{} < {} > {}".format(
                conversion_tool, input_file_path, output_file_path)
            self._logger.debug("Running command: {}".format(command))
            proc = subprocess.Popen(command, shell=True)
            proc.wait()
            if proc.returncode != 0:
                raise Exception(
                    "Failed to convert {}: {}".format(input_file_path, command))

    @oasis_log
    def check_inputs_directory(self, directory_to_check, do_il):
        """
        Check that all the required csv files are present in the directory.
        Args:
            ``directory`` (string): the directory containing the CSV files.
            ``do_il`` (bool): do insured loss. If True, FM file must be present.
        Returns:
            None
        """
        file_path = os.path.join(directory_to_check, self.TAR_FILE)
        if os.path.exists(file_path):
            raise OasisException("Inputs tar file already exists: {}".format(file_path))

        if do_il:
            input_files = chain(self.GUL_INPUTS_FILES, self.IL_INPUTS_FILES)
        else:
            input_files = self.GUL_INPUTS_FILES

        for input_file in input_files:
            file_path = os.path.join(directory_to_check, input_file + ".csv")
            if not os.path.exists(file_path):
                raise OasisException("Failed to find {}".format(file_path))

            file_path = os.path.join(directory_to_check, input_file + ".bin")
            if os.path.exists(file_path):
                raise OasisException("Binary file already exists: {}".format(file_path))

    @oasis_log
    def health_check(self, poll_attempts=1, retry_delay=5):
        for attempt in range(poll_attempts):
            try:
                if attempt > 0:
                    time.sleep(retry_delay)

                resp = requests.get('{}/healthcheck'.format(self._oasis_api_url))
                if resp.status_code == 200:
                    return True
            except RequestException as e:
                pass
        else:
            self._logger.error(
                'Could not connect to the api server after {} attempts. Check it is running and try again later.'.format(
                    poll_attempts,
                )
            )
            return False

    def _validate_inputs(self, directory):
        ''' Validate the input files.'''
        # TODO
        pass

    def _create_tar_file(self, directory):
        ''' Package the binaries in a gzipped tar. '''
        original_cwd = os.getcwd()
        os.chdir(directory)

        with tarfile.open(self.TAR_FILE, "w:gz") as tar:
            for file in chain(self.GUL_INPUTS_FILES, self.IL_INPUTS_FILES, self.OPTIONAL_INPUTS_FILES):
                bin_file = file + ".bin"
                if os.path.exists(bin_file):
                    tar.add(bin_file)
        os.chdir(original_cwd)

    def _clean_directory(self, directory_to_check):
        ''' Clean the tar and binary files. '''
        file_path = os.path.join(directory_to_check, self.TAR_FILE)
        if os.path.exists(file_path):
            os.remove(file_path)
        for file in chain(self.GUL_INPUTS_FILES, self.IL_INPUTS_FILES, self.OPTIONAL_INPUTS_FILES):
            file_path = os.path.join(directory_to_check, file + ".bin")
            if os.path.exists(file_path):
                os.remove(file_path)

    def _get_summaries(self, summary_file):
        ''' Get a list representation of a summary file. '''
        summaries_dict = dict()
        with open(summary_file, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                id = int(row[0])
                if id not in list(summaries_dict.keys):
                    summaries_dict[id] = dict()
                    summaries_dict[id]['leccalc'] = dict()
                if row[1].startswith('leccalc'):
                    summaries_dict[id]['leccalc'][row[1]] = bool(row[2])
                else:
                    summaries_dict[id][row[1]] = bool(row[2])
        summaries = list()
        for id in summaries_dict.keys():
            summaries_dict[id]['id'] = id
            summaries.append(summaries_dict[id])

        return summaries
