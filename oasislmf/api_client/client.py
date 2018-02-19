"""
    Provides a simple class to build Oasis API clients.
"""
import logging

from requests import RequestException
from six.moves import urllib

from ..model_execution.files import TAR_FILE
from ..model_execution.bin import create_binary_tar_file, create_binary_files, check_inputs_directory, \
    cleanup_bin_directory, check_conversion_tools
from ..utils.exceptions import OasisException

__all__ = [
    'OasisAPIClient'
]


# Python 2 standard imports
import os
import io
import time

# Python 3rd party imports
import requests

from requests_toolbelt.multipart.encoder import MultipartEncoder

# Oasis imports
from ..utils.log import oasis_log
from ..utils.status import STATUS_PENDING, STATUS_SUCCESS, STATUS_FAILURE


class OasisAPIClient(object):
    """
    Class for interacting with the oasis api server

    :param oasis_api_url: The root URL for the API. This should
        include the scheme and port (eg. http://localhost:8001)
    :type oasis_api_url: str

    :param logger: The logger to use for message logging. If None
        the root logger is used.
    :type logger: Logger
    """
    #: The chunk size to use when streaming data from the server
    DOWNLOAD_CHUCK_SIZE_IN_BYTES = 1024

    def __init__(self, oasis_api_url, logger=None):
        """
        Construct the client.
        """

        self._oasis_api_url = oasis_api_url
        self._logger = logger or logging.getLogger()

    def build_uri(self, path):
        """
        Builds the uri for the requested resource

        :param path: The path to the resource
        :type path: str

        :return: The fully qualified uri for the resource
        """
        return urllib.parse.urljoin(self._oasis_api_url, path)

    @oasis_log
    def upload_inputs_from_directory(self, directory, bin_directory=None, do_il=False, do_build=False, do_clean=False):
        """
        Upload the CSV files from a specified directory.

        :param directory: the directory containing the CSVs.
        :type directory: str

        :param bin_directory: the directory to build the binary files in, if not set ``directory`` is used
        :type bin_directory: str

        :param do_il: if True, require files for insured loss (IL) calculation.
        :type do_il: bool

        :param do_build: if True, the input tar will be built
        :type do_build: bool

        :param do_clean: if True, remove the tar and bin files
        :type do_clean: bool

        :return: The location of the uploaded inputs.
        """
        bin_directory = bin_directory or directory

        try:
            if do_build:
                check_inputs_directory(directory, do_il=do_il)
                check_conversion_tools(do_il=do_il)
                create_binary_files(directory, bin_directory, do_il=do_il)
                create_binary_tar_file(bin_directory)

            self._logger.debug("Uploading inputs")
            inputs_tar_to_upload = os.path.join(bin_directory, TAR_FILE)

            with io.open(inputs_tar_to_upload, 'rb') as f:
                inputs_multipart_data = MultipartEncoder(
                    fields={
                        'file': (TAR_FILE, f, 'text/plain')
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
                raise OasisException("Failed to save exposure.")

            exposure_location = response.json()['exposures'][0]['location']
            self._logger.debug("Uploaded exposure. Location: " + exposure_location)

            # Return the location of the uploaded inputs
            return exposure_location
        finally:
            # Tidy up
            if do_clean:
                cleanup_bin_directory(bin_directory)

    @oasis_log
    def run_analysis(self, analysis_settings_json, input_location):
        """
        Starts an analysis running. Calling code will need to poll for
        completion or failure, and handle results.

        :param analysis_setting_json: The analysis settings encoded as JSON.
        :type analysis_setting_json: str

        :param input_location: The location of the inputs resource.
        :type input_location: str

        :return: The location of analysis status, to poll.
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
        """
        Fetches the analysis status for the requested analysis.

        :param analysis_status_location: The location the analysis status.
            This is the value returned from ``run_analysis``
        :type analysis_status_location: str

        :raises OasisException: If the analysis fails.
        :raises OasisException: If the http request is not successful

        :return: A 2 tuple of the status and output location. If the
            status is pending the output location is empty.
        """
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
        Run an analysis to completion or failure. resources on the server are
        cleaned up once the analysis is complete.

        :param analysis_setting_json: The analysis settings encoded as JSON.
        :type analysis_settings_json: str

        :param input_location: The local directory containing the inputs resources.
        :type input_location: str

        :param outputs_directory: The local directory to save the outputs to.
        :type outputs_directory: str

        :param analysis_poll_interval: The interval time to wait between status checks
        :type analysis_poll_interval: int

        :raises OasisException: If the analysis fails.
        :raises OasisException: If a http request is not successful
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

    def delete_resource(self, path):
        """
        Cleans up a resource on the server

        :param path: The path to the resource to delete
        :type path: str
        """
        response = requests.delete(self.build_uri(path))
        if response.status_code != 200:
            self._logger.warning("DELETE {} failed: {}".format(response.request.url, response.status_code))
        else:
            self._logger.info("Deleted {}".format(response.request.url))

    @oasis_log
    def delete_exposure(self, input_location):
        """
        Cleans up an exposure on the server

        :param input_location: The input location for the analysis
        :type input_location: str
        """
        self._logger.info("Deleting exposure")
        self.delete_resource('/exposure/' + input_location)

    @oasis_log
    def delete_outputs(self, outputs_location):
        """
        Cleans up ouptut files on the server

        :param output_location: The output location for the analysis
        :type output_location: str
        """
        self._logger.info("Deleting outputs")
        self.delete_resource('/outputs/' + outputs_location)

    def download_resource(self, path, localfile):
        """
        Streams a resource from the server to a local file

        :param path: The path of the resource to download
        :type path: str

        :param localfile: The path to the local file to download the
            resource to.
        :type localfile: str
        """
        if os.path.exists(localfile):
            error_message = 'Local file alreday exists: {}'.format(localfile)
            self._logger.error(error_message)
            raise OasisException(error_message)

        response = requests.get(self.build_uri(path), stream=True)
        if not response.ok:
            exception_message = 'GET {} failed: {}'.format(response.request.url, response.status_code)
            self._logger.error(exception_message)
            raise OasisException(exception_message)

        with io.open(localfile, 'wb') as f:
            for chunk in response.iter_content(chunk_size=self.DOWNLOAD_CHUCK_SIZE_IN_BYTES):
                f.write(chunk)

    @oasis_log
    def download_exposure(self, exposure_location, localfile):
        """
        Download exposure data to a specified local file.

        :param exposure_location: The location of the exposure resource.
        :type exposure_location: str

        :param localfile: The path to the local file to download the
            exposure to.
        :type localfile: str
        """
        self.download_resource('/exposure/' + exposure_location, localfile)

    @oasis_log
    def download_outputs(self, outputs_location, localfile):
        """
        Download outputs data to a specified local file.

        :param outputs_location: The location of the ouptuts resource.
        :type outputs_location: str

        :param localfile: The path to the local file to download the
            exposure to.
        :type localfile: str
        """
        self.download_resource('/outputs/' + outputs_location, localfile)

    @oasis_log
    def health_check(self, poll_attempts=1, retry_delay=5):
        """
        Checks the health of the server.

        :param poll_attempts: The maximum number of checks to make
        :type poll_attempts: int

        :param retry_delay: The amount of time to wait between retry attempts
        :type retry_delay: int

        :return: True If the server is healthy, otherwise False
        """
        for attempt in range(poll_attempts):
            try:
                if attempt > 0:
                    time.sleep(retry_delay)

                resp = requests.get('{}/healthcheck'.format(self._oasis_api_url))
                if resp.status_code == 200:
                    return True
            except RequestException:
                pass
        else:
            self._logger.error(
                'Could not connect to the api server after {} attempts. Check it is running and try again later.'.format(
                    poll_attempts,
                )
            )
            return False
