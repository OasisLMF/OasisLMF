"""
    Provides a simple class to build Oasis API clients.
"""
import logging

from requests import RequestException
from six.moves import urllib

from oasislmf.utils.exceptions import OasisException

__all__ = [
    'OasisAPIClient'
]


# Python 2 standard imports
import os
import time

# Python 3rd party imports
import requests

from requests_toolbelt.multipart.encoder import MultipartEncoder

# Oasis imports
from ..utils.log import oasis_log
from ..utils.status import STATUS_PENDING, STATUS_SUCCESS, STATUS_FAILURE


class OasisAPIClient(object):
    DOWNLOAD_CHUCK_SIZE_IN_BYTES = 1024

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
            self.create_binary_files(directory, do_il)
            self.create_binary_tar_file(directory)

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

    def delete_resource(self, url):
        response = requests.delete(self.build_uri(url))
        if response.status_code != 200:
            self._logger.warning("DELETE {} failed: {}".format(response.request.url, response.status_code))
        else:
            self._logger.info("Deleted {}".format(response.request.url))

    @oasis_log
    def delete_exposure(self, input_location):
        self._logger.info("Deleting exposure")
        self.delete_resource('/exposure/' + input_location)

    @oasis_log
    def delete_outputs(self, outputs_location):
        self._logger.info("Deleting outputs")
        self.delete_resource('/outputs/' + outputs_location)

    def download_resource(self, url, localfile):
        if os.path.exists(localfile):
            error_message = 'Local file alreday exists: {}'.format(localfile)
            self._logger.error(error_message)
            raise OasisException(error_message)

        response = requests.get(self.build_uri(url), stream=True)
        if not response.ok:
            exception_message = 'GET {} failed: {}'.format(response.request.url, response.status_code)
            self._logger.error(exception_message)
            raise OasisException(exception_message)

        with open(localfile, 'wb') as f:
            for chunk in response.iter_content(chunk_size=self.DOWNLOAD_CHUCK_SIZE_IN_BYTES):
                f.write(chunk)

    @oasis_log
    def download_exposure(self, exposure_location, localfile):
        """
        Download exposure data to a specified local file.
        Args:
            ``exposure_location`` (string): The location of the exposure resource.
            ``localfile` (string): The localfile to download to.
        """
        self.download_resource('/exposure/' + exposure_location, localfile)

    @oasis_log
    def download_outputs(self, outputs_location, localfile):
        """
        Download outputs data to a specified local file.
        Args:
            ``outputs_location`` (string): The location of the outputs resource.
            ``localfile`` (string): The localfile to download to.
        """
        self.download_resource('/outputs/' + outputs_location, localfile)

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
