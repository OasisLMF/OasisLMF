__all__ = [
    'csv_validity_test',
]

import os
import logging
import subprocess

from ..utils.exceptions import OasisException

# Model files to test
INPUT_FILES = {
    'damage_bin_dict': {
        'name': 'damage_bin_dict',
        'validation_tool': 'validatedamagebin',
        'flag': '-d'
    },
    'footprint': {
        'name': 'footprint',
        'validation_tool': 'validatefootprint',
        'flag': '-f'
    },
    'vulnerability': {
        'name': 'vulnerability',
        'validation_tool': 'validatevulnerability',
        'flag': '-s'
    }
}


def csv_validity_test(model_data_fp):
    """
    Assess validity of model data.

    :param model_data_fp: directory containing csv files
    :type model_data_fp: str

    :raises OasisException: if one of the tests fail
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    model_data_dir = os.path.abspath(model_data_fp)

    # Check individual files
    for input_file in INPUT_FILES.values():
        validation_tool = input_file['validation_tool']
        input_file_path = os.path.join(
            model_data_dir,
            '{}.csv'.format(input_file['name'])
        )

        logger.info("Testing {}.csv".format(input_file['name']))
        cmd_str = "{} < {}".format(validation_tool, input_file_path)

        try:
            subprocess.check_call(cmd_str, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            raise OasisException("Exception raised in 'csv_validity_test'", e)

    # Execute cross checks
    logger.info("Executing cross checks")
    cmd_str = "crossvalidation"
    for input_file in INPUT_FILES.values():
        flag = input_file['flag']
        input_file_path = os.path.join(
            model_data_dir,
            '{}.csv'.format(input_file['name'])
        )
        cmd_str += " {} {}".format(flag, input_file_path)

    try:
        subprocess.check_call(cmd_str, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        raise OasisException("Exception raised in 'csv_validity_test'", e)
