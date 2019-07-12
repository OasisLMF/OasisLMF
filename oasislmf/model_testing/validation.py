import os
import subprocess

from ..utils.exceptions import OasisException

# Model files to test
INPUT_FILES = {
    'damage_bin_dict': {
        'name': 'damage_bin_dict', 'conversion_tool': 'damagebintobin'
    }
}

def csv_to_bin_validity_test(model_data_fp):
    """
    Create a set of binary files to assess validity of model data.

    :param model_data_fp: directory containing csv files
    :type model_data_fp: str

    :raises OasisException: if one of the conversions fails
    """

    model_data_dir = os.path.abspath(model_data_fp)

    for input_file in INPUT_FILES.values():
        conversion_tool = input_file['conversion_tool']
        input_file_path = os.path.join(
            model_data_dir,
            '{}.csv'.format(input_file['name'])
        )

        output_file_path = os.path.join(
            model_data_dir,
            '{}.bin'.format(input_file['name'])
        )

        cmd_str = "{} < {} > {}".format(conversion_tool, input_file_path, output_file_path)

        try:
            subprocess.check_call(cmd_str, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            raise OasisException from e
