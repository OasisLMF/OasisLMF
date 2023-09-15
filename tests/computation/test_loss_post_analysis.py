import os
from tempfile import TemporaryDirectory

import pytest
import pathlib

from oasislmf.manager import OasisManager
from oasislmf.utils.defaults import SOURCE_FILENAMES
from oasislmf.utils.exceptions import OasisException

input_oed_location = """PortNumber,AccNumber,LocNumber,BuildingTIV
1,A11111,10002082046,1
1,A11111,10002082047,2
1,A11111,10002082048,3
1,A11111,10002082049,4
1,A11111,10002082050,5
"""

output_oed_location = """PortNumber,AccNumber,LocNumber,BuildingTIV,ContentsTIV,BITIV
1,A11111,10002082046,2.0,0.0,0.0
1,A11111,10002082047,4.0,0.0,0.0
1,A11111,10002082048,6.0,0.0,0.0
1,A11111,10002082049,8.0,0.0,0.0
1,A11111,10002082050,10.0,0.0,0.0
"""


def write_simple_post_analysis_module(module_path, class_name='PostAnalysis'):
    with open(module_path, 'w') as f:
        f.write(f'''
class {class_name}:
    """
    Example of custom module called by oasislmf/model_preparation/PostAnalysis.py
    """

    def __init__(self, raw_output_dir, post_processed_output_dir, **kwargs):
        self.raw_output_dir = raw_output_dir
        self.post_processed_output_dir = post_processed_output_dir

    def run(self):
        # create a file 
        Path(self.post_processed_output_dir) / "my_file.txt"
''')


def write_oed_location(oed_location_csv):
    with open(oed_location_csv, 'w') as f:
        f.write(input_oed_location)


def write_exposure_pre_analysis_setting_json(exposure_pre_analysis_setting_json):
    with open(exposure_pre_analysis_setting_json, 'w') as f:
        f.write('{"BuildingTIV_multiplyer":  2}')


def test_exposure_pre_analysis_simple_example():
    with TemporaryDirectory() as d:
        kwargs = {'oasis_files_dir': d,
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_simple.py'),
                  'oed_location_csv': os.path.join(d, 'input_{}'.format(SOURCE_FILENAMES['oed_location_csv'])),
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'),
                  'check_oed': False}

        write_simple_epa_module(kwargs['exposure_pre_analysis_module'])
        write_oed_location(kwargs['oed_location_csv'])
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        OasisManager().exposure_pre_analysis(**kwargs)

        with open(os.path.join(d, SOURCE_FILENAMES['oed_location_csv'])) as new_oed_location_csv:
            new_oed_location_csv_data = new_oed_location_csv.read()
            assert new_oed_location_csv_data == output_oed_location


def test_exposure_pre_analysis_class_name():
    with TemporaryDirectory() as d:
        kwargs = {'oasis_files_dir': d,
                  'exposure_pre_analysis_class_name': 'foobar',
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_simple_foobar.py'),
                  'oed_location_csv': os.path.join(d, 'input_{}'.format(SOURCE_FILENAMES['oed_location_csv'])),
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'),
                  'check_oed': False}

        write_simple_epa_module(kwargs['exposure_pre_analysis_module'], kwargs['exposure_pre_analysis_class_name'])
        write_oed_location(kwargs['oed_location_csv'])
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        OasisManager().exposure_pre_analysis(**kwargs)

        with open(os.path.join(d, SOURCE_FILENAMES['oed_location_csv'])) as new_oed_location_csv:
            new_oed_location_csv_data = new_oed_location_csv.read()
            assert new_oed_location_csv_data == output_oed_location


def test_missing_module():
    with pytest.raises(OasisException, match="parameter exposure_pre_analysis_module is required for Computation Step ExposurePreAnalysis"):
        OasisManager().exposure_pre_analysis()


def test_wrong_class():
    with TemporaryDirectory() as d:
        kwargs = {'oasis_files_dir': d,
                  'exposure_pre_analysis_class_name': 'foobar',
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_simple.py'),
                  'oed_location_csv': os.path.join(d, 'input_{}'.format(SOURCE_FILENAMES['oed_location_csv'])),
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'),
                  'check_oed': False}

        write_simple_epa_module(kwargs['exposure_pre_analysis_module'])
        write_oed_location(kwargs['oed_location_csv'])
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        with pytest.raises(OasisException, match=f"class {kwargs['exposure_pre_analysis_class_name']} "
                                                 f"is not defined in module {kwargs['exposure_pre_analysis_module']}"):
            OasisManager().exposure_pre_analysis(**kwargs)
