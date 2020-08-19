import os
import pytest

from backports.tempfile import TemporaryDirectory

from oasislmf.manager import OasisManager
from oasislmf.utils.defaults import KEY_NAME_TO_FILE_NAME
from oasislmf.utils.exceptions import OasisException

input_oed_location = """BuildingTIV
1
2
3
4
5
"""

output_oed_location = """BuildingTIV
2
4
6
8
10
"""


def write_simple_epa_module(module_path, class_name='ExposurePreAnalysis'):
    with open(module_path, 'w') as f:
        f.write(f'''
import pandas as pd

class {class_name}:
    """
    Example of custum module called by oasislmf/model_preparation/ExposurePreAnalysis.py
    """

    def __init__(self, raw_oed_location_csv, oed_location_csv, exposure_pre_analysis_setting, **kwargs):
        self.raw_oed_location_csv = raw_oed_location_csv
        self.oed_location_csv = oed_location_csv
        self.exposure_pre_analysis_setting = exposure_pre_analysis_setting

    def run(self):
        panda_df = pd.read_csv(self.raw_oed_location_csv, memory_map=True)
        panda_df['BuildingTIV'] = panda_df['BuildingTIV'] * self.exposure_pre_analysis_setting['BuildingTIV_multiplyer']
        panda_df.to_csv(self.oed_location_csv, index=False)
''')


def write_oed_location(oed_location_csv):
    with open(oed_location_csv, 'w') as f:
        f.write(input_oed_location)


def write_exposure_pre_analysis_setting_json(exposure_pre_analysis_setting_json):
    with open(exposure_pre_analysis_setting_json, 'w') as f:
        f.write('{"BuildingTIV_multiplyer":  2}')


def test_exposure_pre_analysis_simple_example():
    with TemporaryDirectory() as d:
        kwargs = {'model_run_dir': d,
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_simple.py'),
                  'oed_location_csv': os.path.join(d, KEY_NAME_TO_FILE_NAME['oed_location_csv']),
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'),}

        write_simple_epa_module(kwargs['exposure_pre_analysis_module'])
        write_oed_location(kwargs['oed_location_csv'] )
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        OasisManager().exposure_pre_analysis(**kwargs)

        with open(os.path.join(d, 'input', KEY_NAME_TO_FILE_NAME['oed_location_csv'])) as new_oed_location_csv:
            new_oed_location_csv_data = new_oed_location_csv.read()
            assert new_oed_location_csv_data == output_oed_location


def test_exposure_pre_analysis_class_name():
    with TemporaryDirectory() as d:
        kwargs = {'model_run_dir': d,
                  'exposure_pre_analysis_class_name': 'foobar',
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_simple_foobar.py'),
                  'oed_location_csv': os.path.join(d, KEY_NAME_TO_FILE_NAME['oed_location_csv']),
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'), }

        write_simple_epa_module(kwargs['exposure_pre_analysis_module'], kwargs['exposure_pre_analysis_class_name'])
        write_oed_location(kwargs['oed_location_csv'])
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        OasisManager().exposure_pre_analysis(**kwargs)

        with open(os.path.join(d, 'input', KEY_NAME_TO_FILE_NAME['oed_location_csv'])) as new_oed_location_csv:
            new_oed_location_csv_data = new_oed_location_csv.read()
            assert new_oed_location_csv_data == output_oed_location


def test_missing_module():
    with pytest.raises(OasisException, match="parameter exposure_pre_analysis_module is required for Computation Step ExposurePreAnalysis"):
        OasisManager().exposure_pre_analysis()


def test_wrong_class():
    with TemporaryDirectory() as d:
        kwargs = {'model_run_dir': d,
                  'exposure_pre_analysis_class_name': 'foobar',
                  'exposure_pre_analysis_module': os.path.join(d, 'exposure_pre_analysis_simple.py'),
                  'oed_location_csv': os.path.join(d, KEY_NAME_TO_FILE_NAME['oed_location_csv']),
                  'exposure_pre_analysis_setting_json': os.path.join(d, 'exposure_pre_analysis_setting.json'), }

        write_simple_epa_module(kwargs['exposure_pre_analysis_module'])
        write_oed_location(kwargs['oed_location_csv'])
        write_exposure_pre_analysis_setting_json(kwargs['exposure_pre_analysis_setting_json'])

        with pytest.raises(OasisException, match=f"class {kwargs['exposure_pre_analysis_class_name']} "
                                                 f"is not defined in module {kwargs['exposure_pre_analysis_module']}"):
            OasisManager().exposure_pre_analysis(**kwargs)
