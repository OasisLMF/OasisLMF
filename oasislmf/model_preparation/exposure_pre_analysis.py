__all__ = [
    'ExposurePreAnalysis'
]

import json
import os
import pathlib
import shutil

from .computation_step import ComputationStep
from ..utils.path import get_custom_module
from ..utils.exceptions import OasisException
from ..utils.defaults import store_exposure_fp, KEY_NAME_TO_FILE_NAME


class ExposurePreAnalysis(ComputationStep):
    """
    Computation step that will be call before the gulcalc.
    Add the ability to specify a model specific pre-analysis hook for exposure modification,
    Allows OED to be processed by some custom code.
    Example of usage include geo-coding, exposure enhancement, or dis-aggregation...

    when the run method is call it will :
    - load the module specified at exposure_pre_analysis_module
    - init the class named exposure_pre_analysis_class_name with all the non null args in step_params as key arguments
    - call the method run of the object
    - return the output of the method

    you can find an example of such custom module in OasisPyWind/custom_module/exposure_pre_analysis.py

    """
    step_params = [{'name': 'exposure_pre_analysis_module', 'required': True, 'is_path': True, 'pre_exist': True,
                    'help': 'Exposure Pre-Analysis lookup module path'},
                   {'name': 'exposure_pre_analysis_class_name', 'default': 'ExposurePreAnalysis',
                    'help': 'Name of the class to use for the exposure_pre_analysis'},
                   {'name': 'exposure_pre_analysis_setting_json', 'is_path': True, 'pre_exist': True,
                    'help': 'Exposure Pre-Analysis config JSON file path'},
                   {'name': 'oed_location_csv', 'is_path': True, 'pre_exist': True,
                    'help': 'Source location CSV file path'},
                   {'name': 'oed_accounts_csv', 'is_path': True, 'pre_exist': True,
                    'help': 'Source accounts CSV file path'},
                   {'name': 'oed_info_csv', 'is_path': True, 'pre_exist': True,
                    'help': 'Reinsurance info. CSV file path'},
                   {'name': 'oed_scope_csv', 'is_path': True, 'pre_exist': True,
                    'help': 'Reinsurance scope CSV file path'},
                   {'name': 'model_run_dir', 'is_path': True, 'pre_exist': False,
                    'help': 'Model run directory path'},
                   ]

    run_dir_key = 'pre-analysis'

    def run(self):
        """
        import exposure_pre_analysis_module and call the run method
        """
        if self.model_run_dir is None:
            self.model_run_dir = self.get_default_run_dir()

        input_dir = os.path.join(self.model_run_dir, 'input')
        pathlib.Path(input_dir).mkdir(parents=True, exist_ok=True)

        kwargs = {}
        for input_name in ('oed_location_csv', 'oed_accounts_csv', 'oed_info_csv', 'oed_scope_csv'):
            file_path_in = getattr(self, input_name)
            if file_path_in is not None:
                file_path_raw = os.path.join(input_dir, f'epa_{KEY_NAME_TO_FILE_NAME[input_name]}')
                file_path_out = os.path.join(input_dir, KEY_NAME_TO_FILE_NAME[input_name])
                kwargs[f'raw_{input_name}'] = file_path_raw
                kwargs[input_name] = file_path_out

                shutil.copyfile(file_path_in, file_path_raw)
                shutil.copyfile(file_path_in, file_path_out)

        if self.exposure_pre_analysis_setting_json:
            with open(self.exposure_pre_analysis_setting_json) as exposure_pre_analysis_setting_file:
                kwargs['exposure_pre_analysis_setting'] = json.load(exposure_pre_analysis_setting_file)

        _module = get_custom_module(self.exposure_pre_analysis_module, 'Exposure Pre-Analysis lookup module path')

        try:
            _class = getattr(_module, self.exposure_pre_analysis_class_name)
        except AttributeError as e:
            raise OasisException(f"class {self.exposure_pre_analysis_class_name} "
                                 f"is not defined in module {self.exposure_pre_analysis_module}") from e.__cause__

        return _class(**kwargs).run()
