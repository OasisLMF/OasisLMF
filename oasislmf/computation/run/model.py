__all__ = [
    'RunModel'
]

import os
from tqdm import tqdm

from ..base import ComputationStep

from ..generate.files import GenerateFiles
from ..generate.losses import GenerateLosses
from ..hooks.pre_analysis import ExposurePreAnalysis

from ...utils.exceptions import OasisException
from ...utils.data import (
    get_analysis_settings,
    get_model_settings, get_exposure_data,
)

from ...utils.path import empty_dir


class RunModel(ComputationStep):
    """
    Run models end to end.
    """

    # Override params
    step_params = [
        {'name': 'oasis_files_dir', 'flag': '-o', 'is_path': True, 'pre_exist': False,
            'help': 'Path to the directory in which to generate the Oasis files'},
        {'name': 'exposure_pre_analysis_module', 'required': False, 'is_path': True,
            'pre_exist': True, 'help': 'Exposure Pre-Analysis lookup module path'},
    ]
    # Add params from each sub command not in 'step_params'
    chained_commands = [
        GenerateLosses,
        GenerateFiles,
        ExposurePreAnalysis,
    ]

    def get_exposure_data_config(self):
        return {
            'location': self.oed_location_csv,
            'account': self.oed_accounts_csv,
            'ri_info': self.oed_info_csv,
            'ri_scope': self.oed_scope_csv,
            'currency_conversion': self.currency_conversion_json,
            'check_oed': self.check_oed,
            'use_field': True
        }

    def run(self):

        # setup output dir
        if not self.model_run_dir:
            self.model_run_dir = GenerateLosses._get_output_dir(self)
        if os.path.exists(self.model_run_dir):
            empty_dir(self.model_run_dir)
        os.makedirs(os.path.join(self.model_run_dir, 'input'))

        self.kwargs['model_run_dir'] = self.model_run_dir
        # TODO: input oasis_files_dir is actually not use in the code
        self.kwargs['oasis_files_dir'] = os.path.join(self.model_run_dir, 'input')
        self.oasis_files_dir = self.kwargs['oasis_files_dir']

        # Validate JSON files (Fail at entry point not after input generation)
        get_analysis_settings(self.analysis_settings_json)
        if self.model_settings_json:
            get_model_settings(self.model_settings_json)

        # Check input exposure
        required_ri_paths = [self.oed_info_csv, self.oed_scope_csv]
        il = True if self.oed_accounts_csv else False
        ri = all(required_ri_paths) and il
        if any(required_ri_paths) and not ri:
            raise OasisException(
                'RI option indicated by provision of some RI related assets, but other assets are missing. '
                'To generate RI inputs you need to provide all of the assets required to generate direct '
                'Oasis files (GUL + FM input files) plus all of the following assets: '
                '    reinsurance info. file path, '
                '    reinsurance scope file path.'
            )

        self.kwargs['exposure_data'] = get_exposure_data(self, add_internal_col=True)

        # Run chain
        if self.exposure_pre_analysis_module:
            cmds = [(ExposurePreAnalysis, self.kwargs), (GenerateFiles, self.kwargs), (GenerateLosses, self.kwargs)]
        else:
            cmds = [(GenerateFiles, self.kwargs), (GenerateLosses, self.kwargs)]

        with tqdm(total=len(cmds)) as pbar:
            for cmd in cmds:
                cmd[0](**cmd[1]).run()
                pbar.update(1)

        self.logger.info('\nModel run completed successfully in {}'.format(self.model_run_dir))
