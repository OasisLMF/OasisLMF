__all__ = [
    'RunModel'
]

import os
from tqdm import tqdm

from ..base import ComputationStep

from ..generate.files import GenerateOasisFiles
from ..generate.losses import GenerateLosses
from ..hooks.pre_analysis import HookPreAnalysis


from ...utils.data import (
    get_analysis_settings,
    get_model_settings,
)


class RunModel(ComputationStep):
    """
    Run models end to end.
    """

    chained_commands = [
        GenerateLosses,
        GenerateOasisFiles,
        HookPreAnalysis,
    ]

    # Combine all arguments for each sub-command
    step_params = list()
    for cmd in chained_commands:
        step_params += cmd.step_params

    # Remove the requirment for pre_analysis_module
    for param in step_params:
        if param['name'] == 'exposure_pre_analysis_module':
            param['required'] = False
            break

    def run(self):

        # setup output paths
        if not self.model_run_dir:
            self.model_run_dir = GenerateLosses._get_output_dir(self)
        self.kwargs['model_run_dir'] = self.model_run_dir
       # self.kwargs['oasis_files_dir'] = os.path.join(self.model_run_dir, 'input')

        # Validate JSON files (Fail at entry point not after input generation)
        if self.analysis_settings_json:
            get_analysis_settings(self.analysis_settings_json)
        if self.model_settings_json:
            get_model_settings(self.model_settings_json)

        # Check input expsoure
        required_ri_paths = [self.oed_info_csv , self.oed_scope_csv]
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

        # Run chain 
        if self.exposure_pre_analysis_module:
            cmds = [HookPreAnalysis(**self.kwargs), GenerateOasisFiles(**self.kwargs), GenerateLosses(**self.kwargs)]
        else:
            cmds = [GenerateOasisFiles(**self.kwargs), GenerateLosses(**self.kwargs)]

        with tqdm(total=len(cmds)) as pbar:
            for cmd in cmds:
                cmd.run()
                pbar.update(1)

        self.logger.info('\nModel run completed successfully in {}'.format(self.model_run_dir))
