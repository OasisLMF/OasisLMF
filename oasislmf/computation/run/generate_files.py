
__all__ = [
    'GenerateOasisFiles'
]

import os

from tqdm import tqdm

from ..base import ComputationStep
from ..generate.files import GenerateFiles
from ..hooks.pre_analysis import ExposurePreAnalysis
from ...utils.defaults import store_exposure_fp


class GenerateOasisFiles(ComputationStep):
    """
    Run Oasis file geneartion with optional PreAnalysis hook.
    """

    # Override params
    step_params = [
        {'name': 'exposure_pre_analysis_module', 'required': False, 'is_path': True, 'pre_exist': True, 'help': 'Exposure Pre-Analysis lookup module path'},
    ]
    # Add params from each sub command not in 'step_params'
    chained_commands = [
        GenerateFiles,
        ExposurePreAnalysis,
    ]

    def pre_analysis_kwargs(self):
        updated_inputs = {}
        input_dir = self.kwargs['oasis_files_dir']

        for input_name in ('oed_location_csv', 'oed_accounts_csv', 'oed_info_csv', 'oed_scope_csv'):
            if self.kwargs[input_name]:
                updated_inputs[input_name] = os.path.join(
                    input_dir,
                    store_exposure_fp(self.kwargs[input_name], input_name)
                )
        return {**self.kwargs, **updated_inputs}

    def run(self):
        # setup input dir
        if not self.oasis_files_dir:
            self.oasis_files_dir = GenerateFiles._get_output_dir(self)

        # create input dir
        if not os.path.exists(self.oasis_files_dir):
            os.makedirs(self.oasis_files_dir)

        # Run chain
        self.kwargs['oasis_files_dir'] = self.oasis_files_dir
        if self.exposure_pre_analysis_module:
            cmds = [(ExposurePreAnalysis, self.kwargs), (GenerateFiles, self.pre_analysis_kwargs())]
        else:
            cmds = [(GenerateFiles, self.kwargs)]

        with tqdm(total=len(cmds)) as pbar:
            for cmd in cmds:
                cmd[0](**cmd[1]).run()
                pbar.update(1)

        self.logger.info('\nGenerate Files completed successfully in {}'.format(self.oasis_files_dir))
