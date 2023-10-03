
__all__ = [
    'GenerateOasisLosses'
]

from tqdm import tqdm

from ..base import ComputationStep
from ..generate.losses import GenerateLosses
from ..hooks.post_analysis import PostAnalysis


class GenerateOasisLosses(ComputationStep):
    """
    Run Oasis file geneartion with optional PreAnalysis hook.
    """

    # Override params
    step_params = [
        {'name': 'post_analysis_module', 'required': False, 'is_path': True, 'pre_exist': True,
         'help': 'Post-Analysis module path'},
    ]
    # Add params from each sub command not in 'step_params'
    chained_commands = [
        GenerateLosses,
        PostAnalysis,
    ]

    def run(self):

        # setup output dir
        if not self.model_run_dir:
            self.model_run_dir = GenerateLosses._get_output_dir(self)
        self.kwargs['model_run_dir'] = self.model_run_dir

        # Run chain
        cmds = [(GenerateLosses, self.kwargs)]
        if self.post_analysis_module:
            cmds += [(PostAnalysis, self.kwargs)]

        with tqdm(total=len(cmds)) as pbar:
            for cmd in cmds:
                cmd[0](**cmd[1]).run()
                pbar.update(1)

        self.logger.info(f'Losses generated in {self.model_run_dir}')
