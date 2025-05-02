
__all__ = [
    'GenerateDocumentation'
]

import os

from ..base import ComputationStep
from ..generate.doc import GenerateModelDocumentation


class GenerateDocumentation(ComputationStep):
    """
    Generate Documentation for model from the config file
    """

    # Override params
    step_params = []
    # Add params from each sub command not in 'step_params'
    chained_commands = [
        GenerateModelDocumentation,
    ]

    def run(self):
        #  setup documentation output dir
        if not self.doc_out_dir:
            self.doc_out_dir = GenerateModelDocumentation._get_output_dir(self)

        # create documentation output dir
        if not os.path.exists(self.doc_out_dir):
            os.makedirs(self.doc_out_dir)

        # generate Model Documentation
        self.kwargs['doc_out_dir'] = self.doc_out_dir
        GenerateModelDocumentation(**self.kwargs).run()

        # logger info
        self.logger.info('\nGenerate Documentation completed successfully in {}'.format(self.doc_out_dir))
