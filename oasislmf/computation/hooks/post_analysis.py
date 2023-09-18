__all__ = [
    'PostAnalysis'
]

import os
from pathlib import Path

from ..base import ComputationStep
from ...utils.path import get_custom_module
from ...utils.exceptions import OasisException


class PostAnalysis(ComputationStep):
    """
    """
    step_params = [
        {'name': 'post_analysis_module', 'required': True, 'is_path': True, 'pre_exist': True,
         'help': 'Post-Analysis module path'},
        {'name': 'post_analysis_class_name', 'default': 'PostAnalysis',
         'help': 'Name of the class to use for the post_analysis'},
        {'name': 'model_run_dir', 'is_path': True, 'pre_exist': False,
         'help': 'Model run directory path'},
    ]

    run_dir_key = 'post-analysis'

    def run(self):
        _module = get_custom_module(self.post_analysis_module, 'Post-Analysis module path')
        kwargs = dict()

        raw_output_dir = os.path.join(self.model_run_dir, "output")
        kwargs['raw_output_dir'] = raw_output_dir
        post_processed_output_dir = os.path.join(raw_output_dir, "postprocessed")
        kwargs['post_processed_output_dir'] = post_processed_output_dir

        Path(post_processed_output_dir).mkdir()

        try:
            _class = getattr(_module, self.post_analysis_class_name)
        except AttributeError as e:
            raise OasisException(f"Class {self.post_analysis_class_name} "
                                 f"is not defined in module {self.post_analysis_module}") from e.__cause__

        _class(**kwargs).run()
