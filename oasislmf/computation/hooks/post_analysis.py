__all__ = [
    'PostAnalysis'
]

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
        {'name': 'raw_output_dir', 'default': 'output', 'is_path': True, 'pre_exist': True,
         'help': 'path to oasis output directory'},
        {'name': 'post_processed_output_dir', 'default': 'postprocessed_output', 'is_path': True,
         'pre_exist': False, 'help': 'path to post-processed output directory'},
    ]

    run_dir_key = 'post-analysis'

    def run(self):
        _module = get_custom_module(self.post_analysis_module, 'Post-Analysis module path')
        kwargs = dict()

        kwargs['raw_output_dir'] = self.raw_output_dir
        kwargs['post_processed_output_dir'] = self.post_processed_output_dir

        Path(self.post_processed_output_dir).mkdir()

        try:
            _class = getattr(_module, self.post_analysis_class_name)
        except AttributeError as e:
            raise OasisException(f"Class {self.post_analysis_class_name} "
                                 f"is not defined in module {self.post_analysis_module}") from e.__cause__

        _class(**kwargs).run()
