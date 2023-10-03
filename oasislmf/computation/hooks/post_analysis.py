__all__ = [
    'PostAnalysis'
]

from ..base import ComputationStep
from ...utils.path import get_custom_module
from ...utils.exceptions import OasisException


class PostAnalysis(ComputationStep):
    """Computation step that is called after loss calculations.

    It passes the output directory to a customisable function that might modify or add to the
    standard output files.
    """
    step_params = [
        {'name': 'post_analysis_module', 'required': True, 'is_path': True, 'pre_exist': True,
         'help': 'Post-Analysis module path'},
        {'name': 'post_analysis_class_name', 'default': 'PostAnalysis',
         'help': 'Name of the class to use for the post_analysis'},
        {'name': 'model_run_dir', 'is_path': True, 'pre_exist': False,
         'help': 'Model run directory path'},
        {'name': 'model_data_dir', 'is_path': True, 'pre_exist': True,
         'help': 'Model data directory path'},
        {'name': 'analysis_settings_json', 'is_path': True, 'pre_exist': True,
         'help': 'Analysis settings JSON file path'},
        {'name': 'model_data_dir', 'is_path': True, 'pre_exist': True,
         'help': 'Model data directory path'},
    ]

    run_dir_key = 'post-analysis'

    def run(self):
        kwargs = {
            'model_data_dir': self.model_data_dir,
            'analysis_settings_json': self.analysis_settings_json,
            'model_run_dir': self.model_run_dir,
        }

        _module = get_custom_module(self.post_analysis_module, 'Post-Analysis module path')
        try:
            _class = getattr(_module, self.post_analysis_class_name)
        except AttributeError as e:
            raise OasisException(f"Class {self.post_analysis_class_name} "
                                 f"is not defined in module {self.post_analysis_module}") from e.__cause__

        _class(**kwargs).run()
