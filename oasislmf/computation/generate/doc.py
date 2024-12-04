import os
from oasislmf.computation.base import ComputationStep
from oasislmf.utils.data import get_utctimestamp


class GenerateModelDocumentation(ComputationStep):
    """
    Generates Model Documentation from schema provided in the model config file
    """
    # Command line options
    step_params = [
        {'name': 'doc_out_dir', 'flag': '-o', 'is_path': True, 'pre_exist': False,
         'help': 'Path to the directory in which to generate the Documentation files'},
        {'name': 'doc_json', 'flag': '-d', 'is_path': True, 'pre_exist': True, 'required': False,
         'help': 'The json file containing model meta-data for documentation'},
        {'name': 'doc_schema_info', 'flag': '-s', 'is_path': True, 'pre_exist': True, 'required': False,
         'help': 'The schema for the model meta-data json'},

    ]
    chained_commands = []

    def _get_output_dir(self):
        if self.doc_out_dir:
            return self.doc_out_dir
        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        return os.path.join(os.getcwd(), 'docs', 'files-{}'.format(utcnow))

    def run(self):
        if not self.doc_json or not os.path.exists(self.doc_json):
            self.logger.warn(f'WARNING: Could not locate doc_json file: {self.doc_json}, Cannot generate documentation')
        if not self.doc_schema_info or not os.path.exists(self.doc_schema_info):
            self.logger.warn(f'WARNING: Could not locate doc_schema_info file: {self.doc_schema_info}, Cannot generate documentation')

        with open(os.path.join(self.doc_out_dir, 'doc.md'), 'w') as md_file:
            md_file.write("# Documentation goes here\n")
