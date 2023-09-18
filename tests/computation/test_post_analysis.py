import pytest
from oasislmf.manager import OasisManager


def write_simple_post_analysis_module(module_path, class_name='PostAnalysis'):
    with open(module_path, 'w') as f:
        f.write(f'''
from pathlib import Path
import shutil

class {class_name}:
    """
    Example of custom module called by oasislmf/model_preparation/PostAnalysis.py
    """
    def __init__(self, output_dir, **kwargs):
        self.output_dir = output_dir

    def run(self):
        post_processed_output_dir = Path(self.output_dir) / "postprocessed"

        # Copy all outputs to new directory.
        shutil.copytree(self.output_dir, post_processed_output_dir, dirs_exist_ok=True)

        # Create an extra file.
        (post_processed_output_dir / "my_file.txt").write_text("Sample Data for Testing")
''')


def test_create_output_file(tmp_path):
    model_run_dir = tmp_path
    raw_output_dir = model_run_dir / 'output'
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    (raw_output_dir / 'gul_S1_aalcalc.csv').write_text("999")  # Create one output file.

    post_processed_output_dir = raw_output_dir / 'postprocessed'

    kwargs = {'post_analysis_module': (tmp_path / 'post_analysis_simple.py').as_posix(),
              'model_run_dir': tmp_path.as_posix()}

    write_simple_post_analysis_module(kwargs['post_analysis_module'])

    OasisManager().post_analysis(**kwargs)

    assert (post_processed_output_dir / 'my_file.txt').read_text() == "Sample Data for Testing"
    assert (post_processed_output_dir / 'gul_S1_aalcalc.csv').read_text() == "999"


def test_create_output_file_non_defaults(tmp_path):
    model_run_dir = tmp_path
    raw_output_dir = model_run_dir / 'output'
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    (raw_output_dir / 'gul_S1_aalcalc.csv').write_text("999")  # Create one output file.

    post_processed_output_dir = raw_output_dir / 'postprocessed'

    kwargs = {'post_analysis_module': (tmp_path / 'post_analysis_simple.py').as_posix(),
              'model_run_dir': tmp_path.as_posix(),
              'post_analysis_class_name': 'MyClass'}

    write_simple_post_analysis_module(kwargs['post_analysis_module'], class_name='MyClass')

    OasisManager().post_analysis(**kwargs)

    assert (post_processed_output_dir / 'my_file.txt').read_text() == "Sample Data for Testing"
    assert (post_processed_output_dir / 'gul_S1_aalcalc.csv').read_text() == "999"
