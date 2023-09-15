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
    def __init__(self, raw_output_dir, post_processed_output_dir, **kwargs):
        self.raw_output_dir = raw_output_dir
        self.post_processed_output_dir = post_processed_output_dir

    def run(self):
        # Copy all outputs to new directory.
        shutil.copytree(self.raw_output_dir, self.post_processed_output_dir, dirs_exist_ok=True)

        # Create an extra file.
        new_file_path = Path(self.post_processed_output_dir) / "my_file.txt"
        new_file_path.write_text("Sample Data for Testing")
''')


@pytest.fixture(autouse=True)
def change_test_dir(tmp_path, monkeypatch):
    """Run tests in a temp directory to mimic the behaviour of model runs."""  # TODO Confirm this???
    monkeypatch.chdir(tmp_path)


def test_create_output_file(tmp_path):
    raw_output_dir = tmp_path / 'output'  # default path
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    (raw_output_dir / 'gul_S1_aalcalc.csv').write_text("999")  # Create one output file.

    post_processed_output_dir = tmp_path / 'postprocessed_output'  # default path

    kwargs = {'post_analysis_module': (tmp_path / 'post_analysis_simple.py').as_posix()}

    write_simple_post_analysis_module(kwargs['post_analysis_module'])

    OasisManager().post_analysis(**kwargs)

    assert (post_processed_output_dir / 'my_file.txt').read_text() == "Sample Data for Testing"
    assert (post_processed_output_dir / 'gul_S1_aalcalc.csv').read_text() == "999"


def test_create_output_file_non_default_paths(tmp_path):
    raw_output_dir = tmp_path / 'raw_output'  # non-default name
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    (raw_output_dir / 'gul_S1_aalcalc.csv').write_text("999")  # Create one output file.

    post_processed_output_dir = tmp_path / 'processed_output'  # non-default name

    kwargs = {'post_analysis_module': (tmp_path / 'post_analysis_simple.py').as_posix(),
              'raw_output_dir': raw_output_dir.as_posix(),
              'post_processed_output_dir': post_processed_output_dir.as_posix()}

    write_simple_post_analysis_module(kwargs['post_analysis_module'])

    OasisManager().post_analysis(**kwargs)

    assert (post_processed_output_dir / 'my_file.txt').read_text() == "Sample Data for Testing"
    assert (post_processed_output_dir / 'gul_S1_aalcalc.csv').read_text() == "999"
