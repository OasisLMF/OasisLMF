import json

import pytest

from oasislmf.manager import OasisManager
from oasislmf.utils.exceptions import OasisException


def write_simple_post_analysis_module(module_path, class_name='PostAnalysis'):
    with open(module_path, 'w') as f:
        f.write(f'''
from pathlib import Path
import shutil
import json

class {class_name}:
    """
    Example of custom module called by oasislmf/model_preparation/PostAnalysis.py
    """
    def __init__(self, output_dir, analysis_settings_json=None, **kwargs):
        self.output_dir = output_dir
        self.analysis_settings_json = analysis_settings_json

    def run(self):
        post_processed_output_dir = Path(self.output_dir) / "postprocessed"

        # Copy all outputs to new directory.
        shutil.copytree(self.output_dir, post_processed_output_dir, dirs_exist_ok=True)

        # Create an extra file.
        (post_processed_output_dir / "my_file.txt").write_text("Sample Data for Testing")

        # Read settings and create file
        if self.analysis_settings_json:
            settings = json.loads(Path(self.analysis_settings_json).read_text())
            (post_processed_output_dir / settings["file_to_create"]).touch()
''')


def test_postanalysis(tmp_path):
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


def test_postanalysis_non_defaults(tmp_path):
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


def test_postanalysis_wrong_module_name(tmp_path):
    model_run_dir = tmp_path
    raw_output_dir = model_run_dir / 'output'
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    (raw_output_dir / 'gul_S1_aalcalc.csv').write_text("999")  # Create one output file.

    kwargs = {'post_analysis_module': (tmp_path / 'post_analysis_simple.py').as_posix(),
              'model_run_dir': tmp_path.as_posix(),
              'post_analysis_class_name': 'IncorrectClassName'}

    write_simple_post_analysis_module(kwargs['post_analysis_module'])

    with pytest.raises(OasisException, match="Class IncorrectClassName is not defined in module*"):
        OasisManager().post_analysis(**kwargs)


def test_postanalysis_with_settings(tmp_path):
    model_run_dir = tmp_path
    raw_output_dir = model_run_dir / 'output'
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    (raw_output_dir / 'gul_S1_aalcalc.csv').write_text("999")  # Create one output file.

    settings_path = tmp_path / "analysis_settings.json"
    settings_path.write_text(json.dumps({"file_to_create": "hello.txt"}))

    post_processed_output_dir = raw_output_dir / 'postprocessed'

    kwargs = {'post_analysis_module': (tmp_path / 'post_analysis_simple.py').as_posix(),
              'model_run_dir': tmp_path.as_posix(),
              'analysis_settings_json': settings_path.as_posix()}

    write_simple_post_analysis_module(kwargs['post_analysis_module'])

    OasisManager().post_analysis(**kwargs)

    assert (post_processed_output_dir / 'my_file.txt').read_text() == "Sample Data for Testing"
    assert (post_processed_output_dir / 'gul_S1_aalcalc.csv').read_text() == "999"
    assert (post_processed_output_dir / 'hello.txt').is_file()
