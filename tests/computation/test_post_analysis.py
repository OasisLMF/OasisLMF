from oasislmf.manager import OasisManager


def write_simple_post_analysis_module(module_path, class_name='PostAnalysis'):
    with open(module_path, 'w') as f:
        f.write(f'''
from pathlib import Path
class {class_name}:
    """
    Example of custom module called by oasislmf/model_preparation/PostAnalysis.py
    """
    def __init__(self, raw_output_dir, post_processed_output_dir, **kwargs):
        self.raw_output_dir = raw_output_dir
        self.post_processed_output_dir = post_processed_output_dir

    def run(self):
        # create a file
        new_file_path = Path(self.post_processed_output_dir) / "my_file.txt"
        new_file_path.touch()

        # Write sample data to the file
        with open(new_file_path, 'w') as file:
            file.write("Sample Data for Testing")
''')


def test_create_output_file(tmp_path):
    raw_output_dir = tmp_path / 'raw_output'
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    post_processed_output_dir = tmp_path / 'post_processed_output'

    kwargs = {'post_analysis_module': (tmp_path / 'post_analysis_simple.py').as_posix(),
              'raw_output_dir': raw_output_dir.as_posix(),
              'post_processed_output_dir': post_processed_output_dir.as_posix()}

    write_simple_post_analysis_module(kwargs['post_analysis_module'])

    OasisManager().post_analysis(**kwargs)

    assert (post_processed_output_dir / 'my_file.txt').exists()

    # Check file is accessible
    with open(post_processed_output_dir / 'my_file.txt', 'r') as file:
        data = file.read()
        assert data == "Sample Data for Testing"
