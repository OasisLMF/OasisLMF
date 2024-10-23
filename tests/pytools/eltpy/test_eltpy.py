import filecmp
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_eltpy")

