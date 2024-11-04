import filecmp
import shutil
from tempfile import TemporaryDirectory
import numpy as np
from pathlib import Path
from unittest.mock import patch

from oasislmf.pytools.plt.manager import main
from oasislmf.pytools.common.data import (oasis_int, oasis_float)

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_pltpy")


def test_splt_output():
    # case_runner("splt")
    pass


def test_mplt_output():
    # case_runner("mplt")
    pass


def test_qplt_output():
    # case_runner("qplt")
    pass
