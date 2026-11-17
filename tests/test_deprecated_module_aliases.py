import subprocess
import sys
from pathlib import Path

import pytest

IDENTITY_CHECK = """
import sys
import warnings
warnings.simplefilter("ignore")
import {first}
import {second}
import oasislmf.execution
import oasislmf.preparation
import oasislmf.platform_api
assert sys.modules["oasislmf.model_execution.bash"] is sys.modules["oasislmf.execution.bash"] is oasislmf.execution.bash
assert sys.modules["oasislmf.model_preparation.gul_inputs"] is sys.modules["oasislmf.preparation.gul_inputs"]
assert oasislmf.preparation.gul_inputs is sys.modules["oasislmf.preparation.gul_inputs"]
assert sys.modules["oasislmf.api.client"] is sys.modules["oasislmf.platform_api.client"] is oasislmf.platform_api.client
"""

ALIASES = "oasislmf.model_execution.bash, oasislmf.model_preparation.gul_inputs, oasislmf.api.client"
REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES = "oasislmf.execution.bash, oasislmf.preparation.gul_inputs, oasislmf.platform_api.client"


@pytest.mark.parametrize("first, second", [(ALIASES, MODULES), (MODULES, ALIASES)], ids=["alias_first", "module_first"])
def test_deprecated_alias_is_the_same_module(first, second):
    """A deprecated alias resolves to the module object already registered under its new name.

    Importing ``oasislmf.model_execution.bash`` used to load a second copy of ``execution/bash.py`` and rebind
    ``oasislmf.execution.bash`` to it, so ``mock.patch('oasislmf.execution.bash.x')`` could patch the wrong copy.
    Checked in a fresh interpreter, as other tests may already have imported these modules.
    """
    result = subprocess.run([sys.executable, "-c", IDENTITY_CHECK.format(first=first, second=second)],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr


def test_unknown_submodule_is_not_aliased():
    with pytest.raises(ModuleNotFoundError):
        __import__("oasislmf.no_such_module")
