__version__ = '2.5.10'

import sys
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader
import importlib
import warnings

import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
handler = NullHandler()
handler.name = 'oasislmf'
logger.addHandler(handler)


if sys.platform == 'win32':
    warnings.warn(
        "oasislmf is not supported on Windows and may not function correctly.",
        RuntimeWarning,
        stacklevel=2
    )


class MyLoader(Loader):

    def __init__(self, sub_module):
        self.sub_module = sub_module

    def exec_module(self, module):
        old_name = module.__name__
        new_name = f"oasislmf.{self.sub_module}"
        sys.modules[old_name] = importlib.import_module(new_name)
        return sys.modules[old_name]


class MyImport(MetaPathFinder):
    """Support alias of deprecated sub-modules

    * model_execution   -> execution
    * model_preparation -> preparation
    * api, platform     -> platform_api

    Example:
        `from oasislmf.model_execution.bash import genbash`
            is the same as calling the new name
        `from oasislmf.execution.bash import genbash`
        https://docs.python.org/3/library/importlib.html#importlib.machinery.PathFinder
    """

    def __init__(self):
        self.deprecated_modules = {
            "model_execution": "execution",
            "model_preparation": "preparation",
            "api": "platform_api",
            "platform": "platform_api",
        }

    def find_spec(self, fullname, path=None, target=None):
        """Return a spec aliasing ``fullname`` to its new module, or None if it is not deprecated.

        The finder sits first on ``sys.meta_path``: an alias's sub-module (``oasislmf.model_execution.bash``)
        must not reach the path finder, which would find ``execution/bash.py`` through the aliased
        package's ``__path__`` and load a second copy of it under the alias name.
        """
        package, _, import_path = fullname.partition(".")
        if package != "oasislmf":
            return None
        deprecated, _, sub_module = import_path.partition(".")
        new = self.deprecated_modules.get(deprecated)
        if new is None:
            return None
        if not sub_module:  # a sub-module is always imported after its package: warn once, on the package
            warnings.warn(f"imports from 'oasislmf.{deprecated}' are deprecated. Import by using 'oasislmf.{new}' instead.")
        return spec_from_loader(fullname, MyLoader(f"{new}.{sub_module}" if sub_module else new))


sys.meta_path.insert(0, MyImport())
