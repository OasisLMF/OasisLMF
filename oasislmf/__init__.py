__version__ = '2.5.9'

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

    def module_repr(self, module):
        return repr(module)

    def exec_module(self, module):
        old_name = module.__name__
        new_name = f"oasislmf.{self.sub_module}"
        sys.modules[old_name] = importlib.import_module(new_name)
        return sys.modules[old_name]


class MyImport(MetaPathFinder):
    """Support alias of depreciated sub-modules

    * model_execution   -> execution
    * model_preparation -> preparation
    * api               -> platform

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
            "api": "platform",
            "platform": "platform_api"
        }

    def find_spec(self, fullname, path=None, target=None):
        """Return a spec aliasing ``fullname`` to its new module, or None if it is not deprecated.

        The finder sits first on ``sys.meta_path``: an alias's sub-module (``oasislmf.model_execution.bash``)
        must not reach the path finder, which would find ``execution/bash.py`` through the aliased
        package's ``__path__`` and load a second copy of it under the alias name.
        """
        package, _, import_path = fullname.partition(".")
        if package != "oasislmf" or not import_path:
            return None
        aliased = False
        for deprecated, new in self.deprecated_modules.items():
            if deprecated == import_path or import_path.startswith(deprecated + '.'):
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(
                        f"imports from 'oasislmf.{deprecated}' are deprecated. Import by using 'oasislmf.{new}' instead."
                    )
                import_path = new + import_path[len(deprecated):]
                aliased = True
        if not aliased:
            return None
        return spec_from_loader(fullname, MyLoader(import_path))


sys.meta_path.insert(0, MyImport())
