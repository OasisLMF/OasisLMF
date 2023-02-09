__version__ = '1.26.6'

import sys
import os
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader
import importlib
import warnings

import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
handler = NullHandler()
handler.name='oasislmf'
logger.addHandler(handler)


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
    """ Support alias of depreciated sub-modules

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
        self.depricated_modules = {
            "model_execution": "execution",
            "model_preparation": "preparation",
            "api": "platform",
        }

    def find_spec(self, fullname, path=None, target=None):
        import_path = fullname.split(".",1)
        if fullname.startswith("oasislmf") and len(import_path) > 1:
            import_path = import_path[1]
            for deprecated in self.depricated_modules:
                if deprecated == import_path or import_path.startswith(deprecated+'.'):
                    warnings.simplefilter("always")
                    warnings.warn(
                        f"imports from 'oasislmf.{deprecated}' are deprecated. Import by using 'oasislmf.{self.depricated_modules[deprecated]}' instead."
                    )
                    import_path = import_path.replace(deprecated, self.depricated_modules[deprecated])

            return spec_from_loader(fullname, MyLoader(import_path))

sys.meta_path.append(MyImport())
