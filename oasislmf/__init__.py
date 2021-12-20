__version__ = '1.15.22'

import sys
import os
from importlib.abc import MetaPathFinder, Loader
import importlib
import warnings

import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


class MyLoader(Loader):

    def __init__(self, sub_module):
        self.sub_module = sub_module

    def module_repr(self, module):
        return repr(module)

    def load_module(self, fullname):
        old_name = fullname
        names = fullname.split(".")
        names[1] = self.sub_module
        fullname = ".".join(names)
        module = importlib.import_module(fullname)
        sys.modules[old_name] = module
        return module


class MyImport(MetaPathFinder):
    """ Support alias of depreciated sub-modules

        * model_execution   -> execution
        * model_preparation -> preparation
        * api               -> platform

        Example:
            `from oasislmf.model_execution.bash import genbash`
                is the same as calling the new name
            `from oasislmf.execution.bash import genbash`
    """

    def __init__(self):
        self.depricated_modules = {
            "model_execution": "execution",
            "model_preparation": "preparation",
            "api": "platform",
        }

    def find_module(self, fullname, path=None):
        names = fullname.split(".")
        if len(names) >= 2:
            if names[0] == "oasislmf" and names[1] in self.depricated_modules:
                deprecated = "imports from 'oasislmf.{}' are deprecated. Import by using 'oasislmf.{}' instead.".format(
                    names[1],
                    self.depricated_modules[names[1]]
                )
                warnings.simplefilter("always")
                warnings.warn(deprecated)
                return MyLoader(self.depricated_modules[names[1]])

sys.meta_path.append(MyImport())
