__all__ = [
    'OasisManager'
]

import re

from .utils.log import oasis_log

from .computation.hooks.pre_analysis import ExposurePreAnalysis
from .computation.generate.files import (
    GenerateFiles,
    GenerateDummyModelFiles,
    GenerateDummyOasisFiles
)
from .computation.generate.keys import GenerateKeys, GenerateKeysDeterministic
from .computation.generate.losses import (
    GenerateLosses,
    GenerateLossesDir,
    GenerateLossesPartial,
    GenerateLossesOutput,
    GenerateLossesDeterministic,
    GenerateLossesDummyModel
)
from .computation.helper.autocomplete import HelperTabComplete
from .computation.helper.cookiecutter import CreateModelRepo, CreateComplexModelRepo
from .computation.run.generate_files import GenerateOasisFiles
from .computation.run.model import RunModel
from .computation.run.exposure import RunExposure, RunFmTest
from .computation.run.platform import (
    PlatformList,
    PlatformRun,
    PlatformDelete,
    PlatformGet
)


class OasisManager(object):
    computation_classes = [
        ExposurePreAnalysis,
        GenerateFiles,
        GenerateOasisFiles,
        GenerateKeys,
        GenerateKeysDeterministic,

        GenerateLosses,
        GenerateLossesDir,
        GenerateLossesPartial,
        GenerateLossesOutput,

        GenerateLossesDeterministic,
        GenerateLossesDummyModel,
        GenerateDummyModelFiles,
        GenerateDummyOasisFiles,
        RunModel,
        RunExposure,
        RunFmTest,
        PlatformList,
        PlatformRun,
        PlatformDelete,
        PlatformGet,
        HelperTabComplete,
        CreateModelRepo,
        CreateComplexModelRepo,
    ]
    computations_params = {}

    @staticmethod
    def computation_name_to_method(name):
        """
        generate the name of the method in manager for a given ComputationStep name
        taken from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case

        >>> OasisManager.computation_name_to_method('ExposurePreAnalysis')
        'exposure_pre_analysis'
        >>> OasisManager.computation_name_to_method('EODFile')
        'eod_file'
        >>> OasisManager.computation_name_to_method('Model1Data')
        'model1_data'
        """
        return re.sub('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))', r'_\1', name).lower()

    def consolidate_input(self, computation_cls, kwargs):
        for param in computation_cls.get_params():
            if kwargs.get(param['name']) is None:
                kwargs[param['name']] = getattr(self, param['name'], None)
        return kwargs


def __interface_factory(computation_cls):
    @oasis_log
    def interface(self, **kwargs):
        self.consolidate_input(computation_cls, kwargs)
        return computation_cls(**kwargs).run()

    OasisManager.computations_params[computation_cls.__name__] = computation_cls.get_params()
    interface.__signature__ = computation_cls.get_signature()
    interface.__doc__ = computation_cls.__doc__
    return interface


for computation_cls in OasisManager.computation_classes:
    # set run() computation_cls functions as attributes 
    setattr(OasisManager, OasisManager.computation_name_to_method(computation_cls.__name__),
            __interface_factory(computation_cls))

    # set get_arguments() computation_cls funcs as attributes
    setattr(OasisManager, '_params_' + OasisManager.computation_name_to_method(computation_cls.__name__),
        computation_cls.get_arguments)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
