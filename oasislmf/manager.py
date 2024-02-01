__all__ = [
    'OasisManager'
]

import re

from oasislmf.computation.generate.files import (GenerateDummyModelFiles,
                                                 GenerateDummyOasisFiles,
                                                 GenerateFiles)
from oasislmf.computation.generate.keys import (GenerateKeys,
                                                GenerateKeysDeterministic)
from oasislmf.computation.generate.losses import (GenerateLosses,
                                                  GenerateLossesDeterministic,
                                                  GenerateLossesDir,
                                                  GenerateLossesDummyModel,
                                                  GenerateLossesOutput,
                                                  GenerateLossesPartial)
from oasislmf.computation.helper.autocomplete import HelperTabComplete
from oasislmf.computation.hooks.pre_analysis import ExposurePreAnalysis
from oasislmf.computation.hooks.post_analysis import PostAnalysis
from oasislmf.computation.run.exposure import RunExposure, RunFmTest
from oasislmf.computation.run.generate_files import GenerateOasisFiles
from oasislmf.computation.run.generate_losses import GenerateOasisLosses
from oasislmf.computation.run.model import RunModel
from oasislmf.computation.run.platform import (PlatformDelete, PlatformGet,
                                               PlatformList, PlatformRun,
                                               PlatformRunInputs,
                                               PlatformRunLosses)
from oasislmf.utils.log import oasis_log


class OasisManager(object):
    computation_classes = [
        ExposurePreAnalysis,
        GenerateFiles,
        GenerateOasisFiles,
        GenerateOasisLosses,
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
        PostAnalysis,
        RunExposure,
        RunFmTest,
        PlatformList,
        PlatformRun,
        PlatformRunInputs,
        PlatformRunLosses,
        PlatformDelete,
        PlatformGet,
        HelperTabComplete,
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
