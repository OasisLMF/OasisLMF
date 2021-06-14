__all__ = [
    'ComputationStep',
]

import os
import pathlib
import logging
import json
import inspect
from collections import OrderedDict

from ..utils.data import get_utctimestamp
from ..utils.exceptions import OasisException


class ComputationStep:
    """
    "Abstract" Class for all Computation Step (ExposurePreAnalysis, GulCalc, ...)
    initialise the object with all specified param un step_param and sub- ComputationStep
    provide a generic interface to get the all those parameter definitions (get_params)

    the Run method must be implemented and contain le business execution logic.

    """

    step_params = []
    chained_commands = []

    def __init__(self, **kwargs):
        """
        initialise the ComputationStep objects:
         - do the basic check for required parameter (required)
         - provide default value if defined (default)
         - check path existence (pre_exist)
         - create necessary directories (is_dir, is_path)
        """
        self.logger = logging.getLogger()
        self.kwargs = kwargs
        self.logger.debug(f"{self.__class__.__name__}: " + json.dumps(self.kwargs, indent=4))

        for param in self.get_params():
            param_value = kwargs.get(param['name'])
            if param_value in [None, ""]:
                if param.get('required'):
                    raise OasisException(f"parameter {param['name']} is required "
                                         f"for Computation Step {self.__class__.__name__}")
                else:
                    param_value = param.get('default')

            if param.get('is_path') and param_value is not None:
                if param.get('pre_exist') and not os.path.exists(param_value):
                    raise OasisException(
                        f"The path {param_value} ({param['help']}) "
                        f"must exist for Computation Step {self.__class__.__name__}")
                else:
                    if param.get('is_dir'):
                        pathlib.Path(param_value).mkdir(parents=True, exist_ok=True)
                    else:
                        pathlib.Path(os.path.dirname(param_value)).mkdir(parents=True, exist_ok=True)
            setattr(self, param['name'], param_value)

    @classmethod
    def get_default_run_dir(cls):
        return os.path.join(os.getcwd(), 'runs', f'{cls.run_dir_key}-{get_utctimestamp(fmt="%Y%m%d%H%M%S")}')

    @classmethod
    def get_params(cls):
        """
        return all the params of the computation step defined in step_params
        and the params from the sub_computation step in chained_commands
        if two params have the same name, return the param definition of the first param found only
        this allow to overwrite the param definition of sub step if necessary.
        """
        params = []
        param_names = set()

        def all_params():
            for param in cls.step_params:
                yield param
            for command in cls.chained_commands:
                for param in command.get_params():
                    yield param

        for param in all_params():
            if param['name'] not in param_names:
                param_names.add(param['name'])
                params.append(param)

        return params

    @classmethod 
    def get_arguments(cls, **kwargs):
        """
        Return a list of default arguments values for the functions parameters 
        If given arg values in 'kwargs' these will override the defaults 
        """
        func_args = {el['name']: el.get('default', None) for el in cls.get_params()}
        for param in kwargs: 
            if param in func_args:
                func_args[param] = kwargs[param]
        return func_args


    @classmethod
    def get_signature(cls):
        """ Create a function signature based on the 'get_params()' return
        """
        try:
            # Create keyword params (without default values)
            params = ["{}=None".format(p.get('name')) for p in cls.get_params() if not p.get('default')]

            # Create keyword params (with default values)
            for p in [p for p in cls.get_params() if p.get('default')]:
                if isinstance(p.get('default'), str):
                    params.append("{}='{}'".format(p.get('name'), p.get('default')))
                elif isinstance(p.get('default'), dict):
                    params.append("{}=dict()".format(p.get('name'), p.get('default')))
                elif isinstance(p.get('default'), OrderedDict):
                    params.append("{}=OrderedDict()".format(p.get('name'), p.get('default')))
                else:
                    params.append("{}={}".format(p.get('name'), p.get('default')))

            exec('def func_sig({}): pass'.format(", ".join(params)))
            return inspect.signature(locals()['func_sig'])
        except Exception:
            # ignore any errors in signature creation and return blank
            return None


    def run(self):
        """method that will be call by all the interface to execute the computation step"""
        raise NotImplemented(f'Methode run must be implemented in {self.__class__.__name__}')
