__all__ = [
    'ComputationStep',
]

import os
import pathlib
import logging
import json
import inspect
from ods_tools.oed import OedSource
from ods_tools.oed.settings import Settings
from collections import OrderedDict

from ..utils.data import get_utctimestamp
from ..utils.exceptions import OasisException
from ..utils.inputs import update_config, str2bool, has_oasis_env, get_oasis_env, ArgumentTypeError
from oasislmf.utils.log import oasis_log


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
        self.logger = logging.getLogger(__name__)
        self.kwargs = kwargs
        self.logger.debug(f"{self.__class__.__name__}: " + json.dumps(self.kwargs, indent=4, default=str))
        self.run = oasis_log(self.run)

        # set initial value for mdk params
        for param in self.get_params():
            param_value = self._get_init_value(param, kwargs)
            setattr(self, param['name'], param_value)

        # read and merge settings files
        settings = Settings()
        for settings_info in self.get_params(param_type="settings"):
            setting_fp = kwargs.get(settings_info['name'])
            if setting_fp:
                new_settings = settings_info['loader'](setting_fp)
                settings.add_settings(new_settings, settings_info.get('user_role'))
        self.settings = settings.get_settings()

    def _get_init_value(self, param, kwargs):
        param_value = kwargs.get(param['name'])
        if param_value in [None, ""]:
            if param.get('required'):
                raise OasisException(f"parameter {param['name']} is required "
                                     f"for Computation Step {self.__class__.__name__}")
            else:
                param_value = param.get('default')

        if (getattr(param.get('type'), '__name__', None) == 'str2bool') and (not isinstance(param_value, bool)):
            try:
                param_value = str2bool(param_value)
            except ArgumentTypeError:
                raise OasisException(
                    f"The parameter '{param.get('name')}' has an invalid value '{param_value}' for boolean. Valid strings are (case insensitive):"
                    "\n  True:  ['yes', 'true', 't', 'y', '1']"
                    "\n  False: ['no', 'false', 'f', 'n', '0']"
                )

        if (param.get('is_path')
                and param_value is not None
                and not isinstance(param_value, OedSource)):
            if param.get('pre_exist') and not os.path.exists(param_value):
                raise OasisException(
                    f"The path {param_value} ({param['help']}) "
                    f"must exist for Computation Step {self.__class__.__name__}")
            else:
                if param.get('is_dir'):
                    pathlib.Path(param_value).mkdir(parents=True, exist_ok=True)
                else:
                    pathlib.Path(os.path.dirname(param_value)).mkdir(parents=True, exist_ok=True)
            param_value = str(param_value)
        return param_value

    @classmethod
    def get_default_run_dir(cls):
        return os.path.join(os.getcwd(), 'runs', f'{cls.run_dir_key}-{get_utctimestamp(fmt="%Y%m%d%H%M%S")}')

    @classmethod
    def get_params(cls, param_type="step"):
        """
        return all the params of the computation step defined in step_params
        and the params from the sub_computation step in chained_commands
        if two params have the same name, return the param definition of the first param found only
        this allow to overwrite the param definition of sub step if necessary.
        """
        params = {}

        def all_params():
            for _param in getattr(cls, f"{param_type}_params", []):
                yield _param
            for command in cls.chained_commands:
                for _param in command.get_params(param_type=param_type):
                    yield _param

        for param in all_params():
            if param['name'] not in params:
                params[param['name']] = param

        return list(params.values())

    @classmethod
    def get_arguments(cls, **kwargs):
        """
        Return a list of default arguments values for the functions parameters
        If given arg values in 'kwargs' these will override the defaults
        """
        func_args = {el['name']: el.get('default', None) for el in cls.get_params()}
        type_map = {el['name']: el.get('type', None) for el in cls.get_params()}

        func_kwargs = update_config(kwargs)
        env_override = str2bool(os.getenv('OASIS_ENV_OVERRIDE', default=False))

        for param in func_args:
            if env_override and has_oasis_env(param):
                func_args[param] = get_oasis_env(param, type_map[param])
            elif param in func_kwargs:
                func_args[param] = func_kwargs[param]
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

    @classmethod
    def get_computation_settings_json_schema(cls):
        """
            return a json schema equivalent to validate the input of the command line
        """

        arg_type_to_json_type = {
            str: "string",
            int: "number",
            float: "number",
            str2bool: "boolean",
        }

        def get_json_type(_param):
            if _param.get('type') in arg_type_to_json_type:
                return arg_type_to_json_type[_param.get('type')]
            elif _param.get('is_path'):
                return "string"
            elif _param.get('default') in [True, False]:
                return "boolean"
            elif isinstance(_param.get('default'), dict):
                return "object"
            elif isinstance(_param.get('default'), list):
                return "array"
            elif isinstance(_param.get('default'), str):
                return "string"
            else:
                return "string"

        json_schema = {
            "$schema": "http://oasislmf.org/computation_settings/draft/schema#",
            "type": "object",
            "title": "Computation settings.",
            "description": "Specifies the computation settings and outputs for an analysis.",
            "additionalProperties": False,
            "properties": {}
        }
        settings_param_names = [param['name'] for param in cls.get_params(param_type="settings")]
        for param in cls.get_params():
            if param['name'] in settings_param_names:  # param is a json settings and therefore cannot be in the settings schema
                continue
            param_schema = {"type": get_json_type(param)}
            if param.get('help'):
                param_schema["description"] = param['help']
            if param.get('choices'):
                param_schema["enum"] = param.get('choices')
            json_schema["properties"][param['name']] = param_schema
        return json_schema

    def run(self):
        """method that will be call by all the interface to execute the computation step"""
        raise NotImplemented(f'Methode run must be implemented in {self.__class__.__name__}')
