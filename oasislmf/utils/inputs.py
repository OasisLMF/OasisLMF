__all__ = [
    'InputValues',
    'update_config',
    'has_oasis_env',
    'get_oasis_env',
    'str2bool'
]

import io
import os
import json
import logging

from ods_tools.oed.settings import SettingHandler
from ..utils.defaults import get_config_profile
from ..utils.exceptions import OasisException
from json.decoder import JSONDecodeError
from argparse import ArgumentTypeError


def update_config(config_data, config_map=get_config_profile()):
    return SettingHandler(compatibility_profiles=[config_map]).update_obsolete_keys(config_data)


def has_oasis_env(name):
    return f'OASIS_{name.upper()}' in os.environ


def get_oasis_env(name, dtype=None, default=None):
    env_var = os.getenv(f'OASIS_{name.upper()}', default=default)
    if dtype and env_var:
        return dtype(env_var)
    else:
        return env_var


class InputValues(object):
    """
    Helper class for accessing the input values from either
    the command line or the configuration file.

    internal_update

    """

    def __init__(self, args, update_keys=True):
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.config = {}
        self.config_fp = self.get('config', is_path=True)
        self.config_mapping = get_config_profile()

        if self.config_fp is not None:
            try:
                self.config = update_config(self.load_config_file())
                self.config_dir = os.path.dirname(self.config_fp)
                self.list_unknown_keys()
            except JSONDecodeError as e:
                raise OasisException(f"Configuration file {self.config_fp} is not a valid json file", e)

    def list_unknown_keys(self):
        """
        List all Unknown keys set in the 'oasislmf.json' file
        """
        valid_arg_names = set(arg[0] for arg in self.args._get_kwargs())
        config_arg_names = set(self.config.keys())
        unknown_args = config_arg_names - valid_arg_names - set(self.config_mapping.keys())

        if unknown_args:
            self.logger.warning('Warning: Unknown options(s) set in MDK config:')
            for k in unknown_args:
                self.logger.warning('   {} : {}'.format(
                    k,
                    self.config[k]
                ))

    def load_config_file(self):
        try:
            with io.open(self.config_fp, 'r', encoding='utf-8') as f:
                return {k.lower(): v for k, v in json.load(f).items()}
        except FileNotFoundError:
            raise OasisException('MDK config. file path {} provided does not exist'.format(self.config_fp))

    def write_config_file(self, config_fp):
        with io.open(config_fp, 'w', encoding='utf-8') as f:
            f.write(u'{}'.format(json.dumps(self.config, sort_keys=True, indent=4, ensure_ascii=False)))

    def confirm_action(self, question_str, no_confirm=False):
        self.logger.debug('Prompt user for confirmation')
        if no_confirm:
            return True
        try:
            check = str(input("%s (Y/N): " % question_str)).lower().strip()
            if check[:1] == 'y':
                return True
            elif check[:1] == 'n':
                return False
            else:
                self.logger.error('Enter "y" for Yes, "n" for No or Ctrl-C to exit.\n')
                return self.confirm_action(question_str)
        except KeyboardInterrupt:
            self.logger.error('\nexiting.')

    def get(self, name, default=None, required=False, is_path=False, dtype=None):
        """
        Gets the name parameter until found from:
          - the command line arguments.
          - the configuration file
          - the environment variable (put in uppercase)

        If it is not found then ``default`` is returned
        unless ``required`` is True in which case an ``OasisException`` is raised.

        :param name: The name of the parameter to lookup
        :type name: str

        :param default: The default value to return if the name is not
            found on the command line or in the configuration file.

        :param required: Flag whether the value is required, if so and
            the parameter is not found on the command line or in the
            configuration file an error is raised.
        :type required: bool

        :param is_path: Flag whether the value should be treated as a path and return an abspath,
            use config_dir as base dir if value comes from the config
        :type is_path: bool

        :param dtype: the class <type> of the value, if 'None' load as string by default
        :type: class

        :raise OasisException: If the value is not found and ``required``
            is True

        :return: The found value or the default
        """
        # Load order 0:  Get from CLI flag
        source = 'arg'
        value = getattr(self.args, name, None)

        # Load order 1: ENV override (intended for worker images)
        if str2bool(os.getenv('OASIS_ENV_OVERRIDE', default=False)) and has_oasis_env(name):
            source = 'env_override'
            value = get_oasis_env(name, dtype)

        # Load order 2: Get from config JSON
        if value is None:
            source = 'config'
            value = self.config.get(name)

        # Load order 3: Get from environment variable
        if value is None:
            source = 'env'
            value = get_oasis_env(name, dtype)

        if value is None and required:
            raise OasisException(
                'Required argument {} could not be found in the command args or the MDK config. file'.format(name)
            )

        # Load order 4: Get default value
        if value is None:
            source = 'default'
            value = default

        if is_path and value not in [None, ""] and not os.path.isabs(value):
            if source == 'config':
                value = os.path.join(self.config_dir, value)
            else:
                value = os.path.abspath(value)

        # Warn user of environment variable load
        if source == 'env_override':
            self.logger.warning(f'Warning - environment variable override: OASIS_{name.upper()}={value}')

        return value


def str2bool(v):
    """ Func type for loading strings to boolean values using argparse
        https://stackoverflow.com/a/43357954

        step_params:
            use: `'default': False, 'type': str2bool, 'const':True, 'nargs':'?', ...`

        CLI:
            oasislmf --some-flag
            oasislmf --some-flag <bool>

        oasislmf.json
        {"some_flag": true, ...}
    """
    if v is None:
        return v
    elif isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')
