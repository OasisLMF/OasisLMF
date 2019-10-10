__all__ = [
    'InputValues',
]

import io
import os
import json
import logging

from ..utils.defaults import get_config_profile
from ..utils.exceptions import OasisException


class InputValues(object):
    """
    Helper class for accessing the input values from either
    the command line or the configuration file.

    internal_update

    """
    def __init__(self, args, update_keys=True):
        self.args = args
        self.config_fp = None
        self.config = {}

        self.logger = logging.getLogger()
        self.config_mapping = get_config_profile()
        self.obsolete_keys = None

        try:
            self.config_fp = os.path.abspath(args.config)
        except (AttributeError, OSError, TypeError):
            pass
        else:
            self.config_dir = os.path.dirname(self.config_fp)
            self.config = self.load_config_file(self.config_dir)
            self.obsolete_keys = (
                set(self.config.keys()).intersection(set(self.config_mapping.keys()))
            )
            if update_keys:
                self.update_config_keys()

    def list_obsolete_keys(self):
        if len(self.obsolete_keys) > 0:
            self.logger.warn('Depricated key(s) in MDK config:')
            for k in self.obsolete_keys:
                self.logger.warn('   {} : {}'.format(
                    k,
                    self.config_mapping[k],
                ))
            self.logger.warn('')

    def has_obsolete_keys(self):
        return len(self.obsolete_keys) > 0

    def update_config_keys(self, warn_user=True):
        """
        If command line flags change between package versions, update them intenrally and warn the user
        """
        if self.has_obsolete_keys():
            for key in self.obsolete_keys:
                if not self.config_mapping[key]['deleted']:
                    self.config[self.config_mapping[key]['updated_to']] = self.config[key]
                del self.config[key]

            if warn_user:
                self.logger.warn(
                    'Depricated key names found in MDK config: {} \n'
                    '  To fix run: oasislmf config update'.format(self.config_fp)
                )

    def load_config_file(self, config_fp):
        try:
            with io.open(self.config_fp, 'r', encoding='utf-8') as f:
                return json.load(f)
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

    def _select_by_type(self, val_cmd, val_default, val_config, types):
        if val_cmd or isinstance(val_cmd, types):
            return val_cmd
        if val_default or isinstance(val_default, types):
            return val_default
        if val_config or isinstance(val_config, types):
            return val_config
        return None

    def get(self, name, default=None, type=None, required=False, is_path=False):
        """
        Gets the names parameter from the command line arguments.

        If it is not set on the command line the configuration file
        is checked.

        If it is also not present in the configuration file then
        ``default`` is returned unless ``required`` is false in which
        case an ``OasisException`` is raised.

        :param name: The name of the parameter to lookup
        :type name: str

        :param default: The default value to return if the name is not
            found on the command line or in the configuration file.

        :param required: Flag whether the value is required, if so and
            the parameter is not found on the command line or in the
            configuration file an error is raised.
        :type required: bool

        :param is_path: Flag whether the value should be treated as a path,
            is so the value is processed as relative to the config file.
        :type is_path: bool

        :raise OasisException: If the value is not found and ``required``
            is True

        :return: The found value or the default
        """
        value = None

        cmd_value = getattr(self.args, name, None)
        config_value = self.config.get(name)

        if type is None:
            value = cmd_value or config_value or default
        else:
            value = self._select_by_type(cmd_value, config_value, default, type)

        if (cmd_value or default) and is_path and not os.path.isabs(value):
            value = os.path.abspath(value)
        elif config_value and is_path and not os.path.isabs(value):
            value = os.path.join(self.config_dir, value)

        if required and value is None:
            raise OasisException(
                'Required argument {} could not be found in the command args or the MDK config. file'.format(name)
            )

        if value is None:
            value = default

        if is_path and value is not None and not os.path.isabs(value):
            value = os.path.abspath(value) if cmd_value else os.path.join(self.config_dir, value)

        return value
