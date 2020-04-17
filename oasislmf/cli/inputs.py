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
        self.logger = logging.getLogger()

        self.args = args
        self.config = {}
        self.config_fp = self.get('config', is_path=True)
        self.config_mapping = get_config_profile()

        if self.config_fp is not None:
            self.config = self.load_config_file()
            self.config_dir = os.path.dirname(self.config_fp)

        self.obsolete_keys = set(self.config) & set(self.config_mapping)
    
        ##! Not sure why this is needed?
        #self.logger.warning(str(self.config))
    
        self.list_obsolete_keys()
        if update_keys:
            self.update_config_keys()

    def list_obsolete_keys(self, fix_warning=True):
        if self.obsolete_keys:
            self.logger.warning('Depricated key(s) in MDK config:')
            for k in self.obsolete_keys:
                self.logger.warning('   {} : {}'.format(
                    k,
                    self.config_mapping[k],
                ))
            self.logger.warning('')
            if fix_warning:
                self.logger.warning('  To fix run: oasislmf config update'.format(self.config_fp))

    def update_config_keys(self):
        """
        If command line flags change between package versions, update them internally
        """
        for key in self.obsolete_keys:
            if not self.config_mapping[key]['deleted']:
                self.config[self.config_mapping[key]['updated_to']] = self.config[key]
            del self.config[key]

    def load_config_file(self):
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

    def get(self, name, default=None, required=False, is_path=False):
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

        :raise OasisException: If the value is not found and ``required``
            is True

        :return: The found value or the default
        """

        value = getattr(self.args, name, None)
        source = 'arg'
        if value is None:
            value = self.config.get(name)
            source = 'config'
        if value is None and required:
            raise OasisException(
                'Required argument {} could not be found in the command args or the MDK config. file'.format(name)
            )
        if value is None:
            value = default
            source = 'default'

        if is_path and value is not None and not os.path.isabs(value):
            if source == 'config':
                value = os.path.join(self.config_dir, value)
            else:
                value = os.path.abspath(value)

        return value
