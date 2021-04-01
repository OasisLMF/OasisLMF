__all__ = [
    'AbstractBasicKeyLookup',
    'MultiprocLookupMixin',
    'OasisBaseLookup',
]


# 'OasisBuiltinBaseLookup' -> 'OasisBaseLookup'

import abc
import itertools
import json
import os
import types

import pandas as pd

from ..utils.path import as_path
from ..utils.exceptions import OasisException

UNKNOWN_ID = -1


''' Basic abstract classes that facilitate the implementation of KeyLookupInterface  
'''


class AbstractBasicKeyLookup:
    """Basic abstract class for KeyLookup"""

    interface_version = "1"

    def __init__(self, config, config_dir=None, user_data_dir=None, output_dir=None):
        self.config = config
        self.config_dir = config_dir or '.'
        self.user_data_dir = user_data_dir
        self.output_dir = output_dir

    @abc.abstractmethod
    def process_locations(self, locations):
        """
        Process location rows - passed in as a pandas dataframe.
        Results can be list, tuple, generator or a pandas dataframe.
        """
        raise NotImplementedError


class MultiprocLookupMixin:
    """
    Simple mixin class for multiprocessing

    implement the process_locations_multiproc by transforming the result of process_locations into a pandas DataFrame
    """

    def process_locations_multiproc(self, loc_df_part):
        result = self.process_locations(loc_df_part)
        if isinstance(result, list) or isinstance(result, tuple):
            return pd.DataFrame(result)
        elif isinstance(result, types.GeneratorType):
            return pd.DataFrame.from_records(result)
        elif isinstance(result, pd.DataFrame):
            return result
        else:
            raise OasisException("Unrecognised type for results: {type(results)}. expected ")


class OasisBaseLookup(AbstractBasicKeyLookup, MultiprocLookupMixin):
    """
    Abstract class that help with the implementation of the KeyServerInterface.
    require lookup method to be implemented.
    Lookup will be call to create a key for each peril id and coverage type
    """
    multiproc_enabled = True

    def __init__(self, config=None, config_json=None, config_fp=None, config_dir=None, user_data_dir=None, output_dir=None):
        if config:
            config_dir = config_dir or '.'
        elif config_json:
            config = json.loads(config_json)
            config_dir = config_dir or '.'
        elif config_fp:
            config_dir = config_dir or os.path.dirname(config_fp)
            _config_fp = as_path(config_fp, 'config_fp')
            with open(_config_fp, 'r', encoding='utf-8') as f:
                config = json.load(f)

        keys_data_path = config.get('keys_data_path')
        keys_data_path = os.path.join(config_dir, keys_data_path) if keys_data_path else ''

        config['keys_data_path'] = as_path(keys_data_path, 'keys_data_path', preexists=(True if keys_data_path else False))

        super().__init__(config, config_dir=config_dir, user_data_dir=user_data_dir, output_dir=output_dir)

        peril_config = self.config.get('peril') or {}

        self.peril_ids = tuple(peril_config.get('peril_ids') or ())

        self.peril_id_col = peril_config.get('peril_id_col') or 'peril_id'

        coverage_config = self.config.get('coverage') or {}

        self.coverage_types = tuple(coverage_config.get('coverage_types') or ())

        self.coverage_type_col = peril_config.get('coverage_type_col') or 'coverage_type'

        self.config.setdefault('exposure', self.config.get('exposure') or self.config.get('locations') or {})

    def __tweak_config_data__(self):
        for section in ('exposure', 'peril', 'vulnerability',):
            section_config = self._config.get(section) or {}
            for k, v in section_config.items():
                if isinstance(v, str) and '%%KEYS_DATA_PATH%%' in v:
                    self._config[section][k] = v.replace('%%KEYS_DATA_PATH%%', self._config['keys_data_path'])
                elif type(v) == list:
                    self._config[section][k] = tuple(v)
                elif isinstance(v, dict):
                    for _k, _v in v.items():
                        if isinstance(_v, str) and '%%KEYS_DATA_PATH%%' in _v:
                            self._config[section][k][_k] = _v.replace('%%KEYS_DATA_PATH%%', self._config['keys_data_path'])

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, c):
        self._config = c
        self.__tweak_config_data__()

    def lookup(self, loc, peril_id, coverage_type, **kwargs):
        """
        Lookup for an individual location item, which could be a dict or a
        Pandas series object.
        """
        raise NotImplementedError

    def process_locations(self, locs):
        """
        Bulk vulnerability lookup for a list, tuple, generator, pandas data
        frame or dict of location items, which can be dicts or Pandas series
        objects or any object which has as a dict-like interface.

        Generates results using ``yield``.
        """
        locs_seq = None

        if isinstance(locs, list) or isinstance(locs, tuple):
            locs_seq = (loc for loc in locs)
        elif isinstance(locs, types.GeneratorType):
            locs_seq = locs
        elif isinstance(locs, dict):
            locs_seq = locs.values()
        elif isinstance(locs, pd.DataFrame):
            locs_seq = (loc for _, loc in locs.iterrows())

        for loc, peril_id, coverage_type in itertools.product(locs_seq, self.peril_ids, self.coverage_types):
            yield self.lookup(loc, peril_id, coverage_type)
