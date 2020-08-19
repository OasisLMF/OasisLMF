__all__ = [
    'OasisBaseLookup',
]


# 'OasisBuiltinBaseLookup' -> 'OasisBaseLookup'

import io
import itertools
import json
import os
import types

import pandas as pd

from ..utils.log import oasis_log
from ..utils.path import as_path

UNKNOWN_ID = -1


''' Interface class for developing built in lookup code (Rtree)
'''


class OasisBaseLookup(object):

    @oasis_log()
    def __init__(self, config=None, config_json=None, config_fp=None, config_dir=None):
        if config:
            self._config = config
            self.config_dir = config_dir or '.'
        elif config_json:
            self._config = json.loads(config_json)
            self.config_dir = config_dir or '.'
        elif config_fp:
            self.config_dir = config_dir or os.path.dirname(config_fp)
            _config_fp = as_path(config_fp, 'config_fp')
            with io.open(_config_fp, 'r', encoding='utf-8') as f:
                self._config = json.load(f)

        keys_data_path = self._config.get('keys_data_path')
        keys_data_path = os.path.join(self.config_dir, keys_data_path) if keys_data_path else ''

        self._config['keys_data_path'] = as_path(keys_data_path, 'keys_data_path', preexists=(True if keys_data_path else False))

        peril_config = self._config.get('peril') or {}

        self._peril_ids = tuple(peril_config.get('peril_ids') or ())

        self._peril_id_col = peril_config.get('peril_id_col') or 'peril_id'

        coverage_config = self._config.get('coverage') or {}

        self._coverage_types = tuple(coverage_config.get('coverage_types') or ())

        self._coverage_type_col = peril_config.get('coverage_type_col') or 'coverage_type'

        self._config.setdefault('exposure', self._config.get('exposure') or self._config.get('locations') or {})

        self.__tweak_config_data__()

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

    @property
    def peril_ids(self):
        return self._peril_ids

    @property
    def peril_id_col(self):
        return self._peril_id_col

    @property
    def coverage_types(self):
        return self._coverage_types

    @property
    def coverage_type_col(self):
        return self._coverage_type_col

    def lookup(self, loc, peril_id, coverage_type, **kwargs):
        """
        Lookup for an individual location item, which could be a dict or a
        Pandas series object.
        """
        pass

    def process_locations_multiproc(self, loc_df):
        """
        Process and return the lookup results a location row
        Used in multiprocessing based query

        location_row is of type <class 'pandas.core.series.Series'>

        """
        locs_seq = (loc for _, loc in loc_df.iterrows())
        return [self.lookup(loc, peril_id, coverage_type) for
                loc, peril_id, coverage_type in
                itertools.product(locs_seq, self.peril_ids, self.coverage_types)]

    @oasis_log()
    def bulk_lookup(self, locs, **kwargs):
        """
        Bulk vulnerability lookup for a list, tuple, generator, pandas data
        frame or dict of location items, which can be dicts or Pandas series
        objects or any object which has as a dict-like interface.

        Generates results using ``yield``.
        """
        locs_seq = None

        if (isinstance(locs, list) or isinstance(locs, tuple)):
            locs_seq = (loc for loc in locs)
        elif isinstance(locs, types.GeneratorType):
            locs_seq = locs
        elif (isinstance(locs, dict)):
            locs_seq = locs.values()
        elif isinstance(locs, pd.DataFrame):
            locs_seq = (loc for _, loc in locs.iterrows())

        for loc, peril_id, coverage_type in itertools.product(locs_seq, self.peril_ids, self.coverage_types):
            yield self.lookup(loc, peril_id, coverage_type)
