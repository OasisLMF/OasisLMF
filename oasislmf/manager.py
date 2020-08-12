__all__ = [
    'OasisManager'
]

import io
import importlib
import json
import logging
import os
import re
import sys
import warnings
import csv
import shutil

from itertools import chain
from filecmp import cmp as compare_files

from builtins import str

from itertools import (
    product,
)

from subprocess import CalledProcessError

import pandas as pd

from pathlib2 import Path

from .execution import runner
from .execution.bin import (
    csv_to_bin,
    prepare_run_directory,
    prepare_run_inputs,
)
from .preparation.gul_inputs import (
    get_gul_input_items,
    write_gul_input_files,
)
from .preparation.il_inputs import (
    get_il_input_items,
    get_oed_hierarchy,
    write_il_input_files,
)
from .preparation.summaries import (
    get_summary_mapping,
    generate_summaryxref_files,
    merge_oed_to_mapping,
    write_mapping_file,
    write_exposure_summary,
    write_summary_levels,
)
from .preparation.lookup import OasisLookupFactory as olf
from .preparation.oed import load_oed_dfs
from .preparation.dir_inputs import prepare_input_files_directory
from .preparation.reinsurance_layer import write_files_for_reinsurance
from .utils.data import (
    get_analysis_settings,
    get_model_settings,
    get_dataframe,
    get_location_df,
    get_json,
    get_utctimestamp,
    print_dataframe,
)

from .utils.exceptions import OasisException
from .utils.log import oasis_log
from .utils.defaults import (
    get_default_accounts_profile,
    get_default_deterministic_analysis_settings,
    get_default_exposure_profile,
    get_default_fm_aggregation_profile,
    GROUP_ID_COLS,
    KTOOLS_NUM_PROCESSES,
    KTOOLS_FIFO_RELATIVE,
    KTOOLS_ERR_GUARD,
    KTOOLS_ALLOC_GUL_MAX,
    KTOOLS_ALLOC_GUL_DEFAULT,
    KTOOLS_ALLOC_FM_MAX,
    KTOOLS_ALLOC_IL_DEFAULT,
    KTOOLS_ALLOC_RI_DEFAULT,
    KTOOLS_GUL_LEGACY_STREAM,
    KTOOLS_DEBUG,
    OASIS_FILES_PREFIXES,
    WRITE_CHUNKSIZE,
)
from .lookup.rtree import PerilAreasIndex
from .utils.path import (
    as_path,
    setcwd,
)
from .utils.coverages import SUPPORTED_COVERAGE_TYPES

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

## IMPORT Computation commands (MOVE THIS LATER)
from .computation.hooks.pre_analysis import HookPreAnalysis
from .computation.generate.files import GenerateOasisFiles 
from .computation.generate.keys import GenerateKeys 
from .computation.generate.losses import GenerateLosses 
from .computation.run.model import RunModel
from .computation.run.exposure import RunExposure, RunFmTest
#from .computation._ import _ 

class OasisManager(object):
    computation_classes = [
        HookPreAnalysis,
        GenerateOasisFiles,
        GenerateKeys,
        GenerateLosses,
        RunModel,
        RunExposure,
        RunFmTest,
    ]
    computations_params = {}

    @oasis_log
    def __init__(
        self,
        exposure_profile=None,
        supported_oed_coverage_types=None,
        accounts_profile=None,
        fm_aggregation_profile=None,
        deterministic_analysis_settings=None,
        ktools_num_processes=None,
        ktools_fifo_relative=None,
        ktools_alloc_rule_gul=None,
        ktools_alloc_rule_il=None,
        ktools_alloc_rule_ri=None,
        ktools_debug=None,
        ktools_error_guard=None,
        ktools_gul_legacy_stream=None,
        oasis_files_prefixes=None,
        write_chunksize=None,
        group_id_cols=None
    ):
        # Set defaults for static data or runtime parameters
        self._exposure_profile = exposure_profile or get_default_exposure_profile()
        self._supported_oed_coverage_types = supported_oed_coverage_types or tuple(v['id'] for v in SUPPORTED_COVERAGE_TYPES.values())
        self._accounts_profile = accounts_profile or get_default_accounts_profile()
        self._fm_aggregation_profile = fm_aggregation_profile or get_default_fm_aggregation_profile()
        self._deterministic_analysis_settings = deterministic_analysis_settings or get_default_deterministic_analysis_settings()
        self._ktools_num_processes = ktools_num_processes or KTOOLS_NUM_PROCESSES
        self._ktools_fifo_relative = ktools_fifo_relative or KTOOLS_FIFO_RELATIVE
        self._ktools_alloc_rule_gul = self.get_alloc_rule(ktools_alloc_rule_gul, KTOOLS_ALLOC_GUL_MAX, fallback=KTOOLS_ALLOC_GUL_DEFAULT)
        self._ktools_alloc_rule_il = self.get_alloc_rule(ktools_alloc_rule_il, KTOOLS_ALLOC_FM_MAX, fallback=KTOOLS_ALLOC_IL_DEFAULT)
        self._ktools_alloc_rule_ri = self.get_alloc_rule(ktools_alloc_rule_ri, KTOOLS_ALLOC_FM_MAX, fallback=KTOOLS_ALLOC_RI_DEFAULT)
        self._ktools_debug = ktools_debug or KTOOLS_DEBUG
        self._ktools_error_guard = ktools_error_guard or KTOOLS_ERR_GUARD
        self._ktools_gul_legacy_stream = ktools_gul_legacy_stream or KTOOLS_GUL_LEGACY_STREAM
        self._oasis_files_prefixes = oasis_files_prefixes or OASIS_FILES_PREFIXES
        self._write_chunksize = write_chunksize or WRITE_CHUNKSIZE
        self._group_id_cols = group_id_cols or GROUP_ID_COLS
        self.logger = logging.getLogger()

    @property
    def exposure_profile(self):
        return self._exposure_profile

    @property
    def supported_oed_coverage_types(self):
        return self._supported_oed_coverage_types

    @property
    def accounts_profile(self):
        return self._accounts_profile

    @property
    def fm_aggregation_profile(self):
        return self._fm_aggregation_profile

    @property
    def deterministic_analysis_settings(self):
        return self._deterministic_analysis_settings

    @property
    def oasis_files_prefixes(self):
        return self._oasis_files_prefixes

    @property
    def write_chunksize(self):
        return self._write_chunksize

    @property
    def group_id_cols(self):
        return self._group_id_cols

    @property
    def ktools_num_processes(self):
        return self._ktools_num_processes

    @property
    def ktools_fifo_relative(self):
        return self._ktools_fifo_relative

    @property
    def ktools_alloc_rule_gul(self):
        return self._ktools_alloc_rule_gul

    @property
    def ktools_alloc_rule_il(self):
        return self._ktools_alloc_rule_il

    @property
    def ktools_alloc_rule_ri(self):
        return self._ktools_alloc_rule_ri

    @property
    def ktools_debug(self):
        return self._ktools_debug

    @property
    def ktools_error_guard(self):
        return self._ktools_error_guard

    @property
    def ktools_gul_legacy_stream(self):
        return self._ktools_gul_legacy_stream

    def consolidate_input(self, computation_cls, kwargs):
        for param in computation_cls.get_params():
            if kwargs.get(param['name']) is None:
                kwargs[param['name']] = getattr(self, param['name'], None)
        return kwargs

    @staticmethod
    def get_alloc_rule(alloc_given, alloc_max, err_msg='Invalid alloc rule', fallback=None):
        if not isinstance(alloc_given, int):
            return fallback if fallback is not None else alloc_max
        elif alloc_given > alloc_max:
            raise OasisException('{}: {} larger than max value "{}"'.format(
                err_msg,
                alloc_given,
                alloc_max,
            ))
        else:
            return alloc_given

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

    @oasis_log
    def generate_peril_areas_rtree_file_index(
        self,
        keys_data_fp,
        areas_rtree_index_fp,
        lookup_config_fp=None,
        lookup_config=None,
    ):

        # Convert paths to absolute
        keys_data_fp = as_path(keys_data_fp, 'Lookup Data directory', is_dir=True, preexists=True)
        areas_rtree_index_fp = as_path(areas_rtree_index_fp, 'Index output file path', preexists=False)
        lookup_config_fp = as_path(lookup_config_fp, 'Built-in lookup config file path', preexists=True)

        if not (lookup_config or lookup_config_fp):
            raise OasisException('Either a built-in lookup config. or config. file path is required')

        config = get_json(src_fp=lookup_config_fp) if lookup_config_fp else lookup_config

        config_dir = os.path.dirname(lookup_config_fp) if lookup_config_fp else keys_data_fp

        peril_config = config.get('peril')

        if not peril_config:
            raise OasisException(
                'The lookup config must contain a peril-related subdictionary with a key named '
                '`peril` defining area-peril-related model information'
            )

        areas_fp = peril_config.get('file_path')

        if not areas_fp:
            raise OasisException(
                'The lookup peril config must define the path of a peril areas '
                '(or area peril) file with the key name `file_path`'
            )

        if areas_fp.startswith('%%KEYS_DATA_PATH%%'):
            areas_fp = areas_fp.replace('%%KEYS_DATA_PATH%%', keys_data_fp)

        if not os.path.isabs(areas_fp):
            areas_fp = os.path.join(config_dir, areas_fp)
            areas_fp = as_path(areas_fp, 'areas_fp')

        src_type = str.lower(str(peril_config.get('file_type')) or '') or 'csv'

        peril_id_col = str.lower(str(peril_config.get('peril_id_col')) or '') or 'peril_id'

        coverage_config = config.get('coverage')

        if not coverage_config:
            raise OasisException(
                'The lookup config must contain a coverage-related subdictionary with a key named '
                '`coverage` defining coverage related model information'
            )

        coverage_type_col = str.lower(str(coverage_config.get('coverage_type_col')) or '') or 'coverage_type'

        peril_area_id_col = str.lower(str(peril_config.get('peril_area_id_col')) or '') or 'area_peril_id'

        area_poly_coords_cols = peril_config.get('area_poly_coords_cols')

        if not area_poly_coords_cols:
            raise OasisException(
                'The lookup peril config must define the column names of '
                'the coordinates used to define areas in the peril areas '
                '(area peril) file using the key `area_poly_coords_cols`'
            )

        non_na_cols = (
            tuple(col.lower() for col in peril_config['non_na_cols']) if peril_config.get('non_na_cols')
            else tuple(col.lower() for col in [peril_area_id_col] + area_poly_coords_cols.values())
        )

        col_dtypes = peril_config.get('col_dtypes') or {peril_area_id_col: int}

        sort_cols = peril_config.get('sort_cols') or peril_area_id_col

        area_poly_coords_seq_start_idx = peril_config.get('area_poly_coords_seq_start_idx') or 1

        area_reg_poly_radius = peril_config.get('area_reg_poly_radius') or 0.00166

        index_props = peril_config.get('rtree_index')
        index_props.pop('filename')

        return PerilAreasIndex.create_from_peril_areas_file(
            src_fp=areas_fp,
            src_type=src_type,
            peril_id_col=peril_id_col,
            coverage_type_col=coverage_type_col,
            peril_area_id_col=peril_area_id_col,
            non_na_cols=non_na_cols,
            col_dtypes=col_dtypes,
            sort_cols=sort_cols,
            area_poly_coords_cols=area_poly_coords_cols,
            area_poly_coords_seq_start_idx=area_poly_coords_seq_start_idx,
            area_reg_poly_radius=area_reg_poly_radius,
            index_fp=areas_rtree_index_fp,
            index_props=index_props
        )


def __interface_factory(computation_cls):
    @oasis_log
    def interface(self, **kwargs):
        self.consolidate_input(computation_cls, kwargs)
        return computation_cls(**kwargs).run()

    OasisManager.computations_params[computation_cls.__name__] = computation_cls.get_params()
    interface.__doc__ =  computation_cls.__doc__
    return interface

for computation_cls in OasisManager.computation_classes:
    setattr(OasisManager, OasisManager.computation_name_to_method(computation_cls.__name__),
            __interface_factory(computation_cls))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
