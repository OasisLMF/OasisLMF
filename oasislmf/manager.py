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
from .utils.deterministic_loss import generate_deterministic_losses
from .lookup.rtree import PerilAreasIndex
from .utils.path import (
    as_path,
    setcwd,
)
from .utils.coverages import SUPPORTED_COVERAGE_TYPES

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

## IMPORT Computation commands (MOVE THIS LATER)
from .computation.hooks.pre_analysis import ExposurePreAnalysis
from .computation.generate.inputs import OasisFiles 
from .computation.generate.keys import OasisKeys 
from .computation.generate.losses import Losses 
#from .computation._ import _ 
#from .computation._ import _ 

class OasisManager(object):
    computation_classes = [
        ExposurePreAnalysis,
        OasisFiles,
        OasisKeys,
        Losses
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
        'generate_exposure_pre_analysis'
        >>> OasisManager.computation_name_to_method('EODFile')
        'generate_eod_file'
        >>> OasisManager.computation_name_to_method('Model1Data')
        'generate_model1_data'
        """
        return 'generate_' + re.sub('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))', r'_\1', name).lower()

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


    @oasis_log
    def run_exposure(
            self,
            src_dir,
            run_dir,
            loss_factors,
            net_ri,
            il_alloc_rule,
            ri_alloc_rule,
            output_level,
            output_file,
            include_loss_factor=True,
            print_summary=False):
        """
        Generates insured losses from preexisting Oasis files with specified
        loss factors (loss % of TIV).
        """

        src_contents = [fn.lower() for fn in os.listdir(src_dir)]

        if 'location.csv' not in src_contents:
            raise OasisException(
                'No location/exposure file found in source directory - '
                'a file named `location.csv` is expected'
            )

        il = ril = False
        il = ('account.csv' in src_contents)
        ril = il and ('ri_info.csv' in src_contents) and ('ri_scope.csv' in src_contents)

        self.logger.debug('\nRunning deterministic losses (GUL=True, IL={}, RIL={})\n'.format(il, ril))

        if not os.path.exists(run_dir):
            Path(run_dir).mkdir(parents=True, exist_ok=True)

        contents = [fn.lower() for fn in os.listdir(src_dir)]
        exposure_fp = [os.path.join(src_dir, fn) for fn in contents if fn == 'location.csv'][0]
        accounts_fp = [os.path.join(src_dir, fn) for fn in contents if fn == 'account.csv'][0]
        ri_info_fp = ri_scope_fp = None
        try:
            ri_info_fp = [os.path.join(src_dir, fn) for fn in contents if fn == 'ri_info.csv'][0]
        except IndexError:
            pass
        else:
            try:
                ri_scope_fp = [os.path.join(src_dir, fn) for fn in contents if fn == 'ri_scope.csv'][0]
            except IndexError:
                ri_info_fp = None

        il_alloc_rule = self.get_alloc_rule(
            alloc_given=il_alloc_rule,
            alloc_max=KTOOLS_ALLOC_FM_MAX,
            err_msg='Invalid alloc IL rule',
            fallback=self.ktools_alloc_rule_il
        )
        ri_alloc_rule = self.get_alloc_rule(
            alloc_given=ri_alloc_rule,
            alloc_max=KTOOLS_ALLOC_FM_MAX,
            err_msg='Invalid alloc RI rule',
            fallback=self.ktools_alloc_rule_ri
        )

        # Start Oasis files generation
        self.generate_oasis_files(
            run_dir,
            exposure_fp,
            accounts_fp=accounts_fp,
            ri_info_fp=ri_info_fp,
            ri_scope_fp=ri_scope_fp
        )

        losses = generate_deterministic_losses(
            run_dir,
            output_dir=os.path.join(run_dir, 'output'),
            include_loss_factor=include_loss_factor,
            loss_factors=loss_factors,
            net_ri=net_ri,
            il_alloc_rule=il_alloc_rule,
            ri_alloc_rule=ri_alloc_rule
        )

        guls_df = losses['gul']
        ils_df = losses['il']
        rils_df = losses['ri']

        # Read in the summary map
        summaries_df = get_dataframe(src_fp=os.path.join(run_dir, 'fm_summary_map.csv'))

        guls_df.to_csv(path_or_buf=os.path.join(run_dir, 'guls.csv'), index=False, encoding='utf-8')
        guls_df.rename(columns={'loss': 'loss_gul'}, inplace=True)

        guls_df = guls_df.merge(
            right=summaries_df,
            left_on=["item_id"],
            right_on=["agg_id"]
        )

        if include_loss_factor:
            join_cols = ["event_id", "output_id", "loss_factor_idx"]
        else:
            join_cols = ["event_id", "output_id"]

        if il:
            ils_df.to_csv(path_or_buf=os.path.join(run_dir, 'ils.csv'), index=False, encoding='utf-8')
            ils_df.rename(columns={'loss': 'loss_il'}, inplace=True)
            all_losses_df = guls_df.merge(
                how='left',
                right=ils_df,
                on=join_cols,
                suffixes=["_gul", "_il"]
            )
        if ril:
            rils_df.to_csv(path_or_buf=os.path.join(run_dir, 'rils.csv'), index=False, encoding='utf-8')
            rils_df.rename(columns={'loss': 'loss_ri'}, inplace=True)
            all_losses_df = all_losses_df.merge(
                how='left',
                right=rils_df,
                on=join_cols
            )

        oed_hierarchy = get_oed_hierarchy()
        portfolio_num = oed_hierarchy['portnum']['ProfileElementName'].lower()
        acc_num = oed_hierarchy['accnum']['ProfileElementName'].lower()
        loc_num = oed_hierarchy['locnum']['ProfileElementName'].lower()
        policy_num = oed_hierarchy['polnum']['ProfileElementName'].lower()

        if output_level == 'port':
            summary_cols = [portfolio_num]
        elif output_level == 'acc':
            summary_cols = [portfolio_num, acc_num]
        elif output_level == 'pol':
            summary_cols = [portfolio_num, acc_num, policy_num]
        elif output_level == 'loc':
            summary_cols = [portfolio_num, acc_num, loc_num]
        elif output_level == 'item':
            summary_cols = [
                'output_id', portfolio_num, acc_num, loc_num, policy_num,
                'coverage_type_id']

        if include_loss_factor:
            group_by_cols = summary_cols + ['loss_factor_idx']
        else:
            group_by_cols = summary_cols
        guls_df = guls_df.loc[:, group_by_cols + ['loss_gul']]

        if not il and not ril:
            all_loss_cols = group_by_cols + ['loss_gul']
            all_losses_df = guls_df.loc[:, all_loss_cols]
            all_losses_df.drop_duplicates(keep=False, inplace=True)
        elif not ril:
            all_loss_cols = group_by_cols + ['loss_gul', 'loss_il']
            all_losses_df = all_losses_df.loc[:, all_loss_cols]
            summary_gul_df = pd.DataFrame(
                {'loss_gul': guls_df.groupby(group_by_cols)['loss_gul'].sum()}).reset_index()
            summary_il_df = pd.DataFrame(
                {'loss_il': all_losses_df.groupby(group_by_cols)['loss_il'].sum()}).reset_index()
            all_losses_df = summary_gul_df.merge(how='left', right=summary_il_df, on=group_by_cols)
        else:
            all_loss_cols = group_by_cols + ['loss_gul', 'loss_il', 'loss_ri']
            all_losses_df = all_losses_df.loc[:, all_loss_cols]
            summary_gul_df = pd.DataFrame(
                {'loss_gul': guls_df.groupby(group_by_cols)['loss_gul'].sum()}).reset_index()
            summary_il_df = pd.DataFrame(
                {'loss_il': all_losses_df.groupby(group_by_cols)['loss_il'].sum()}).reset_index()
            summary_ri_df = pd.DataFrame(
                {'loss_ri': all_losses_df.groupby(group_by_cols)['loss_ri'].sum()}).reset_index()
            all_losses_df = summary_gul_df.merge(how='left', right=summary_il_df, on=group_by_cols)
            all_losses_df = all_losses_df.merge(how='left', right=summary_ri_df, on=group_by_cols)

        for i in range(len(loss_factors)):

            if include_loss_factor:
                total_gul = guls_df[guls_df.loss_factor_idx == i].loss_gul.sum()
            else:
                total_gul = guls_df.loss_gul.sum()

            if not il and not ril:
                all_loss_cols = all_loss_cols + ['loss_gul']
                all_losses_df = guls_df.loc[:, all_loss_cols]
                all_losses_df.drop_duplicates(keep=False, inplace=True)
                header = \
                    'Losses (loss factor={:.2%}; total gul={:,.00f})'.format(
                        loss_factors[i],
                        total_gul)
            elif not ril:
                if include_loss_factor:
                    total_il = ils_df[ils_df.loss_factor_idx == i].loss_il.sum()
                else:
                    total_il = ils_df.loss_il.sum()

                header = \
                    'Losses (loss factor={:.2%}; total gul={:,.00f}; total il={:,.00f})'.format(
                        loss_factors[i],
                        total_gul, total_il)
            else:
                if include_loss_factor:
                    total_il = ils_df[ils_df.loss_factor_idx == i].loss_il.sum()
                    total_ri_net = rils_df[rils_df.loss_factor_idx == i].loss_ri.sum()
                else:
                    total_il = ils_df.loss_il.sum()
                    total_ri_net = rils_df.loss_ri.sum()
                total_ri_ceded = total_il - total_ri_net
                header = \
                    'Losses (loss factor={:.2%}; total gul={:,.00f}; total il={:,.00f}; total ri ceded={:,.00f})'.format(
                        loss_factors[i],
                        total_gul, total_il, total_ri_ceded)

            # Convert output cols to strings for formatting
            for c in group_by_cols:
                all_losses_df[c] = all_losses_df[c].apply(str)

            if print_summary:
                cols_to_print = all_loss_cols.copy()
                if False:
                    cols_to_print.remove('loss_factor_idx')
                if include_loss_factor:
                    print_dataframe(
                        all_losses_df[all_losses_df.loss_factor_idx == str(i)],
                        frame_header=header,
                        cols=cols_to_print)
                else:
                    print_dataframe(
                        all_losses_df,
                        frame_header=header,
                        cols=cols_to_print)

        if output_file:
            all_losses_df.to_csv(output_file, index=False, encoding='utf-8')

        return (il, ril)

    def run_fm_test(self, test_case_dir, run_dir, update_expected=False):
        """
        Runs an FM test case and validates generated
        losses against expected losses.

        only use 'update_expected' for debugging
        it replaces the expected file with generated
        """

        net_ri = True
        il_alloc_rule = KTOOLS_ALLOC_IL_DEFAULT
        ri_alloc_rule = KTOOLS_ALLOC_RI_DEFAULT
        output_level = 'loc'

        loss_factor_fp = os.path.join(test_case_dir, 'loss_factors.csv')
        loss_factor = []
        include_loss_factor = False
        if os.path.exists(loss_factor_fp):
            loss_factor = []
            include_loss_factor = True
            try:
                with open(loss_factor_fp, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        loss_factor.append(
                            float(row['loss_factor']))
            except Exception as e:
                raise OasisException(f"Failed to read {loss_factor_fp}", e)
        else:
            loss_factor.append(1.0)

        output_file = os.path.join(run_dir, 'loc_summary.csv')
        (il, ril) = self.run_exposure(
            test_case_dir, run_dir, loss_factor, net_ri,
            il_alloc_rule, ri_alloc_rule, output_level, output_file,
            include_loss_factor)

        expected_data_dir = os.path.join(test_case_dir, 'expected')
        if not os.path.exists(expected_data_dir):
            raise OasisException(
                'No subfolder named `expected` found in the input directory - '
                'this subfolder should contain the expected set of GUL + IL '
                'input files, optionally the RI input files, and the expected '
                'set of GUL, IL and optionally the RI loss files'
            )

        files = ['keys.csv', 'loc_summary.csv']
        files += [
            '{}.csv'.format(fn)
            for ft, fn in chain(OASIS_FILES_PREFIXES['gul'].items(), OASIS_FILES_PREFIXES['il'].items())
        ]
        files += ['gul_summary_map.csv', 'guls.csv']
        if il:
            files += ['fm_summary_map.csv', 'ils.csv']
        if ril:
            files += ['rils.csv']

        test_result = True
        for f in files:
            generated = os.path.join(run_dir, f)
            expected = os.path.join(expected_data_dir, f)

            if not os.path.exists(expected):
                continue

            try:
                pd.testing.assert_frame_equal(
                    pd.read_csv(expected),
                    pd.read_csv(generated)
                )
            except AssertionError:
                if update_expected:
                    shutil.copyfile(generated, expected)
                else:
                    print("Expected:")
                    with open(expected) as f:
                        print(f.read())
                    print("Generated:")
                    with open(generated) as f:
                        print(f.read())
                    raise OasisException(
                        f'\n FAIL: generated {generated} vs expected {expected}'
                    )
                test_result = False
        return test_result


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
