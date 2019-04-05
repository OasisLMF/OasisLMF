# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import open as io_open
from builtins import str

from future import standard_library
standard_library.install_aliases()

__all__ = [
    'OasisManager'
]

import copy
import io
import filecmp
import json
import logging
import multiprocessing
import os
import re
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from builtins import str
from collections import OrderedDict
from future.utils import viewitems

try:
    from json import JSONDecodeError
except ImportError:
    from builtins import ValueError as JSONDecodeError

from subprocess32 import (
    CalledProcessError,
    check_call,
    run,
)
import sys
import importlib
from itertools import (
    chain,
    product,
)

import pandas as pd
pd.options.mode.chained_assignment = None

from pathlib2 import Path
from six import text_type as _unicode
from tabulate import tabulate

from .model_execution import runner
from .model_execution.bin import (
    csv_to_bin,
    prepare_run_directory,
    prepare_run_inputs,
)
from .model_preparation import oed
from .model_preparation.gul_inputs import (
    get_gul_input_items,
    write_gul_input_files,
)
from .model_preparation.il_inputs import (
    get_il_input_items,
    unified_id_terms,
    write_il_input_files,
)
from .model_preparation.lookup import OasisLookupFactory as olf
from .model_preparation.utils import prepare_input_files_directory
from .model_preparation.reinsurance_layer import write_ri_input_files
from .utils.concurrency import (
    multiprocess,
    multithread,
    Task,
)
from .utils.data import (
    get_dataframe,
    get_json,
    get_utctimestamp,
)
from .utils.exceptions import OasisException
from .utils.log import oasis_log
from .utils.metadata import COVERAGE_TYPES
from .utils.defaults import (
    get_default_accounts_profile,
    get_default_deterministic_analysis_settings,
    get_default_exposure_profile,
    get_default_fm_aggregation_profile,
    KTOOLS_NUM_PROCESSES,
    KTOOLS_MEM_LIMIT,
    KTOOLS_FIFO_RELATIVE,
    KTOOLS_ALLOC_RULE,
    OASIS_FILES_PREFIXES,
)
from .utils.peril import PerilAreasIndex
from .utils.path import (
    as_path,
    empty_dir,
    setcwd,
)


class OasisManager(object):

    @oasis_log
    def __init__(
        self,
        exposure_profile=None,
        supported_oed_coverage_types=None,
        accounts_profile=None,
        fm_aggregation_profile=None,
        deterministic_analysis_settings=None,
        ktools_num_processes=None,
        ktools_mem_limit=None,
        ktools_fifo_relative=None,
        ktools_alloc_rule=None,
        oasis_files_prefixes=None
    ):
        # Set defaults for static data or runtime parameters
        self._exposure_profile = exposure_profile or get_default_exposure_profile()
        self._supported_oed_coverage_types = supported_oed_coverage_types or tuple(COVERAGE_TYPES[k]['id'] for k in COVERAGE_TYPES if k not in ['pd', 'all'])
        self._accounts_profile = accounts_profile or get_default_accounts_profile()
        self._fm_aggregation_profile = fm_aggregation_profile or get_default_fm_aggregation_profile()
        self._deterministic_analysis_settings = deterministic_analysis_settings or get_default_deterministic_analysis_settings()
        self._ktools_num_processes = ktools_num_processes or KTOOLS_NUM_PROCESSES
        self._ktools_mem_limit = ktools_mem_limit or KTOOLS_MEM_LIMIT
        self._ktools_fifo_relative = ktools_fifo_relative or KTOOLS_FIFO_RELATIVE
        self._ktools_alloc_rule = ktools_alloc_rule or KTOOLS_ALLOC_RULE
        self._oasis_files_prefixes = oasis_files_prefixes or OASIS_FILES_PREFIXES

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
    def ktools_num_processes(self):
        return self._ktools_num_processes

    @property
    def ktools_mem_limit(self):
        return self._ktools_mem_limit

    @property
    def ktools_fifo_relative(self):
        return self._ktools_fifo_relative

    @property
    def ktools_alloc_rule(self):
        return self._ktools_alloc_rule

    @oasis_log
    def generate_peril_areas_rtree_file_index(
        self,
        keys_data_fp,
        areas_rtree_index_fp,
        lookup_config_fp=None,
        lookup_config=None,
    ):
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
    def generate_keys(
        self,
        exposure_fp,
        lookup_config_fp=None,
        keys_data_fp=None,
        model_version_fp=None,
        lookup_package_fp=None,
        keys_fp=None,
        keys_errors_fp=None,
        keys_id_col='locnumber',
        keys_format=None
    ):
        model_info, lookup = olf.create(
            lookup_config_fp=lookup_config_fp,
            model_keys_data_path=keys_data_fp,
            model_version_file_path=model_version_fp,
            lookup_package_path=lookup_package_fp
        )

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')

        keys_fp = keys_fp or '{}-keys.csv'.format(utcnow)
        keys_errors_fp = keys_errors_fp or '{}-keys-errors.csv'.format(utcnow)

        return olf.save_results(
            lookup,
            loc_id_col=keys_id_col,
            successes_fp=keys_fp,
            errors_fp=keys_errors_fp,
            source_exposure_fp=exposure_fp,
            format=keys_format
        )

    @oasis_log
    def generate_oasis_files(
        self,
        target_dir,
        exposure_fp,
        exposure_profile=None,
        exposure_profile_fp=None,
        keys_fp=None,
        lookup_config=None,
        lookup_config_fp=None,
        keys_data_fp=None,
        model_version_fp=None,
        lookup_package_fp=None,
        supported_oed_coverage_types=None,
        accounts_fp=None,
        accounts_profile=None,
        accounts_profile_fp=None,
        fm_aggregation_profile=None,
        fm_aggregation_profile_fp=None,
        ri_info_fp=None,
        ri_scope_fp=None,
        oasis_files_prefixes=None
    ):
        # Check whether the invocation indicates a deterministic or model
        # analysis/run - the CLI supports deterministic analyses via a command
        # `oasislmf exposure run` which requires a preexisting input files
        # directory, which is usually the same as the analysis/output directory
        deterministic = not(keys_fp or (lookup_config or lookup_config_fp) or (keys_data_fp and model_version_fp and lookup_package_fp))

        # Prepare the target directory and copy the source files, profiles and
        # model version file into it
        target_dir = prepare_input_files_directory(
            target_dir,
            exposure_fp,
            exposure_profile_fp=exposure_profile_fp,
            keys_fp=keys_fp,
            lookup_config_fp=lookup_config_fp,
            model_version_fp=model_version_fp,
            accounts_fp=accounts_fp,
            accounts_profile_fp=accounts_profile_fp,
            fm_aggregation_profile_fp=fm_aggregation_profile_fp,
            ri_info_fp=ri_info_fp,
            ri_scope_fp=ri_scope_fp
        )

        # Get the profiles defining the exposure and accounts files, ID related
        # terms in these files, and FM aggregation hierarchy
        exposure_profile = exposure_profile or (get_json(src_fp=exposure_profile_fp) if exposure_profile_fp else self.exposure_profile)
        accounts_profile = accounts_profile or (get_json(src_fp=accounts_profile_fp) if accounts_profile_fp else self.accounts_profile)
        id_terms = unified_id_terms(profiles=(exposure_profile, accounts_profile,))
        loc_id = id_terms['locid']
        acc_id = id_terms['accid']
        portfolio_num = id_terms['portid']
        fm_aggregation_profile = (
            fm_aggregation_profile or
            ({int(k): v for k, v in viewitems(get_json(src_fp=fm_aggregation_profile_fp))} if fm_aggregation_profile_fp else {}) or
            self.fm_aggregation_profile
        )

        # If a pre-generated keys file path has not been provided,
        # then it is asssumed some model lookup assets have been provided, so
        # as to allow the lookup to be instantiated and called to generated
        # the keys file. Otherwise if no model keys file path or lookup assets
        # were provided then a "deterministic" keys file is generated.
        _keys_fp = _keys_errors_fp = None
        if not keys_fp:
            _keys_fp = os.path.join(target_dir, 'keys.csv')
            _keys_errors_fp = os.path.join(target_dir, 'keys-errors.csv')

            cov_types = supported_oed_coverage_types or self.supported_oed_coverage_types

            if deterministic:
                loc_numbers = (loc_num[loc_id] for _, loc_num in get_dataframe(
                    src_fp=exposure_fp,
                    col_dtypes={loc_id: 'str', acc_id: 'str', portfolio_num: 'str'},
                    empty_data_error_msg='No exposure found in the source exposure (loc.) file'
                )[[loc_id]].iterrows())
                keys = [
                    {loc_id: loc_num, 'peril_id': 1, 'coverage_type': cov_type, 'area_peril_id': i + 1, 'vulnerability_id': i + 1}
                    for i, (loc_num, cov_type) in enumerate(product(loc_numbers, cov_types))
                ]
                _, _ = olf.write_oasis_keys_file(keys, _keys_fp)
            else:
                lookup_config = get_json(src_fp=lookup_config_fp) if lookup_config_fp else lookup_config
                if lookup_config:
                    lookup_config['keys_data_path'] = os.path.abspath(os.path.dirname(lookup_config_fp))

                _, lookup = olf.create(
                    lookup_config=lookup_config,
                    model_keys_data_path=keys_data_fp,
                    model_version_file_path=model_version_fp,
                    lookup_package_path=lookup_package_fp
                )
                f1, n1, f2, n2 = olf.save_results(
                    lookup,
                    loc_id_col=loc_id,
                    successes_fp=_keys_fp,
                    errors_fp=_keys_errors_fp,
                    source_exposure_fp=exposure_fp
                )
        else:
            _keys_fp = os.path.join(target_dir, os.path.basename(keys_fp))

        # Get the GUL input items and exposure dataframes
        gul_inputs_df, exposure_df = get_gul_input_items(
            exposure_fp, _keys_fp, exposure_profile=exposure_profile
        )

        # Write the GUL input files
        files_prefixes = oasis_files_prefixes or self.oasis_files_prefixes
        gul_input_files = write_gul_input_files(
            gul_inputs_df,
            target_dir,
            oasis_files_prefixes=files_prefixes['gul']
        )

        # If no source accounts file path has been provided assume that IL
        # input files, and therefore also RI input files, are not needed
        if not accounts_fp:
            return gul_input_files

        # Get the IL input items
        il_inputs_df, _ = get_il_input_items(
            exposure_df,
            gul_inputs_df,
            accounts_fp=accounts_fp,
            exposure_profile=exposure_profile,
            accounts_profile=accounts_profile,
            fm_aggregation_profile=fm_aggregation_profile
        )

        # Write the IL/FM input files
        il_input_files = write_il_input_files(
            il_inputs_df,
            target_dir,
            oasis_files_prefixes=files_prefixes['il']
        )

        # Combine the GUL and IL input file paths into a single dict (for convenience)
        oasis_files = {k: v for k, v in chain(gul_input_files.items(), il_input_files.items())}

        # If no RI input file paths (info. and scope) have been provided then
        # no RI input files are needed, just return the GUL and IL Oasis files
        if not (ri_info_fp or ri_scope_fp):
            return oasis_files

        # Write the RI input files, and write the returned RI layer info. as a
        # file, which can be reused by the model runner (in the model execution
        # stage) to set the number of RI iterations
        ri_layers = write_ri_input_files(
            exposure_fp,
            accounts_fp,
            oasis_files['items'],
            oasis_files['coverages'],
            oasis_files['gulsummaryxref'],
            oasis_files['fm_xref'],
            oasis_files['fmsummaryxref'],
            ri_info_fp,
            ri_scope_fp,
            target_dir
        )
        with io_open(os.path.join(target_dir, 'ri_layers.json'), 'w', encoding='utf-8') as f:
            f.write(_unicode(json.dumps(ri_layers, ensure_ascii=False, indent=4)))
            oasis_files['ri_layers'] = os.path.abspath(f.name)
            for layer, layer_info in viewitems(ri_layers):
                oasis_files['RI_{}'.format(layer)] = layer_info['directory']

        return oasis_files

    @oasis_log
    def generate_model_losses(
        self,
        model_run_fp,
        oasis_fp,
        analysis_settings_fp,
        model_data_fp,
        model_package_fp=None,
        ktools_num_processes=None,
        ktools_mem_limit=None,
        ktools_fifo_relative=None,
        ktools_alloc_rule=None
    ):

        il = all(p in os.listdir(oasis_fp) for p in ['fm_policytc.csv', 'fm_profile.csv', 'fm_programme.csv', 'fm_xref.csv'])

        ri = False
        if os.path.basename(oasis_fp) == 'csv':
            ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(os.path.dirname(oasis_fp)))
        else:
            ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(oasis_fp))

        if not os.path.exists(model_run_fp):
            Path(model_run_fp).mkdir(parents=True, exist_ok=True)

        prepare_run_directory(
            model_run_fp,
            oasis_fp,
            model_data_fp,
            analysis_settings_fp,
            ri=ri
        )

        if not ri:
            csv_to_bin(oasis_fp, os.path.join(model_run_fp, 'input'), il=il)
        else:
            contents = os.listdir(model_run_fp)
            for fp in [os.path.join(model_run_fp, fn) for fn in contents if re.match(r'RI_\d+$', fn) or re.match(r'input$', fn)]:
                csv_to_bin(fp, fp, il=True, ri=True)

        analysis_settings_fn = 'analysis_settings.json'
        _analysis_settings_fp = os.path.join(model_run_fp, analysis_settings_fn)
        try:
            with io_open(_analysis_settings_fp, 'r', encoding='utf-8') as f:
                analysis_settings = json.load(f)

            if analysis_settings.get('analysis_settings'):
                analysis_settings = analysis_settings['analysis_settings']

            if il:
                analysis_settings['il_output'] = True
            else:
                analysis_settings['il_output'] = False
                analysis_settings['il_summaries'] = []
            
            if ri:
                analysis_settings['ri_output'] = True
            else:
                analysis_settings['ri_output'] = False
                analysis_settings['ri_summaries'] = []
        except (IOError, TypeError, ValueError):
            raise OasisException('Invalid analysis settings file or file path: {}.'.format(_analysis_settings_fp))

        prepare_run_inputs(analysis_settings, model_run_fp, ri=ri)

        script_fp = os.path.join(model_run_fp, 'run_ktools.sh')

        if model_package_fp and os.path.exists(os.path.join(model_package_fp, 'supplier_model_runner.py')):
            path, package_name = os.path.split(model_package_fp)
            sys.path.append(path)
            model_runner_module = importlib.import_module('{}.supplier_model_runner'.format(package_name))
        else:
            model_runner_module = runner

        print(runner)

        with setcwd(model_run_fp) as cwd_path:
            ri_layers = 0
            if ri:
                try:
                    with io_open(os.path.join(model_run_fp, 'ri_layers.json'), 'r', encoding='utf-8') as f:
                        ri_layers = len(json.load(f))
                except IOError:
                    with io_open(os.path.join(model_run_fp, 'input', 'ri_layers.json'), 'r', encoding='utf-8') as f:
                        ri_layers = len(json.load(f))

            model_runner_module.run(
                analysis_settings,
                number_of_processes=(ktools_num_processes or self.ktools_num_processes),
                filename=script_fp,
                num_reinsurance_iterations=ri_layers,
                ktools_mem_limit=(ktools_mem_limit or self.ktools_mem_limit),
                set_alloc_rule=(ktools_alloc_rule or self.ktools_alloc_rule), 
                fifo_tmp_dir=(not (ktools_fifo_relative or self.ktools_fifo_relative))
            )

        return model_run_fp

    @oasis_log
    def generate_deterministic_losses(
        self,
        input_dir,
        output_dir=None,
        loss_percentage_of_tiv=1.0,
        net=False
    ):
        losses = OrderedDict({
            'gul': None, 'il': None, 'ri': None
        })

        output_dir = output_dir or input_dir

        il = all(p in os.listdir(input_dir) for p in ['fm_policytc.csv', 'fm_profile.csv', 'fm_programme.csv', 'fm_xref.csv'])

        ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(input_dir))

        csv_to_bin(input_dir, output_dir, il=il, ri=ri)

        # Generate an items and coverages dataframe and set column types (important!!)
        items_df = pd.merge(
            pd.read_csv(os.path.join(input_dir, 'items.csv')),
            pd.read_csv(os.path.join(input_dir, 'coverages.csv'))
        )
        for col in items_df:
            if col != 'tiv':
                items_df[col] = items_df[col].astype(int)
            else:
                items_df[col] = items_df[col].astype(float)

        guls_items = []
        for item_id, tiv in zip(items_df['item_id'], items_df['tiv']):
            event_loss = loss_percentage_of_tiv * tiv
            guls_items += [
                oed.GulRecord(event_id=1, item_id=item_id, sidx=-1, loss=event_loss),
                oed.GulRecord(event_id=1, item_id=item_id, sidx=-2, loss=0),
                oed.GulRecord(event_id=1, item_id=item_id, sidx=1, loss=event_loss)
            ]

        guls = pd.DataFrame(guls_items)
        guls_fp = os.path.join(output_dir, "guls.csv")
        guls.to_csv(guls_fp, index=False)

        net_flag = "-n" if net else ""
        ils_fp = os.path.join(output_dir, 'ils.csv')
        cmd = 'gultobin -S 1 < {} | fmcalc -p {} {} -a {} | tee ils.bin | fmtocsv > {}'.format(
            guls_fp, output_dir, net_flag, oed.ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID, ils_fp
        )
        print("\nGenerating deterministic ground-up and direct insured losses with command: {}\n".format(cmd))
        try:
            check_call(cmd, shell=True)
        except CalledProcessError as e:
            raise OasisException(e)

        guls.drop(guls[guls['sidx'] != 1].index, inplace=True)
        guls.reset_index(drop=True, inplace=True)
        guls.drop('sidx', axis=1, inplace=True)
        guls = guls[(guls[['loss']] != 0).any(axis=1)]
        guls['item_id'] = range(1, len(guls) + 1)
        losses['gul'] = guls

        ils = pd.read_csv(ils_fp)
        ils.drop(ils[ils['sidx'] != 1].index, inplace=True)
        ils.reset_index(drop=True, inplace=True)
        ils.drop('sidx', axis=1, inplace=True)
        ils = ils[(ils[['loss']] != 0).any(axis=1)]
        ils['output_id'] = range(1, len(ils) + 1)
        losses['il'] = ils

        if ri:
            try:
                [fn for fn in os.listdir(input_dir) if fn == 'ri_layers.json'][0]
            except IndexError:
                raise OasisException(
                    'No RI layers JSON file "ri_layers.json " found in the '
                    'input directory despite presence of RI input files'
                )
            else:
                try:
                    with io_open(os.path.join(input_dir, 'ri_layers.json'), 'r', encoding='utf-8') as f:
                        ri_layers = len(json.load(f))
                except (IOError, JSONDecodeError, OSError, TypeError) as e:
                    raise OasisException('Error trying to read the RI layers file: {}'.format(e))
                else:
                    def run_ri_layer(layer):
                        layer_inputs_fp = os.path.join(input_dir, 'RI_{}'.format(layer))
                        _input = 'gultobin -S 1 < {} | fmcalc -p {} -a {} | tee ils.bin |'.format(guls_fp, input_dir, oed.ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID) if layer == 1 else ''
                        pipe_in_previous_layer = '< ri{}.bin'.format(layer - 1) if layer > 1 else ''
                        ri_layer_fp = os.path.join(output_dir, 'ri{}.csv'.format(layer))
                        cmd = '{} fmcalc -p {} -n -a {} {}| tee ri{}.bin | fmtocsv > {}'.format(
                            _input,
                            layer_inputs_fp,
                            oed.ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID,
                            pipe_in_previous_layer,
                            layer,
                            ri_layer_fp
                        )
                        print("\nGenerating deterministic RI layer {} losses with command: {}\n".format(layer, cmd))
                        try:
                            check_call(cmd, shell=True)
                        except CalledProcessError as e:
                            raise OasisException(e)
                        rils = pd.read_csv(ri_layer_fp)
                        rils.drop(rils[rils['sidx'] != 1].index, inplace=True)
                        rils.drop('sidx', axis=1, inplace=True)
                        rils.reset_index(drop=True, inplace=True)
                        rils = rils[(rils[['loss']] != 0).any(axis=1)]

                        return rils

                    for i in range(1, ri_layers + 1):
                        rils = run_ri_layer(i)
                        if i in [1, ri_layers]:
                            rils['output_id'] = range(1, len(rils) + 1)
                            losses['ri'] = rils

        return losses

    @oasis_log
    def run_deterministic(
        self,
        input_dir,
        output_dir=None,
        loss_percentage_of_tiv=1.0,
        net=False
    ):
        """
        Generates insured losses from preexisting Oasis files with a specified
        damage ratio (loss % of TIV).
        """
        if not os.path.exists(output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        contents = [p.lower() for p in os.listdir(input_dir)]
        exposure_fp = [os.path.join(input_dir, p) for p in contents if 'location' in p][0]
        accounts_fp = [os.path.join(input_dir, p) for p in contents if 'account' in p][0]

        ri_info_fp = ri_scope_fp = None
        try:
            ri_info_fp = [os.path.join(input_dir, p) for p in contents if p.startswith('ri_info') or 'reinsinfo' in p][0]
        except IndexError:
            pass
        else:
            try:
                ri_scope_fp = [os.path.join(input_dir, p) for p in contents if p.startswith('ri_scope') or 'reinsscope' in p][0]
            except IndexError:
                ri_info_fp = None

        # Start Oasis files generation
        self.generate_oasis_files(
            input_dir,
            exposure_fp,
            accounts_fp=accounts_fp,
            ri_info_fp=ri_info_fp,
            ri_scope_fp=ri_scope_fp
        )

        losses = self.generate_deterministic_losses(
            input_dir,
            output_dir=output_dir,
            loss_percentage_of_tiv=loss_percentage_of_tiv,
            net=net
        )

        return losses['gul'], losses['il'], losses['ri']

    @oasis_log
    def run_model(
        self,
        exposure_fp,
        model_run_fp,
        analysis_settings_fp,
        model_data_fp,
        exposure_profile=None,
        exposure_profile_fp=None,
        lookup_config=None,
        lookup_config_fp=None,
        keys_data_fp=None,
        model_version_fp=None,
        lookup_package_fp=None,
        supported_oed_coverage_types=None,
        accounts_fp=None,
        accounts_profile=None,
        accounts_profile_fp=None,
        fm_aggregation_profile=None,
        fm_aggregation_profile_fp=None,
        ri_info_fp=None,
        ri_scope_fp=None,
        oasis_files_prefixes=None,
        model_package_fp=None,
        ktools_num_processes=None,
        ktools_mem_limit=None,
        ktools_fifo_relative=None,
        ktools_alloc_rule=None
    ):
        il = True if accounts_fp else False

        required_ri_paths = [ri_info_fp, ri_scope_fp]

        ri = all(required_ri_paths) and il

        if not os.path.exists(model_run_fp):
            Path(model_run_fp).mkdir(parents=True, exist_ok=True)
        else:
            empty_dir(model_run_dir)

        oasis_fp = os.path.join(model_run_fp, 'input') if ri else os.path.join(model_run_fp, 'input', 'csv')
        Path(oasis_fp).mkdir(parents=True, exist_ok=True)

        oasis_files = self.generate_oasis_files(
            exposure_fp,
            oasis_fp,
            exposure_profile=(exposure_profile or self.exposure_profile),
            exposure_profile_fp=exposure_profile_fp,
            lookup_config=lookup_config,
            lookup_config_fp=lookup_config_fp,
            keys_data_fp=keys_data_fp,
            model_version_fp=model_version_fp,
            lookup_package_fp=lookup_package_fp,
            supported_oed_coverage_types=supported_oed_coverage_types,
            accounts_fp=accounts_fp,
            accounts_profile=(accounts_profile or self.accounts_profile),
            accounts_profile_fp=accounts_profile_fp,
            fm_aggregation_profile=(fm_aggregation_profile or self.fm_aggregation_profile),
            fm_aggregation_profile_fp=fm_aggregation_profile_fp,
            ri_info_fp=ri_info_fp,
            ri_scope_fp=ri_scope_fp,
            oasis_files_prefixes=(oasis_files_prefixes or self.oasis_files_prefixes)
        )

        model_run_fp = self.generate_losses(
            oasis_fp,
            model_run_fp,
            analysis_settings_fp,
            model_data_fp,
            model_package_fp=model_package_fp,
            ktools_num_processes=(ktools_num_processes or self.ktools_num_processes),
            ktools_mem_limit=(ktools_mem_limit or self.ktools_mem_limit),
            ktools_fifo_relative=(ktools_fifo_relative or self.ktools_fifo_relative),
            ktools_alloc_rule=(ktools_alloc_rule or self.ktools_alloc_rule)
        )

        return model_run_fp
