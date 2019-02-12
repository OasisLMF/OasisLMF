from __future__ import (
    print_function,
    unicode_literals,
)

__all__ = [
    'OasisManager'
]

import copy
import io
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess

from builtins import str
from collections import OrderedDict
from future.utils import viewitems

from itertools import (
    chain,
    product,
)

import pandas as pd

from pathlib2 import Path
from six import u as _unicode
from tabulate import tabulate

from .cli.base import InputValues
from .model_execution import runner
from .model_execution.bin import (
    csv_to_bin,
    generate_binary_inputs,
    prepare_model_run_directory,
    prepare_model_run_inputs,
)
from .model_preparation.lookup import OasisLookupFactory as olf
from .model_preparation.gul_inputs import write_gul_input_files
from .model_preparation.il_inputs import write_il_input_files
from .model_preparation import oed
from .model_preparation.reinsurance_layer import write_ri_input_files
from .utils.concurrency import (
    multiprocess,
    multithread,
    Task,
)
from .utils.data import (
    get_json,
    get_utctimestamp,
)
from .utils.exceptions import OasisException
from .utils.log import oasis_log
from .utils.metadata import OED_COVERAGE_TYPES
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
from .utils.path import setcwd


class OasisManager(object):

    @oasis_log
    def __init__(
        self,
        exposure_profile=None,
        accounts_profile=None,
        fm_aggregation_profile=None,
        deterministic_analysis_settings=None
    ):
        # Set defaults for static data or runtime parameters
        self._default_exposure_profile = exposure_profile or get_default_exposure_profile()
        self._default_supported_oed_coverage_types = tuple(OED_COVERAGE_TYPES[k]['id'] for k in OED_COVERAGE_TYPES if k not in ['pd', 'all'])
        self._default_accounts_profile = accounts_profile or get_default_accounts_profile()
        self._default_fm_aggregation_profile = fm_aggregation_profile or get_default_fm_aggregation_profile()
        self._default_deterministic_analysis_settings = deterministic_analysis_settings or get_default_deterministic_analysis_settings()
        self._ktools_num_processes = KTOOLS_NUM_PROCESSES
        self._ktools_mem_limit = KTOOLS_MEM_LIMIT
        self._ktools_fifo_relative = KTOOLS_FIFO_RELATIVE
        self._ktools_alloc_rule = KTOOLS_ALLOC_RULE
        self._oasis_files_prefixes = OASIS_FILES_PREFIXES

    @property
    def default_exposure_profile(self):
        return self._default_exposure_profile

    @property
    def default_supported_oed_coverage_types(self):
        return self._default_supported_oed_coverage_types

    @property
    def default_accounts_profile(self):
        return self._default_accounts_profile

    @property
    def default_fm_aggregation_profile(self):
        return self._default_fm_aggregation_profile

    @property
    def default_deterministic_analysis_settings(self):
        return self._default_deterministic_analysis_settings

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


    def generate_peril_areas_rtree_file_index(
        self,
        keys_data_fp,
        areas_rtree_index_fp,
        lookup_config_fp=None,
        lookup_config=None,
    ):
        if not (lookup_config or lookup_config_fp):
            raise OasisException('Either a built-in lookup config. or config. file path is required')

        config = lookup_config or get_json(src_fp=lookup_config_fp)

        config_dir = os.path.dirname(config_fp)

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

        sort_col = peril_config.get('sort_col') or peril_area_id_col

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
            sort_col=sort_col,
            area_poly_coords_cols=area_poly_coords_cols,
            area_poly_coords_seq_start_idx=area_poly_coords_seq_start_idx,
            area_reg_poly_radius=area_reg_poly_radius,
            index_fp=areas_rtree_index_fp,
            index_props=index_props
        )

    def generate_keys(
        self,
        exposure_fp,
        lookup_config_fp=None,
        keys_data_fp=None,
        model_version_fp=None,
        lookup_package_fp=None,
        keys_fp=None,
        keys_errors_fp=None,
        keys_id_col=None,
        keys_format=None
    ):
        model_info, lookup = olf.create(
            lookup_config_fp=lookup_config_fp,
            model_keys_data_path=keys_data_path,
            model_version_file_path=model_version_file_path,
            lookup_package_path=lookup_package_path
        )

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')

        keys_fp = keys_fp or '{}-keys.csv'.format(utcnow)
        keys_errors_fp = keys_errors_fp or '{}-keys-errors.csv'.format(utcnow)

        return olf.save_results(
            lookup,
            keys_id_col=keys_id_col,
            successes_fp=keys_fp,
            errors_fp=keys_errors_fp,
            source_exposure_fp=exposure_fp,
            format=keys_format
        )

    def generate_oasis_files(
        self,
        target_dir,
        exposure_fp,
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
        oasis_files_prefixes=None
    ):
        # Prepare the target directory and copy the source files, profiles and
        # model version file into it
        if not os.path.exists(target_dir):
            Path(target_dir).mkdir(parents=True, exist_ok=True)

        for p in (exposure_fp, exposure_profile_fp, accounts_fp, accounts_profile_fp, fm_aggregation_profile_fp, lookup_config_fp, model_version_fp, ri_info_fp, ri_scope_fp):
            if p in os.listdir(target_dir):
                os.remove(p)
            if p and os.path.exists(p) and p != os.path.join(target_dir, os.path.basename(p)):
                    shutil.copy2(p, target_dir)

        # Get the exposure + accounts + FM aggregation profiles + lookup
        # config. profiles either from the optional arguments if present, or
        # then manager defaults
        exposure_profile = exposure_profile or get_json(src_fp=exposure_profile_fp) or self.exposure_profile
        accounts_profile = accounts_profile or get_json(src_fp=accounts_profile_fp) or self.accounts_profile
        fm_aggregation_profile = fm_aggregation_profile or get_json(src_fp=fm_aggregation_profile_fp, key_transform=int) or self.fm_aggregation_profile
        lookup_config = get_json(src_fp=lookup_config_fp) or lookup_config
        if lookup_config:
            lookup_config['keys_data_path'] = os.path.dirname(lookup_config_fp)

        # Generate keys and keys errors files - if no lookup assets provided
        # then assume the caller is trying to generate Oasis files for
        # deterministic losses
        keys_fp = os.path.join(target_dir, 'keys.csv')
        keys_errors_fp = os.path.join(target_dir, 'keys-errors.csv')

        cov_types = supported_oed_coverage_types or self.default_supported_oed_coverage_types
        if not (lookup_config or keys_data_fp or model_version_fp or lookup_package_fp):
            n = len(pd.read_csv(exposure_fp))
            keys = [
                {'locnumber': i + 1, 'peril_id': 1, 'coverage_type': j, 'area_peril_id': i + 1, 'vulnerability_id': i + 1}
                for i, j in product(range(n), cov_types)
            ]
            _, _ = olf.write_oasis_keys_file(keys, keys_fp)
        else:
            _, lookup = olf.create(
                lookup_config=lookup_config,
                model_keys_data_path=keys_data_fp,
                model_version_file_path=model_version_fp,
                lookup_package_path=lookup_package_fp
            )
            f1, n1, f2, n2 = olf.save_results(
                lookup,
                keys_id_col='locnumber',
                successes_fp=keys_fp,
                errors_fp=keys_errors_fp,
                source_exposure_fp=exposure_fp
            )

        # Write the GUL input files
        files_prefixes = oasis_files_prefixes or self.oasis_files_prefixes
        gul_input_files, gul_inputs_df, exposure_df = write_gul_input_files(
            exposure_fp,
            keys_fp,
            target_dir,
            exposure_profile=exposure_profile,
            oasis_files_prefixes=files_prefixes['gul']
        )

        # If no source accounts file path has been provided assume that IL
        # input files, and therefore also RI input files, are not needed
        if not accounts_fp:
            return gul_input_files

        # Write the IL/FM input files
        il_input_files, _, _ = write_il_input_files(
            exposure_df,
            gul_inputs_df,
            accounts_fp,
            target_dir,
            exposure_profile=exposure_profile,
            accounts_profile=accounts_profile,
            fm_aggregation_profile=fm_aggregation_profile,
            oasis_files_prefixes=files_prefixes['il']
        )

        oasis_files = {k: v for k, v in chain(gul_input_files.items(), il_input_files.items())}

        # If no RI input file paths (info. and scope) have been provided then
        # no RI input files are needed
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
        with io.open(os.path.join(target_dir, 'ri_layers.json'), 'w', encoding='utf-8') as f:
            f.write(_unicode(json.dumps(ri_layers, ensure_ascii=False, indent=4)))
            oasis_files['ri_layers'] = os.path.abspath(f.name)
            for layer, layer_info in viewitems(ri_layers):
                oasis_files['RI_{}'.format(layer)] = layer_info['directory']

        return oasis_files

    def generate_losses(
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
        if os.path.basename(oasis_fp) == 'input':
            ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(oasis_fp))
        elif os.path.basename(oasis_fp) == 'csv':
            ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(os.path.dirname(oasis_fp)))

        if not os.path.exists(model_run_fp):
            Path(model_run_fp).mkdir(parents=True, exist_ok=True)

        shutil.copy2(analysis_settings_fp, model_run_fp)

        prepare_model_run_directory(
            model_run_fp,
            oasis_fp=oasis_fp,
            ri=ri,
            analysis_settings_fp=analysis_settings_fp,
            model_data_fp=model_data_fp
        )

        if not ri:
            csv_to_bin(oasis_fp, os.path.join(model_run_fp, 'input'), il=il)
        else:
            contents = os.listdir(model_run_fp)
            for fp in [os.path.join(model_run_fp, fn) for fn in contents if re.match(r'RI_\d+$', fn) or re.match(r'input$', fn)]:
                csv_to_bin(fp, fp, il=True, ri=True)

        analysis_settings_fp = os.path.join(model_run_fp, 'analysis_settings.json')
        try:
            with io.open(analysis_settings_fp, 'r', encoding='utf-8') as f:
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
            raise OasisException('Invalid analysis settings file or file path: {}.'.format(analysis_settings_fp))

        prepare_model_run_inputs(analysis_settings, model_run_fp, ri=ri)

        script_fp = os.path.join(model_run_fp, 'run_ktools.sh')

        if model_package_fp and os.path.exists(os.path.join(model_package_fp, 'supplier_model_runner.py')):
            path, package_name = model_package_path.rsplit('/')
            sys.path.append(path)
            model_runner_module = importlib.import_module('{}.supplier_model_runner'.format(package_name))
        else:
            model_runner_module = runner

        with setcwd(model_run_fp) as cwd_path:
            ri_layers = 0
            if ri:
                try:
                    with io.open(os.path.join(model_run_fp, 'ri_layers.json'), 'r', encoding='utf-8') as f:
                        ri_layers = len(json.load(f))
                except IOError:
                    with io.open(os.path.join(model_run_fp, 'input', 'ri_layers.json'), 'r', encoding='utf-8') as f:
                        ri_layers = len(json.load(f))

            model_runner_module.run(
                analysis_settings, 
                (ktools_num_processes or self.ktools_num_processes),
                filename=script_fp,
                num_reinsurance_iterations=ri_layers,
                ktools_mem_limit=(ktools_mem_limit or self.ktools_mem_limit),
                set_alloc_rule=(ktools_alloc_rule or self.ktools_alloc_rule), 
                fifo_tmp_dir=(not (ktools_fifo_relative or self.ktools_fifo_relative))
            )

    def run_deterministic(
        self,
        input_dir,
        output_dir=None,
        loss_percentage_of_tiv=1.0,
        net=False,
        print_losses=True
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

        generate_binary_inputs(input_dir, output_dir)

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

        guls_list = []
        for item_id, tiv in zip(items_df['item_id'], items_df['tiv']):
            event_loss = loss_percentage_of_tiv * tiv
            guls_list += [
                oed.GulRecord(event_id=1, item_id=item_id, sidx=-1, loss=event_loss),
                oed.GulRecord(event_id=1, item_id=item_id, sidx=-2, loss=0),
                oed.GulRecord(event_id=1, item_id=item_id, sidx=1, loss=event_loss)
            ]

        guls_df = pd.DataFrame(guls_list)
        guls_fp = os.path.join(output_dir, "guls.csv")
        guls_df.to_csv(guls_fp, index=False)

        net_flag = "-n" if net else ""
        ils_fp = os.path.join(output_dir, 'ils.csv')
        command = "gultobin -S 1 < {} | fmcalc -p {} {} -a {} | tee ils.bin | fmtocsv > {}".format(
            guls_fp, output_dir, net_flag, oed.ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID, ils_fp)
        print("\nRunning command: {}\n".format(command))
        proc = subprocess.Popen(command, shell=True)
        proc.wait()
        if proc.returncode != 0:
            raise OasisException("Failed to run fm")

        losses_df = pd.read_csv(ils_fp)
        losses_df.drop(losses_df[losses_df.sidx != 1].index, inplace=True)
        losses_df.reset_index(drop=True, inplace=True)
        del losses_df['sidx']

        if print_losses:
            # Set ``event_id`` and ``output_id`` column data types to ``object``
            # to prevent ``tabulate`` from int -> float conversion during console printing
            losses_df['event_id'] = losses_df['event_id'].astype(object)
            losses_df['output_id'] = losses_df['output_id'].astype(object)

            print(tabulate(losses_df, headers='keys', tablefmt='psql', floatfmt=".2f"))

            # Reset event ID and output ID column dtypes to int
            losses_df['event_id'] = losses_df['event_id'].astype(int)
            losses_df['output_id'] = losses_df['output_id'].astype(int)

        return losses_df

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

        oasis_fp = os.path.join(model_run_fp, 'input') if ri else os.path.join(model_run_fp, 'input', 'csv')
        Path(oasis_fp).mkdir(parents=True, exist_ok=True)

        self.generate_oasis_files(
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

        self.generate_losses(
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