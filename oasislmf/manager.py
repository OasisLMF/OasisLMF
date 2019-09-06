__all__ = [
    'OasisManager'
]

import io
import importlib
import json
import os
import re
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from builtins import str

from itertools import (
    product,
)

import pandas as pd
pd.options.mode.chained_assignment = None

from pathlib2 import Path

from .model_execution import runner
from .model_execution.bin import (
    csv_to_bin,
    prepare_run_directory,
    prepare_run_inputs,
)
from .model_preparation.gul_inputs import (
    get_gul_input_items,
    write_gul_input_files,
)
from .model_preparation.il_inputs import (
    get_il_input_items,
    get_oed_hierarchy,
    write_il_input_files,
)
from .model_preparation.summaries import (
    get_summary_mapping,
    generate_summaryxref_files,
    merge_oed_to_mapping,
    write_mapping_file,
    write_exposure_summary,
    write_summary_levels,
)
from .model_preparation.lookup import OasisLookupFactory as olf
from .model_preparation.oed import load_oed_dfs
from .model_preparation.utils import prepare_input_files_directory
from .model_preparation.reinsurance_layer import write_files_for_reinsurance
from .utils.data import (
    get_dataframe,
    get_ids,
    get_json,
    get_utctimestamp,
)
from .utils.exceptions import OasisException
from .utils.log import oasis_log
from .utils.defaults import (
    get_default_accounts_profile,
    get_default_deterministic_analysis_settings,
    get_default_exposure_profile,
    get_default_fm_aggregation_profile,
    KTOOLS_NUM_PROCESSES,
    KTOOLS_MEM_LIMIT,
    KTOOLS_FIFO_RELATIVE,
    KTOOLS_ALLOC_RULE_GUL,
    KTOOLS_ALLOC_RULE_IL,
    KTOOLS_DEBUG,
    OASIS_FILES_PREFIXES,
    WRITE_CHUNKSIZE,
)
from .utils.deterministic_loss import generate_deterministic_losses
from .utils.peril import PerilAreasIndex
from .utils.path import (
    as_path,
    setcwd,
)
from .utils.coverages import SUPPORTED_COVERAGE_TYPES


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
        ktools_alloc_rule_gul=None,
        ktools_alloc_rule_il=None,
        ktools_debug=None,
        oasis_files_prefixes=None,
        write_chunksize=None
    ):
        # Set defaults for static data or runtime parameters
        self._exposure_profile = exposure_profile or get_default_exposure_profile()
        self._supported_oed_coverage_types = supported_oed_coverage_types or tuple(v['id'] for v in SUPPORTED_COVERAGE_TYPES.values())
        self._accounts_profile = accounts_profile or get_default_accounts_profile()
        self._fm_aggregation_profile = fm_aggregation_profile or get_default_fm_aggregation_profile()
        self._deterministic_analysis_settings = deterministic_analysis_settings or get_default_deterministic_analysis_settings()
        self._ktools_num_processes = ktools_num_processes or KTOOLS_NUM_PROCESSES
        self._ktools_mem_limit = ktools_mem_limit or KTOOLS_MEM_LIMIT
        self._ktools_fifo_relative = ktools_fifo_relative or KTOOLS_FIFO_RELATIVE
        self._ktools_alloc_rule_gul = ktools_alloc_rule_gul if isinstance(ktools_alloc_rule_gul, int) else KTOOLS_ALLOC_RULE_GUL
        self._ktools_alloc_rule_il = ktools_alloc_rule_il if isinstance(ktools_alloc_rule_il, int) else KTOOLS_ALLOC_RULE_IL
        self._ktools_debug = ktools_debug or KTOOLS_DEBUG
        self._oasis_files_prefixes = oasis_files_prefixes or OASIS_FILES_PREFIXES
        self._write_chunksize = write_chunksize or WRITE_CHUNKSIZE

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
    def ktools_num_processes(self):
        return self._ktools_num_processes

    @property
    def ktools_mem_limit(self):
        return self._ktools_mem_limit

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
    def ktools_debug(self):
        return self._ktools_debug

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
        complex_lookup_config_fp=None,
        keys_fp=None,
        keys_errors_fp=None,
        keys_format=None
    ):
        if keys_fp:
            lookup_extra_outputs_dir = os.path.basename(keys_fp)
        else:
            lookup_extra_outputs_dir = os.getcwd()

        model_info, lookup = olf.create(
            lookup_config_fp=lookup_config_fp,
            model_keys_data_path=keys_data_fp,
            model_version_file_path=model_version_fp,
            lookup_package_path=lookup_package_fp,
            complex_lookup_config_fp=complex_lookup_config_fp,
            output_directory=lookup_extra_outputs_dir
        )

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')

        keys_fp = keys_fp or '{}-keys.csv'.format(utcnow)
        keys_errors_fp = keys_errors_fp or '{}-keys-errors.csv'.format(utcnow)

        return olf.save_results(
            lookup,
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
        complex_lookup_config_fp=None,
        user_data_dir=None,
        supported_oed_coverage_types=None,
        summarise_exposure=None,
        write_chunksize=None,
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
        target_dir = prepare_input_files_directory(
            target_dir,
            exposure_fp,
            exposure_profile_fp=exposure_profile_fp,
            keys_fp=keys_fp,
            lookup_config_fp=lookup_config_fp,
            model_version_fp=model_version_fp,
            complex_lookup_config_fp=complex_lookup_config_fp,
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
        oed_hierarchy = get_oed_hierarchy(exposure_profile, accounts_profile)
        loc_num = oed_hierarchy['locnum']['ProfileElementName'].lower()
        loc_grp = oed_hierarchy['locgrp']['ProfileElementName'].lower()
        acc_num = oed_hierarchy['accnum']['ProfileElementName'].lower()
        portfolio_num = oed_hierarchy['portnum']['ProfileElementName'].lower()
        fm_aggregation_profile = (
            fm_aggregation_profile or
            ({int(k): v for k, v in get_json(src_fp=fm_aggregation_profile_fp).items()} if fm_aggregation_profile_fp else {}) or
            self.fm_aggregation_profile
        )

        # The chunksize to use when writing the GUL and IL inputs dataframes
        # to file
        write_chunksize = write_chunksize or self.write_chunksize

        # Check whether the files generation is for deterministic or model losses
        deterministic = not(
            (lookup_config or lookup_config_fp) or
            (keys_data_fp and model_version_fp and lookup_package_fp) or
            keys_fp
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
                exposure_df = get_dataframe(
                    src_fp=exposure_fp,
                    empty_data_error_msg='No exposure found in the source exposure (loc.) file'
                )
                exposure_df['loc_id'] = get_ids(exposure_df, [portfolio_num, acc_num, loc_num])
                loc_ids = (loc_it['loc_id'] for _, loc_it in exposure_df.loc[:, ['loc_id']].iterrows())
                keys = [
                    {'loc_id': _loc_id, 'peril_id': 1, 'coverage_type': cov_type, 'area_peril_id': i + 1, 'vulnerability_id': i + 1}
                    for i, (_loc_id, cov_type) in enumerate(product(loc_ids, cov_types))
                ]
                _, _ = olf.write_oasis_keys_file(keys, _keys_fp)
            else:
                lookup_config = get_json(src_fp=lookup_config_fp) if lookup_config_fp else lookup_config
                if lookup_config and lookup_config['keys_data_path'] in ['.', './']:
                    lookup_config['keys_data_path'] = os.path.join(os.path.dirname(lookup_config_fp))
                elif lookup_config and not os.path.isabs(lookup_config['keys_data_path']):
                    lookup_config['keys_data_path'] = os.path.join(os.path.dirname(lookup_config_fp), keys_data_fp)

                _, lookup = olf.create(
                    lookup_config=lookup_config,
                    model_keys_data_path=keys_data_fp,
                    model_version_file_path=model_version_fp,
                    lookup_package_path=lookup_package_fp,
                    complex_lookup_config_fp=complex_lookup_config_fp,
                    user_data_dir=user_data_dir,
                    output_directory=target_dir
                )
                f1, _, f2, _ = olf.save_results(
                    lookup,
                    successes_fp=_keys_fp,
                    errors_fp=_keys_errors_fp,
                    source_exposure_fp=exposure_fp
                )
        else:
            _keys_fp = os.path.join(target_dir, os.path.basename(keys_fp))

        # Get the GUL input items and exposure dataframes
        gul_inputs_df, exposure_df = get_gul_input_items(
            exposure_fp,
            _keys_fp,
            exposure_profile=exposure_profile
        )

        # If not in det. loss gen. scenario, write exposure summary file
        if summarise_exposure and not deterministic:
            write_exposure_summary(
                target_dir,
                gul_inputs_df,
                exposure_df,
                exposure_fp,
                keys_errors_fp=_keys_errors_fp,
                exposure_profile=exposure_profile,
                oed_hierarchy=oed_hierarchy
            )

        # If exposure summary set, write valid columns for summary levels to file
        if summarise_exposure:
            write_summary_levels(exposure_df, accounts_fp, target_dir)

        # Write the GUL input files
        files_prefixes = oasis_files_prefixes or self.oasis_files_prefixes
        gul_input_files = write_gul_input_files(
            gul_inputs_df,
            target_dir,
            oasis_files_prefixes=files_prefixes['gul'],
            chunksize=write_chunksize
        )
        gul_summary_mapping = get_summary_mapping(gul_inputs_df, oed_hierarchy)
        write_mapping_file(gul_summary_mapping, target_dir)

        # If no source accounts file path has been provided assume that IL
        # input files, and therefore also RI input files, are not needed
        if not accounts_fp:
            # Write `summary_map.csv` for GUL only
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
            oasis_files_prefixes=files_prefixes['il'],
            chunksize=write_chunksize
        )
        fm_summary_mapping = get_summary_mapping(il_inputs_df, oed_hierarchy, is_fm_summary=True)
        write_mapping_file(fm_summary_mapping, target_dir, is_fm_summary=True)

        # Combine the GUL and IL input file paths into a single dict (for convenience)
        oasis_files = {**gul_input_files, **il_input_files}

        # If no RI input file paths (info. and scope) have been provided then
        # no RI input files are needed, just return the GUL and IL Oasis files
        if not (ri_info_fp or ri_scope_fp):
            return oasis_files

        # Write the RI input files, and write the returned RI layer info. as a
        # file, which can be reused by the model runner (in the model execution
        # stage) to set the number of RI iterations

        xref_descriptions_df = merge_oed_to_mapping(
            fm_summary_mapping,
            exposure_df,
            oed_column_set=[loc_grp],
            defaults={loc_grp: 1}
        )

        ri_info_df, ri_scope_df, _ = load_oed_dfs(ri_info_fp, ri_scope_fp)
        ri_layers = write_files_for_reinsurance(
            gul_inputs_df,
            xref_descriptions_df,
            ri_info_df,
            ri_scope_df,
            oasis_files['fm_xref'],
            target_dir
        )

        with io.open(os.path.join(target_dir, 'ri_layers.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(ri_layers, ensure_ascii=False, indent=4))
            oasis_files['ri_layers'] = os.path.abspath(f.name)
            for layer, layer_info in ri_layers.items():
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
        ktools_alloc_rule_gul=None,
        ktools_alloc_rule_il=None,
        ktools_debug=None,
        user_data_dir=None
    ):
        il = all(p in os.listdir(oasis_fp) for p in ['fm_policytc.csv', 'fm_profile.csv', 'fm_programme.csv', 'fm_xref.csv'])
        ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(os.path.dirname(oasis_fp)) + os.listdir(oasis_fp))
        gul_item_stream = False if (ktools_alloc_rule_gul is 0) or (self.ktools_alloc_rule_gul is 0) else True

        if not os.path.exists(model_run_fp):
            Path(model_run_fp).mkdir(parents=True, exist_ok=True)

        prepare_run_directory(
            model_run_fp,
            oasis_fp,
            model_data_fp,
            analysis_settings_fp,
            user_data_dir=user_data_dir,
            ri=ri
        )

        # Load analysis_settings file
        try:
            analysis_settings_fn = 'analysis_settings.json'
            _analysis_settings_fp = os.path.join(model_run_fp, analysis_settings_fn)
            with io.open(_analysis_settings_fp, 'r', encoding='utf-8') as f:
                analysis_settings = json.load(f)
            if analysis_settings.get('analysis_settings'):
                analysis_settings = analysis_settings['analysis_settings']
        except (IOError, TypeError, ValueError):
            raise OasisException('Invalid analysis settings file or file path: {}.'.format(_analysis_settings_fp))

        generate_summaryxref_files(model_run_fp, 
                                   analysis_settings, 
                                   gul_item_stream=gul_item_stream,
                                   il=il, 
                                   ri=ri)

        if not ri:
            fp = os.path.join(model_run_fp, 'input')
            csv_to_bin(fp, fp, il=il)
        else:
            contents = os.listdir(model_run_fp)
            for fp in [os.path.join(model_run_fp, fn) for fn in contents if re.match(r'RI_\d+$', fn) or re.match(r'input$', fn)]:
                csv_to_bin(fp, fp, il=True, ri=True)

        if not il:
            analysis_settings['il_output'] = False
            analysis_settings['il_summaries'] = []

        if not ri:
            analysis_settings['ri_output'] = False
            analysis_settings['ri_summaries'] = []

        prepare_run_inputs(analysis_settings, model_run_fp, ri=ri)

        script_fp = os.path.join(model_run_fp, 'run_ktools.sh')

        if model_package_fp and os.path.exists(os.path.join(model_package_fp, 'supplier_model_runner.py')):
            path, package_name = os.path.split(model_package_fp)
            sys.path.append(path)
            model_runner_module = importlib.import_module('{}.supplier_model_runner'.format(package_name))
        else:
            model_runner_module = runner

        with setcwd(model_run_fp):
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
                number_of_processes=(ktools_num_processes or self.ktools_num_processes),
                filename=script_fp,
                num_reinsurance_iterations=ri_layers,
                ktools_mem_limit=(ktools_mem_limit or self.ktools_mem_limit),
                set_alloc_rule_gul=(ktools_alloc_rule_gul if isinstance(ktools_alloc_rule_gul, int) else self.ktools_alloc_rule_gul), 
                set_alloc_rule_il=(ktools_alloc_rule_il if isinstance(ktools_alloc_rule_il, int) else self.ktools_alloc_rule_il),
                run_debug=(ktools_debug or self.ktools_debug),
                fifo_tmp_dir=(not (ktools_fifo_relative or self.ktools_fifo_relative))
            )

        return model_run_fp

    @oasis_log
    def run_deterministic(
        self,
        src_dir,
        run_dir=None,
        loss_percentage_of_tiv=1.0,
        alloc_rule=KTOOLS_ALLOC_RULE_IL,
        net_ri=False
    ):
        """
        Generates insured losses from preexisting Oasis files with a specified
        damage ratio (loss % of TIV).
        """
        if not run_dir:
            run_dir = os.path.join(src_dir, 'run')
        elif not os.path.exists(run_dir):
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
            loss_percentage_of_tiv=loss_percentage_of_tiv,
            net_ri=net_ri,
            alloc_rule=alloc_rule
        )

        return losses['gul'], losses['il'], losses['ri']
