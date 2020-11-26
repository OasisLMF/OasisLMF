__all__ = [
    'GenerateLosses',
    'GenerateLossesDeterministic'
]

import importlib
import io
import json
import os
import pandas as pd
import re
import sys
import warnings

from collections import OrderedDict
from itertools import product
from json import JSONDecodeError
from pathlib2 import Path
from subprocess import CalledProcessError, check_call

from ..base import ComputationStep
from ...execution import runner
from ...execution.bin import (
    csv_to_bin,
    prepare_run_directory,
    prepare_run_inputs,
)
from ...preparation.summaries import generate_summaryxref_files
from ...utils.exceptions import OasisException
from ...utils.path import setcwd
from ...utils.inputs import str2bool

from ...utils.data import (
    fast_zip_dataframe_columns,
    get_analysis_settings,
    get_dataframe,
    get_utctimestamp,
    merge_dataframes,
    set_dataframe_column_dtypes,
)
from ...utils.defaults import (
    KTOOLS_ALLOC_FM_MAX,
    KTOOLS_ALLOC_GUL_DEFAULT,
    KTOOLS_ALLOC_GUL_MAX,
    KTOOLS_ALLOC_IL_DEFAULT,
    KTOOLS_ALLOC_RI_DEFAULT,
    KTOOLS_DEBUG,
    KTOOLS_DISABLE_ERR_GUARD,
    KTOOLS_FIFO_RELATIVE,
    KTOOLS_GUL_LEGACY_STREAM,
    KTOOLS_MEAN_SAMPLE_IDX,
    KTOOLS_NUM_PROCESSES,
    KTOOL_N_GUL_PER_LB,
    KTOOL_N_FM_PER_LB,
    KTOOLS_STD_DEV_SAMPLE_IDX,
    KTOOLS_TIV_SAMPLE_IDX
)

warnings.simplefilter(action='ignore', category=FutureWarning)


class GenerateLosses(ComputationStep):
    """
    Generates losses using the installed ktools framework given Oasis files,
    model analysis settings JSON file, model data and model package data.

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    Run ``oasislmf config --help`` for more information.

    The script creates a time-stamped folder in the model run directory and
    sets that as the new model run directory, copies the analysis settings
    JSON file into the run directory and creates the following folder
    structure
    ::

        |-- analysis_settings.json
        |-- fifo
        |-- input
        |-- output
        |-- RI_1
        |-- ri_layers.json
        |-- run_ktools.sh
        |-- static
        `-- work

    Depending on the OS type the model data is symlinked (Linux, Darwin) or
    copied (Cygwin, Windows) into the ``static`` subfolder. The input files
    are kept in the ``input`` subfolder and the losses are generated as CSV
    files in the ``output`` subfolder.
    """
    step_params = [
        # Command line options
        {'name': 'oasis_files_dir',        'flag':'-o', 'is_path': True, 'pre_exist': True, 'required': True, 'help': 'Path to the directory in which to generate the Oasis files'},
        {'name': 'analysis_settings_json', 'flag':'-a', 'is_path': True, 'pre_exist': True, 'required': True,  'help': 'Analysis settings JSON file path'},
        {'name': 'user_data_dir',          'flag':'-D', 'is_path': True, 'pre_exist': False, 'help': 'Directory containing additional model data files which varies between analysis runs'},
        {'name': 'model_data_dir',         'flag':'-d', 'is_path': True, 'pre_exist': True,  'help': 'Model data directory path'},
        {'name': 'model_run_dir',          'flag':'-r', 'is_path': True, 'pre_exist': False, 'help': 'Model run directory path'},
        {'name': 'model_package_dir',      'flag':'-p', 'is_path': True, 'pre_exist': False, 'help': 'Path containing model specific package'},
        {'name': 'ktools_num_processes',   'flag':'-n', 'type':int,   'default': KTOOLS_NUM_PROCESSES, 'help': 'Number of ktools calculation processes to use'},
        {'name': 'ktools_alloc_rule_gul',  'default': KTOOLS_ALLOC_GUL_DEFAULT, 'type':int, 'help': 'Set the allocation used in gulcalc'},
        {'name': 'ktools_alloc_rule_il',   'default': KTOOLS_ALLOC_IL_DEFAULT,  'type':int, 'help': 'Set the fmcalc allocation rule used in direct insured loss'},
        {'name': 'ktools_alloc_rule_ri',   'default': KTOOLS_ALLOC_RI_DEFAULT,  'type':int, 'help': 'Set the fmcalc allocation rule used in reinsurance'},
        {'name': 'ktools_num_gul_per_lb',  'default': KTOOL_N_GUL_PER_LB,       'type':int, 'help': 'Number of gul per load balancer (0 means no load balancer)'},
        {'name': 'ktools_num_fm_per_lb',   'default': KTOOL_N_FM_PER_LB,        'type':int, 'help': 'Number of fm per load balancer (0 means no load balancer)'},
        {'name': 'ktools_disable_guard',   'default': False, 'type': str2bool, 'const':True, 'nargs':'?', 'help': 'Disables error handling in the ktools run script (abort on non-zero exitcode or output on stderr)'},
        {'name': 'ktools_fifo_relative',   'default': False, 'type': str2bool, 'const':True, 'nargs':'?', 'help': 'Create ktools fifo queues under the ./fifo dir'},
        {'name': 'fmpy',                   'default': False, 'type': str2bool, 'const':True, 'nargs':'?', 'help': 'use fmcalc python version instead of c++ version'},
        {'name': 'fmpy_low_memory',        'default': False, 'type': str2bool, 'const':True, 'nargs':'?', 'help': 'use memory map instead of RAM to store loss array (may decrease performance but reduce RAM usage drastically)'},

        # Manager only options (pass data directy instead of filepaths)
        {'name': 'verbose',              'default': KTOOLS_DEBUG},
        {'name': 'ktools_legacy_stream', 'default': KTOOLS_GUL_LEGACY_STREAM},
        {'name': 'model_custom_gulcalc', 'default': None},
    ]

    def _get_output_dir(self):
        run_dir = None
        if self.model_run_dir:
            run_dir = self.model_run_dir
        else:
            utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
            run_dir = os.path.join(os.getcwd(), 'runs', 'losses-{}'.format(utcnow))

        if not os.path.exists(run_dir):
            Path(run_dir).mkdir(parents=True, exist_ok=True)
        return run_dir

    def _check_alloc_rules(self):
        alloc_ranges = {
            'ktools_alloc_rule_gul': KTOOLS_ALLOC_GUL_MAX,
            'ktools_alloc_rule_il': KTOOLS_ALLOC_FM_MAX,
            'ktools_alloc_rule_ri': KTOOLS_ALLOC_FM_MAX}
        for rule in alloc_ranges:
            alloc_val = getattr(self, rule)
            if (alloc_val < 0) or (alloc_val > alloc_ranges[rule]):
                raise OasisException(f'Error: {rule}={alloc_val} - Not withing valid range [0..{alloc_ranges[rule]}]')

    def run(self):
        model_run_fp = self._get_output_dir()

        il = all(p in os.listdir(self.oasis_files_dir) for p in ['fm_policytc.csv', 'fm_profile.csv', 'fm_programme.csv', 'fm_xref.csv'])
        ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(os.path.dirname(self.oasis_files_dir)) + os.listdir(self.oasis_files_dir))
        gul_item_stream = (not self.ktools_legacy_stream)
        self.logger.info('\nGenerating losses (GUL=True, IL={}, RIL={})'.format(il, ri))

        self._check_alloc_rules()
        analysis_settings = get_analysis_settings(self.analysis_settings_json)

        prepare_run_directory(
            model_run_fp,
            self.oasis_files_dir,
            self.model_data_dir,
            self.analysis_settings_json,
            user_data_dir=self.user_data_dir,
            ri=ri
        )
        generate_summaryxref_files(
            model_run_fp,
            analysis_settings,
            gul_item_stream=gul_item_stream,
            il=il,
            ri=ri
        )

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

        if not any(analysis_settings.get(output) for output in ['gul_output', 'il_output', 'ri_output']):
            raise OasisException(
                'No valid output settings in: {}'.format(self.analysis_settings_json))

        prepare_run_inputs(analysis_settings, model_run_fp, ri=ri)
        script_fp = os.path.join(os.path.abspath(model_run_fp), 'run_ktools.sh')

        if self.model_package_dir and os.path.exists(os.path.join(self.model_package_dir, 'supplier_model_runner.py')):
            path, package_name = os.path.split(self.model_package_dir)
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

            try:
                try:
                    model_runner_module.run(
                        analysis_settings,
                        number_of_processes=self.ktools_num_processes,
                        filename=script_fp,
                        num_reinsurance_iterations=ri_layers,
                        set_alloc_rule_gul=self.ktools_alloc_rule_gul,
                        set_alloc_rule_il=self.ktools_alloc_rule_il,
                        set_alloc_rule_ri=self.ktools_alloc_rule_ri,
                        num_gul_per_lb=self.ktools_num_gul_per_lb,
                        num_fm_per_lb=self.ktools_num_fm_per_lb,
                        run_debug=self.verbose,
                        stderr_guard=not self.ktools_disable_guard,
                        gul_legacy_stream=self.ktools_legacy_stream,
                        fifo_tmp_dir=not self.ktools_fifo_relative,
                        custom_gulcalc_cmd=self.model_custom_gulcalc,
                        fmpy=self.fmpy,
                        fmpy_low_memory=self.fmpy_low_memory,
                    )
                except TypeError:
                    warnings.simplefilter("always")
                    warnings.warn(f"{package_name}.supplier_model_runner doesn't accept new runner arguments, please add **kwargs to the run function signature")
                    model_runner_module.run(
                        analysis_settings,
                        number_of_processes=self.ktools_num_processes,
                        filename=script_fp,
                        num_reinsurance_iterations=ri_layers,
                        set_alloc_rule_gul=self.ktools_alloc_rule_gul,
                        set_alloc_rule_il=self.ktools_alloc_rule_il,
                        set_alloc_rule_ri=self.ktools_alloc_rule_ri,
                        run_debug=self.verbose,
                        stderr_guard=not self.ktools_disable_guard,
                        gul_legacy_stream=self.ktools_legacy_stream,
                        fifo_tmp_dir=not self.ktools_fifo_relative,
                        custom_gulcalc_cmd=self.model_custom_gulcalc,
                    )
            except CalledProcessError as e:
                bash_trace_fp = os.path.join(model_run_fp, 'log', 'bash.log')
                if os.path.isfile(bash_trace_fp):
                   with io.open(bash_trace_fp, 'r', encoding='utf-8') as f:
                        self.logger.info('\nBASH_TRACE:\n' + "".join(f.readlines()))

                stderror_fp = os.path.join(model_run_fp, 'log', 'stderror.err')
                if os.path.isfile(stderror_fp):
                    with io.open(stderror_fp, 'r', encoding='utf-8') as f:
                        self.logger.info('\nKTOOLS_STDERR:\n' + "".join(f.readlines()))

                gul_stderror_fp = os.path.join(model_run_fp, 'log', 'gul_stderror.err')
                if os.path.isfile(gul_stderror_fp):
                    with io.open(gul_stderror_fp, 'r', encoding='utf-8') as f:
                        self.logger.info('\nGUL_STDERR:\n' + "".join(f.readlines()))

                self.logger.info('\nSTDOUT:\n' + e.output.decode('utf-8').strip())

                raise OasisException(
                    'Ktools run Error: non-zero exit code or output detected on STDERR\n'
                    'Logs stored in: {}/log'.format(model_run_fp)
                )

        self.logger.info('Losses generated in {}'.format(model_run_fp))
        return model_run_fp


class GenerateLossesDeterministic(ComputationStep):


    step_params = [
        {'name': 'oasis_files_dir',  'is_path': True, 'pre_exist': True},
        {'name': 'output_dir',           'default': None},
        {'name': 'include_loss_factor',  'default': True},
        {'name': 'loss_factor',          'default': [1.0]},
        {'name': 'net_ri',               'default': False},
        {'name': 'ktools_alloc_rule_il', 'default': KTOOLS_ALLOC_IL_DEFAULT},
        {'name': 'ktools_alloc_rule_ri', 'default': KTOOLS_ALLOC_RI_DEFAULT},
        {'name': 'num_subperils',        'default': 1}
    ]

    def run(self):

        losses = OrderedDict({'gul': None, 'il': None, 'ri': None})
        output_dir = self.output_dir or self.oasis_files_dir

        il = all(p in os.listdir(self.oasis_files_dir) for p in ['fm_policytc.csv', 'fm_profile.csv', 'fm_programme.csv', 'fm_xref.csv'])
        ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(self.oasis_files_dir))

        step_flag = ''
        try:
            pd.read_csv(os.path.join(self.oasis_files_dir, 'fm_profile.csv'))['step_id']
        except (OSError, FileNotFoundError, KeyError):
            pass
        else:
            step_flag = '-S'

        csv_to_bin(self.oasis_files_dir, output_dir, il=il, ri=ri)

        # Generate an items and coverages dataframe and set column types (important!!)
        items = merge_dataframes(
            pd.read_csv(os.path.join(self.oasis_files_dir, 'items.csv')),
            pd.read_csv(os.path.join(self.oasis_files_dir, 'coverages.csv')),
            on=['coverage_id'], how='left'
        )

        dtypes = {t: ('uint32' if t != 'tiv' else 'float32') for t in items.columns}
        items = set_dataframe_column_dtypes(items, dtypes)

        gulcalc_sidxs = \
            [KTOOLS_MEAN_SAMPLE_IDX, KTOOLS_STD_DEV_SAMPLE_IDX, KTOOLS_TIV_SAMPLE_IDX] + \
            list(range(1, len(self.loss_factor) + 1))

        # Set damage percentages corresponing to the special indexes.
        # We don't care about mean and std_dev, but
        # TIV needs to be set correctly.
        special_loss_factors = {
            KTOOLS_MEAN_SAMPLE_IDX: 0.,
            KTOOLS_STD_DEV_SAMPLE_IDX: 0.,
            KTOOLS_TIV_SAMPLE_IDX: 1.
        }

        guls_items = [
            OrderedDict({
                'event_id': 1,
                'item_id': item_id,
                'sidx': sidx,
                'loss':
                (tiv * special_loss_factors[sidx]) / self.num_subperils if sidx < 0
                else (tiv * self.loss_factor[sidx - 1]) / self.num_subperils
            })
            for (item_id, tiv), sidx in product(
                fast_zip_dataframe_columns(items, ['item_id', 'tiv']), gulcalc_sidxs
            )
        ]

        guls = get_dataframe(
            src_data=guls_items,
            col_dtypes={
                'event_id': int,
                'item_id': int,
                'sidx': int,
                'loss': float})
        guls_fp = os.path.join(output_dir, "raw_guls.csv")
        guls.to_csv(guls_fp, index=False)

        ils_fp = os.path.join(output_dir, 'raw_ils.csv')
        cmd = 'gultobin -S {} -t1 < {} | fmcalc -p {} -a {} {} | tee ils.bin | fmtocsv > {}'.format(
            len(self.loss_factor), guls_fp, output_dir, self.ktools_alloc_rule_il, step_flag, ils_fp
        )
        try:
            self.logger.debug("RUN: " + cmd)
            check_call(cmd, shell=True)
        except CalledProcessError as e:
            raise OasisException("Exception raised in 'generate_deterministic_losses'", e)

        guls.drop(guls[guls['sidx'] < 1].index, inplace=True)
        guls.reset_index(drop=True, inplace=True)
        if self.include_loss_factor:
            guls['loss_factor_idx'] = guls.apply(
                lambda r: int(r['sidx'] - 1), axis='columns')
        guls.drop('sidx', axis=1, inplace=True)
        guls = guls[(guls[['loss']] != 0).any(axis=1)]

        losses['gul'] = guls

        ils = get_dataframe(src_fp=ils_fp)
        ils.drop(ils[ils['sidx'] < 0].index, inplace=True)
        ils.reset_index(drop=True, inplace=True)
        if self.include_loss_factor:
            ils['loss_factor_idx'] = ils.apply(
                lambda r: int(r['sidx'] - 1), axis='columns')
        ils.drop('sidx', axis=1, inplace=True)
        ils = ils[(ils[['loss']] != 0).any(axis=1)]
        losses['il'] = ils

        if ri:
            try:
                [fn for fn in os.listdir(self.oasis_files_dir) if fn == 'ri_layers.json'][0]
            except IndexError:
                raise OasisException(
                    'No RI layers JSON file "ri_layers.json " found in the '
                    'input directory despite presence of RI input files'
                )
            else:
                try:
                    with io.open(os.path.join(self.oasis_files_dir, 'ri_layers.json'), 'r', encoding='utf-8') as f:
                        ri_layers = len(json.load(f))
                except (IOError, JSONDecodeError, OSError, TypeError) as e:
                    raise OasisException('Error trying to read the RI layers file: {}'.format(e))
                else:
                    def run_ri_layer(layer):
                        layer_inputs_fp = os.path.join(output_dir, 'RI_{}'.format(layer))
                        _input = 'gultobin -S 1 -t1 < {} | fmcalc -p {} -a {} {} | tee ils.bin |'.format(
                            guls_fp, output_dir, self.ktools_alloc_rule_il, step_flag
                        ) if layer == 1 else ''
                        pipe_in_previous_layer = '< ri{}.bin'.format(layer - 1) if layer > 1 else ''
                        ri_layer_fp = os.path.join(output_dir, 'ri{}.csv'.format(layer))
                        net_flag = "-n" if self.net_ri else ""
                        cmd = '{} fmcalc -p {} {} -a {} {} {} | tee ri{}.bin | fmtocsv > {}'.format(
                            _input,
                            layer_inputs_fp,
                            net_flag,
                            self.ktools_alloc_rule_ri,
                            pipe_in_previous_layer,
                            step_flag,
                            layer,
                            ri_layer_fp
                        )
                        try:
                            self.logger.debug("RUN: " + cmd)
                            check_call(cmd, shell=True)
                        except CalledProcessError as e:
                            raise OasisException("Exception raised in 'generate_deterministic_losses'", e)
                        rils = get_dataframe(src_fp=ri_layer_fp)
                        rils.drop(rils[rils['sidx'] < 0].index, inplace=True)
                        if self.include_loss_factor:
                            rils['loss_factor_idx'] = rils.apply(
                                lambda r: int(r['sidx'] - 1), axis='columns')

                        rils.drop('sidx', axis=1, inplace=True)
                        rils.reset_index(drop=True, inplace=True)
                        rils = rils[(rils[['loss']] != 0).any(axis=1)]

                        return rils

                    for i in range(1, ri_layers + 1):
                        rils = run_ri_layer(i)
                        if i in [1, ri_layers]:
                            losses['ri'] = rils

        return losses
