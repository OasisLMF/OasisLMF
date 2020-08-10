__all__ = [
    'Losses'
]

import importlib
import io
import json
import os
import re
import sys
import warnings

from subprocess import CalledProcessError

from pathlib2 import Path

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

from ...utils.data import (
    get_analysis_settings,
    get_utctimestamp,
)
from ...utils.defaults import (
    KTOOLS_NUM_PROCESSES,
    KTOOLS_FIFO_RELATIVE,
    KTOOLS_ERR_GUARD,
    KTOOLS_ALLOC_GUL_DEFAULT,
    KTOOLS_ALLOC_IL_DEFAULT,
    KTOOLS_ALLOC_RI_DEFAULT,
    KTOOLS_ALLOC_GUL_MAX,
    KTOOLS_ALLOC_FM_MAX,
    KTOOLS_GUL_LEGACY_STREAM,
    KTOOLS_DEBUG,
)

warnings.simplefilter(action='ignore', category=FutureWarning)

class Losses(ComputationStep):
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
        {'name': 'oasis_files_dir',        'flag':'-o', 'is_path': True, 'pre_exist': False, 'help': 'Path to the directory in which to generate the Oasis files'},
        {'name': 'analysis_settings_json', 'flag':'-a', 'is_path': True, 'pre_exist': True,  'help': 'Analysis settings JSON file path'},
        {'name': 'user_data_dir',          'flag':'-D', 'is_path': True, 'pre_exist': False, 'help': 'Directory containing additional model data files which varies between analysis runs'},
        {'name': 'model_data_dir',         'flag':'-d', 'is_path': True, 'pre_exist': True,  'help': 'Model data directory path'},
        {'name': 'model_run_dir',          'flag':'-r', 'is_path': True, 'pre_exist': False, 'help': 'Model run directory path'},
        {'name': 'model_package_dir',      'flag':'-p', 'is_path': True, 'pre_exist': False, 'help': 'Path containing model specific package'},
        {'name': 'ktools_num_processes',   'flag':'-n', 'type':int,   'default': KTOOLS_NUM_PROCESSES, 'help': 'Number of ktools calculation processes to use'},
        {'name': 'ktools_alloc_rule_gul',  'default': KTOOLS_ALLOC_GUL_DEFAULT, 'type':int, 'help': 'Set the allocation used in gulcalc'},
        {'name': 'ktools_alloc_rule_il',   'default': KTOOLS_ALLOC_IL_DEFAULT,  'type':int, 'help': 'Set the fmcalc allocation rule used in direct insured loss'},
        {'name': 'ktools_alloc_rule_ri',   'default': KTOOLS_ALLOC_RI_DEFAULT,  'type':int, 'help': 'Set the fmcalc allocation rule used in reinsurance'},
        {'name': 'ktools_disable_guard',   'default': not KTOOLS_ERR_GUARD, 'action': 'store_true', 'help': 'Disables error handling in the ktools run script (abort on non-zero exitcode or output on stderr)'},
        {'name': 'ktools_fifo_relative',   'default': KTOOLS_FIFO_RELATIVE, 'action': 'store_true', 'help': 'Create ktools fifo queues under the ./fifo dir'},

        # Manager only options (pass data directy instead of filepaths)
        {'name': 'ktools_debug',         'default': KTOOLS_DEBUG},
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

        il = all(p in os.listdir(self.oasis_files_dir) for p in ['fm_policytc.csv', 'fm_profile.csv', 'fm_programme.csv', 'fm_xref.csv'])
        ri = any(re.match(r'RI_\d+$', fn) for fn in os.listdir(os.path.dirname(self.oasis_files_dir)) + os.listdir(self.oasis_files_dir))
        gul_item_stream = (not self.ktools_legacy_stream)
        self.logger.info('\nGenerating losses (GUL=True, IL={}, RIL={})'.format(il, ri))

        model_run_fp = self._get_output_dir()
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

        if not any([
            analysis_settings['gul_output'] if 'gul_output' in analysis_settings else False,
            analysis_settings['il_output'] if 'il_output' in analysis_settings else False,
            analysis_settings['ri_output'] if 'ri_output' in analysis_settings else False,
        ]):
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
                model_runner_module.run(
                    analysis_settings,
                    number_of_processes=self.ktools_num_processes,
                    filename=script_fp,
                    num_reinsurance_iterations=ri_layers,
                    set_alloc_rule_gul=self.ktools_alloc_rule_gul,
                    set_alloc_rule_il=self.ktools_alloc_rule_il,
                    set_alloc_rule_ri=self.ktools_alloc_rule_ri,
                    run_debug=self.ktools_debug,
                    stderr_guard= not self.ktools_disable_guard,
                    gul_legacy_stream=self.ktools_legacy_stream,
                    fifo_tmp_dir=self.ktools_fifo_relative,
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
