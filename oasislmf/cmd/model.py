# -*- coding: utf-8 -*-

import io
import json

import os
import subprocess

import shutil
from argparse import RawDescriptionHelpFormatter

from oasislmf.exposures.manager import OasisExposuresManager
from oasislmf.models.model import OasisModelFactory
from ..model_execution.bin import create_binary_files, prepare_model_run_directory, prepare_model_run_inputs
from ..utils.exceptions import OasisException
from ..utils.values import get_utctimestamp
from ..keys.lookup import OasisKeysLookupFactory
from .cleaners import PathCleaner, as_path
from .base import OasisBaseCommand, InputValues


class GenerateKeysCmd(OasisBaseCommand):
    """
    Generate and write Oasis keys (area peril ID, vulnerability ID) for a model.

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    Run ``oasislmf config --help`` for more information.

    Keys records returned by an Oasis keys lookup service (see the PiWind
    lookup service for reference) will be Python dicts with the following
    structure
    ::

        {
            "id": <loc. ID>,
            "peril_id": <Oasis peril type ID - oasis_utils/oasis_utils.py>,
            "coverage": <Oasis coverage type ID - see oasis_utils/oasis_utils.py>,
            "area_peril_id": <area peril ID>,
            "vulnerability_id": <vulnerability ID>,
            "message": <lookup status message>,
            "status": <lookup status code - see oasis_utils/oasis_utils.py>
        }

    The script can generate keys records in this format, and write them to file.

    For model loss calculations however ``ktools`` requires a keys CSV file with
    the following format
    ::

        LocID,PerilID,CoverageID,AreaPerilID,VulnerabilityID
        ..
        ..

    where the headers correspond to the relevant Oasis keys record fields.
    The script can also generate and write Oasis keys files.
    """
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(GenerateKeysCmd, self).add_args(parser)

        parser.add_argument(
            'output-file-path', type=PathCleaner('Output file', preexists=False),
            help='Keys records output file path',
        )
        parser.add_argument(
            '-k', '--keys-data-path', default=None,
            help='Keys data directory path',
        )
        parser.add_argument(
            '-v', '--model-version-file-path', default=None,
            help='Model version file path',
        )
        parser.add_argument(
            '-l', '--lookup-package-file-path', default=None,
            help='Keys data directory path',
        )
        parser.add_argument(
            '-t', '--output-format', choices=['oasis_keys', 'list_keys'],
            help='Keys records file output format',
        )
        parser.add_argument(
            '-e', '--model-exposures-file-path', default=None, help='Keys records file output format',
        )
        parser.add_argument(
            '-s', '--successes-only', action='store_true', help='Only record successful entries',
        )

    def action(self, args):
        """
        Generate and write Oasis keys (area peril ID, vulnerability ID) for a model.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)
        model_exposures_file_path = as_path(inputs.get('model_exposures_file_path', required=True, is_path=True), 'Model exposures')
        keys_data_path = as_path(inputs.get('keys_data_path', required=True, is_path=True), 'Keys data')
        version_file_path = as_path(inputs.get('model_version_file_path', required=True, is_path=True), 'Version file')
        lookup_package_path = as_path(inputs.get('lookup_package_path', required=True, is_path=True), 'Lookup package')

        self.logger.info('Getting model info and creating lookup service instance')
        model_info, model_klc = OasisKeysLookupFactory.create(
            model_keys_data_path=keys_data_path,
            model_version_file_path=version_file_path,
            lookup_package_path=lookup_package_path,
        )
        self.logger.info('\t{}, {}'.format(model_info, model_klc))

        self.logger.info('Saving keys records to file')
        f, n = OasisKeysLookupFactory.save_keys(
            model_klc,
            model_exposures_file_path=model_exposures_file_path,
            output_file_path=args.output_file_path,
            success_only=args.success_only,
        )
        self.logger.info('{} keys records saved to file {}'.format(n, f))


class GenerateLossesCmd(OasisBaseCommand):
    """
    Generate losses using the installed ktools framework.

    Given Oasis files, model analysis settings JSON file, model data, and
    some other parameters. can generate losses using the installed ktools framework.

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    Run ``oasislmf config --help`` for more information.

    The script creates a time-stamped folder in the model run directory and
    sets that as the new model run directory, copies the analysis settings
    JSON file into the run directory and creates the following folder
    structure
    ::

        ├── analysis_settings.json
        ├── fifo/
        ├── input/
        ├── output/
        ├── static/
        └── work/

    Depending on the OS type the model data is symlinked (Linux, Darwin) or
    copied (Cygwin, Windows) into the ``static`` subfolder. The input files
    are kept in the ``input`` subfolder and the losses are generated as CSV
    files in the ``output`` subfolder.

    By default executing the generated ktools losses script will automatically
    execute, this can be overridden by providing the ``--no-execute`` flag.
    """
    formatter_class = RawDescriptionHelpFormatter

    wait_proocessing_switches = {
        'full_uncertainty_aep': '-F',
        'wheatsheaf_aep': '-W',
        'sample_mean_aep': '-S',
        'full_uncertainty_oep': '-f',
        'wheatsheaf_oep': '-w',
        'sample_mean_oep': '-s',
        'wheatsheaf_mean_aep': '-M',
        'wheatsheaf_mean_oep': '-m',
    }

    def __init__(self, *args, **kwargs):
        self.pid_monitor_count = 0
        self.apid_monitor_count = 0
        self.lpid_monitor_count = 0
        self.kpid_monitor_count = 0
        self.command_file = ''

        super(GenerateLossesCmd, self).__init__(*args, **kwargs)

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(GenerateLossesCmd, self).add_args(parser)

        parser.add_argument('-o', '--oasis-files-path', default=None, help='Path to Oasis files')
        parser.add_argument(
            '-j', '--analysis-settings-json-file-path', default=None,
            help='Relative or absolute path of the model analysis settings JSON file'
        )
        parser.add_argument('-m', '--model-data-path', default=None, help='Model data source path')
        parser.add_argument('-r', '--model-run-dir-path', default=None, help='Model run directory path')
        parser.add_argument(
            '-s', '--ktools-script-name', default=None,
            help='Name of the ktools output script (should not contain any filetype extension)'
        )
        parser.add_argument('-n', '--ktools-num-processes', default=2, help='Number of ktools calculation processes to use')
        parser.add_argument('-x', '--no-execute', action='store_false', help='Whether to execute generated ktools script')

    def print_command(self, cmd):
        """
        Writes the supplied command to the end of the generated script

        :param cmd: The command to append
        """
        with io.open(self.command_file, "a", encoding='utf-8') as myfile:
            myfile.writelines(cmd + "\n")

    def leccalc_enabled(self, lec_options):
        """
        Checks if leccalc is enabled in the leccalc options

        :param lec_options: The leccalc options from the analysis settings
        :type lec_options: dict

        :return: True is leccalc is enables, False otherwise.
        """
        for option in lec_options["outputs"]:
            if lec_options["outputs"][option]:
                return True
        return False

    def do_post_wait_processing(self, runtype, analysis_settings):
        if '{}_summaries'.format(runtype) not in analysis_settings:
            return

        for summary in analysis_settings['{}_summaries'.format(runtype)]:
            if "id" in summary:
                summary_set = summary['id']
                if summary.get('aalcalc'):
                    self.apid_monitor_count += 1
                    self.print_command('aalsummary -K{0}_S{1}_aalcalc > output/{0}_S{1}_aalcalc.csv & apid{2}=$!'.format(
                        runtype,
                        summary_set,
                        self.apid_monitor_count
                    ))

                if summary.get('lec_output'):
                    leccalc = summary.get('leccalc', {})
                    if leccalc and self.leccalc_enabled(leccalc):
                        cmd = 'leccalc {} -K{}_S{}_summaryleccalc'.format(
                            '-r' if leccalc.get('return_period_file') else '',
                            runtype,
                            summary_set
                        )

                        lpid_monitor_count = self.lpid_monitor_count + 1
                        for option, value in leccalc['outputs'].items():
                            if value:
                                switch = self.wait_proocessing_switches.get(value, '')
                                cmd = '{} {} output/{}_S{}_leccalc_{}.csv'.format(cmd, switch, runtype, summary_set, option)

                        cmd = '{} &  lpid{}=$!'.format(cmd, lpid_monitor_count)
                        self.print_command(cmd)

    def do_fifos(self, action, runtype, analysis_settings, process_id):
        summaries = analysis_settings.get('{}_summaries'.format(runtype))
        if not summaries:
            return

        self.print_command('{} fifo/{}_P{}'.format(action, runtype, process_id))
        self.print_command('')
        for summary in summaries:
            if 'id' in summary:
                summary_set = summary['id']
                self.print_command('{} fifo/{}_S{}_summary_P{}'.format(action, runtype, summary_set, process_id))

                if summary.get('eltcalc'):
                    self.print_command('{} fifo/{}_S{}_summaryeltcalc_P{}'.format(action, runtype, summary_set, process_id))
                    self.print_command('{} fifo/{}_S{}_eltcalc_P{}'.format(action, runtype, summary_set, process_id))

                if summary.get('summarycalc'):
                    self.print_command('{} fifo/{}_S{}_summarysummarycalc_P{}'.format(action, runtype, summary_set, process_id))
                    self.print_command('{} fifo/{}_S{}_summarycalc_P{}'.format(action, runtype, summary_set, process_id))

                if summary.get('pltcalc'):
                    self.print_command('{} fifo/{}_S{}_summarypltcalc_P{}'.format(action, runtype, summary_set, process_id))
                    self.print_command('{} fifo/{}_S{}_pltcalc_P{}'.format(action, runtype, summary_set, process_id))

                if summary.get('aalcalc'):
                    self.print_command('{} fifo/{}_S{}_summaryaalcalc_P{}'.format(action, runtype, summary_set, process_id))

        self.print_command('')

    def create_workfolders(self, runtype, analysis_settings):
        summaries = analysis_settings.get('{}_summaries'.format(runtype))
        if not summaries:
            return

        for summary in summaries:
            if 'id' in summary:
                summary_set = summary['id']
                if summary.get('lec_output'):
                    if self.leccalc_enabled(summary['leccalc']):
                        self.print_command("mkdir work/{}_S{}_summaryleccalc".format(runtype, summary_set))

                if summary.get('aalcalc'):
                    self.print_command('mkdir work/{}_S{}_aalcalc'.format(runtype, summary_set))

    def remove_workfolders(self, runtype, analysis_settings):
        self.print_command('rm -rf work/kat')

        summaries = analysis_settings.get('{}_summaries'.format(runtype))
        if not summaries:
            return

        for summary in summaries:
            if 'id' in summary:
                summary_set = summary['id']
                if summary.get('lec_output'):
                    if self.leccalc_enabled(summary['leccalc']):
                        self.print_command('rm work/{}_S{}_summaryleccalc/*'.format(runtype, summary_set))
                        self.print_command('rmdir work/{}_S{}_summaryleccalc'.format(runtype, summary_set))

                if summary.get('aalcalc'):
                    self.print_command('rm work/{}_S{}_aalcalc/*'.format(runtype, summary_set))
                    self.print_command('rmdir work/{}_S{}_aalcalc'.format(runtype, summary_set))

    def do_make_fifos(self, runtype, analysis_settings, process_id):
        self.do_fifos('mkfifo', runtype, analysis_settings, process_id)

    def do_remove_fifos(self, runtype, analysis_settings, process_id):
        self.do_fifos('rm', runtype, analysis_settings, process_id)

    def do_kats(self, runtype, analysis_settings, max_process_id):
        summaries = analysis_settings.get('{}_summaries'.format(runtype))
        if not summaries:
            return False

        anykats = False
        for summary in summaries:
            if 'id' in summary:
                summary_set = summary['id']

                if summary.get('eltcalc'):
                    anykats = True

                    cmd = 'kat'
                    for process_id in range(1, max_process_id + 1):
                        cmd = '{} work/kat/{}_S{}_eltcalc_P{} '.format(cmd, runtype, summary_set, process_id)

                    self.kpid_monitor_count += 1
                    cmd = '{} > output/{}_S{}_eltcalc.csv & kpid{}=$!'.format(cmd, runtype, summary_set, self.kpid_monitor_count)
                    self.print_command(cmd)

                if summary.get('pltcalc'):
                    anykats = True

                    cmd = 'kat'
                    for process_id in range(1, max_process_id + 1):
                        cmd = '{} work/kat/{}_S{}_pltcalc_P{}'.format(cmd, runtype, summary_set, process_id)

                    self.kpid_monitor_count += 1
                    cmd = '{} > output/{}_S{}_pltcalc.csv & kpid{}=$!'.format(cmd, runtype, summary_set, self.kpid_monitor_count)
                    self.print_command(cmd)

                if summary.get("summarycalc"):
                    anykats = True

                    cmd = 'kat'
                    for process_id in range(1, max_process_id + 1):
                        cmd = "work/kat/{}_S{}_summarycalc_P{} ".format(cmd, runtype, summary_set, process_id)

                    self.kpid_monitor_count += 1
                    cmd = '{} > output/{}_S{}_summarycalc.csv & kpid{}=$!'.format(cmd, runtype, summary_set, self.kpid_monitor_count)
                    self.print_command(cmd)

        return anykats

    def do_summarycalcs(self, runtype, analysis_settings, process_id):
        summaries = analysis_settings.get('{}_summaries'.format(runtype))
        if not summaries:
            return

        summarycalc_switch = '-g'
        if runtype == 'il':
            summarycalc_switch = '-f'

        cmd = 'summarycalc {}'.format(summarycalc_switch)
        for summary in summaries:
            if 'id' in summary:
                summary_set = summary['id']
                cmd = '{} -{0} fifo/{1}_S{0}_summary_P{2}'.format(cmd, summary_set, runtype, process_id)

        cmd = '{} < fifo/{}_P{} &'.format(cmd, runtype, process_id)
        self.print_command(cmd)

    def do_tees(self, runtype, analysis_settings, process_id):
        summaries = analysis_settings.get('{}_summaries'.format(runtype))
        if not summaries:
            return

        for summary in summaries:
            if 'id' in summary:
                self.pid_monitor_count += 1
                summary_set = summary['id']
                cmd = "tee < fifo/{}_S{}_summary_P{} ".format(runtype, summary_set, process_id)

                if summary.get('eltcalc'):
                    cmd = '{} fifo/{}_S{}_summaryeltcalc_P{}'.format(cmd, runtype, summary_set, process_id)

                if summary.get('pltcalc'):
                    cmd = '{} fifo/{}_S{}_summarypltcalc_P{}'.format(cmd, runtype, summary_set, process_id)

                if summary.get('summarycalc'):
                    cmd = '{} fifo/{}_S{}_summarysummarycalc_P{}'.format(cmd, runtype, summary_set, process_id)

                if summary.get('aalcalc'):
                    cmd = '{} fifo/{}_S{}_summaryaalcalc_P{}'.format(cmd, runtype, summary_set, process_id)

                if summary.get('lec_output') and self.leccalc_enabled(summary['leccalc']):
                    cmd = '{} work/{}_S{}_summaryleccalc/P{}.bin'.format(cmd, runtype, summary_set, process_id)

                cmd = '{} > /dev/null & pid{}=$!'.format(cmd, self.pid_monitor_count)
                self.print_command(cmd)

    def do_any(self, runtype, analysis_settings, process_id):
        summaries = analysis_settings.get('{}_summaries'.format(runtype))
        if not summaries:
            return

        for summary in summaries:
            if 'id' in summary:
                summary_set = summary['id']
                if summary.get('eltcalc'):
                    cmd = 'eltcalc -s'
                    if process_id == 1:
                        cmd = 'eltcalc'

                    self.pid_monitor_count += 1
                    self.print_command(
                        "{3} < fifo/{0}_S{1}_summaryeltcalc_P{2} > work/kat/{0}_S{1}_eltcalc_P{2} & pid{4}=$!".format(
                            runtype, summary_set, process_id, cmd, self.pid_monitor_count
                        )
                    )

                if summary.get("summarycalc"):
                    cmd = 'summarycalctocsv -s'
                    if process_id == 1:
                        cmd = 'summarycalctocsv'

                    self.pid_monitor_count += 1
                    self.print_command(
                        '{3} < fifo/{0}_S{1}_summarysummarycalc_P{2} > work/kat/{0}_S{1}_summarycalc_P{2} & pid{4}=$!'.format(
                            runtype, summary_set, process_id, cmd, self.pid_monitor_count
                        )
                    )

                if summary.get('pltcalc'):
                    cmd = 'pltcalc -s'
                    if process_id == 1:
                        cmd = 'pltcalc'

                    self.pid_monitor_count += 1
                    self.print_command(
                        '{3} < fifo/{0}_S{1}_summarypltcalc_P{2} > work/kat/{0}_S{1}_pltcalc_P{2} & pid{4}=$!'.format(
                            runtype, summary_set, process_id, cmd, self.pid_monitor_count
                        )
                    )

                if summary.get('aalcalc'):
                    self.pid_monitor_count += 1
                    self.print_command(
                        'aalcalc < fifo/{0}_S{1}_summaryaalcalc_P{2} > work/{0}_S{1}_aalcalc/P{2}.bin & pid{3}=$!'.format(
                            runtype, summary_set, process_id, self.pid_monitor_count
                        )
                    )

            self.print_command('')

    def do_il(self, analysis_settings, max_process_id):
        for process_id in range(1, max_process_id + 1):
            self.do_any('il', analysis_settings, process_id)

        for process_id in range(1, max_process_id + 1):
            self.do_tees('il', analysis_settings, process_id)

        for process_id in range(1, max_process_id + 1):
            self.do_summarycalcs('il', analysis_settings, process_id)

    def do_gul(self, analysis_settings, max_process_id):
        for process_id in range(1, max_process_id + 1):
            self.do_any('gul', analysis_settings, process_id)

        for process_id in range(1, max_process_id + 1):
            self.do_tees('gul', analysis_settings, process_id)

        for process_id in range(1, max_process_id + 1):
            self.do_summarycalcs('gul', analysis_settings, process_id)

    def do_il_make_fifo(self, analysis_settings, max_process_id):
        for process_id in range(1, max_process_id + 1):
            self.do_make_fifos('il', analysis_settings, process_id)

    def do_gul_make_fifo(self, analysis_settings, max_process_id):
        for process_id in range(1, max_process_id + 1):
            self.do_make_fifos('gul', analysis_settings, process_id)

    def do_il_remove_fifo(self, analysis_settings, max_process_id):
        for process_id in range(1, max_process_id + 1):
            self.do_remove_fifos('il', analysis_settings, process_id)

    def do_gul_remove_fifo(self, analysis_settings, max_process_id):
        for process_id in range(1, max_process_id + 1):
            self.do_remove_fifos('gul', analysis_settings, process_id)

    def do_waits(self, wait_variable, wait_count):
        """
        Add waits to the script

        :param wait_variable: The type of wait
        :type wait_variable: str

        :param wait_count: The number of processes to wait for
        :type wait_count: int
        """
        if wait_count > 0:
            cmd = 'wait'
            for pid in range(1, wait_count + 1):
                cmd = '{} ${}{}'.format(cmd, wait_variable, pid)

            self.print_command(cmd)
            self.print_command('')

    def do_pwaits(self):
        """
        Add pwaits to the script
        """
        self.do_waits('pid', self.pid_monitor_count)

    def do_awaits(self):
        """
        Add awaits to the script
        """
        self.do_waits('apid', self.apid_monitor_count)

    def do_lwaits(self):
        """
        Add lwaits to the script
        """
        self.do_waits('lpid', self.lpid_monitor_count)

    def do_kwaits(self):
        """
        Add kwaits to the script
        """
        self.do_waits('kpid', self.kpid_monitor_count)

    def get_getmodel_cmd(self, number_of_samples, gul_threshold, use_random_number_file, coverage_output, item_output):
        """
        Gets the getmodel ktools command

        :param number_of_samples: The number of samples to run
        :type number_of_samples: int

        :param gul_threshold: The GUL threshold to use
        :type gul_threshold: float

        :param use_random_number_file: flag to use the random number file
        :type use_random_number_file: bool

        :param coverage_output: The coverage output
        :type coverage_output: str

        :param item_output: The item output
        :type item_output: str

        :return: The generated getmodel command
        """

        cmd = 'getmodel | gulcalc -S{} -L{}'.format(number_of_samples, gul_threshold)

        if use_random_number_file:
            cmd = '{} -r'.format(cmd)
        if coverage_output != '':
            cmd = '{} -c {}'.format(cmd, coverage_output)
        if item_output != '':
            cmd = '{} -i {}'.format(cmd, item_output)

        return cmd

    def genbash(self, max_process_id=None, analysis_settings=None, get_getmodel_cmd=None):
        """
        Generates a bash script containing ktools calculation instructions for an
        Oasis model.

        :param max_process_id: The number of processes to create
        :type max_process_id: int

        :param analysis_settings: The analysis settings
        :type analysis_settings: dict

        :param get_getmodel_cmd: Method for getting the getmodel command, by default
            ``GenerateLossesCmd.get_getmodel_cmd`` is used.
        :type get_getmodel_cmd: callable
        """
        get_getmodel_cmd = get_getmodel_cmd or self.get_getmodel_cmd

        use_random_number_file = False
        gul_output = False
        il_output = False

        gul_threshold = analysis_settings.get('gul_threshold', 0)
        number_of_samples = analysis_settings.get('number_of_samples', 0)

        if 'model_settings' in analysis_settings and analysis_settings['model_settings'].get('use_random_number_file'):
            use_random_number_file = True

        if 'gul_output' in analysis_settings:
            gul_output = analysis_settings['gul_output']

        if 'il_output' in analysis_settings:
            il_output = analysis_settings['il_output']

        self.print_command('#!/bin/bash')

        self.print_command('')

        self.print_command('rm -R -f output/*')
        self.print_command('rm -R -f fifo/*')
        self.print_command('rm -R -f work/*')
        self.print_command('')

        self.print_command('mkdir work/kat')

        if gul_output:
            self.do_gul_make_fifo(analysis_settings, max_process_id)
            self.create_workfolders('gul', analysis_settings)

        self.print_command('')

        if il_output:
            self.do_il_make_fifo(analysis_settings, max_process_id)
            self.create_workfolders('il', analysis_settings)

        self.print_command('')
        self.print_command('# --- Do insured loss computes ---')
        self.print_command('')
        if il_output:
            self.do_il(analysis_settings, max_process_id)

        self.print_command('')
        self.print_command('# --- Do ground up loss  computes ---')
        self.print_command('')
        if gul_output:
            self.do_gul(analysis_settings, max_process_id)

        self.print_command('')

        for process_id in range(1, max_process_id + 1):
            if gul_output and il_output:
                getmodel_cmd = get_getmodel_cmd(
                    number_of_samples,
                    gul_threshold,
                    use_random_number_file,
                    'fifo/gul_P{}'.format(process_id),
                    '-'
                )

                self.print_command('eve {0} {1} | {2} | fmcalc > fifo/il_P{0}  &'.format(process_id, max_process_id, getmodel_cmd))
            else:
                #  Now the mainprocessing
                if gul_output:
                    if 'gul_summaries' in analysis_settings:
                        getmodel_cmd = get_getmodel_cmd(
                            number_of_samples,
                            gul_threshold,
                            use_random_number_file,
                            '-',
                            '')

                        self.print_command('eve {0} {1} | {2} > fifo/gul_P{0}  &'.format(process_id, max_process_id, getmodel_cmd))

                if il_output:
                    if 'il_summaries' in analysis_settings:
                        getmodel_cmd = get_getmodel_cmd(
                            number_of_samples,
                            gul_threshold,
                            use_random_number_file,
                            '',
                            '-'
                        )

                        self.print_command("eve {0} {1} | {2} | fmcalc > fifo/il_P{0}  &".format(process_id, max_process_id, getmodel_cmd))

        self.print_command('')

        self.do_pwaits()

        self.print_command('')
        self.print_command('# --- Do insured loss kats ---')
        self.print_command('')
        if il_output:
            self.do_kats('il', analysis_settings, max_process_id)

        self.print_command('')
        self.print_command('# --- Do ground up loss kats ---')
        self.print_command('')
        if gul_output:
            self.do_kats('gul', analysis_settings, max_process_id)

        self.do_kwaits()

        self.print_command('')
        self.do_post_wait_processing('il', analysis_settings)
        self.do_post_wait_processing('gul', analysis_settings)

        self.do_awaits()  # waits for aalcalc
        self.do_lwaits()  # waits for leccalc

        if gul_output:
            self.do_gul_remove_fifo(analysis_settings, max_process_id)
            self.remove_workfolders('gul', analysis_settings)

        self.print_command('')

        if il_output:
            self.do_il_remove_fifo(analysis_settings, max_process_id)
            self.remove_workfolders('il', analysis_settings)

    def action(self, args):
        """
        Generate losses using the installed ktools framework.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)
        oasis_files_path = as_path(inputs.get('oasis_files_path', required=True, is_path=True), 'Oasis files')
        analysis_settings_json_file_path = as_path(
            inputs.get('analysis_settings_json_file_path', required=True, is_path=True),
            'Analysis settings file'
        )
        model_data_path = as_path(inputs.get('model_data_path', required=True, is_path=True), 'Model data')
        model_run_dir_path = as_path(inputs.get('model_run_dir_path', required=True, is_path=True), 'Model run directory')
        ktools_script_name = inputs.get('ktools_script_name', default='run_ktools')
        no_execute = inputs.get('no_execute', default=False)

        if not os.path.exists(model_run_dir_path):
            os.mkdir(model_run_dir_path)

        utcnow = get_utctimestamp(fmt='%Y%m%d%H%M%S')
        model_run_dir_path = os.path.join(args['model_run_dir_path'], 'ProgOasis-{}'.format(utcnow))
        self.logger.info('Creating time-stamped model run folder {}'.format(model_run_dir_path))
        os.mkdir(model_run_dir_path)

        self.logger.info(
            'Preparing model run directory {} - copying Oasis files, analysis settings JSON file and linking model data'.format(model_run_dir_path)
        )
        prepare_model_run_directory(
            model_run_dir_path,
            oasis_files_path,
            analysis_settings_json_file_path,
            model_data_path
        )

        self.logger.info('Converting Oasis files to ktools binary files')
        oasis_files_path = os.path.join(model_run_dir_path, 'input', 'csv')
        binary_files_path = os.path.join(model_run_dir_path, 'input')
        create_binary_files(oasis_files_path, binary_files_path)

        analysis_settings_json_file_path = os.path.join(model_run_dir_path, 'analysis_settings.json')
        try:
            self.logger.info('Reading analysis settings JSON file')
            with io.open(analysis_settings_json_file_path, 'r', encoding='utf-8') as f:
                analysis_settings = json.load(f)
                if 'analysis_settings' in analysis_settings:
                    analysis_settings = analysis_settings['analysis_settings']
        except (IOError, TypeError, ValueError):
            raise OasisException('Invalid analysis settings JSON file or file path: {}.'.format(analysis_settings_json_file_path))

        self.logger.info('Loaded analysis settings JSON: {}'.format(analysis_settings))

        self.logger.info('Preparing model run inputs')
        prepare_model_run_inputs(analysis_settings, model_run_dir_path)

        self.command_file = os.path.join(model_run_dir_path, '{}.sh'.format(ktools_script_name))
        try:
            self.logger.info('Generating ktools losses script')
            self.genbash(
                max_process_id=args.ktools_num_processes,
                analysis_settings=analysis_settings,
            )
        except Exception as e:
            raise OasisException(e)

        try:
            self.logger.info('Making ktools losses script executable')
            subprocess.check_call('chmod +x {}'.format(self.command_file), stderr=subprocess.STDOUT, shell=True)
        except (OSError, IOError) as e:
            raise OasisException(e)

        self.logger.info('Generated ktools losses script {}'.format(self.command_file))

        if not no_execute:
            try:
                os.chdir(model_run_dir_path)
                cmd_str = 'bash {}.sh'.format(ktools_script_name)
                self.logger.info('Running ktools losses script {}'.format(self.command_file))
                subprocess.check_call(cmd_str, stderr=subprocess.STDOUT, shell=True)
            except (OSError, IOError, subprocess.CalledProcessError) as e:
                raise OasisException(e)

        self.logger.info('Loss outputs generated in {}'.format(os.path.join(model_run_dir_path, 'output')))


class GenerateOasisFilesCmd(OasisBaseCommand):
    """
    Generate Oasis files (items, coverages, GUL summary) for a model

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    """
    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(GenerateOasisFilesCmd, self).add_args(parser)

        parser.add_argument('oasis_files_path', default=None, help='Path to Oasis files', nargs='?')
        parser.add_argument('-k', '--keys-data-path', default=None, help='Path to Oasis files')
        parser.add_argument('-v', '--model-version-file-path', default=None, help='Model version file path')
        parser.add_argument('-l', '--lookup-package-file-path', default=None, help='Keys data directory path')
        parser.add_argument(
            '-p', '--canonical-exposures-profile-json-path', default=None,
            help='Path of the supplier canonical exposures profile JSON file'
        )
        parser.add_argument('-e', '--source-exposures-file-path', default=None, help='Source exposures file path')
        parser.add_argument(
            '-a', '--source-exposures-validation-file-path', default=None,
            help='Source exposures validation file (XSD) path'
        )
        parser.add_argument(
            '-b', '--source-to-canonical-exposures-transformation-file-path', default=None,
            help='Source -> canonical exposures transformation file (XSLT) path'
        )
        parser.add_argument(
            '-c', '--canonical-exposures-validation-file-path', default=None,
            help='Canonical exposures validation file (XSD) path'
        )
        parser.add_argument(
            '-d', '--canonical-to-model-exposures-transformation-file-path', default=None,
            help='Canonical exposures validation file (XSD) path'
        )

    def action(self, args):
        """
        Generate Oasis files (items, coverages, GUL summary) for a model

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)
        oasis_files_path = as_path(inputs.get('oasis_files_path', required=True, is_path=True), 'Oasis file', preexists=False)
        keys_data_path = as_path(inputs.get('keys_data_path', required=True, is_path=True), 'Keys data')
        model_version_file_path = as_path(inputs.get('model_version_file_path', required=True, is_path=True), 'Model version file')
        lookup_package_file_path = as_path(inputs.get('lookup_package_file_path', required=True, is_path=True), 'Lookup package file')
        canonical_exposures_profile_json_path = as_path(
            inputs.get('canonical_exposures_profile_json_path', required=True, is_path=True),
            'Canonical exposures profile json'
        )
        source_exposures_file_path = as_path(inputs.get('source_exposures_file_path', required=True, is_path=True), 'Source exposures')
        source_exposures_validation_file_path = as_path(
            inputs.get('source_exposures_validation_file_path', required=True, is_path=True),
            'Source exposures validation file'
        )
        source_to_canonical_exposures_transformation_file_path = as_path(
            inputs.get('source_to_canonical_exposures_transformation_file_path', required=True, is_path=True),
            'Source to canonical exposures transformation'
        )
        canonical_exposures_validation_file_path = as_path(
            inputs.get('canonical_exposures_validation_file_path', required=True, is_path=True),
            'Canonical exposures validation file'
        )
        canonical_to_model_exposures_transformation_file_path = as_path(
            inputs.get('canonical_to_model_exposures_transformation_file_path', required=True, is_path=True),
            'Canonical to model exposures transformation file'
        )

        self.logger.info('Getting model info and creating lookup service instance')
        model_info, model_klc = OasisKeysLookupFactory.create(
            model_keys_data_path=keys_data_path,
            model_version_file_path=model_version_file_path,
            lookup_package_path=lookup_package_file_path,
        )
        self.logger.info('\t{}, {}'.format(model_info, model_klc))

        self.logger.info('Creating Oasis model object')
        model = OasisModelFactory.create(
            model_supplier_id=model_info['supplier_id'],
            model_id=model_info['model_id'],
            model_version_id=model_info['model_version_id'],
            resources={
                'oasis_files_path': oasis_files_path,
                'canonical_exposures_profile_json_path': canonical_exposures_profile_json_path,
                'source_exposures_validation_file_path': source_exposures_validation_file_path,
                'source_to_canonical_exposures_transformation_file_path': source_to_canonical_exposures_transformation_file_path,
                'canonical_exposures_validation_file_path': canonical_exposures_validation_file_path,
                'canonical_to_model_exposures_transformation_file_path': canonical_to_model_exposures_transformation_file_path,
            }
        )
        self.logger.info('\t{}'.format(model))

        self.logger.info('Setting up Oasis files directory for model {}'.format(model.key))
        if not os.path.exists(oasis_files_path):
            os.mkdir(oasis_files_path)

        self.logger.info('Generating Oasis files for model')
        oasis_files = OasisExposuresManager.start_files_pipeline(
            oasis_model=model,
            oasis_files_path=oasis_files_path,
            source_exposures_path=source_exposures_file_path,
            logger=self.logger,
        )

        self.logger.info('Generated Oasis files for model: {}'.format(oasis_files))


class RunCmd(OasisBaseCommand):
    """
    Run models end to end.

    The command line arguments can be supplied in the configuration file
    (``oasislmf.json`` by default or specified with the ``--config`` flag).
    """
    def add_args(self, parser):
        """
        Run models end to end.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(RunCmd, self).add_args(parser)

        parser.add_argument('-k', '--keys-data-path', default=None, help='Path to Oasis files')
        parser.add_argument('-v', '--model-version-file-path', default=None, help='Model version file path')
        parser.add_argument('-l', '--lookup-package-file-path', default=None, help='Keys data directory path')
        parser.add_argument(
            '-p', '--canonical-exposures-profile-json-path', default=None,
            help='Path of the supplier canonical exposures profile JSON file'
        )
        parser.add_argument('-e', '--source-exposures-file-path', default=None, help='Source exposures file path')
        parser.add_argument(
            '-a', '--source-exposures-validation-file-path', default=None,
            help='Source exposures validation file (XSD) path'
        )
        parser.add_argument(
            '-b', '--source-to-canonical-exposures-transformation-file-path', default=None,
            help='Source -> canonical exposures transformation file (XSLT) path'
        )
        parser.add_argument(
            '-c', '--canonical-exposures-validation-file-path', default=None,
            help='Canonical exposures validation file (XSD) path'
        )
        parser.add_argument(
            '-d', '--canonical-to-model-exposures-transformation-file-path', default=None,
            help='Canonical exposures validation file (XSD) path'
        )
        parser.add_argument(
            '-j', '--analysis-settings-json-file-path', default=None,
            help='Model analysis settings JSON file path'
        )
        parser.add_argument('-m', '--model-data-path', default=None, help='Model data source path')
        parser.add_argument('-r', '--model-run-dir-path', default=None, help='Model run directory path')
        parser.add_argument(
            '-s', '--ktools-script-name', default=None,
            help='Name of the ktools output script (should not contain any filetype extension)'
        )
        parser.add_argument('-n', '--ktools-num-processes', default=2, help='Number of ktools calculation processes to use')

    def action(self, args):
        """
        Generate Oasis files (items, coverages, GUL summary) for a model

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)
        model_run_dir_path = as_path(inputs.get('model_run_dir_path', required=True), 'Model run path', preexists=False)

        self.logger.info('Creating temporary folder {} for Oasis files'.format(args.oasis_files_path))
        args.oasis_files_path = os.path.join(model_run_dir_path, 'tmp')
        if not os.path.exists(args.oasis_files_path):
            os.mkdir(args.oasis_files_path)

        gen_oasis_files_cmd = GenerateOasisFilesCmd()
        gen_oasis_files_cmd._logger = self.logger
        gen_oasis_files_cmd.action(args)

        gen_losses_cmd = GenerateLossesCmd()
        gen_losses_cmd._logger = self.logger
        gen_losses_cmd.action(args)

        shutil.rmtree(model_run_dir_path)


class ModelsCmd(OasisBaseCommand):
    sub_commands = {
        'generate-keys': GenerateKeysCmd,
        'generate-losses': GenerateLossesCmd,
        'generate-oasis-files': GenerateOasisFilesCmd,
        'run': RunCmd,
    }
