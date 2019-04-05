# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import io
import os
import random
import re
import string

from collections import Counter

from ..model_preparation.oed import (
    ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID, # Alloc Rule 2 (Default)
    ALLOCATE_TO_ITEMS_BY_GUL_ALLOC_ID,            # Alloc Rule 1
    NO_ALLOCATION_ALLOC_ID,                       # Alloc Rule 0
)

RUNTYPE_GROUNDUP_LOSS = 'gul'
RUNTYPE_INSURED_LOSS = 'il'
RUNTYPE_REINSURANCE_LOSS = 'ri'

WAIT_PROCESSING_SWITCHES = {
    'full_uncertainty_aep': '-F',
    'wheatsheaf_aep': '-W',
    'sample_mean_aep': '-S',
    'full_uncertainty_oep': '-f',
    'wheatsheaf_oep': '-w',
    'sample_mean_oep': '-s',
    'wheatsheaf_mean_aep': '-M',
    'wheatsheaf_mean_oep': '-m',
}

def print_command(command_file, cmd):
    """
    Writes the supplied command to the end of the generated script

    :param cmd: The command to append
    """
    with io.open(command_file, "a", encoding='utf-8') as myfile:
        myfile.writelines(cmd + "\n")


def leccalc_enabled(lec_options):
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


def do_post_wait_processing(runtype, analysis_settings, filename, process_counter):
    if '{}_summaries'.format(runtype) not in analysis_settings:
        return

    for summary in analysis_settings['{}_summaries'.format(runtype)]:
        if "id" in summary:
            summary_set = summary['id']
            if summary.get('aalcalc'):
                cmd = 'aalcalc -K{}_S{}_summaryaalcalc'.format(
                    runtype,
                    summary_set
                )

                process_counter['lpid_monitor_count'] += 1
                cmd = '{} > output/{}_S{}_aalcalc.csv'.format(cmd, runtype, summary_set)
                cmd = '{} & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                print_command(filename, cmd)

            if summary.get('lec_output'):
                leccalc = summary.get('leccalc', {})
                if leccalc and leccalc_enabled(leccalc):
                    cmd = 'leccalc {} -K{}_S{}_summaryleccalc'.format(
                        '-r' if leccalc.get('return_period_file') else '',
                        runtype,
                        summary_set
                    )

                    process_counter['lpid_monitor_count'] += 1
                    for option, active in sorted(leccalc['outputs'].items()):
                        if active:
                            switch = WAIT_PROCESSING_SWITCHES.get(option, '')
                            cmd = '{} {} output/{}_S{}_leccalc_{}.csv'.format(cmd, switch, runtype, summary_set,
                                                                              option)

                    cmd = '{} & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                    print_command(filename, cmd)


def do_fifos(action, runtype, analysis_settings, process_id, filename, fifo_dir=''):
    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    print_command(filename, '{} {}fifo/{}_P{}'.format(action, fifo_dir, runtype, process_id))
    print_command(filename, '')
    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']
            print_command(filename, '{} {}fifo/{}_S{}_summary_P{}'.format(action, fifo_dir, runtype, summary_set, process_id))

            if summary.get('eltcalc'):
                print_command(
                    filename,
                    '{} {}fifo/{}_S{}_summaryeltcalc_P{}'.format(action, fifo_dir, runtype, summary_set, process_id)
                )
                print_command(
                    filename,
                    '{} {}fifo/{}_S{}_eltcalc_P{}'.format(action, fifo_dir, runtype, summary_set, process_id)
                )

            if summary.get('summarycalc'):
                print_command(
                    filename,
                    '{} {}fifo/{}_S{}_summarysummarycalc_P{}'.format(action, fifo_dir, runtype, summary_set, process_id)
                )
                print_command(
                    filename,
                    '{} {}fifo/{}_S{}_summarycalc_P{}'.format(action, fifo_dir, runtype, summary_set, process_id)
                )

            if summary.get('pltcalc'):
                print_command(
                    filename,
                    '{} {}fifo/{}_S{}_summarypltcalc_P{}'.format(action, fifo_dir, runtype, summary_set, process_id)
                )
                print_command(
                    filename,
                    '{} {}fifo/{}_S{}_pltcalc_P{}'.format(action, fifo_dir, runtype, summary_set, process_id)
                )

    print_command(filename, '')


def create_workfolders(runtype, analysis_settings, filename):
    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']
            if summary.get('lec_output'):
                if leccalc_enabled(summary['leccalc']):
                    print_command(filename, "mkdir work/{}_S{}_summaryleccalc".format(runtype, summary_set))

            if summary.get('aalcalc'):
                print_command(filename, 'mkdir work/{}_S{}_summaryaalcalc'.format(runtype, summary_set))


def remove_workfolders(runtype, analysis_settings, filename):
    print_command(filename, 'rm -rf work/kat')

    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']
            if summary.get('lec_output'):
                if leccalc_enabled(summary['leccalc']):
                    print_command(filename, 'rm work/{}_S{}_summaryleccalc/*'.format(runtype, summary_set))
                    print_command(filename, 'rmdir work/{}_S{}_summaryleccalc'.format(runtype, summary_set))

            if summary.get('aalcalc'):
                print_command(filename, 'rm -rf work/{}_S{}_summaryaalcalc/*'.format(runtype, summary_set))
                print_command(filename, 'rmdir work/{}_S{}_summaryaalcalc'.format(runtype, summary_set))


def do_make_fifos(runtype, analysis_settings, process_id, filename, fifo_dir=''):
    do_fifos('mkfifo', runtype, analysis_settings, process_id, filename, fifo_dir)

def do_ktools_mem_limit(max_process_id ,filename):
    """
        Set each Ktools pipeline to trap and terminate on hitting its
        memory allocation limit

        Limit: Maximum avalible memory / max_process_id
    """
    cmd_mem_limit = 'ulimit -v $(ktgetmem {})'.format(max_process_id)
    print_command(filename, cmd_mem_limit)

def do_remove_fifos(runtype, analysis_settings, process_id, filename, fifo_dir=''):
    do_fifos('rm', runtype, analysis_settings, process_id, filename, fifo_dir)


def do_kats(runtype, analysis_settings, max_process_id, filename, process_counter):
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
                    cmd = '{} work/kat/{}_S{}_eltcalc_P{}'.format(cmd, runtype, summary_set, process_id)

                process_counter['kpid_monitor_count'] += 1
                cmd = '{} > output/{}_S{}_eltcalc.csv & kpid{}=$!'.format(cmd, runtype, summary_set,
                                                                          process_counter['kpid_monitor_count'])
                print_command(filename, cmd)

            if summary.get('pltcalc'):
                anykats = True

                cmd = 'kat'
                for process_id in range(1, max_process_id + 1):
                    cmd = '{} work/kat/{}_S{}_pltcalc_P{}'.format(cmd, runtype, summary_set, process_id)

                process_counter['kpid_monitor_count'] += 1
                cmd = '{} > output/{}_S{}_pltcalc.csv & kpid{}=$!'.format(cmd, runtype, summary_set,
                                                                          process_counter['kpid_monitor_count'])
                print_command(filename, cmd)

            if summary.get("summarycalc"):
                anykats = True

                cmd = 'kat'
                for process_id in range(1, max_process_id + 1):
                    cmd = '{} work/kat/{}_S{}_summarycalc_P{}'.format(cmd, runtype, summary_set, process_id)

                process_counter['kpid_monitor_count'] += 1
                cmd = '{} > output/{}_S{}_summarycalc.csv & kpid{}=$!'.format(cmd, runtype, summary_set,
                                                                              process_counter['kpid_monitor_count'])
                print_command(filename, cmd)

    return anykats


def do_summarycalcs(
    runtype, analysis_settings, process_id, filename, fifo_dir='', num_reinsurance_iterations=0):

    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    summarycalc_switch = '-f'
    if runtype == RUNTYPE_GROUNDUP_LOSS:
        summarycalc_switch = '-g'

    summarycalc_directory_switch = ""
    if runtype == RUNTYPE_REINSURANCE_LOSS:
        i = num_reinsurance_iterations
        summarycalc_directory_switch = "-p RI_{0}".format(i)

    cmd = 'summarycalc {} {}'.format(summarycalc_switch, summarycalc_directory_switch)
    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']
            cmd = '{0} -{1} {4}fifo/{2}_S{1}_summary_P{3}'.format(cmd, summary_set, runtype, process_id, fifo_dir)

    cmd = '{0} < {1}fifo/{2}_P{3} &'.format(cmd, fifo_dir, runtype, process_id)
    print_command(filename, cmd)


def do_tees(runtype, analysis_settings, process_id, filename, process_counter, fifo_dir=''):
    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    for summary in summaries:
        if 'id' in summary:
            process_counter['pid_monitor_count'] += 1
            summary_set = summary['id']
            cmd = 'tee < {}fifo/{}_S{}_summary_P{}'.format(fifo_dir, runtype, summary_set, process_id)

            if summary.get('eltcalc'):
                cmd = '{} {}fifo/{}_S{}_summaryeltcalc_P{}'.format(cmd, fifo_dir, runtype, summary_set, process_id)

            if summary.get('pltcalc'):
                cmd = '{} {}fifo/{}_S{}_summarypltcalc_P{}'.format(cmd, fifo_dir, runtype, summary_set, process_id)

            if summary.get('summarycalc'):
                cmd = '{} {}fifo/{}_S{}_summarysummarycalc_P{}'.format(cmd, fifo_dir, runtype, summary_set, process_id)

            if summary.get('aalcalc'):
                cmd = '{} work/{}_S{}_summaryaalcalc/P{}.bin'.format(cmd, runtype, summary_set, process_id)

            if summary.get('lec_output') and leccalc_enabled(summary['leccalc']):
                cmd = '{} work/{}_S{}_summaryleccalc/P{}.bin'.format(cmd, runtype, summary_set, process_id)

            cmd = '{} > /dev/null & pid{}=$!'.format(cmd, process_counter['pid_monitor_count'])
            print_command(filename, cmd)


def do_any(runtype, analysis_settings, process_id, filename, process_counter, fifo_dir=''):
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

                process_counter['pid_monitor_count'] += 1
                print_command(
                    filename,
                    "{3} < {5}fifo/{0}_S{1}_summaryeltcalc_P{2} > work/kat/{0}_S{1}_eltcalc_P{2} & pid{4}=$!".format(
                        runtype, summary_set, process_id, cmd, process_counter['pid_monitor_count'], fifo_dir
                    )
                )

            if summary.get("summarycalc"):
                cmd = 'summarycalctocsv -s'
                if process_id == 1:
                    cmd = 'summarycalctocsv'

                process_counter['pid_monitor_count'] += 1
                print_command(
                    filename,
                    '{3} < {5}fifo/{0}_S{1}_summarysummarycalc_P{2} > work/kat/{0}_S{1}_summarycalc_P{2} & pid{4}=$!'.format(
                        runtype, summary_set, process_id, cmd, process_counter['pid_monitor_count'], fifo_dir
                    )
                )

            if summary.get('pltcalc'):
                cmd = 'pltcalc -s'
                if process_id == 1:
                    cmd = 'pltcalc'

                process_counter['pid_monitor_count'] += 1
                print_command(
                    filename,
                    '{3} < {5}fifo/{0}_S{1}_summarypltcalc_P{2} > work/kat/{0}_S{1}_pltcalc_P{2} & pid{4}=$!'.format(
                        runtype, summary_set, process_id, cmd, process_counter['pid_monitor_count'], fifo_dir
                    )
                )


        print_command(filename, '')


def ri(analysis_settings, max_process_id, filename, process_counter, num_reinsurance_iterations, fifo_dir=''):
    for process_id in range(1, max_process_id + 1):
        do_any(RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir)

    for process_id in range(1, max_process_id + 1):
        do_tees(RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir)

    for process_id in range(1, max_process_id + 1):
        do_summarycalcs(
            RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_id, filename, fifo_dir, num_reinsurance_iterations)


def il(analysis_settings, max_process_id, filename, process_counter, fifo_dir=''):
    for process_id in range(1, max_process_id + 1):
        do_any(RUNTYPE_INSURED_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir)

    for process_id in range(1, max_process_id + 1):
        do_tees(RUNTYPE_INSURED_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir)

    for process_id in range(1, max_process_id + 1):
        do_summarycalcs(RUNTYPE_INSURED_LOSS, analysis_settings, process_id, filename, fifo_dir)


def do_gul(analysis_settings, max_process_id, filename, process_counter, fifo_dir=''):
    for process_id in range(1, max_process_id + 1):
        do_any(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir)

    for process_id in range(1, max_process_id + 1):
        do_tees(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir)

    for process_id in range(1, max_process_id + 1):
        do_summarycalcs(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename, fifo_dir)


def il_make_fifo(analysis_settings, max_process_id, filename, fifo_dir=''):
    for process_id in range(1, max_process_id + 1):
        do_make_fifos(RUNTYPE_INSURED_LOSS, analysis_settings, process_id, filename, fifo_dir)


def do_gul_make_fifo(analysis_settings, max_process_id, filename, fifo_dir=''):
    for process_id in range(1, max_process_id + 1):
        do_make_fifos(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename, fifo_dir)


def ri_make_fifo(analysis_settings, max_process_id, filename, fifo_dir=''):
    for process_id in range(1, max_process_id + 1):
        do_make_fifos(RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_id, filename, fifo_dir)


def il_remove_fifo(analysis_settings, max_process_id, filename, fifo_dir=''):
    for process_id in range(1, max_process_id + 1):
        do_remove_fifos(RUNTYPE_INSURED_LOSS, analysis_settings, process_id, filename, fifo_dir)


def do_gul_remove_fifo(analysis_settings, max_process_id, filename, fifo_dir=''):
    for process_id in range(1, max_process_id + 1):
        do_remove_fifos(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename, fifo_dir)


def ri_remove_fifo(analysis_settings, max_process_id, filename, fifo_dir=''):
    for process_id in range(1, max_process_id + 1):
        do_remove_fifos(RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_id, filename, fifo_dir)


def do_waits(wait_variable, wait_count, filename):
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

        print_command(filename, cmd)
        print_command(filename, '')


def do_pwaits(filename, process_counter):
    """
    Add pwaits to the script
    """
    do_waits('pid', process_counter['pid_monitor_count'], filename)


def do_awaits(filename, process_counter):
    """
    Add awaits to the script
    """
    do_waits('apid', process_counter['apid_monitor_count'], filename)


def do_lwaits(filename, process_counter):
    """
    Add lwaits to the script
    """
    do_waits('lpid', process_counter['lpid_monitor_count'], filename)


def do_kwaits(filename, process_counter):
    """
    Add kwaits to the script
    """
    do_waits('kpid', process_counter['kpid_monitor_count'], filename)


def get_getmodel_cmd(
    number_of_samples, gul_threshold, use_random_number_file, 
    coverage_output, item_output,
    process_id, max_process_id, **kwargs):
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

    cmd = 'eve {0} {1} | getmodel | gulcalc -S{2} -L{3}'.format(
        process_id, max_process_id,
        number_of_samples, gul_threshold)

    if use_random_number_file:
        cmd = '{} -r'.format(cmd)
    if coverage_output != '':
        cmd = '{} -c {}'.format(cmd, coverage_output)
    if item_output != '':
        cmd = '{} -i {}'.format(cmd, item_output)

    return cmd


def genbash(
    max_process_id,
    analysis_settings,
    num_reinsurance_iterations=0,
    fifo_tmp_dir=True,
    mem_limit=False,
    alloc_rule=None,
    filename='run_kools.sh',
    _get_getmodel_cmd=get_getmodel_cmd,
    custom_args={}
):
    """
    Generates a bash script containing ktools calculation instructions for an
    Oasis model.

    :param max_process_id: The number of processes to create
    :type max_process_id: int

    :param analysis_settings: The analysis settings
    :type analysis_settings: dict

    :param filename: The output file name
    :type filename: string

    :param num_reinsurance_iterations: The number of reinsurance iterations
    :type num_reinsurance_iterations: int

    :param fifo_tmp_dir: When set to True, Create and use FIFO quese in `/tmp/[A-Z,0-9]/fifo`, if False run in './fifo'
    :type fifo_tmp_dir: boolean

    :param alloc_rule: override for the Ktools Allocation rule (1 or 2)
    :type alloc_rule: Int

    :param mem_limit: Flag to set a max memory limit for each ktools process
    :type mem_limit: boolean

    :param get_getmodel_cmd: Method for getting the getmodel command, by default
        ``GenerateLossesCmd.get_getmodel_cmd`` is used.
    :type get_getmodel_cmd: callable
    """
    process_counter = Counter()

    use_random_number_file = False
    gul_output = False
    il_output = False
    ri_output = False
    fifo_queue_dir = ""

    # Alloc Rule input guard - default to '2' if invalid value given
    if alloc_rule not in [ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID,
                          ALLOCATE_TO_ITEMS_BY_GUL_ALLOC_ID,
                          NO_ALLOCATION_ALLOC_ID]:
        alloc_rule = ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID

    # remove the file if it already exists
    if os.path.exists(filename):
        os.remove(filename)

    gul_threshold = analysis_settings.get('gul_threshold', 0)
    number_of_samples = analysis_settings.get('number_of_samples', 0)

    if 'model_settings' in analysis_settings and analysis_settings['model_settings'].get('use_random_number_file'):
        use_random_number_file = True

    if 'gul_output' in analysis_settings:
        gul_output = analysis_settings['gul_output']

    if 'il_output' in analysis_settings:
        il_output = analysis_settings['il_output']

    if 'ri_output' in analysis_settings:
        ri_output = analysis_settings['ri_output']

    print_command(filename, '#!/bin/bash')

    print_command(filename, '')

    print_command(filename, 'rm -R -f output/*')
    if not fifo_tmp_dir:
        print_command(filename, 'rm -R -f fifo/*')
    print_command(filename, 'rm -R -f work/*')
    print_command(filename, '')

    print_command(filename, 'mkdir work/kat')

    # Create tmp dir for FIFO queues (Windows support)
    if fifo_tmp_dir:
        fifo_queue_dir = '/tmp/{}/'.format(''.join(
                          random.choice(string.ascii_letters + string.digits) for _ in range(10)))
        print_command(filename, 'mkdir -p {}fifo'.format(fifo_queue_dir))

    if gul_output:
        do_gul_make_fifo(analysis_settings, max_process_id, filename, fifo_queue_dir)
        create_workfolders(RUNTYPE_GROUNDUP_LOSS, analysis_settings, filename)

    if il_output:
        il_make_fifo(analysis_settings, max_process_id, filename, fifo_queue_dir)
        create_workfolders(RUNTYPE_INSURED_LOSS, analysis_settings, filename)

    if ri_output:
        ri_make_fifo(analysis_settings, max_process_id, filename, fifo_queue_dir)
        create_workfolders(RUNTYPE_REINSURANCE_LOSS, analysis_settings, filename)
        print_command(filename, '')

    if ri_output:
        print_command(filename, '')
        print_command(filename, '# --- Do reinsurance loss computes ---')
        print_command(filename, '')
        ri(analysis_settings, max_process_id, filename, process_counter, num_reinsurance_iterations, fifo_queue_dir)

    if il_output:
        print_command(filename, '')
        print_command(filename, '# --- Do insured loss computes ---')
        print_command(filename, '')
        il(analysis_settings, max_process_id, filename, process_counter, fifo_queue_dir)

    if mem_limit:
        print_command(filename, '')
        print_command(filename, '# --- Use Ktools per process memory limit ---')
        print_command(filename, '')
        do_ktools_mem_limit(max_process_id, filename)

    if gul_output:
        print_command(filename, '')
        print_command(filename, '# --- Do ground up loss computes ---')
        print_command(filename, '')
        do_gul(analysis_settings, max_process_id, filename, process_counter, fifo_queue_dir)

    print_command(filename, '')

    for process_id in range(1, max_process_id + 1):

        ##! Should be able to streamline the logic a little
        if num_reinsurance_iterations > 0 and ri_output:

            getmodel_args = {
                'number_of_samples'      : number_of_samples,
                'gul_threshold'          : gul_threshold,
                'use_random_number_file' : use_random_number_file,
                'coverage_output'        : '{}fifo/gul_P{}'.format(fifo_queue_dir, process_id),
                'item_output'            : '-',
                'process_id'             : process_id,
                'max_process_id'         : max_process_id
            }
            getmodel_args.update(custom_args)
            getmodel_cmd = _get_getmodel_cmd(**getmodel_args)
            main_cmd = '{2} | fmcalc -a {3} | tee {4}fifo/il_P{0}'.format(
                process_id, max_process_id, getmodel_cmd,
                alloc_rule,
                fifo_queue_dir)

            for i in range(1, num_reinsurance_iterations + 1):
                main_cmd = "{0} | fmcalc -a {3} -n -p RI_{2}".format(
                    main_cmd, os.sep, i, ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID)

            main_cmd = "{0} > {1}fifo/ri_P{2} &".format(main_cmd, fifo_queue_dir, process_id)

            print_command(
                filename,
                main_cmd
            )

        elif gul_output and il_output:
            getmodel_args = {
                'number_of_samples'      : number_of_samples,
                'gul_threshold'          : gul_threshold,
                'use_random_number_file' : use_random_number_file,
                'coverage_output'        : '{}fifo/gul_P{}'.format(fifo_queue_dir, process_id),
                'item_output'            : '-',
                'process_id'             : process_id,
                'max_process_id'         : max_process_id
            }
            getmodel_args.update(custom_args)
            getmodel_cmd = _get_getmodel_cmd(**getmodel_args)
            main_cmd = '{2} | fmcalc -a {3} > {4}fifo/il_P{0}  &'.format(
                process_id, max_process_id, getmodel_cmd,
                alloc_rule,
                fifo_queue_dir)

            print_command(
                filename,
                main_cmd
            )
        else:
            if gul_output and 'gul_summaries' in analysis_settings:
                getmodel_args = {
                    'number_of_samples'      : number_of_samples,
                    'gul_threshold'          : gul_threshold,
                    'use_random_number_file' : use_random_number_file,
                    'coverage_output'        : '-',
                    'item_output'            : '',
                    'process_id'             : process_id,
                    'max_process_id'         : max_process_id
                }
                getmodel_args.update(custom_args)
                getmodel_cmd = _get_getmodel_cmd(**getmodel_args)
                print_command(
                    filename,
                    '{2} > {3}fifo/gul_P{0}  &'.format(process_id, max_process_id, getmodel_cmd, fifo_queue_dir)
                )
            if il_output and 'il_summaries' in analysis_settings:
                getmodel_args = {
                    'number_of_samples'      : number_of_samples,
                    'gul_threshold'          : gul_threshold,
                    'use_random_number_file' : use_random_number_file,
                    'coverage_output'        : '',
                    'item_output'            : '-',
                    'process_id'             : process_id,
                    'max_process_id'         : max_process_id
                }
                getmodel_args.update(custom_args)
                getmodel_cmd = _get_getmodel_cmd(**getmodel_args)
                print_command(
                    filename,
                    "{2} | fmcalc -a {3} > {4}fifo/il_P{0}  &".format(
                        process_id, max_process_id, getmodel_cmd,
                        alloc_rule,
                        fifo_queue_dir)
                )

    print_command(filename, '')

    do_pwaits(filename, process_counter)

    if ri_output:
        print_command(filename, '')
        print_command(filename, '# --- Do reinsurance loss kats ---')
        print_command(filename, '')
        do_kats(RUNTYPE_REINSURANCE_LOSS, analysis_settings, max_process_id, filename, process_counter)

    if il_output:
        print_command(filename, '')
        print_command(filename, '# --- Do insured loss kats ---')
        print_command(filename, '')
        do_kats(RUNTYPE_INSURED_LOSS, analysis_settings, max_process_id, filename, process_counter)

    if gul_output:
        print_command(filename, '')
        print_command(filename, '# --- Do ground up loss kats ---')
        print_command(filename, '')
        do_kats(RUNTYPE_GROUNDUP_LOSS, analysis_settings, max_process_id, filename, process_counter)

    do_kwaits(filename, process_counter)

    if mem_limit:
        print_command(filename, '')
        print_command(filename, '# --- Remove per process memory limit ---')
        print_command(filename, '')
        do_ktools_mem_limit(1, filename)

    print_command(filename, '')
    do_post_wait_processing(RUNTYPE_REINSURANCE_LOSS, analysis_settings, filename, process_counter)
    do_post_wait_processing(RUNTYPE_INSURED_LOSS, analysis_settings, filename, process_counter)
    do_post_wait_processing(RUNTYPE_GROUNDUP_LOSS, analysis_settings, filename, process_counter)

    do_awaits(filename, process_counter)  # waits for aalcalc
    do_lwaits(filename, process_counter)  # waits for leccalc

    if gul_output:
        do_gul_remove_fifo(analysis_settings, max_process_id, filename, fifo_queue_dir)
        remove_workfolders(RUNTYPE_GROUNDUP_LOSS, analysis_settings, filename)

    print_command(filename, '')

    if ri_output:
        ri_remove_fifo(analysis_settings, max_process_id, filename, fifo_queue_dir)
        remove_workfolders(RUNTYPE_REINSURANCE_LOSS, analysis_settings, filename)

    if il_output:
        il_remove_fifo(analysis_settings, max_process_id, filename, fifo_queue_dir)
        remove_workfolders(RUNTYPE_INSURED_LOSS, analysis_settings, filename)

    # If fifo dir is in /tmp/*/ then clean up
    if re.search(r"((/tmp/)[A-Za-z0-9_-]+(/))", fifo_queue_dir) and fifo_tmp_dir:
        print_command(filename, 'rm {}fifo/*'.format(fifo_queue_dir))
        print_command(filename, 'rmdir {}fifo'.format(fifo_queue_dir))
        print_command(filename, 'rmdir {}'.format(fifo_queue_dir))
