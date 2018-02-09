from __future__ import unicode_literals

from collections import Counter

import os
import io


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
                process_counter['apid_monitor_count'] += 1
                print_command(
                    filename,
                    'aalsummary -K{0}_S{1}_aalcalc > output/{0}_S{1}_aalcalc.csv & apid{2}=$!'.format(
                        runtype,
                        summary_set,
                        process_counter['apid_monitor_count']
                    )
                )

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
                            switch = wait_proocessing_switches.get(option, '')
                            cmd = '{} {} output/{}_S{}_leccalc_{}.csv'.format(cmd, switch, runtype, summary_set,
                                                                              option)

                    cmd = '{} & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                    print_command(filename, cmd)


def do_fifos(action, runtype, analysis_settings, process_id, filename):
    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    print_command(filename, '{} fifo/{}_P{}'.format(action, runtype, process_id))
    print_command(filename, '')
    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']
            print_command(filename, '{} fifo/{}_S{}_summary_P{}'.format(action, runtype, summary_set, process_id))

            if summary.get('eltcalc'):
                print_command(
                    filename,
                    '{} fifo/{}_S{}_summaryeltcalc_P{}'.format(action, runtype, summary_set, process_id)
                )
                print_command(
                    filename,
                    '{} fifo/{}_S{}_eltcalc_P{}'.format(action, runtype, summary_set, process_id)
                )

            if summary.get('summarycalc'):
                print_command(
                    filename,
                    '{} fifo/{}_S{}_summarysummarycalc_P{}'.format(action, runtype, summary_set, process_id)
                )
                print_command(
                    filename,
                    '{} fifo/{}_S{}_summarycalc_P{}'.format(action, runtype, summary_set, process_id)
                )

            if summary.get('pltcalc'):
                print_command(
                    filename,
                    '{} fifo/{}_S{}_summarypltcalc_P{}'.format(action, runtype, summary_set, process_id)
                )
                print_command(
                    filename,
                    '{} fifo/{}_S{}_pltcalc_P{}'.format(action, runtype, summary_set, process_id)
                )

            if summary.get('aalcalc'):
                print_command(
                    filename,
                    '{} fifo/{}_S{}_summaryaalcalc_P{}'.format(action, runtype, summary_set, process_id)
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
                print_command(filename, 'mkdir work/{}_S{}_aalcalc'.format(runtype, summary_set))


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
                print_command(filename, 'rm work/{}_S{}_aalcalc/*'.format(runtype, summary_set))
                print_command(filename, 'rmdir work/{}_S{}_aalcalc'.format(runtype, summary_set))


def do_make_fifos(runtype, analysis_settings, process_id, filename):
    do_fifos('mkfifo', runtype, analysis_settings, process_id, filename)


def do_remove_fifos(runtype, analysis_settings, process_id, filename):
    do_fifos('rm', runtype, analysis_settings, process_id, filename)


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


def do_summarycalcs(runtype, analysis_settings, process_id, filename):
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
            cmd = '{0} -{1} fifo/{2}_S{1}_summary_P{3}'.format(cmd, summary_set, runtype, process_id)

    cmd = '{} < fifo/{}_P{} &'.format(cmd, runtype, process_id)
    print_command(filename, cmd)


def do_tees(runtype, analysis_settings, process_id, filename, process_counter):
    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    for summary in summaries:
        if 'id' in summary:
            process_counter['pid_monitor_count'] += 1
            summary_set = summary['id']
            cmd = 'tee < fifo/{}_S{}_summary_P{}'.format(runtype, summary_set, process_id)

            if summary.get('eltcalc'):
                cmd = '{} fifo/{}_S{}_summaryeltcalc_P{}'.format(cmd, runtype, summary_set, process_id)

            if summary.get('pltcalc'):
                cmd = '{} fifo/{}_S{}_summarypltcalc_P{}'.format(cmd, runtype, summary_set, process_id)

            if summary.get('summarycalc'):
                cmd = '{} fifo/{}_S{}_summarysummarycalc_P{}'.format(cmd, runtype, summary_set, process_id)

            if summary.get('aalcalc'):
                cmd = '{} fifo/{}_S{}_summaryaalcalc_P{}'.format(cmd, runtype, summary_set, process_id)

            if summary.get('lec_output') and leccalc_enabled(summary['leccalc']):
                cmd = '{} work/{}_S{}_summaryleccalc/P{}.bin'.format(cmd, runtype, summary_set, process_id)

            cmd = '{} > /dev/null & pid{}=$!'.format(cmd, process_counter['pid_monitor_count'])
            print_command(filename, cmd)


def do_any(runtype, analysis_settings, process_id, filename, process_counter):
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
                    "{3} < fifo/{0}_S{1}_summaryeltcalc_P{2} > work/kat/{0}_S{1}_eltcalc_P{2} & pid{4}=$!".format(
                        runtype, summary_set, process_id, cmd, process_counter['pid_monitor_count']
                    )
                )

            if summary.get("summarycalc"):
                cmd = 'summarycalctocsv -s'
                if process_id == 1:
                    cmd = 'summarycalctocsv'

                process_counter['pid_monitor_count'] += 1
                print_command(
                    filename,
                    '{3} < fifo/{0}_S{1}_summarysummarycalc_P{2} > work/kat/{0}_S{1}_summarycalc_P{2} & pid{4}=$!'.format(
                        runtype, summary_set, process_id, cmd, process_counter['pid_monitor_count']
                    )
                )

            if summary.get('pltcalc'):
                cmd = 'pltcalc -s'
                if process_id == 1:
                    cmd = 'pltcalc'

                process_counter['pid_monitor_count'] += 1
                print_command(
                    filename,
                    '{3} < fifo/{0}_S{1}_summarypltcalc_P{2} > work/kat/{0}_S{1}_pltcalc_P{2} & pid{4}=$!'.format(
                        runtype, summary_set, process_id, cmd, process_counter['pid_monitor_count']
                    )
                )

            if summary.get('aalcalc'):
                process_counter['pid_monitor_count'] += 1
                print_command(
                    filename,
                    'aalcalc < fifo/{0}_S{1}_summaryaalcalc_P{2} > work/{0}_S{1}_aalcalc/P{2}.bin & pid{3}=$!'.format(
                        runtype, summary_set, process_id, process_counter['pid_monitor_count']
                    )
                )

        print_command(filename, '')


def do_il(analysis_settings, max_process_id, filename, process_counter):
    for process_id in range(1, max_process_id + 1):
        do_any('il', analysis_settings, process_id, filename, process_counter)

    for process_id in range(1, max_process_id + 1):
        do_tees('il', analysis_settings, process_id, filename, process_counter)

    for process_id in range(1, max_process_id + 1):
        do_summarycalcs('il', analysis_settings, process_id, filename)


def do_gul(analysis_settings, max_process_id, filename, process_counter):
    for process_id in range(1, max_process_id + 1):
        do_any('gul', analysis_settings, process_id, filename, process_counter)

    for process_id in range(1, max_process_id + 1):
        do_tees('gul', analysis_settings, process_id, filename, process_counter)

    for process_id in range(1, max_process_id + 1):
        do_summarycalcs('gul', analysis_settings, process_id, filename)


def do_il_make_fifo(analysis_settings, max_process_id, filename):
    for process_id in range(1, max_process_id + 1):
        do_make_fifos('il', analysis_settings, process_id, filename)


def do_gul_make_fifo(analysis_settings, max_process_id, filename):
    for process_id in range(1, max_process_id + 1):
        do_make_fifos('gul', analysis_settings, process_id, filename)


def do_il_remove_fifo(analysis_settings, max_process_id, filename):
    for process_id in range(1, max_process_id + 1):
        do_remove_fifos('il', analysis_settings, process_id, filename)


def do_gul_remove_fifo(analysis_settings, max_process_id, filename):
    for process_id in range(1, max_process_id + 1):
        do_remove_fifos('gul', analysis_settings, process_id, filename)


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


def get_getmodel_cmd(number_of_samples, gul_threshold, use_random_number_file, coverage_output, item_output):
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


def genbash(max_process_id, analysis_settings, filename, _get_getmodel_cmd=None):
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
    process_counter = Counter()
    _get_getmodel_cmd = _get_getmodel_cmd or get_getmodel_cmd

    use_random_number_file = False
    gul_output = False
    il_output = False

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

    print_command(filename, '#!/bin/bash')

    print_command(filename, '')

    print_command(filename, 'rm -R -f output/*')
    print_command(filename, 'rm -R -f fifo/*')
    print_command(filename, 'rm -R -f work/*')
    print_command(filename, '')

    print_command(filename, 'mkdir work/kat')

    if gul_output:
        do_gul_make_fifo(analysis_settings, max_process_id, filename)
        create_workfolders('gul', analysis_settings, filename)

    print_command(filename, '')

    if il_output:
        do_il_make_fifo(analysis_settings, max_process_id, filename)
        create_workfolders('il', analysis_settings, filename)

    print_command(filename, '')
    print_command(filename, '# --- Do insured loss computes ---')
    print_command(filename, '')
    if il_output:
        do_il(analysis_settings, max_process_id, filename, process_counter)

    print_command(filename, '')
    print_command(filename, '# --- Do ground up loss  computes ---')
    print_command(filename, '')
    if gul_output:
        do_gul(analysis_settings, max_process_id, filename, process_counter)

    print_command(filename, '')

    for process_id in range(1, max_process_id + 1):
        if gul_output and il_output:
            getmodel_cmd = _get_getmodel_cmd(
                number_of_samples,
                gul_threshold,
                use_random_number_file,
                'fifo/gul_P{}'.format(process_id),
                '-'
            )

            print_command(
                filename,
                'eve {0} {1} | {2} | fmcalc > fifo/il_P{0}  &'.format(process_id, max_process_id, getmodel_cmd)
            )
        else:
            #  Now the mainprocessing
            if gul_output:
                if 'gul_summaries' in analysis_settings:
                    getmodel_cmd = _get_getmodel_cmd(
                        number_of_samples,
                        gul_threshold,
                        use_random_number_file,
                        '-',
                        '')

                    print_command(
                        filename,
                        'eve {0} {1} | {2} > fifo/gul_P{0}  &'.format(process_id, max_process_id, getmodel_cmd)
                    )

            if il_output:
                if 'il_summaries' in analysis_settings:
                    getmodel_cmd = _get_getmodel_cmd(
                        number_of_samples,
                        gul_threshold,
                        use_random_number_file,
                        '',
                        '-'
                    )

                    print_command(
                        filename,
                        "eve {0} {1} | {2} | fmcalc > fifo/il_P{0}  &".format(process_id, max_process_id, getmodel_cmd)
                    )

    print_command(filename, '')

    do_pwaits(filename, process_counter)

    print_command(filename, '')
    print_command(filename, '# --- Do insured loss kats ---')
    print_command(filename, '')
    if il_output:
        do_kats('il', analysis_settings, max_process_id, filename, process_counter)

    print_command(filename, '')
    print_command(filename, '# --- Do ground up loss kats ---')
    print_command(filename, '')
    if gul_output:
        do_kats('gul', analysis_settings, max_process_id, filename, process_counter)

    do_kwaits(filename, process_counter)

    print_command(filename, '')
    do_post_wait_processing('il', analysis_settings, filename, process_counter)
    do_post_wait_processing('gul', analysis_settings, filename, process_counter)

    do_awaits(filename, process_counter)  # waits for aalcalc
    do_lwaits(filename, process_counter)  # waits for leccalc

    if gul_output:
        do_gul_remove_fifo(analysis_settings, max_process_id, filename)
        remove_workfolders('gul', analysis_settings, filename)

    print_command(filename, '')

    if il_output:
        do_il_remove_fifo(analysis_settings, max_process_id, filename)
        remove_workfolders('il', analysis_settings, filename)
