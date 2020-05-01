import io
import os
import random
import re
import string

from collections import Counter

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

    :param command_file: File to append command to.
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

    # Note: Backwards compatibility of "outputs" in lec_options
    if "outputs" in lec_options:
        lec_options = lec_options["outputs"]

    for option in lec_options:
        if option in WAIT_PROCESSING_SWITCHES and lec_options[option]:
            return True
    return False


def do_post_wait_processing(
    runtype,
    analysis_settings,
    filename,
    process_counter,
    work_sub_dir='',
    output_dir='output/'
):
    if '{}_summaries'.format(runtype) not in analysis_settings:
        return

    for summary in analysis_settings['{}_summaries'.format(runtype)]:
        if "id" in summary:
            summary_set = summary['id']
            if summary.get('aalcalc'):
                cmd = 'aalcalc -K{}{}_S{}_summaryaalcalc'.format(
                    work_sub_dir,
                    runtype,
                    summary_set
                )

                process_counter['lpid_monitor_count'] += 1
                cmd = '{} > {}{}_S{}_aalcalc.csv'.format(
                    cmd, output_dir, runtype, summary_set
                )
                cmd = '{} & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                print_command(filename, cmd)

            if summary.get('lec_output'):
                leccalc = summary.get('leccalc', {})
                if leccalc and leccalc_enabled(leccalc):
                    cmd = 'leccalc {} -K{}{}_S{}_summaryleccalc'.format(
                        '-r' if leccalc.get('return_period_file') else '',
                        work_sub_dir,
                        runtype,
                        summary_set
                    )

                    # Note: Backwards compatibility of "outputs" in lec_options
                    if "outputs" in leccalc:
                        leccalc = leccalc["outputs"]

                    process_counter['lpid_monitor_count'] += 1
                    for option, active in sorted(leccalc.items()):
                        if active and option in WAIT_PROCESSING_SWITCHES:
                            switch = WAIT_PROCESSING_SWITCHES.get(option, '')
                            cmd = '{} {} {}{}_S{}_leccalc_{}.csv'.format(
                                cmd, switch, output_dir, runtype, summary_set,
                                option
                            )

                    cmd = '{} & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                    print_command(filename, cmd)


def do_fifos_exec(runtype, max_process_id, filename, fifo_dir, action='mkfifo'):
    for process_id in range(1, max_process_id + 1):
        print_command(filename, '{} {}{}_P{}'.format(action, fifo_dir, runtype, process_id))
    print_command(filename, '')


def do_fifos_calc(runtype, analysis_settings, max_process_id,
                  filename, fifo_dir='fifo/', action='mkfifo'):

    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    for process_id in range(1, max_process_id + 1):
        for summary in summaries:
            if 'id' in summary:
                summary_set = summary['id']
                print_command(filename, '{} {}{}_S{}_summary_P{}'.format(action, fifo_dir, runtype, summary_set, process_id))

                if summary.get('eltcalc'):
                    print_command(
                        filename,
                        '{} {}{}_S{}_summaryeltcalc_P{}'.format(action, fifo_dir, runtype, summary_set, process_id)
                    )
                    print_command(
                        filename,
                        '{} {}{}_S{}_eltcalc_P{}'.format(action, fifo_dir, runtype, summary_set, process_id)
                    )

                if summary.get('summarycalc'):
                    print_command(
                        filename,
                        '{} {}{}_S{}_summarysummarycalc_P{}'.format(action, fifo_dir, runtype, summary_set, process_id)
                    )
                    print_command(
                        filename,
                        '{} {}{}_S{}_summarycalc_P{}'.format(action, fifo_dir, runtype, summary_set, process_id)
                    )

                if summary.get('pltcalc'):
                    print_command(
                        filename,
                        '{} {}{}_S{}_summarypltcalc_P{}'.format(action, fifo_dir, runtype, summary_set, process_id)
                    )
                    print_command(
                        filename,
                        '{} {}{}_S{}_pltcalc_P{}'.format(action, fifo_dir, runtype, summary_set, process_id)
                    )
        print_command(filename, '')


def create_workfolders(runtype, analysis_settings, filename, work_dir='work/'):
    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']
            if summary.get('lec_output'):
                if leccalc_enabled(summary['leccalc']):
                    print_command(
                        filename,
                        'mkdir {}{}_S{}_summaryleccalc'.format(work_dir, runtype, summary_set)
                    )

            if summary.get('aalcalc'):
                print_command(
                    filename,
                    'mkdir {}{}_S{}_summaryaalcalc'.format(work_dir, runtype, summary_set)
                )


def do_kats(
    runtype,
    analysis_settings,
    max_process_id,
    filename,
    process_counter,
    work_dir='work/kat/',
    output_dir='output/'
):
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
                    cmd = '{} {}{}_S{}_eltcalc_P{}'.format(
                        cmd, work_dir, runtype, summary_set, process_id
                    )

                process_counter['kpid_monitor_count'] += 1
                cmd = '{} > {}{}_S{}_eltcalc.csv & kpid{}=$!'.format(
                    cmd, output_dir, runtype, summary_set,
                    process_counter['kpid_monitor_count']
                )
                print_command(filename, cmd)

            if summary.get('pltcalc'):
                anykats = True

                cmd = 'kat'
                for process_id in range(1, max_process_id + 1):
                    cmd = '{} {}{}_S{}_pltcalc_P{}'.format(
                        cmd, work_dir, runtype, summary_set, process_id
                    )

                process_counter['kpid_monitor_count'] += 1
                cmd = '{} > {}{}_S{}_pltcalc.csv & kpid{}=$!'.format(
                    cmd, output_dir, runtype, summary_set,
                    process_counter['kpid_monitor_count']
                )
                print_command(filename, cmd)

            if summary.get("summarycalc"):
                anykats = True

                cmd = 'kat'
                for process_id in range(1, max_process_id + 1):
                    cmd = '{} {}{}_S{}_summarycalc_P{}'.format(
                        cmd, work_dir, runtype, summary_set, process_id
                    )

                process_counter['kpid_monitor_count'] += 1
                cmd = '{} > {}{}_S{}_summarycalc.csv & kpid{}=$!'.format(
                    cmd, output_dir, runtype, summary_set,
                    process_counter['kpid_monitor_count']
                )
                print_command(filename, cmd)

    return anykats


def do_summarycalcs(
    runtype,
    analysis_settings,
    process_id,
    filename,
    fifo_dir='fifo/',
    stderr_guard=True,
    num_reinsurance_iterations=0,
    gul_legacy_stream=None,
):

    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    if process_id == 1:
        print_command(filename, '')

    summarycalc_switch = '-f'
    if runtype == RUNTYPE_GROUNDUP_LOSS:
        if gul_legacy_stream:
            # gul coverage stream
            summarycalc_switch = '-g'
        else:
            # Accept item stream only
            summarycalc_switch = '-i'

    summarycalc_directory_switch = ""
    if runtype == RUNTYPE_REINSURANCE_LOSS:
        i = num_reinsurance_iterations
        summarycalc_directory_switch = "-p RI_{0}".format(i)

    cmd = 'summarycalc {} {}'.format(summarycalc_switch, summarycalc_directory_switch)
    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']
            cmd = '{0} -{1} {4}{2}_S{1}_summary_P{3}'.format(cmd, summary_set, runtype, process_id, fifo_dir)

    cmd = '{0} < {1}{2}_P{3}'.format(cmd, fifo_dir, runtype, process_id)
    cmd = '( {0} ) 2>> log/stderror.err  &'.format(cmd) if stderr_guard else '{0} &'.format(cmd)      # Wrap in subshell and pipe stderr to file
    print_command(filename, cmd)


def do_tees(runtype, analysis_settings, process_id, filename, process_counter, fifo_dir='fifo/', work_dir='work/'):
    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    if process_id == 1:
        print_command(filename, '')

    for summary in summaries:
        if 'id' in summary:
            process_counter['pid_monitor_count'] += 1
            summary_set = summary['id']
            cmd = 'tee < {}{}_S{}_summary_P{}'.format(fifo_dir, runtype, summary_set, process_id)

            if summary.get('eltcalc'):
                cmd = '{} {}{}_S{}_summaryeltcalc_P{}'.format(cmd, fifo_dir, runtype, summary_set, process_id)

            if summary.get('pltcalc'):
                cmd = '{} {}{}_S{}_summarypltcalc_P{}'.format(cmd, fifo_dir, runtype, summary_set, process_id)

            if summary.get('summarycalc'):
                cmd = '{} {}{}_S{}_summarysummarycalc_P{}'.format(cmd, fifo_dir, runtype, summary_set, process_id)

            if summary.get('aalcalc'):
                cmd = '{} {}{}_S{}_summaryaalcalc/P{}.bin'.format(cmd, work_dir, runtype, summary_set, process_id)

            if summary.get('lec_output') and leccalc_enabled(summary['leccalc']):
                cmd = '{} {}{}_S{}_summaryleccalc/P{}.bin'.format(cmd, work_dir, runtype, summary_set, process_id)

            cmd = '{} > /dev/null & pid{}=$!'.format(cmd, process_counter['pid_monitor_count'])
            print_command(filename, cmd)


def do_any(runtype, analysis_settings, process_id, filename, process_counter, fifo_dir='fifo/', work_dir='work/'):
    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    if process_id == 1:
        print_command(filename, '')

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
                    "{3} < {5}{0}_S{1}_summaryeltcalc_P{2} > {6}kat/{0}_S{1}_eltcalc_P{2} & pid{4}=$!".format(
                        runtype, summary_set, process_id, cmd, process_counter['pid_monitor_count'], fifo_dir, work_dir
                    )
                )

            if summary.get("summarycalc"):
                cmd = 'summarycalctocsv -s'
                if process_id == 1:
                    cmd = 'summarycalctocsv'

                process_counter['pid_monitor_count'] += 1
                print_command(
                    filename,
                    '{3} < {5}{0}_S{1}_summarysummarycalc_P{2} > {6}kat/{0}_S{1}_summarycalc_P{2} & pid{4}=$!'.format(
                        runtype, summary_set, process_id, cmd, process_counter['pid_monitor_count'], fifo_dir, work_dir
                    )
                )

            if summary.get('pltcalc'):
                cmd = 'pltcalc -s'
                if process_id == 1:
                    cmd = 'pltcalc'

                process_counter['pid_monitor_count'] += 1
                print_command(
                    filename,
                    '{3} < {5}{0}_S{1}_summarypltcalc_P{2} > {6}kat/{0}_S{1}_pltcalc_P{2} & pid{4}=$!'.format(
                        runtype, summary_set, process_id, cmd, process_counter['pid_monitor_count'], fifo_dir, work_dir
                    )
                )


def ri(analysis_settings, max_process_id, filename, process_counter, num_reinsurance_iterations, fifo_dir='fifo/', work_dir='work/', stderr_guard=True):
    for process_id in range(1, max_process_id + 1):
        do_any(RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir)

    for process_id in range(1, max_process_id + 1):
        do_tees(RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir)

    for process_id in range(1, max_process_id + 1):
        do_summarycalcs(
            runtype=RUNTYPE_REINSURANCE_LOSS,
            analysis_settings=analysis_settings,
            process_id=process_id,
            filename=filename,
            fifo_dir=fifo_dir,
            stderr_guard=stderr_guard,
            num_reinsurance_iterations=num_reinsurance_iterations,
        )


def il(analysis_settings, max_process_id, filename, process_counter, fifo_dir='fifo/', work_dir='work/', stderr_guard=True):
    for process_id in range(1, max_process_id + 1):
        do_any(RUNTYPE_INSURED_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir)

    for process_id in range(1, max_process_id + 1):
        do_tees(RUNTYPE_INSURED_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir)

    for process_id in range(1, max_process_id + 1):
        do_summarycalcs(
            runtype=RUNTYPE_INSURED_LOSS,
            analysis_settings=analysis_settings,
            process_id=process_id,
            filename=filename,
            fifo_dir=fifo_dir,
            stderr_guard=stderr_guard,
        )


def do_gul(
    analysis_settings,
    max_process_id,
    filename,
    process_counter,
    fifo_dir='fifo/',
    work_dir='work/',
    gul_alloc_rule=None,
    gul_legacy_stream=None,
    stderr_guard=True,
    full_correlation=False
):

    for process_id in range(1, max_process_id + 1):
        do_any(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir)

    for process_id in range(1, max_process_id + 1):
        do_tees(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir)

    for process_id in range(1, max_process_id + 1):
        do_summarycalcs(
            runtype=RUNTYPE_GROUNDUP_LOSS,
            analysis_settings=analysis_settings,
            process_id=process_id,
            filename=filename,
            gul_legacy_stream=gul_legacy_stream,
            fifo_dir=fifo_dir,
            stderr_guard=stderr_guard
        )


def do_waits(wait_variable, wait_count, filename):
    """
    Add waits to the script

    :param wait_variable: The type of wait
    :type wait_variable: str

    :param wait_count: The number of processes to wait for
    :type wait_count: int

    :param filename: Script to add waits to
    :type filename: str
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


def do_fcwaits(filename, process_counter):
    """
    Add fcwaits to the script
    """
    do_waits('fcpid', process_counter['fcpid_monitor_count'], filename)


def get_getmodel_itm_cmd(
        number_of_samples, gul_threshold, use_random_number_file,
        gul_alloc_rule, item_output,
        process_id, max_process_id, correlated_output, **kwargs):
    """
    Gets the getmodel ktools command (3.1.0+) Gulcalc item stream
    :param number_of_samples: The number of samples to run
    :type number_of_samples: int
    :param gul_threshold: The GUL threshold to use
    :type gul_threshold: float
    :param use_random_number_file: flag to use the random number file
    :type use_random_number_file: bool
    :param gul_alloc_rule: back allocation rule for gulcalc
    :type gul_alloc_rule: int
    :param item_output: The item output
    :type item_output: str
    :return: The generated getmodel command
    """
    cmd = 'eve {0} {1} | getmodel | gulcalc -S{2} -L{3}'.format(
        process_id, max_process_id,
        number_of_samples, gul_threshold)

    if use_random_number_file:
        cmd = '{} -r'.format(cmd)
    if correlated_output != '':
        cmd = '{} -j {}'.format(cmd, correlated_output)
    cmd = '{} -a{} -i {}'.format(cmd, gul_alloc_rule, item_output)
    return cmd


def get_getmodel_cov_cmd(
        number_of_samples, gul_threshold, use_random_number_file,
        coverage_output, item_output,
        process_id, max_process_id, **kwargs):
    """
    Gets the getmodel ktools command (version < 3.0.8) gulcalc coverage stream
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


def get_main_cmd_ri_stream(
    cmd,
    process_id,
    il_output,
    il_alloc_rule,
    ri_alloc_rule,
    num_reinsurance_iterations,
    fifo_dir='fifo/',
    stderr_guard=True,
    full_correlation=False,
    process_counter=None
):
    """
    Gets the fmcalc ktools command reinsurance stream
    :param cmd: either gulcalc command stream or correlated output file
    :type cmd: str
    :param process_id: ID corresponding to thread
    :type process_id: int
    :param il_output: If insured loss outputs required
    :type il_output: Boolean
    :param il_alloc_rule: insured loss allocation rule for fmcalc
    :type il_alloc_rule: int
    :param ri_alloc_rule: reinsurance allocation rule for fmcalc
    :type ri_alloc_rule: int
    :param num_reinsurance_iterations: number of reinsurance iterations
    :type num_reinsurance_iterations: int
    :param fifo_dir: path to fifo directory
    :type fifo_dir: str
    :param stderr_guard: send stderr output to log file
    :type stderr_guard: bool
    :param full_correlation: execute fmcalc on fully correlated data
    :type full_correlation: bool
    :param process_counter: process counter
    :type process_counter: Counter
    :return: generated fmcalc command as str
    """

    if full_correlation:
        fm_cmd = 'fmcalc -a{1} < {0}'
    else:
        fm_cmd = '{0} | fmcalc -a{1}'
    main_cmd = fm_cmd.format(cmd, il_alloc_rule)

    if il_output:
        main_cmd = "{0} | tee {1}il_P{2}".format(main_cmd, fifo_dir, process_id)

    for i in range(1, num_reinsurance_iterations + 1):
        main_cmd = "{0} | fmcalc -a{2} -n -p RI_{1}".format(
            main_cmd, i, ri_alloc_rule
        )

    main_cmd = "{0} > {1}ri_P{2}".format(main_cmd, fifo_dir, process_id)
    main_cmd = '( {0} ) 2>> log/stderror.err &'.format(main_cmd) if stderr_guard else '{0} &'.format(main_cmd)
    if full_correlation:
        process_counter['fcpid_monitor_count'] += 1
        main_cmd = '{0} fcpid{1}=$!'.format(
            main_cmd, process_counter['fcpid_monitor_count']
        )

    return main_cmd


def get_main_cmd_il_stream(
    cmd,
    process_id,
    il_alloc_rule,
    fifo_dir='fifo/',
    stderr_guard=True,
    full_correlation=False,
    process_counter=None
):
    """
    Gets the fmcalc ktools command insured losses stream
    :param cmd: either gulcalc command stream or correlated output file
    :type cmd: str
    :param process_id: ID corresponding to thread
    :type process_id: int
    :param il_alloc_rule: insured loss allocation rule for fmcalc
    :type il_alloc_rule: int
    :param fifo_dir: path to fifo directory
    :type fifo_dir: str
    :param stderr_guard: send stderr output to log file
    :type stderr_guard: bool
    :param full_correlation: execute fmcalc on fully correlated data
    :type full_correlation: bool
    :param process_counter: process counter
    :type process_counter: Counter
    :return: generated fmcalc command as str
    """

    if full_correlation:
        fm_cmd = 'fmcalc -a{2} < {1} > {3}il_P{0}'
    else:
        fm_cmd = '{1} | fmcalc -a{2} > {3}il_P{0} '
    main_cmd = fm_cmd.format(process_id, cmd, il_alloc_rule, fifo_dir)
    main_cmd = '( {0} ) 2>> log/stderror.err &'.format(main_cmd) if stderr_guard else '{0} &'.format(main_cmd)
    if full_correlation:
        process_counter['fcpid_monitor_count'] += 1
        main_cmd = '{0} fcpid{1}=$!'.format(
            main_cmd, process_counter['fcpid_monitor_count']
        )

    return main_cmd


def get_main_cmd_gul_stream(
    cmd,
    process_id,
    fifo_dir='fifo/',
    stderr_guard=True
):
    """
    Gets the command to output ground up losses
    :param cmd: either gulcalc command stream or correlated output file
    :type cmd: str
    :param process_id: ID corresponding to thread
    :type process_id: int
    :param fifo_dir: path to fifo directory
    :type fifo_dir: str
    :param stderr_guard: send stderr output to log file
    :type stderr_guard: bool
    :return: generated command as str
    """

    gul_cmd = '{1} > {2}gul_P{0} '
    main_cmd = gul_cmd.format(process_id, cmd, fifo_dir)
    main_cmd = '( {0} ) 2>> log/stderror.err &'.format(main_cmd) if stderr_guard else '{0} &'.format(main_cmd)

    return main_cmd


def do_computes(outputs):

    if len(outputs) == 0:
        return

    for output in outputs:
        filename = output['compute_args']['filename']
        print_command(filename, '')
        print_command(
            filename,
            '# --- Do {} loss computes ---'.format(output['loss_type'])
        )
        output['compute_fun'](**output['compute_args'])


def genbash(
    max_process_id,
    analysis_settings,
    num_reinsurance_iterations=0,
    fifo_tmp_dir=True,
    gul_alloc_rule=None,
    il_alloc_rule=None,
    ri_alloc_rule=None,
    stderr_guard=True,
    gul_legacy_stream=False,
    bash_trace=False,
    filename='run_kools.sh',
    _get_getmodel_cmd=None,
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

    :param gul_alloc_rule: Allocation rule (None or 1) for gulcalc, if not set default to coverage stream
    :type gul_alloc_rule: Int

    :param il_alloc_rule: Allocation rule (0, 1 or 2) for fmcalc
    :type il_alloc_rule: Int

    :param ri_alloc_rule: Allocation rule (0, 1 or 2) for fmcalc
    :type ri_alloc_rule: Int

    :param get_getmodel_cmd: Method for getting the getmodel command, by default
        ``GenerateLossesCmd.get_getmodel_cmd`` is used.
    :type get_getmodel_cmd: callable
    """
    process_counter = Counter()

    use_random_number_file = False
    stderr_guard = stderr_guard
    gul_item_stream = not gul_legacy_stream
    full_correlation = False
    gul_output = False
    il_output = False
    ri_output = False
    fifo_queue_dir = ""
    fifo_full_correlation_dir = ""
    work_dir = 'work/'
    work_kat_dir = 'work/kat/'
    work_full_correlation_dir = 'work/full_correlation/'
    work_full_correlation_kat_dir = 'work/full_correlation/kat/'
    output_dir = 'output/'
    output_full_correlation_dir = 'output/full_correlation/'

    # remove the file if it already exists
    if os.path.exists(filename):
        os.remove(filename)

    gul_threshold = analysis_settings.get('gul_threshold', 0)
    number_of_samples = analysis_settings.get('number_of_samples', 0)

    if 'model_settings' in analysis_settings and analysis_settings['model_settings'].get('use_random_number_file'):
        use_random_number_file = True

    if 'full_correlation' in analysis_settings:
        if _get_getmodel_cmd is None and gul_item_stream:
            full_correlation = analysis_settings['full_correlation']

    if 'gul_output' in analysis_settings:
        gul_output = analysis_settings['gul_output']

    if 'il_output' in analysis_settings:
        il_output = analysis_settings['il_output']

    if 'ri_output' in analysis_settings:
        ri_output = analysis_settings['ri_output']

    print_command(filename, '#!/bin/bash')
    print_command(filename, 'SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")')
    print_command(filename, '')

    print_command(filename, '# --- Script Init ---')
    print_command(filename, '')
    print_command(filename, 'set -e')
    print_command(filename, 'set -o pipefail')

    print_command(filename, 'mkdir -p log')
    print_command(filename, 'rm -R -f log/*')
    print_command(filename, '')

    if bash_trace:
        print_command(filename, '# --- Redirect Bash trace to file ---')
        print_command(filename, 'exec   > >(tee -ia log/bash.log)')
        print_command(filename, 'exec  2> >(tee -ia log/bash.log >& 2)')
        print_command(filename, 'exec 19> log/bash.log')
        print_command(filename, 'export BASH_XTRACEFD="19"')
        print_command(filename, '')

    if stderr_guard:
        print_command(filename, 'error_handler(){')
        print_command(filename, "   echo 'Run Error - terminating'")
        print_command(filename, '   exit_code=$?')
        print_command(filename, '   set +x')
        print_command(filename, '   group_pid=$(ps -p $$ -o pgid --no-headers)')
        print_command(filename, '   sess_pid=$(ps -p $$ -o sess --no-headers)')
        print_command(filename, '   printf "Script PID:%d, GPID:%s, SPID:%d" $$ $group_pid $sess_pid >> log/killout.txt')
        print_command(filename, '')
        print_command(filename, '   if hash pstree 2>/dev/null; then')
        print_command(filename, '       pstree -pn $$ >> log/killout.txt')
        print_command(filename, '       PIDS_KILL=$(pstree -pn $$ | grep -o "([[:digit:]]*)" | grep -o "[[:digit:]]*")')
        print_command(filename, '       kill -9 $(echo "$PIDS_KILL" | grep -v $group_pid | grep -v $$) 2>/dev/null')
        print_command(filename, '   else')
        print_command(filename, '       ps f -g $sess_pid > log/subprocess_list')
        print_command(filename, '       PIDS_KILL=$(pgrep -a --pgroup $group_pid | grep -v celery | grep -v $group_pid | grep -v $$)')
        print_command(filename, '       echo "$PIDS_KILL" >> log/killout.txt')
        print_command(filename, '       kill -9 $(echo "$PIDS_KILL" | awk \'BEGIN { FS = "[ \\t\\n]+" }{ print $1 }\') 2>/dev/null')
        print_command(filename, '   fi')
        print_command(filename, '   exit $(( 1 > $exit_code ? 1 : $exit_code ))')
        print_command(filename, '}')
        print_command(filename, 'trap error_handler QUIT HUP INT KILL TERM ERR')
        print_command(filename, '')
        print_command(filename, 'touch log/stderror.err')
        print_command(filename, 'ktools_monitor.sh $$ & pid0=$!')
        print_command(filename, '')

    if bash_trace:
        print_command(filename, 'set -x')

    print_command(filename, '# --- Setup run dirs ---')
    print_command(filename, '')
    print_command(filename, "find output/* ! -name '*summary-info*' -type f -exec rm -f {} +")
    if full_correlation:
        print_command(filename, 'mkdir {}'.format(output_full_correlation_dir))
    print_command(filename, '')
    if not fifo_tmp_dir:
        fifo_queue_dir = 'fifo/'
        print_command(filename, 'rm -R -f {}*'.format(fifo_queue_dir))
        if full_correlation:
            fifo_full_correlation_dir = fifo_queue_dir + 'full_correlation/'
            print_command(
                filename, 'mkdir {}'.format(fifo_full_correlation_dir)
            )
    print_command(filename, 'rm -R -f {}*'.format(work_dir))
    print_command(filename, 'mkdir {}'.format(work_kat_dir))
    if full_correlation:
        print_command(filename, 'mkdir {}'.format(work_full_correlation_dir))
        print_command(
            filename, 'mkdir {}'.format(work_full_correlation_kat_dir)
        )
    print_command(filename, '')

    # Create FIFOS under /tmp/* (Windows support)
    if fifo_tmp_dir:
        fifo_queue_dir = '/tmp/{}/'.format(
            ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
        )
        print_command(filename, 'rm -R -f {}'.format(fifo_queue_dir))
        fifo_queue_dir = fifo_queue_dir + 'fifo/'
        print_command(filename, 'mkdir -p {}'.format(fifo_queue_dir))
        if full_correlation:
            fifo_full_correlation_dir = fifo_queue_dir + 'full_correlation/'
            print_command(
                filename, 'mkdir {}'.format(fifo_full_correlation_dir)
            )

    # Create workfolders
    if gul_output:
        create_workfolders(RUNTYPE_GROUNDUP_LOSS, analysis_settings, filename, work_dir)
        if full_correlation:
            create_workfolders(
                RUNTYPE_GROUNDUP_LOSS, analysis_settings,
                filename, work_full_correlation_dir
            )

    if il_output:
        create_workfolders(RUNTYPE_INSURED_LOSS, analysis_settings, filename, work_dir)
        if full_correlation:
            create_workfolders(
                RUNTYPE_INSURED_LOSS, analysis_settings,
                filename, work_full_correlation_dir
            )

    if ri_output:
        create_workfolders(RUNTYPE_REINSURANCE_LOSS, analysis_settings, filename, work_dir)
        if full_correlation:
            create_workfolders(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings,
                filename, work_full_correlation_dir
            )
    print_command(filename, '')

    # Create Execution Pipeline FIFOs
    if gul_output:
        do_fifos_exec(RUNTYPE_GROUNDUP_LOSS, max_process_id, filename, fifo_queue_dir)
    if il_output:
        do_fifos_exec(RUNTYPE_INSURED_LOSS, max_process_id, filename, fifo_queue_dir)
    if ri_output:
        do_fifos_exec(RUNTYPE_REINSURANCE_LOSS, max_process_id, filename, fifo_queue_dir)

    # Create Summarycalc FIFOs
    if gul_output:
        do_fifos_calc(RUNTYPE_GROUNDUP_LOSS, analysis_settings, max_process_id,
                      filename, fifo_queue_dir)
    if il_output:
        do_fifos_calc(RUNTYPE_INSURED_LOSS, analysis_settings,
                      max_process_id, filename, fifo_queue_dir)
    if ri_output:
        do_fifos_calc(RUNTYPE_REINSURANCE_LOSS, analysis_settings,
                      max_process_id, filename, fifo_queue_dir)

    # Create Full correlation FIFO
    if full_correlation:
        if gul_output:
            do_fifos_calc(
                RUNTYPE_GROUNDUP_LOSS, analysis_settings,
                max_process_id, filename, fifo_full_correlation_dir
            )
        if il_output:
            do_fifos_calc(
                RUNTYPE_INSURED_LOSS, analysis_settings,
                max_process_id, filename, fifo_full_correlation_dir
            )
        if ri_output:
            do_fifos_calc(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings,
                max_process_id, filename, fifo_full_correlation_dir
            )

    print_command(filename, '')
    compute_outputs = []
    if ri_output:
        ri_computes = {
            'loss_type': 'reinsurance',
            'compute_fun': ri,
            'compute_args': {
                'analysis_settings': analysis_settings,
                'max_process_id': max_process_id,
                'filename': filename,
                'process_counter': process_counter,
                'num_reinsurance_iterations': num_reinsurance_iterations,
                'fifo_dir': fifo_queue_dir,
                'work_dir': work_dir,
                'stderr_guard': stderr_guard
            }
        }
        compute_outputs.append(ri_computes)

    if il_output:
        il_computes = {
            'loss_type': 'insured',
            'compute_fun': il,
            'compute_args': {
                'analysis_settings': analysis_settings,
                'max_process_id': max_process_id,
                'filename': filename,
                'process_counter': process_counter,
                'fifo_dir': fifo_queue_dir,
                'work_dir': work_dir,
                'stderr_guard': stderr_guard
            }
        }
        compute_outputs.append(il_computes)

    if gul_output:
        gul_computes = {
            'loss_type': 'ground up',
            'compute_fun': do_gul,
            'compute_args': {
                'analysis_settings': analysis_settings,
                'max_process_id': max_process_id,
                'filename': filename,
                'process_counter': process_counter,
                'fifo_dir': fifo_queue_dir,
                'work_dir': work_dir,
                'gul_alloc_rule': gul_alloc_rule,
                'gul_legacy_stream': gul_legacy_stream,
                'stderr_guard': stderr_guard
            }
        }
        compute_outputs.append(gul_computes)

    do_computes(compute_outputs)

    print_command(filename, '')

    for process_id in range(1, max_process_id + 1):
        # gulcalc output file for fully correlated output
        if full_correlation:
            correlated_output_file = '{0}gul_P{1}'.format(
                fifo_full_correlation_dir,
                process_id
            )
        else:
            correlated_output_file = ''

        getmodel_args = {
            'number_of_samples': number_of_samples,
            'gul_threshold': gul_threshold,
            'use_random_number_file': use_random_number_file,
            'coverage_output': '{0}gul_P{1}'.format(fifo_queue_dir, process_id),
            'item_output': '-',
            'gul_alloc_rule': gul_alloc_rule,
            'gul_legacy_stream': gul_legacy_stream,
            'process_id': process_id,
            'max_process_id': max_process_id,
            'correlated_output': correlated_output_file,
            'stderr_guard': stderr_guard
        }

        # GUL coverage & item stream (Older)
        if gul_item_stream:
            if gul_output:
                getmodel_args['item_output'] = '- | tee {0}gul_P{1}'.format(fifo_queue_dir, process_id)

            _get_getmodel_cmd = (_get_getmodel_cmd or get_getmodel_itm_cmd)
        else:
            if not gul_output:
                getmodel_args['coverage_output'] = ""
            _get_getmodel_cmd = (_get_getmodel_cmd or get_getmodel_cov_cmd)

        # ! Should be able to streamline the logic a little
        if num_reinsurance_iterations > 0 and ri_output:
            getmodel_args.update(custom_args)
            getmodel_cmd = _get_getmodel_cmd(**getmodel_args)
            main_cmd = get_main_cmd_ri_stream(
                getmodel_cmd,
                process_id,
                il_output,
                il_alloc_rule,
                ri_alloc_rule,
                num_reinsurance_iterations,
                fifo_queue_dir,
                stderr_guard
            )
            print_command(filename, main_cmd)

        elif gul_output and il_output:
            getmodel_args.update(custom_args)
            getmodel_cmd = _get_getmodel_cmd(**getmodel_args)
            main_cmd = get_main_cmd_il_stream(
                getmodel_cmd, process_id, il_alloc_rule, fifo_queue_dir,
                stderr_guard
            )
            print_command(filename, main_cmd)

        else:
            if gul_output and 'gul_summaries' in analysis_settings:
                getmodel_args['coverage_output'] = '-'
                getmodel_args['item_output'] = ''

                if gul_item_stream:
                    getmodel_args['item_output'] = '-'

                getmodel_args.update(custom_args)
                getmodel_cmd = _get_getmodel_cmd(**getmodel_args)
                main_cmd = get_main_cmd_gul_stream(
                    getmodel_cmd, process_id, fifo_queue_dir, stderr_guard
                )
                print_command(filename, main_cmd)

            if il_output and 'il_summaries' in analysis_settings:
                getmodel_args['coverage_output'] = ''
                getmodel_args['item_output'] = '-'

                getmodel_args.update(custom_args)
                getmodel_cmd = _get_getmodel_cmd(**getmodel_args)
                main_cmd = get_main_cmd_il_stream(
                    getmodel_cmd, process_id, il_alloc_rule, fifo_queue_dir,
                    stderr_guard
                )
                print_command(filename, main_cmd)

    print_command(filename, '')

    do_pwaits(filename, process_counter)

    if full_correlation:
        print_command(
            filename, '# --- Do computes for fully correlated output ---'
        )
        print_command(filename, '')

        for process_id in range(1, max_process_id + 1):
            # Set up file name for full correlation file
            correlated_output_file = '{0}gul_P{1}'.format(
                fifo_full_correlation_dir,
                process_id
            )

            if num_reinsurance_iterations > 0 and ri_output:
                main_cmd = get_main_cmd_ri_stream(
                    correlated_output_file,
                    process_id,
                    il_output,
                    il_alloc_rule,
                    ri_alloc_rule,
                    num_reinsurance_iterations,
                    fifo_full_correlation_dir,
                    stderr_guard,
                    full_correlation,
                    process_counter
                )

                print_command(filename, main_cmd)
            elif gul_output and il_output:
                main_cmd = get_main_cmd_il_stream(
                    correlated_output_file, process_id, il_alloc_rule,
                    fifo_full_correlation_dir, stderr_guard, full_correlation,
                    process_counter
                )
                print_command(filename, main_cmd)
            else:
                if il_output and 'il_summaries' in analysis_settings:

                    main_cmd = get_main_cmd_il_stream(
                        correlated_output_file, process_id, il_alloc_rule,
                        fifo_full_correlation_dir, stderr_guard,
                        full_correlation, process_counter
                    )
                    print_command(filename, main_cmd)

        print_command(filename, '')

        do_fcwaits(filename, process_counter)

        process_counter['pid_monitor_count'] = 0
        compute_outputs = []
        if ri_output:
            ri_computes = {
                'loss_type': 'reinsurance',
                'compute_fun': ri,
                'compute_args': {
                    'analysis_settings': analysis_settings,
                    'max_process_id': max_process_id,
                    'filename': filename,
                    'process_counter': process_counter,
                    'num_reinsurance_iterations': num_reinsurance_iterations,
                    'fifo_dir': fifo_full_correlation_dir,
                    'work_dir': work_full_correlation_dir,
                    'stderr_guard': stderr_guard
                }
            }
            compute_outputs.append(ri_computes)
        if il_output:
            il_computes = {
                'loss_type': 'insured',
                'compute_fun': il,
                'compute_args': {
                    'analysis_settings': analysis_settings,
                    'max_process_id': max_process_id,
                    'filename': filename,
                    'process_counter': process_counter,
                    'fifo_dir': fifo_full_correlation_dir,
                    'work_dir': work_full_correlation_dir,
                    'stderr_guard': stderr_guard
                }
            }
            compute_outputs.append(il_computes)
        if gul_output:
            gul_computes = {
                'loss_type': 'ground up',
                'compute_fun': do_gul,
                'compute_args': {
                    'analysis_settings': analysis_settings,
                    'max_process_id': max_process_id,
                    'filename': filename,
                    'process_counter': process_counter,
                    'fifo_dir': fifo_full_correlation_dir,
                    'work_dir': work_full_correlation_dir,
                    'gul_alloc_rule': gul_alloc_rule,
                    'gul_legacy_stream': gul_legacy_stream,
                    'stderr_guard': stderr_guard,
                    'full_correlation': full_correlation
                }
            }
            compute_outputs.append(gul_computes)

        do_computes(compute_outputs)

        print_command(filename, '')

        do_pwaits(filename, process_counter)

    if ri_output:
        print_command(filename, '')
        print_command(filename, '# --- Do reinsurance loss kats ---')
        print_command(filename, '')
        do_kats(
            RUNTYPE_REINSURANCE_LOSS, analysis_settings, max_process_id,
            filename, process_counter, work_kat_dir, output_dir
        )
        if full_correlation:
            print_command(filename, '')
            print_command(
                filename,
                '# --- Do reinsurance loss kats for fully correlated output ---'
            )
            print_command(filename, '')
            do_kats(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings, max_process_id,
                filename, process_counter, work_full_correlation_kat_dir,
                output_full_correlation_dir
            )

    if il_output:
        print_command(filename, '')
        print_command(filename, '# --- Do insured loss kats ---')
        print_command(filename, '')
        do_kats(
            RUNTYPE_INSURED_LOSS, analysis_settings, max_process_id, filename,
            process_counter, work_kat_dir, output_dir
        )
        if full_correlation:
            print_command(filename, '')
            print_command(
                filename,
                '# --- Do insured loss kats for fully correlated output ---'
            )
            print_command(filename, '')
            do_kats(
                RUNTYPE_INSURED_LOSS, analysis_settings, max_process_id,
                filename, process_counter, work_full_correlation_kat_dir,
                output_full_correlation_dir
            )

    if gul_output:
        print_command(filename, '')
        print_command(filename, '# --- Do ground up loss kats ---')
        print_command(filename, '')
        do_kats(
            RUNTYPE_GROUNDUP_LOSS, analysis_settings, max_process_id, filename,
            process_counter, work_kat_dir, output_dir
        )
        if full_correlation:
            print_command(filename, '')
            print_command(
                filename,
                '# --- Do ground up loss kats for fully correlated output ---'
            )
            print_command(filename, '')
            do_kats(
                RUNTYPE_GROUNDUP_LOSS, analysis_settings, max_process_id,
                filename, process_counter, work_full_correlation_kat_dir,
                output_full_correlation_dir
            )

    do_kwaits(filename, process_counter)

    print_command(filename, '')
    if ri_output:
        do_post_wait_processing(
            RUNTYPE_REINSURANCE_LOSS, analysis_settings, filename, process_counter,
            '', output_dir
        )
    if il_output:
        do_post_wait_processing(
            RUNTYPE_INSURED_LOSS, analysis_settings, filename, process_counter, '',
            output_dir
        )
    if gul_output:
        do_post_wait_processing(
            RUNTYPE_GROUNDUP_LOSS, analysis_settings, filename, process_counter, '',
            output_dir
        )

    if full_correlation:
        work_sub_dir = re.sub('^work/', '', work_full_correlation_dir)
        if ri_output:
            do_post_wait_processing(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings, filename,
                process_counter, work_sub_dir, output_full_correlation_dir
            )
        if il_output:
            do_post_wait_processing(
                RUNTYPE_INSURED_LOSS, analysis_settings, filename, process_counter,
                work_sub_dir, output_full_correlation_dir
            )
        if gul_output:
            do_post_wait_processing(
                RUNTYPE_GROUNDUP_LOSS, analysis_settings, filename, process_counter,
                work_sub_dir, output_full_correlation_dir
            )

    do_awaits(filename, process_counter)  # waits for aalcalc
    do_lwaits(filename, process_counter)  # waits for leccalc

    print_command(filename, 'rm -R -f work/*')
    if fifo_tmp_dir:
        print_command(
            filename, 'rm -R -f {}'.format(re.sub('fifo/$', '', fifo_queue_dir))
        )
    else:
        print_command(filename, 'rm -R -f fifo/*')

    if stderr_guard:
        print_command(filename, '')
        print_command(filename, '# Stop ktools watcher')
        print_command(filename, 'kill -9 $pid0')
