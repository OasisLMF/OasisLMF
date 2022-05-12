import copy
import contextlib
import io
import os
import pandas as pd
import random
import re
import multiprocessing
import string
import logging
from functools import partial
logger = logging.getLogger(__name__)

from collections import Counter

from ..utils.exceptions import OasisException
from ..utils.defaults import (
    KTOOLS_ALLOC_GUL_DEFAULT,
    KTOOLS_ALLOC_IL_DEFAULT,
    KTOOLS_ALLOC_RI_DEFAULT,
    KTOOL_N_GUL_PER_LB,
    KTOOL_N_FM_PER_LB,
    EVE_DEFAULT_SHUFFLE,
    EVE_NO_SHUFFLE,
    EVE_ROUND_ROBIN,
    EVE_FISHER_YATES,
    EVE_STD_SHUFFLE,
)

RUNTYPE_GROUNDUP_LOSS = 'gul'
RUNTYPE_LOAD_BALANCED_LOSS = 'lb'
RUNTYPE_INSURED_LOSS = 'il'
RUNTYPE_REINSURANCE_LOSS = 'ri'
RUNTYPE_FULL_CORRELATION = 'fc'

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

ORD_EPT_OUTPUT_SWITCHES = {
    "ept_full_uncertainty_aep": '-F',
    "ept_full_uncertainty_oep": '-f',
    "ept_mean_sample_aep": '-S',
    "ept_mean_sample_oep": '-s',
    "ept_per_sample_mean_aep": '-M',
    "ept_per_sample_mean_oep": '-m',
}

ORD_PSEPT_OUTPUT_SWITCHES = {
    "psept_aep": '-W',
    "psept_oep": '-w',
}

ORD_LECCALC = {**ORD_EPT_OUTPUT_SWITCHES, **ORD_PSEPT_OUTPUT_SWITCHES}

ORD_ALT_OUTPUT_SWITCHES = {
    "alt_period": {
        'csv_flag': '-o', 'parquet_flag': '-p'
    }
}

ORD_PLT_OUTPUT_SWITCHES = {
    "plt_sample": {
        'csv_flag': '-S', 'ktools_exe': 'pltcalc', 'table_name': 'splt',
        'parquet_flag': '-s', 'kat_flag': '-S'
    },
    "plt_quantile": {
        'csv_flag': '-Q', 'ktools_exe': 'pltcalc', 'table_name': 'qplt',
        'parquet_flag': '-q', 'kat_flag': '-Q'
    },
    "plt_moment": {
        'csv_flag': '-M', 'ktools_exe': 'pltcalc', 'table_name': 'mplt',
        'parquet_flag': '-m', 'kat_flag': '-M'
    }
}

ORD_ELT_OUTPUT_SWITCHES = {
    "elt_quantile": {
        'csv_flag': '-Q', 'ktools_exe': 'eltcalc', 'table_name': 'qelt',
        'parquet_flag': '-q', 'kat_flag': '-q'
    },
    "elt_moment": {
        'csv_flag': '-M', 'ktools_exe': 'eltcalc', 'table_name': 'melt',
        'parquet_flag': '-m', 'kat_flag': '-m'
    }
}

ORD_SELT_OUTPUT_SWITCH = {
    "elt_sample": {
        'csv_flag': '-o', 'ktools_exe': 'summarycalctocsv', 'table_name': 'selt',
        'parquet_flag': '-p', 'kat_flag': '-s'
    }
}

OUTPUT_SWITCHES = {
    "plt_ord": ORD_PLT_OUTPUT_SWITCHES,
    "elt_ord": ORD_ELT_OUTPUT_SWITCHES,
    "selt_ord": ORD_SELT_OUTPUT_SWITCH
}

EVE_SHUFFLE_OPTIONS = {
    EVE_NO_SHUFFLE: {'eve': '-n ', 'kat_sorting': False},
    EVE_ROUND_ROBIN: {'eve': '', 'kat_sorting': True},
    EVE_FISHER_YATES: {'eve': '-r ', 'kat_sorting': False},
    EVE_STD_SHUFFLE: {'eve': '-R ', 'kat_sorting': False}
}

SUMMARY_TYPES = ['eltcalc', 'summarycalc', 'pltcalc']


TRAP_FUNC = """
touch log/stderror.err
ktools_monitor.sh $$ & pid0=$!

exit_handler(){
   exit_code=$?

   # disable handler
   trap - QUIT HUP INT KILL TERM ERR EXIT

   kill -9 $pid0 2> /dev/null
   if [ "$exit_code" -gt 0 ]; then
       # Error - run process clean up
       echo 'Ktools Run Error - exitcode='$exit_code

       set +x
       group_pid=$(ps -p $$ -o pgid --no-headers)
       sess_pid=$(ps -p $$ -o sess --no-headers)
       script_pid=$$
       printf "Script PID:%d, GPID:%s, SPID:%d\n" $script_pid $group_pid $sess_pid >> log/killout.txt

       ps -jf f -g $sess_pid > log/subprocess_list
       PIDS_KILL=$(pgrep -a --pgroup $group_pid | awk \'BEGIN { FS = "[ \\t\\n]+" }{ if ($1 >= \'$script_pid\') print}\' | grep -v celery | egrep -v *\\\.log$  | egrep -v *\\\.sh$ | sort -n -r)
       echo "$PIDS_KILL" >> log/killout.txt
       kill -9 $(echo "$PIDS_KILL" | awk \'BEGIN { FS = "[ \\t\\n]+" }{ print $1 }\') 2>/dev/null
       exit $exit_code
   else
       # script successful
       exit 0
   fi
}
trap exit_handler QUIT HUP INT KILL TERM ERR EXIT"""

CHECK_FUNC = """
check_complete(){
    set +e
    proc_list="eve getmodel gulcalc fmcalc summarycalc eltcalc aalcalc leccalc pltcalc ordleccalc"
    has_error=0
    for p in $proc_list; do
        started=$(find log -name "$p*.log" | wc -l)
        finished=$(find log -name "$p*.log" -exec grep -l "finish" {} + | wc -l)
        if [ "$finished" -lt "$started" ]; then
            echo "[ERROR] $p - $((started-finished)) processes lost"
            has_error=1
        elif [ "$started" -gt 0 ]; then
            echo "[OK] $p"
        fi
    done
    if [ "$has_error" -ne 0 ]; then
        false # raise non-zero exit code
    else
        echo 'Run Completed'
    fi
}"""

BASH_TRACE = """
# --- Redirect Bash trace to file ---
bash_logging_supported(){
    local BASH_VER_MAJOR=${BASH_VERSION:0:1}
    local BASH_VER_MINOR=${BASH_VERSION:2:1}

    if [[ "$BASH_VER_MAJOR" -gt 4 ]]; then
        echo 1; exit
    fi
    if [[ $BASH_VER_MAJOR -eq 4 ]] && [[ $BASH_VER_MINOR -gt 3 ]]; then
        echo 1; exit
    fi
    echo 0
}
if [ $(bash_logging_supported) == 1 ]; then
    exec   > >(tee -ia log/bash.log)
    exec  2> >(tee -ia log/bash.log >& 2)
    exec 19> log/bash.log
    export BASH_XTRACEFD="19"
    set -x
else
    echo "WARNING: logging disabled, bash version '$BASH_VERSION' is not supported, minimum requirement is bash v4.4"
fi """


def process_range(max_process_id, process_number=None):
    """
    Creates an iterable for all the process ids, if process number is set
    then an iterable containing only that number is returned.

    This allows for the loss generation to be ran in different processes
    rather than accross multiple cores.

    :param max_process_id: The largest process number
    :param process_number: If set iterable only containing this number is returned
    :return: iterable containing all the process numbers to process
    """
    if process_number is not None:
        return [process_number]
    else:
        return range(1, max_process_id + 1)


def get_modelcmd(modelpy: bool, server=False) -> str:
    """
    Gets the construct model command line argument for the bash script.

    Args:
        modelpy: (bool) if the getmodel Python setting is True or not
        server: (bool) if set then enable 'TCP' ipc server/client mode

    Returns: C++ getmodel if modelpy is False, Python getmodel if the modelpy if False
    """
    py_cmd = 'modelpy'
    cpp_cmd = 'getmodel'

    if server is True:
        py_cmd = '{} --data-server'.format(py_cmd)

    if modelpy is True:
        return py_cmd
    else:
        return cpp_cmd


def get_fmcmd(fmpy, fmpy_low_memory=False, fmpy_sort_output=False):
    if fmpy:
        cmd = 'fmpy'
        if fmpy_low_memory:
            cmd += ' -l'
        if fmpy_sort_output:
            cmd += ' --sort-output'
        return cmd
    else:
        return 'fmcalc'


def print_command(command_file, cmd):
    """
    Writes the supplied command to the end of the generated script

    :param command_file: File to append command to.
    :param cmd: The command to append
    """
    with io.open(command_file, "a", encoding='utf-8') as myfile:
        myfile.writelines(cmd + "\n")


def leccalc_enabled(summary_options):
    """
    Checks if leccalc is enabled in a summaries section

    :param summary_options: Summaies section from an analysis_settings file
    :type summary_options: dict

    Example:
    {
        "aalcalc": true,
        "eltcalc": true,
        "id": 1,
        "lec_output": true,
        "leccalc": {
            "full_uncertainty_aep": true,
            "full_uncertainty_oep": true,
            "return_period_file": true
        }
    }
    :return: True is leccalc is enables, False otherwise.
    """

    lec_options = summary_options.get('leccalc', {})
    lec_boolean = summary_options.get('lec_output', False)

    # Disabled if leccalc flag is missing or false
    if not lec_boolean:
        return False

    # Backwards compatibility for nested "outputs" keys in lec_options
    if "outputs" in lec_options:
        lec_options = lec_options["outputs"]

    # Enabled if at least one option is selected
    for ouput_opt in lec_options:
        if ouput_opt in WAIT_PROCESSING_SWITCHES and lec_options[ouput_opt]:
            return True
    return False


def ord_enabled(summary_options, ORD_SWITCHES):
    """
    Checks if ORD leccalc is enabled in a summaries section

    :param summary_options: Summaies section from an analysis_settings file
    :type summary_options: dict

    :param ORD_SWITCHES: Options from the analysis_settings 'Summaies' section to search
    :type  ORD_SWITCHES: dict

    Example:
    {
        "id": 1,
        "ord_output": {
            "ept_full_uncertainty_aep": true,
            "ept_full_uncertainty_oep": true,
            "ept_mean_sample_aep": true,
            "ept_mean_sample_oep": true,
            "ept_per_sample_mean_aep": true,
            "ept_per_sample_mean_oep": true,
            "psept_aep": true,
            "psept_oep": true,
            "return_period_file": true
        }
    }

    :return: True is leccalc is enables, False otherwise.
    """

    ord_options = summary_options.get('ord_output', {})
    for ouput_opt in ord_options:
        if ouput_opt in ORD_SWITCHES and ord_options[ouput_opt]:
            return True
    return False



def do_post_wait_processing(
    runtype,
    analysis_settings,
    filename,
    process_counter,
    work_sub_dir='',
    output_dir='output/',
    stderr_guard=True
):
    if '{}_summaries'.format(runtype) not in analysis_settings:
        return

    for summary in analysis_settings['{}_summaries'.format(runtype)]:
        if "id" in summary:
            summary_set = summary['id']

            # ktools ORIG - aalcalc
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
                if stderr_guard:
                    cmd = '( {} ) 2>> log/stderror.err & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                else:
                    cmd = '{} & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                print_command(filename, cmd)


            # ORD - PALT
            if ord_enabled(summary, ORD_ALT_OUTPUT_SWITCHES):
                cmd = 'aalcalc -K{}{}_S{}_summary_palt'.format(
                    work_sub_dir, runtype, summary_set
                )
                palt_outfile_stem = '{}{}_S{}_palt'.format(
                    output_dir, runtype, summary_set
                )

                if summary.get('ord_output', {}).get('parquet_format'):
                    cmd = '{} {}'.format(
                        cmd,
                        ORD_ALT_OUTPUT_SWITCHES.get('alt_period', {}).get('parquet_flag', '')
                    )
                    cmd = '{} {}.parquet'.format(cmd, palt_outfile_stem)
                else:
                    cmd = '{} {}'.format(
                        cmd,
                        ORD_ALT_OUTPUT_SWITCHES.get('alt_period', {}).get('csv_flag', '')
                    )
                    cmd = '{} > {}.csv'.format(cmd, palt_outfile_stem)

                process_counter['lpid_monitor_count'] += 1
                if stderr_guard:
                    cmd = '( {} ) 2>> log/stderror.err & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                else:
                    cmd = '{} & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                print_command(filename, cmd)

            # ORD - PSEPT,EPT
            if ord_enabled(summary, ORD_LECCALC):

                ord_outputs = summary.get('ord_output', {})
                ept_output = False
                psept_output = False

                cmd = 'ordleccalc {} -K{}{}_S{}_summaryleccalc'.format(
                    '-r' if ord_outputs.get('return_period_file') else '',
                    work_sub_dir,
                    runtype,
                    summary_set
                )

                process_counter['lpid_monitor_count'] += 1
                for option, active in sorted(ord_outputs.items()):
                    # Add EPT switches
                    if active and option in ORD_EPT_OUTPUT_SWITCHES:
                        switch = ORD_EPT_OUTPUT_SWITCHES.get(option, '')
                        cmd = '{} {}'.format(cmd, switch)
                        if not ept_output:
                            ept_output = True

                    # Add PSEPT switches
                    if active and option in ORD_PSEPT_OUTPUT_SWITCHES:
                        switch = ORD_PSEPT_OUTPUT_SWITCHES.get(option, '')
                        cmd = '{} {}'.format(cmd, switch)
                        if not psept_output:
                            psept_output = True

                ept_output_flag = '-O'
                psept_output_flag = '-o'
                outfile_ext = 'csv'
                if summary.get('ord_output', {}).get('parquet_format'):
                    ept_output_flag = '-P'
                    psept_output_flag = '-p'
                    outfile_ext = 'parquet'

                if ept_output:
                    cmd = '{} {} {}{}_S{}_ept.{}'.format(
                        cmd, ept_output_flag, output_dir, runtype, summary_set,
                        outfile_ext
                    )

                if psept_output:
                    cmd = '{} {} {}{}_S{}_psept.{}'.format(
                        cmd, psept_output_flag, output_dir, runtype,
                        summary_set, outfile_ext
                    )

                if stderr_guard:
                    cmd = '( {} ) 2>> log/stderror.err & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                else:
                    cmd = '{} & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                print_command(filename, cmd)

            # ktools ORIG - Leccalc
            if leccalc_enabled(summary):
                leccalc = summary.get('leccalc', {})
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

                if stderr_guard:
                    cmd = '( {} ) 2>> log/stderror.err & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                else:
                    cmd = '{} & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                print_command(filename, cmd)


def get_fifo_name(fifo_dir, producer, producer_id, consumer=''):
    """Standard name for FIFO"""
    if consumer:
        return f'{fifo_dir}{producer}_{consumer}_P{producer_id}'
    else:
        return f'{fifo_dir}{producer}_P{producer_id}'


def do_fifo_exec(producer, producer_id, filename, fifo_dir, action='mkfifo', consumer=''):
    print_command(filename, f'{action} {get_fifo_name(fifo_dir, producer, producer_id, consumer)}')


def do_fifos_exec(runtype, max_process_id, filename, fifo_dir, process_number=None, action='mkfifo', consumer=''):
    for process_id in process_range(max_process_id, process_number):
        do_fifo_exec(runtype, process_id, filename, fifo_dir, action, consumer)
    print_command(filename, '')


def do_fifos_exec_full_correlation(
    runtype, max_process_id, filename, fifo_dir, process_number=None, action='mkfifo'):
    for process_id in process_range(max_process_id, process_number):
        print_command(filename, '{} {}{}_sumcalc_P{}'.format(
            action, fifo_dir, runtype, process_id
        ))
    print_command(filename, '')
    for process_id in process_range(max_process_id, process_number):
        print_command(filename, '{} {}{}_fmcalc_P{}'.format(
            action, fifo_dir, runtype, process_id
        ))
    print_command(filename, '')


def do_fifos_calc(runtype, analysis_settings, max_process_id, filename, fifo_dir='fifo/', process_number=None, action='mkfifo'):

    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    for process_id in process_range(max_process_id, process_number):
        for summary in summaries:
            if 'id' in summary:
                summary_set = summary['id']
                do_fifo_exec(runtype, process_id, filename, fifo_dir, action, f'S{summary_set}_summary')
                if leccalc_enabled(summary) or ord_enabled(summary, ORD_LECCALC) or summary.get('aalcalc') or ord_enabled(summary, ORD_ALT_OUTPUT_SWITCHES):
                    idx_fifo = get_fifo_name(fifo_dir, runtype, process_id, f'S{summary_set}_summary')
                    idx_fifo += '.idx'
                    print_command(filename, f'mkfifo {idx_fifo}')

                for summary_type in SUMMARY_TYPES:
                    if summary.get(summary_type):
                        do_fifo_exec(runtype, process_id, filename, fifo_dir, action, f'S{summary_set}_{summary_type}')

                for ord_type, output_switch in OUTPUT_SWITCHES.items():
                    for ord_table in output_switch.keys():
                        if summary.get('ord_output', {}).get(ord_table):
                            do_fifo_exec(runtype, process_id, filename, fifo_dir, action, f'S{summary_set}_{ord_type}')
                            break

        print_command(filename, '')


def create_workfolders(runtype, analysis_settings, filename, work_dir='work/'):
    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']

            # EDIT: leccalc and ordleccalc share the same summarycalc binary data
            # only create the workfolders once if either option is selected
            if leccalc_enabled(summary) or ord_enabled(summary, ORD_LECCALC):
                print_command(
                    filename,
                    'mkdir {}{}_S{}_summaryleccalc'.format(work_dir, runtype, summary_set)
                )

            if summary.get('aalcalc'):
                print_command(
                    filename,
                    'mkdir {}{}_S{}_summaryaalcalc'.format(work_dir, runtype, summary_set)
                )

            if summary.get('ord_output', {}).get('alt_period'):
                print_command(
                    filename,
                    'mkdir {}{}_S{}_summary_palt'.format(work_dir, runtype, summary_set)
                )


def do_kats(
    runtype,
    analysis_settings,
    max_process_id,
    filename,
    process_counter,
    work_dir='work/kat/',
    output_dir='output/',
    sort_by_event=False,
    process_number=None,
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

                cmd = 'kat' if sort_by_event else 'kat -u'
                for process_id in process_range(max_process_id, process_number):

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

                cmd = 'kat' if sort_by_event else 'kat -u'
                for process_id in process_range(max_process_id, process_number):
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

                cmd = 'kat' if sort_by_event else 'kat -u'
                for process_id in process_range(max_process_id, process_number):
                    cmd = '{} {}{}_S{}_summarycalc_P{}'.format(
                        cmd, work_dir, runtype, summary_set, process_id
                    )

                process_counter['kpid_monitor_count'] += 1
                cmd = '{} > {}{}_S{}_summarycalc.csv & kpid{}=$!'.format(
                    cmd, output_dir, runtype, summary_set,
                    process_counter['kpid_monitor_count']
                )
                print_command(filename, cmd)

            for ord_type, output_switch in OUTPUT_SWITCHES.items():
                for ord_table, v in output_switch.items():
                    if summary.get('ord_output', {}).get(ord_table):
                        anykats = True

                        cmd = 'kat' if sort_by_event else 'kat -u'
                        outfile_flag = '>'
                        outfile_ext = 'csv'
                        if summary.get('ord_output', {}).get('parquet_format'):
                            cmd = f'katparquet {v["kat_flag"]}'
                            outfile_flag = '-o'
                            outfile_ext = 'parquet'

                        for process_id in process_range(max_process_id, process_number):
                            cmd = f'{cmd} {work_dir}{runtype}_S{summary_set}_{ord_table}_P{process_id}'

                        process_counter['kpid_monitor_count'] += 1
                        cmd = f'{cmd} {outfile_flag} {output_dir}{runtype}_S{summary_set}_{v["table_name"]}.{outfile_ext}'
                        cmd = f'{cmd} & kpid{process_counter["kpid_monitor_count"]}=$!'
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
    gul_full_correlation=False
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

    input_filename_component = ''
    if gul_full_correlation:
        input_filename_component = '_sumcalc'

    # Use -m flag to create summary index files
    # This is likely to become default in future ktools releases
    cmd = 'summarycalc -m {} {}'.format(summarycalc_switch, summarycalc_directory_switch)
    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']
            cmd = '{0} -{1} {4}{2}_S{1}_summary_P{3}'.format(cmd, summary_set, runtype, process_id, fifo_dir)

    cmd = '{0} < {1}{2}{3}_P{4}'.format(
        cmd, fifo_dir, runtype, input_filename_component, process_id
    )
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


            cmd = f'tee < {get_fifo_name(fifo_dir, runtype, process_id, f"S{summary_set}_summary")}'
            if leccalc_enabled(summary) or ord_enabled(summary, ORD_LECCALC) or summary.get('aalcalc') or ord_enabled(summary, ORD_ALT_OUTPUT_SWITCHES):
                cmd_idx = cmd + '.idx'

            for summary_type in SUMMARY_TYPES:
                if summary.get(summary_type):
                    cmd = f'{cmd} {get_fifo_name(fifo_dir, runtype, process_id, f"S{summary_set}_{summary_type}")}'

            for ord_type, output_switch in OUTPUT_SWITCHES.items():
                for ord_table in output_switch.keys():
                    if summary.get('ord_output', {}).get(ord_table):
                        cmd = f'{cmd} {get_fifo_name(fifo_dir, runtype, process_id, f"S{summary_set}_{ord_type}")}'
                        break

            if summary.get('aalcalc'):
                aalcalc_out = f'{work_dir}{runtype}_S{summary_set}_summaryaalcalc/P{process_id}'
                cmd = f'{cmd} {aalcalc_out}.bin'
                cmd_idx = f'{cmd_idx} {aalcalc_out}.idx'

            if summary.get('ord_output', {}).get('alt_period'):
                aalcalc_ord_out = f'{work_dir}{runtype}_S{summary_set}_summary_palt/P{process_id}'
                cmd = f'{cmd} {aalcalc_ord_out}.bin'
                cmd_idx = f'{cmd_idx} {aalcalc_ord_out}.idx'


            # leccalc and ordleccalc share the same summarycalc binary data
            # only create the workfolders once if either option is selected
            if leccalc_enabled(summary) or ord_enabled(summary, ORD_LECCALC):
                leccalc_out = f'{work_dir}{runtype}_S{summary_set}_summaryleccalc/P{process_id}'
                cmd = f'{cmd} {leccalc_out}.bin'
                cmd_idx = f'{cmd_idx} {leccalc_out}.idx'

            cmd = '{} > /dev/null & pid{}=$!'.format(cmd, process_counter['pid_monitor_count'])
            print_command(filename, cmd)
            if leccalc_enabled(summary) or ord_enabled(summary, ORD_LECCALC) or summary.get('aalcalc') or ord_enabled(summary, ORD_ALT_OUTPUT_SWITCHES):
                process_counter['pid_monitor_count'] += 1
                cmd_idx = '{} > /dev/null & pid{}=$!'.format(cmd_idx, process_counter['pid_monitor_count'])
                print_command(filename, cmd_idx)


def do_tees_fc_sumcalc_fmcalc(process_id, filename, correlated_output_stems):

    if process_id == 1:
        print_command(filename, '')

    cmd = 'tee < {0}{1}'.format(
        correlated_output_stems['gulcalc_output'], process_id
    )
    cmd = '{0} {1}{3} {2}{3} > /dev/null &'.format(
        cmd,
        correlated_output_stems['sumcalc_input'],
        correlated_output_stems['fmcalc_input'],
        process_id
    )

    print_command(filename, cmd)


def get_correlated_output_stems(fifo_dir):

    correlated_output_stems = {}
    correlated_output_stems['gulcalc_output'] = '{0}{1}_P'.format(
        fifo_dir, RUNTYPE_GROUNDUP_LOSS
    )
    correlated_output_stems['fmcalc_input'] = '{0}{1}_fmcalc_P'.format(
        fifo_dir, RUNTYPE_GROUNDUP_LOSS
    )
    correlated_output_stems['sumcalc_input'] = '{0}{1}_sumcalc_P'.format(
        fifo_dir, RUNTYPE_GROUNDUP_LOSS
    )

    return correlated_output_stems


def do_ord(runtype, analysis_settings, process_id, filename, process_counter, fifo_dir='fifo/', work_dir='work/', stderr_guard=True):
    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    if process_id == 1:
        print_command(filename, '')

    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']
            for ord_type, output_switch in OUTPUT_SWITCHES.items():
                cmd = ''
                fifo_out_name = ''
                skip_line = True
                for ord_table, flag_proc in output_switch.items():
                    if summary.get('ord_output', {}).get(ord_table):

                        if process_id != 1 and skip_line:
                            if ord_type == 'plt_ord':
                                cmd += ' -H'
                            else:
                                cmd += ' -s'
                            skip_line = False

                        if summary.get('ord_output', {}).get('parquet_format'):
                            cmd += f' {flag_proc["parquet_flag"]}'
                        else:
                            cmd += f' {flag_proc["csv_flag"]}'

                        fifo_out_name = get_fifo_name(f'{work_dir}kat/', runtype, process_id, f'S{summary_set}_{ord_table}')
                        if ord_type != 'selt_ord' or summary.get('ord_output', {}).get('parquet_format'):
                            cmd = f'{cmd} {fifo_out_name}'

                if cmd:
                    fifo_in_name = get_fifo_name(fifo_dir, runtype, process_id, f'S{summary_set}_{ord_type}')
                    cmd = f'{cmd} < {fifo_in_name}'
                    if ord_type == 'selt_ord' and not summary.get('ord_output', {}).get('parquet_format'):
                        cmd = f'{cmd} > {fifo_out_name}'
                    process_counter['pid_monitor_count'] += 1
                    cmd = f'{flag_proc["ktools_exe"]}{cmd}'
                    if stderr_guard:
                        cmd = f'( {cmd} ) 2>> log/stderror.err & pid{process_counter["pid_monitor_count"]}=$!'
                    else:
                        cmd = f'{cmd} & pid{process_counter["pid_monitor_count"]}=$!'

                    print_command(filename, cmd)


def do_any(runtype, analysis_settings, process_id, filename, process_counter, fifo_dir='fifo/', work_dir='work/', stderr_guard=True):
    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    if process_id == 1:
        print_command(filename, '')

    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']
            for summary_type in SUMMARY_TYPES:
                if summary.get(summary_type):
                    #cmd exception for summarycalc
                    if summary_type == 'summarycalc':
                        cmd = 'summarycalctocsv'
                    else:
                        cmd = summary_type

                    if process_id != 1:
                        if summary_type == 'pltcalc':
                            cmd += ' -H'
                        else:
                            cmd += ' -s'

                    process_counter['pid_monitor_count'] += 1

                    fifo_in_name = get_fifo_name(fifo_dir, runtype, process_id, f'S{summary_set}_{summary_type}')
                    fifo_out_name = get_fifo_name(f'{work_dir}kat/', runtype, process_id, f'S{summary_set}_{summary_type}')
                    cmd = f'{cmd} < {fifo_in_name} > {fifo_out_name}'

                    if stderr_guard:
                        cmd = f'( {cmd} ) 2>> log/stderror.err & pid{process_counter["pid_monitor_count"]}=$!'
                    else:
                        cmd = f'{cmd} & pid{process_counter["pid_monitor_count"]}=$!'

                    print_command(filename, cmd)


def ri(analysis_settings, max_process_id, filename, process_counter, num_reinsurance_iterations, fifo_dir='fifo/', work_dir='work/', stderr_guard=True, process_number=None):
    for process_id in process_range(max_process_id, process_number):
        do_any(RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir, stderr_guard)

    for process_id in process_range(max_process_id, process_number):
        do_ord(RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir, stderr_guard)

    for process_id in process_range(max_process_id, process_number):
        do_tees(RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir)

    # TODO => insert server here

    for process_id in process_range(max_process_id, process_number):
        do_summarycalcs(
            runtype=RUNTYPE_REINSURANCE_LOSS,
            analysis_settings=analysis_settings,
            process_id=process_id,
            filename=filename,
            fifo_dir=fifo_dir,
            stderr_guard=stderr_guard,
            num_reinsurance_iterations=num_reinsurance_iterations,
        )


def il(analysis_settings, max_process_id, filename, process_counter, fifo_dir='fifo/', work_dir='work/', stderr_guard=True, process_number=None):
    for process_id in process_range(max_process_id, process_number):
        do_any(RUNTYPE_INSURED_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir, stderr_guard)

    for process_id in process_range(max_process_id, process_number):
        do_ord(RUNTYPE_INSURED_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir, stderr_guard)

    for process_id in process_range(max_process_id, process_number):
        do_tees(RUNTYPE_INSURED_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir)

    for process_id in process_range(max_process_id, process_number):
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
    gul_legacy_stream=None,
    stderr_guard=True,
    process_number=None,
):

    for process_id in process_range(max_process_id, process_number):
        do_any(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir, stderr_guard)

    for process_id in process_range(max_process_id, process_number):
        do_ord(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir, stderr_guard)

    for process_id in process_range(max_process_id, process_number):
        do_tees(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir)

    for process_id in process_range(max_process_id, process_number):
        do_summarycalcs(
            runtype=RUNTYPE_GROUNDUP_LOSS,
            analysis_settings=analysis_settings,
            process_id=process_id,
            filename=filename,
            gul_legacy_stream=gul_legacy_stream,
            fifo_dir=fifo_dir,
            stderr_guard=stderr_guard
        )


def do_gul_full_correlation(
    analysis_settings,
    max_process_id,
    filename,
    process_counter,
    fifo_dir='fifo/full_correlation/',
    work_dir='work/full_correlation/',
    gul_legacy_stream=None,
    stderr_guard=None,
    process_number=None,
):

    for process_id in process_range(max_process_id, process_number):
        do_any(
            RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename,
            process_counter, fifo_dir, work_dir
        )

    for process_id in process_range(max_process_id, process_number):
        do_tees(
            RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename,
            process_counter, fifo_dir, work_dir
        )

    print_command(filename, '')


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


def get_getmodel_itm_cmd(
        number_of_samples,
        gul_threshold,
        use_random_number_file,
        gul_alloc_rule,
        item_output,
        process_id,
        max_process_id,
        correlated_output,
        eve_shuffle_flag,
        modelpy=False,
        modelpy_server=False,
        **kwargs):
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
    :param eve_shuffle_flag: The event shuffling rule
    :type eve_shuffle_flag: str
    :return: The generated getmodel command
    """
    cmd = f'eve {eve_shuffle_flag}{process_id} {max_process_id} | {get_modelcmd(modelpy, modelpy_server)} | gulcalc -S{number_of_samples} -L{gul_threshold}'

    if use_random_number_file:
        cmd = '{} -r'.format(cmd)
    if correlated_output != '':
        cmd = '{} -j {}'.format(cmd, correlated_output)
    cmd = '{} -a{} -i {}'.format(cmd, gul_alloc_rule, item_output)
    return cmd


def get_getmodel_cov_cmd(
        number_of_samples,
        gul_threshold,
        use_random_number_file,
        coverage_output,
        item_output,
        process_id,
        max_process_id,
        eve_shuffle_flag,
        modelpy=False,
        modelpy_server=False,
        **kwargs) -> str:
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
    :param eve_shuffle_flag: The event shuffling rule
    :type  eve_shuffle_flag: str
    :return: (str) The generated getmodel command
    """

    cmd = f'eve {eve_shuffle_flag}{process_id} {max_process_id} | {get_modelcmd(modelpy, modelpy_server)} | gulcalc -S{number_of_samples} -L{gul_threshold}'

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
    from_file=False,
    fmpy=True,
    fmpy_low_memory=False,
    fmpy_sort_output=False,
    step_flag=''
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
    :param from_file: must be true if cmd is a file and false if it can be piped
    :type from_file: bool
    """
    if from_file:
        main_cmd = f'{get_fmcmd(fmpy, fmpy_low_memory, fmpy_sort_output)} -a{il_alloc_rule}{step_flag} < {cmd}'
    else:
        main_cmd = f'{cmd} | {get_fmcmd(fmpy, fmpy_low_memory, fmpy_sort_output)} -a{il_alloc_rule}{step_flag}'

    if il_output:
        main_cmd += f" | tee {get_fifo_name(fifo_dir, RUNTYPE_INSURED_LOSS, process_id)}"

    for i in range(1, num_reinsurance_iterations + 1):
        main_cmd += f" | {get_fmcmd(fmpy, fmpy_low_memory, fmpy_sort_output)} -a{ri_alloc_rule} -n -p RI_{i}"

    ri_fifo_name = get_fifo_name(fifo_dir, RUNTYPE_REINSURANCE_LOSS, process_id)
    main_cmd += f" > {ri_fifo_name}"
    main_cmd = f'( {main_cmd} ) 2>> log/stderror.err &' if stderr_guard else f'{main_cmd} &'

    return main_cmd


def get_main_cmd_il_stream(
    cmd,
    process_id,
    il_alloc_rule,
    fifo_dir='fifo/',
    stderr_guard=True,
    from_file=False,
    fmpy=True,
    fmpy_low_memory=False,
    fmpy_sort_output=False,
    step_flag=''
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
    :param from_file: must be true if cmd is a file and false if it can be piped
    :type from_file: bool
    :return: generated fmcalc command as str
    """

    il_fifo_name = get_fifo_name(fifo_dir, RUNTYPE_INSURED_LOSS, process_id)

    if from_file:
        main_cmd = f'{get_fmcmd(fmpy, fmpy_low_memory, fmpy_sort_output)} -a{il_alloc_rule}{step_flag} < {cmd} > {il_fifo_name}'
    else:
        main_cmd = f'{cmd} | {get_fmcmd(fmpy, fmpy_low_memory, fmpy_sort_output)} -a{il_alloc_rule}{step_flag} > {il_fifo_name} '#need extra space at the end to pass test

    main_cmd = f'( {main_cmd} ) 2>> log/stderror.err &' if stderr_guard else f'{main_cmd} &'

    return main_cmd


def get_main_cmd_gul_stream(
    cmd,
    process_id,
    fifo_dir='fifo/',
    stderr_guard=True,
    consumer='',
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
    :param consumer: optional name of the consumer of the stream
    :type consumer: string
    :return: generated command as str
    """

    gul_fifo_name = get_fifo_name(fifo_dir, RUNTYPE_GROUNDUP_LOSS, process_id, consumer)
    main_cmd = f'{cmd} > {gul_fifo_name} '
    main_cmd = f'( {main_cmd} ) 2>> log/stderror.err &' if stderr_guard else f'{main_cmd} &'

    return main_cmd


def get_complex_model_cmd(custom_gulcalc_cmd, analysis_settings):
    # If `given_gulcalc_cmd` is set then always run as a complex model
    # and raise an exception when not found in PATH
    if custom_gulcalc_cmd:
        if not shutil.which(custom_gulcalc_cmd):
            raise OasisException(
                'Run error: Custom Gulcalc command "{}" explicitly set but not found in path.'.format(custom_gulcalc_cmd)
            )
    # when not set then fallback to previous behaviour:
    # Check if a custom binary `<supplier>_<model>_gulcalc` exists in PATH
    else:
        inferred_gulcalc_cmd = "{}_{}_gulcalc".format(
            analysis_settings.get('model_supplier_id'),
            analysis_settings.get('model_name_id'))
        if shutil.which(inferred_gulcalc_cmd):
            custom_gulcalc_cmd = inferred_gulcalc_cmd

    if custom_gulcalc_cmd:
        def custom_get_getmodel_cmd(
            number_of_samples,
            gul_threshold,
            use_random_number_file,
            coverage_output,
            item_output,
            process_id,
            max_process_id,
            gul_alloc_rule,
            stderr_guard,
            **kwargs
        ):
            cmd = "{} -e {} {} -a {} -p {}".format(
                custom_gulcalc_cmd,
                process_id,
                max_process_id,
                os.path.abspath("analysis_settings.json"),
                "input")
            if gul_legacy_stream and coverage_output != '':
                cmd = '{} -c {}'.format(cmd, coverage_output)
            if item_output != '':
                cmd = '{} -i {}'.format(cmd, item_output)
            if stderr_guard:
                cmd = '({}) 2>> log/gul_stderror.err'.format(cmd)

            return cmd
    else:
        custom_get_getmodel_cmd = None
    return custom_get_getmodel_cmd


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


def get_main_cmd_lb(num_lb, num_in_per_lb, num_out_per_lb, get_input_stream_name, get_output_stream_name, stderr_guard):
    in_id = 1
    out_id = 1
    for _ in range(num_lb):
        lb_in_l = []
        for _ in range(num_in_per_lb):
            lb_in_fifo_name = get_input_stream_name(producer_id=in_id)
            in_id += 1
            lb_in_l.append(lb_in_fifo_name)
        lb_in = ' '.join(lb_in_l)

        lb_out_l = []
        for _ in range(num_out_per_lb):
            lb_out_fifo_name = get_output_stream_name(producer_id=out_id)
            out_id += 1
            lb_out_l.append(lb_out_fifo_name)
        lb_out = ' '.join(lb_out_l)

        lb_main_cmd = f"load_balancer -i {lb_in} -o {lb_out}"
        lb_main_cmd = f'( {lb_main_cmd} ) 2>> log/stderror.err &' if stderr_guard else f'{lb_main_cmd} &'
        yield lb_main_cmd





def bash_params(
    analysis_settings,
    max_process_id=-1,
    number_of_processes=-1,
    num_reinsurance_iterations=0,
    fifo_tmp_dir=True,
    gul_alloc_rule=None,
    il_alloc_rule=None,
    ri_alloc_rule=None,
    num_gul_per_lb=None,
    num_fm_per_lb=None,
    stderr_guard=True,
    gul_legacy_stream=False,
    bash_trace=False,
    filename='run_kools.sh',
    _get_getmodel_cmd=None,
    custom_gulcalc_cmd=None,
    custom_args={},
    fmpy=True,
    fmpy_low_memory=False,
    fmpy_sort_output=False,
    event_shuffle=None,
    modelpy=False,

    ## new options
    process_number=None,
    remove_working_files=True,
    model_run_dir='',
    model_py_server=False,
    **kwargs
):



    bash_params = {}
    bash_params['max_process_id'] = max_process_id if max_process_id > 0 else multiprocessing.cpu_count()
    bash_params['number_of_processes'] = number_of_processes if number_of_processes > 0 else multiprocessing.cpu_count()
    bash_params['process_counter'] = Counter()
    bash_params['num_reinsurance_iterations'] = num_reinsurance_iterations
    bash_params['fifo_tmp_dir'] = fifo_tmp_dir
    bash_params['gul_legacy_stream'] = gul_legacy_stream
    bash_params['bash_trace'] = bash_trace
    bash_params['filename'] = filename
    bash_params['custom_args'] = custom_args
    bash_params['modelpy'] = modelpy
    bash_params['fmpy'] = fmpy
    bash_params['fmpy_low_memory'] = fmpy_low_memory
    bash_params['fmpy_sort_output'] = fmpy_sort_output
    bash_params['process_number'] = process_number
    bash_params['remove_working_files'] = remove_working_files
    bash_params['model_run_dir'] = model_run_dir
    bash_params['gul_threshold'] = analysis_settings.get('gul_threshold', 0)
    bash_params['number_of_samples'] = analysis_settings.get('number_of_samples', 0)
    bash_params["static_path"] = os.path.join(model_run_dir, "static/")

    bash_params["model_py_server"] = model_py_server
    if model_py_server:
        bash_params['modelpy'] = True


    # set complex model gulcalc command
    if not _get_getmodel_cmd and custom_gulcalc_cmd:
        bash_params['_get_getmodel_cmd'] = get_complex_model_cmd(custom_gulcalc_cmd, analysis_settings)
    else:
        bash_params['_get_getmodel_cmd'] = _get_getmodel_cmd

    ## Set fifo dirs
    if fifo_tmp_dir:
        bash_params['fifo_queue_dir'] = '/tmp/{}/fifo/'.format(''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)))
    else:
        bash_params['fifo_queue_dir'] = os.path.join(model_run_dir, 'fifo/')

    ## set dirs
    bash_params['stderr_guard'] = stderr_guard
    bash_params['gul_item_stream'] = not gul_legacy_stream
    bash_params['work_dir'] = os.path.join(model_run_dir, 'work/')
    bash_params['work_kat_dir'] = os.path.join(model_run_dir, 'work/kat/')
    bash_params['work_full_correlation_dir'] = os.path.join(model_run_dir, 'work/full_correlation/')
    bash_params['work_full_correlation_kat_dir'] = os.path.join(model_run_dir, 'work/full_correlation/kat/')
    bash_params['output_dir'] = os.path.join(model_run_dir, 'output/')
    bash_params['output_full_correlation_dir'] = os.path.join(model_run_dir, 'output/full_correlation/')
    bash_params['fifo_full_correlation_dir'] = os.path.join(bash_params['fifo_queue_dir'], 'full_correlation/')

    # Set default alloc/shuffle rules if missing
    bash_params['gul_alloc_rule'] = gul_alloc_rule if isinstance(gul_alloc_rule, int) else KTOOLS_ALLOC_GUL_DEFAULT
    bash_params['il_alloc_rule'] = il_alloc_rule if isinstance(il_alloc_rule, int) else KTOOLS_ALLOC_IL_DEFAULT
    bash_params['ri_alloc_rule'] = ri_alloc_rule if isinstance(ri_alloc_rule, int) else KTOOLS_ALLOC_RI_DEFAULT
    bash_params['num_gul_per_lb'] = num_gul_per_lb if isinstance(num_gul_per_lb, int) else KTOOL_N_GUL_PER_LB
    bash_params['num_fm_per_lb'] = num_fm_per_lb if isinstance(num_fm_per_lb, int) else KTOOL_N_FM_PER_LB

    # Get event shuffle flags
    event_shuffle_rule = event_shuffle if isinstance(event_shuffle, int) else EVE_DEFAULT_SHUFFLE
    bash_params['event_shuffle'] = event_shuffle_rule
    if event_shuffle_rule in EVE_SHUFFLE_OPTIONS:
        bash_params['eve_shuffle_flag'] = EVE_SHUFFLE_OPTIONS[event_shuffle_rule]['eve']
        bash_params['kat_sort_by_event'] = EVE_SHUFFLE_OPTIONS[event_shuffle_rule]['kat_sorting']
    else:
        raise OasisException(f'Error: Unknown event shuffle rule "{event_shuffle}" expected value between [0..{EVE_STD_SHUFFLE}]')

    # set random num file option
    use_random_number_file = False
    if 'model_settings' in analysis_settings and analysis_settings['model_settings'].get('use_random_number_file'):
        use_random_number_file = True
    bash_params['use_random_number_file'] = use_random_number_file

    # set full_correlation option
    full_correlation = False
    if 'full_correlation' in analysis_settings:
        if _get_getmodel_cmd is None and bash_params['gul_item_stream']:
            full_correlation = analysis_settings['full_correlation']
    bash_params['full_correlation'] = full_correlation

    # Output depends on being enabled AND having at least one summaries section
    # checking output settings coherence
    for mod in ['gul', 'il', 'ri']:
        if analysis_settings.get(f'{mod}_output') and not analysis_settings.get(f'{mod}_summaries'):
            logger.warning(f'{mod}_output set to True but there is no {mod}_summaries')
            analysis_settings[f'{mod}_output'] = False

    # additional check for ri, must have num_reinsurance_iterations
    if analysis_settings.get('ri_output', False) and not num_reinsurance_iterations:
        logger.warning('ri_output set to True but there is no reinsurance layers')
        analysis_settings['ri_output'] = False

    if not any(analysis_settings.get(f'{mod}_output') for mod in ['gul', 'il', 'ri']):
        raise OasisException('No valid output settings')

    ## Get perfecting values from 'analysis_settings' settings)
    bash_params['analysis_settings'] = analysis_settings
    bash_params['gul_output'] = analysis_settings.get('gul_output', False)
    bash_params['il_output'] = analysis_settings.get('il_output', False)
    bash_params['ri_output'] = analysis_settings.get('ri_output', False)
    bash_params['need_summary_fifo_for_gul'] = bash_params['gul_output'] and (
        bash_params['il_output'] or bash_params['ri_output']
    )

    return bash_params


@contextlib.contextmanager
def bash_wrapper(filename, bash_trace, stderr_guard):
    # Header
    print_command(filename, '#!/bin/bash')
    print_command(filename, 'SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")')
    print_command(filename, '')
    print_command(filename, '# --- Script Init ---')
    print_command(filename, 'set -euET -o pipefail')
    print_command(filename, 'shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."')

    print_command(filename, '')
    print_command(filename, 'mkdir -p log')
    print_command(filename, 'rm -R -f log/*')
    print_command(filename, '')

    # Trap func and logging
    if bash_trace:
        print_command(filename, BASH_TRACE)
    if stderr_guard:
        print_command(filename, TRAP_FUNC)
        print_command(filename, CHECK_FUNC)

    # Script content
    yield

    # Script footer
    if stderr_guard:
        print_command(filename, '')
        print_command(filename, 'check_complete')


def create_bash_analysis(
    process_counter,
    max_process_id,
    num_reinsurance_iterations,
    fifo_tmp_dir,
    gul_legacy_stream,
    bash_trace,
    filename,
    _get_getmodel_cmd,
    custom_args,
    fmpy,
    fmpy_low_memory,
    fmpy_sort_output,
    process_number,
    remove_working_files,
    model_run_dir,
    fifo_queue_dir,
    fifo_full_correlation_dir,
    stderr_guard,
    gul_item_stream,
    work_dir,
    work_kat_dir,
    work_full_correlation_dir,
    work_full_correlation_kat_dir,
    output_dir,
    output_full_correlation_dir,
    gul_alloc_rule,
    il_alloc_rule,
    ri_alloc_rule,
    num_gul_per_lb,
    num_fm_per_lb,
    event_shuffle,
    eve_shuffle_flag,
    kat_sort_by_event,
    gul_threshold,
    number_of_samples,
    use_random_number_file,
    full_correlation,
    gul_output,
    il_output,
    ri_output,
    need_summary_fifo_for_gul,
    analysis_settings,
    modelpy,
    model_py_server,
    **kwargs
):

    process_counter = process_counter or Counter()
    custom_args = custom_args or {}

    # WORKAROUND: Disable load balancer if chunk number is set
    # remove this limit when fixed -- need to support the load balancer + analysis chunks
    if process_number is not None:
        num_gul_per_lb = 0
        num_fm_per_lb = 0

    print_command(filename, '# --- Setup run dirs ---')
    print_command(filename, '')
    print_command(filename, "find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +")
    if full_correlation:
        print_command(filename, 'mkdir {}'.format(output_full_correlation_dir))

    print_command(filename, '')
    if not fifo_tmp_dir:
        print_command(filename, 'rm -R -f {}*'.format(fifo_queue_dir))
        if full_correlation:
            print_command( filename, 'mkdir {}'.format(fifo_full_correlation_dir))

    print_command(filename, 'rm -R -f {}*'.format(work_dir))
    print_command(filename, 'mkdir {}'.format(work_kat_dir))
    if full_correlation:
        print_command(filename, 'mkdir {}'.format(work_full_correlation_dir))
        print_command(
            filename, 'mkdir {}'.format(work_full_correlation_kat_dir)
        )

    if model_py_server:
        print_command(filename, '# --- run data server ---')
        print_command(command_file=filename, cmd=f"servedata {kwargs['static_path']} {max_process_id} &")
        # print_command(command_file=filename, cmd="while ! nc -vz localhost 8080 < /dev/null > /dev/null 2>&1; do")
        # print_command(command_file=filename, cmd="  printf '.'")
        # print_command(command_file=filename, cmd="  sleep 2")
        # print_command(command_file=filename, cmd="done")

    print_command(filename, '')

    if fmpy:
        if il_output or ri_output:
            print_command(
                filename, f'{get_fmcmd(fmpy)} -a{il_alloc_rule} --create-financial-structure-files'
            )
        if ri_output:
            for i in range(1, num_reinsurance_iterations + 1):
                print_command(filename, f'{get_fmcmd(fmpy)} -a{ri_alloc_rule} --create-financial-structure-files -p RI_{i}')

    # Create FIFOS under /tmp/* (Windows support)
    if fifo_tmp_dir:

        ## workaround to match bash tests
        if fifo_queue_dir.endswith('fifo/'):
            print_command(filename, 'rm -R -f {}'.format(fifo_queue_dir[:-5]))
        else:
            print_command(filename, 'rm -R -f {}'.format(fifo_queue_dir))

        print_command(filename, 'mkdir -p {}'.format(fifo_queue_dir))
        if full_correlation:
            print_command(filename, 'mkdir {}'.format(fifo_full_correlation_dir))

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

    # infer number of calc block and FIFO to create, (no load balancer for old stream option)
    if num_gul_per_lb and num_fm_per_lb and (il_output or ri_output) and gul_item_stream:
        block_process_size = num_gul_per_lb + (num_fm_per_lb * (2 if ri_output else 1))
        num_lb = (max_process_id - 1) // block_process_size + 1
        num_gul_output = num_lb * num_gul_per_lb
        num_fm_output = num_lb * num_fm_per_lb
    else:
        num_lb = 0
        num_gul_output = num_fm_output = max_process_id

    fifo_dirs = [fifo_queue_dir]
    if full_correlation:
        fifo_dirs.append(fifo_full_correlation_dir)
        if (il_output or ri_output) and (gul_output or not num_lb):
            #create fifo for il or ri full correlation compute
            do_fifos_exec(RUNTYPE_GROUNDUP_LOSS, num_gul_output, filename, fifo_full_correlation_dir, process_number, consumer=RUNTYPE_FULL_CORRELATION)

    for fifo_dir in fifo_dirs:
        # create fifos for Summarycalc
        if gul_output:
            do_fifos_exec(RUNTYPE_GROUNDUP_LOSS, num_gul_output, filename, fifo_dir, process_number)
            do_fifos_calc(RUNTYPE_GROUNDUP_LOSS, analysis_settings, num_gul_output, filename, fifo_dir, process_number)
        if il_output:
            do_fifos_exec(RUNTYPE_INSURED_LOSS, num_fm_output, filename, fifo_dir, process_number)
            do_fifos_calc(RUNTYPE_INSURED_LOSS, analysis_settings, num_fm_output, filename, fifo_dir, process_number)
        if ri_output:
            do_fifos_exec(RUNTYPE_REINSURANCE_LOSS, num_fm_output, filename, fifo_dir, process_number)
            do_fifos_calc(RUNTYPE_REINSURANCE_LOSS, analysis_settings, num_fm_output, filename, fifo_dir, process_number)

        # create fifos for Load balancer
        if num_lb:
            do_fifos_exec(RUNTYPE_GROUNDUP_LOSS, num_gul_output, filename, fifo_dir, process_number, consumer=RUNTYPE_LOAD_BALANCED_LOSS)
            do_fifos_exec(RUNTYPE_LOAD_BALANCED_LOSS, num_fm_output, filename, fifo_dir, process_number, consumer=RUNTYPE_INSURED_LOSS)

    print_command(filename, '')
    dirs = [(fifo_queue_dir, work_dir)]
    if full_correlation:
        dirs.append((fifo_full_correlation_dir, work_full_correlation_dir))

    compute_outputs = []
    for (_fifo_dir, _work_dir) in dirs:  # create Summarycalc
        if ri_output:
            ri_computes = {
                'loss_type': 'reinsurance',
                'compute_fun': ri,
                'compute_args': {
                    'analysis_settings': analysis_settings,
                    'max_process_id': num_fm_output,
                    'filename': filename,
                    'process_counter': process_counter,
                    'num_reinsurance_iterations': num_reinsurance_iterations,
                    'fifo_dir': _fifo_dir,
                    'work_dir': _work_dir,
                    'stderr_guard': stderr_guard,
                    'process_number': process_number
                }
            }
            compute_outputs.append(ri_computes)

        if il_output:
            il_computes = {
                'loss_type': 'insured',
                'compute_fun': il,
                'compute_args': {
                    'analysis_settings': analysis_settings,
                    'max_process_id': num_fm_output,
                    'filename': filename,
                    'process_counter': process_counter,
                    'fifo_dir': _fifo_dir,
                    'work_dir': _work_dir,
                    'stderr_guard': stderr_guard,
                    'process_number': process_number
                }
            }
            compute_outputs.append(il_computes)

        if gul_output:
            gul_computes = {
                'loss_type': 'ground up',
                'compute_fun': do_gul,
                'compute_args': {
                    'analysis_settings': analysis_settings,
                    'max_process_id': num_gul_output,
                    'filename': filename,
                    'process_counter': process_counter,
                    'fifo_dir': _fifo_dir,
                    'work_dir': _work_dir,
                    'gul_legacy_stream': gul_legacy_stream,
                    'stderr_guard': stderr_guard,
                    'process_number': process_number
                }
            }
            compute_outputs.append(gul_computes)

    do_computes(compute_outputs)

    print_command(filename, '')

    # create all gul streams
    get_gul_stream_cmds = {}

    # WARNING: this probably wont work well with the load balancer (needs guard/ edit)
    #for gul_id in range(1, num_gul_output + 1):
    for gul_id in process_range(num_gul_output, process_number):
        getmodel_args = {
            'number_of_samples': number_of_samples,
            'gul_threshold': gul_threshold,
            'use_random_number_file': use_random_number_file,
            'gul_alloc_rule': gul_alloc_rule,
            'gul_legacy_stream': gul_legacy_stream,
            'process_id': gul_id,
            'max_process_id': num_gul_output,
            'stderr_guard': stderr_guard,
            'eve_shuffle_flag': eve_shuffle_flag,
            'modelpy': modelpy,
            'modelpy_server': model_py_server,
        }

        # GUL coverage & item stream (Older)
        gul_fifo_name = get_fifo_name(fifo_queue_dir, RUNTYPE_GROUNDUP_LOSS, gul_id)
        if gul_item_stream:
            if need_summary_fifo_for_gul:
                getmodel_args['coverage_output'] = ''
                getmodel_args['item_output'] = f'- | tee {gul_fifo_name}'
            else:
                getmodel_args['coverage_output'] = ''
                getmodel_args['item_output'] = '-'
            _get_getmodel_cmd = (_get_getmodel_cmd or get_getmodel_itm_cmd)
        else:
            if need_summary_fifo_for_gul:
                getmodel_args['coverage_output'] = f'{gul_fifo_name}'
                getmodel_args['item_output'] = '-'
            elif gul_output:# only gul direct stdout to summary
                getmodel_args['coverage_output'] = '-'
                getmodel_args['item_output'] = ''
            else:# direct stdout to il
                getmodel_args['coverage_output'] = ''
                getmodel_args['item_output'] = '-'
            _get_getmodel_cmd = (_get_getmodel_cmd or get_getmodel_cov_cmd)

        # gulcalc output file for fully correlated output
        if full_correlation:
            fc_gul_fifo_name = get_fifo_name(fifo_full_correlation_dir, RUNTYPE_GROUNDUP_LOSS, gul_id)
            if need_summary_fifo_for_gul:# need both stream for summary and tream for il
                getmodel_args['correlated_output'] = get_fifo_name(fifo_full_correlation_dir, RUNTYPE_GROUNDUP_LOSS, gul_id,
                                                                   consumer=RUNTYPE_FULL_CORRELATION)
                if num_lb:
                    tee_output = get_fifo_name(fifo_full_correlation_dir, RUNTYPE_GROUNDUP_LOSS, gul_id,
                                                                   consumer=RUNTYPE_LOAD_BALANCED_LOSS)
                    tee_cmd = f"tee < {getmodel_args['correlated_output']} {fc_gul_fifo_name} > {tee_output} &"
                    print_command(filename, tee_cmd)

                else:
                    tee_output = get_fifo_name(fifo_full_correlation_dir, RUNTYPE_GROUNDUP_LOSS, gul_id,
                                                                   consumer=RUNTYPE_INSURED_LOSS)
                    tee_cmd = f"tee < {getmodel_args['correlated_output']} {fc_gul_fifo_name} "
                    get_gul_stream_cmds.setdefault(fifo_full_correlation_dir, []).append((tee_cmd, False))

            elif gul_output:# only gul direct correlated_output to summary
                getmodel_args['correlated_output'] = fc_gul_fifo_name
            else:
                if num_lb:
                    getmodel_args['correlated_output'] = get_fifo_name(fifo_full_correlation_dir, RUNTYPE_GROUNDUP_LOSS, gul_id,
                                  consumer=RUNTYPE_LOAD_BALANCED_LOSS)

                else:
                    getmodel_args['correlated_output'] = get_fifo_name(fifo_full_correlation_dir, RUNTYPE_GROUNDUP_LOSS, gul_id,
                                                                   consumer=RUNTYPE_FULL_CORRELATION)
                    get_gul_stream_cmds.setdefault(fifo_full_correlation_dir, []).append((getmodel_args['correlated_output'], True))

        else:
            getmodel_args['correlated_output'] = ''

        getmodel_args.update(custom_args)
        getmodel_cmd = _get_getmodel_cmd(**getmodel_args)
        if num_lb: # print main_cmd_gul_stream, get_gul_stream_cmds will be updated after by the main lb block
            main_cmd_gul_stream = get_main_cmd_gul_stream(
                getmodel_cmd, gul_id, fifo_queue_dir, stderr_guard, RUNTYPE_LOAD_BALANCED_LOSS
            )
            print_command(filename, main_cmd_gul_stream)
        else:
            get_gul_stream_cmds.setdefault(fifo_queue_dir, []).append((getmodel_cmd, False))

    if num_lb: # create load balancer cmds
        for fifo_dir in fifo_dirs:
            get_gul_stream_cmds[fifo_dir] = [
                (get_fifo_name(fifo_dir, RUNTYPE_LOAD_BALANCED_LOSS, fm_id, RUNTYPE_INSURED_LOSS), True) for
                fm_id in range(1, num_fm_output + 1)]

            #print the load balancing command
            get_input_stream_name = partial(get_fifo_name,
                                            fifo_dir=fifo_dir,
                                            producer=RUNTYPE_GROUNDUP_LOSS,
                                            consumer=RUNTYPE_LOAD_BALANCED_LOSS)
            get_output_stream_name = partial(get_fifo_name,
                                             fifo_dir=fifo_dir,
                                             producer=RUNTYPE_LOAD_BALANCED_LOSS,
                                             consumer=RUNTYPE_INSURED_LOSS)
            for lb_main_cmd in get_main_cmd_lb(num_lb, num_gul_per_lb, num_fm_per_lb, get_input_stream_name,
                                               get_output_stream_name, stderr_guard):
                print_command(filename, lb_main_cmd)

    # Establish whether step policies present
    step_flag = ''
    try:
        pd.read_csv(
            os.path.join(os.getcwd(), 'input/fm_profile.csv')
        )['step_id']
    except (OSError, FileNotFoundError, KeyError):
        pass
    else:
        step_flag = ' -S'


    for fifo_dir, gul_streams in get_gul_stream_cmds.items():
        for i, (getmodel_cmd, from_file) in enumerate(gul_streams):

            ## THIS NEEDS EDIT - temp workaround for dist work chunk
            if process_number is not None:
                process_id = process_number
            else:
                process_id = i + 1
            #######################################################


            if ri_output:
                main_cmd = get_main_cmd_ri_stream(
                    getmodel_cmd,
                    process_id,
                    il_output,
                    il_alloc_rule,
                    ri_alloc_rule,
                    num_reinsurance_iterations,
                    fifo_dir,
                    stderr_guard,
                    from_file,
                    fmpy,
                    fmpy_low_memory,
                    fmpy_sort_output,
                    step_flag
                )
                print_command(filename, main_cmd)

            elif il_output:
                main_cmd = get_main_cmd_il_stream(
                    getmodel_cmd, process_id, il_alloc_rule, fifo_dir,
                    stderr_guard,
                    from_file,
                    fmpy,
                    fmpy_low_memory,
                    fmpy_sort_output,
                    step_flag
                )
                print_command(filename, main_cmd)

            else:
                main_cmd = get_main_cmd_gul_stream(
                    getmodel_cmd, process_id, fifo_dir, stderr_guard
                )
                print_command(filename, main_cmd)

    print_command(filename, '')

    do_pwaits(filename, process_counter)


    if ri_output:
        print_command(filename, '')
        print_command(filename, '# --- Do reinsurance loss kats ---')
        print_command(filename, '')
        do_kats(
            RUNTYPE_REINSURANCE_LOSS, analysis_settings, num_fm_output,
            filename, process_counter, work_kat_dir, output_dir, kat_sort_by_event,
            process_number=process_number
        )
        if full_correlation:
            print_command(filename, '')
            print_command(
                filename,
                '# --- Do reinsurance loss kats for fully correlated output ---'
            )
            print_command(filename, '')
            do_kats(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings, num_fm_output,
                filename, process_counter, work_full_correlation_kat_dir,
                output_full_correlation_dir, kat_sort_by_event,
                process_number=process_number
            )

    if il_output:
        print_command(filename, '')
        print_command(filename, '# --- Do insured loss kats ---')
        print_command(filename, '')
        do_kats(
            RUNTYPE_INSURED_LOSS, analysis_settings, num_fm_output, filename,
            process_counter, work_kat_dir, output_dir, kat_sort_by_event,
            process_number=process_number
        )
        if full_correlation:
            print_command(filename, '')
            print_command(
                filename,
                '# --- Do insured loss kats for fully correlated output ---'
            )
            print_command(filename, '')
            do_kats(
                RUNTYPE_INSURED_LOSS, analysis_settings, num_fm_output,
                filename, process_counter, work_full_correlation_kat_dir,
                output_full_correlation_dir, kat_sort_by_event,
                process_number=process_number
            )

    if gul_output:
        print_command(filename, '')
        print_command(filename, '# --- Do ground up loss kats ---')
        print_command(filename, '')
        do_kats(
            RUNTYPE_GROUNDUP_LOSS, analysis_settings, num_gul_output, filename,
            process_counter, work_kat_dir, output_dir, kat_sort_by_event,
            process_number=process_number
        )
        if full_correlation:
            print_command(filename, '')
            print_command(
                filename,
                '# --- Do ground up loss kats for fully correlated output ---'
            )
            print_command(filename, '')
            do_kats(
                RUNTYPE_GROUNDUP_LOSS, analysis_settings, num_gul_output,
                filename, process_counter, work_full_correlation_kat_dir,
                output_full_correlation_dir, kat_sort_by_event,
                process_number=process_number
            )

    do_kwaits(filename, process_counter)


def create_bash_outputs(
    process_counter,
    fifo_tmp_dir,
    filename,
    remove_working_files,
    fifo_queue_dir,
    stderr_guard,
    work_dir,
    work_full_correlation_dir,
    output_dir,
    output_full_correlation_dir,
    full_correlation,
    gul_output,
    il_output,
    ri_output,
    analysis_settings,
    **kwargs
):


    print_command(filename, '')
    if ri_output:
        do_post_wait_processing(
            RUNTYPE_REINSURANCE_LOSS, analysis_settings, filename, process_counter,
            '', output_dir, stderr_guard
        )
    if il_output:
        do_post_wait_processing(
            RUNTYPE_INSURED_LOSS, analysis_settings, filename, process_counter, '',
            output_dir, stderr_guard
        )
    if gul_output:
        do_post_wait_processing(
            RUNTYPE_GROUNDUP_LOSS, analysis_settings, filename, process_counter, '',
            output_dir, stderr_guard
        )

    if full_correlation:
        work_sub_dir = re.sub('^work/', '', work_full_correlation_dir)
        if ri_output:
            do_post_wait_processing(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings, filename, process_counter,
                work_sub_dir, output_full_correlation_dir, stderr_guard
            )
        if il_output:
            do_post_wait_processing(
                RUNTYPE_INSURED_LOSS, analysis_settings, filename, process_counter,
                work_sub_dir, output_full_correlation_dir, stderr_guard
            )
        if gul_output:
            do_post_wait_processing(
                RUNTYPE_GROUNDUP_LOSS, analysis_settings, filename, process_counter,
                work_sub_dir, output_full_correlation_dir, stderr_guard
            )

    do_awaits(filename, process_counter)  # waits for aalcalc
    do_lwaits(filename, process_counter)  # waits for leccalc


    if remove_working_files:
        print_command(filename, 'rm -R -f {}'.format(os.path.join(work_dir, '*')))

        if fifo_tmp_dir:
            ## workaround to match bash tests
            if fifo_queue_dir.endswith('fifo/'):
                print_command(filename, 'rm -R -f {}'.format(fifo_queue_dir[:-5]))
            else:
                print_command(filename, 'rm -R -f {}'.format(fifo_queue_dir))
        else:
            print_command(filename, 'rm -R -f {}'.format(os.path.join(fifo_queue_dir, '*')))



# ========================================================================== #
# COMPATIBILITY ONLY - used to support older model runners
# ========================================================================== #

def genbash(
    max_process_id,
    analysis_settings,
    num_reinsurance_iterations=0,
    fifo_tmp_dir=True,
    gul_alloc_rule=None,
    il_alloc_rule=None,
    ri_alloc_rule=None,
    num_gul_per_lb=None,
    num_fm_per_lb=None,
    stderr_guard=True,
    gul_legacy_stream=False,
    bash_trace=False,
    filename='run_kools.sh',
    _get_getmodel_cmd=None,
    custom_args={},
    fmpy=True,
    fmpy_low_memory=False,
    fmpy_sort_output=False,
    event_shuffle=None,
    modelpy=False,
    model_py_server=False
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

    :param num_gul_in_calc_block: number of gul in calc block
    :type num_gul_in_calc_block: Int

    :param num_fm_in_calc_block: number of gul in calc block
    :type num_fm_in_calc_block: Int

    :param get_getmodel_cmd: Method for getting the getmodel command, by default
        ``GenerateLossesCmd.get_getmodel_cmd`` is used.
    :type get_getmodel_cmd: callable
    """


    params = bash_params(
        max_process_id=max_process_id,
        analysis_settings=analysis_settings,
        num_reinsurance_iterations=num_reinsurance_iterations,
        fifo_tmp_dir=fifo_tmp_dir,
        gul_alloc_rule=gul_alloc_rule,
        il_alloc_rule=il_alloc_rule,
        ri_alloc_rule=ri_alloc_rule,
        num_gul_per_lb=num_gul_per_lb,
        num_fm_per_lb=num_fm_per_lb,
        stderr_guard=stderr_guard,
        gul_legacy_stream=gul_legacy_stream,
        bash_trace=bash_trace,
        filename=filename,
        _get_getmodel_cmd=_get_getmodel_cmd,
        custom_args=custom_args,
        fmpy=fmpy,
        fmpy_low_memory=fmpy_low_memory,
        fmpy_sort_output=fmpy_sort_output,
        event_shuffle=event_shuffle,
        modelpy=modelpy,
        model_py_server=model_py_server
    )

    # remove the file if it already exists
    if os.path.exists(filename):
        os.remove(filename)

    with bash_wrapper(filename, bash_trace, stderr_guard):
        create_bash_analysis(**params)
        create_bash_outputs(**params)
