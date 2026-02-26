"""Bash script generation for the pytools loss calculation pipeline.

This module generates bash scripts that orchestrate the pytools loss calculation
pipeline. The generated scripts coordinate multiple concurrent processes connected
via named pipes (FIFOs) to compute ground-up losses (GUL), insured losses (IL),
reinsurance losses (RI), reinsurance gross losses (RL), and optionally
fully-correlated (FC) and load-balanced (LB) streams.

The high-level flow of a generated bash script is:

    1. **Initialisation** -- set shell options, logging, error traps, and the
       completion-check function.
    2. **Directory and FIFO setup** -- create or clean work, output, and FIFO
       directories; create named pipes for every process/summary combination.
    3. **Consumer commands** (written in reverse pipeline order so readers are
       ready before writers start):
       a. ``do_ord``  -- per-process ORD output consumers (eltpy, pltpy).
       b. ``do_tees`` -- ``tee`` commands that fan summary streams out to
          work folders and ORD FIFOs.
       c. ``do_summarycalcs`` -- ``summarypy`` commands that read per-runtype
          FIFOs and write per-summary FIFOs.
    4. **Main GUL pipeline commands** -- ``evepy | modelpy | gulpy`` (or
       ``gulmc``) piped through ``fmpy`` for IL/RI, writing into the FIFOs
       consumed above.
    5. **Waits** -- ``wait`` on all background PIDs from steps 3-4.
    6. **Kat concatenation** -- ``katpy`` merges per-process binary ORD files
       into single output files.
    7. **Post-wait aggregation** -- ``aalpy`` and ``lecpy`` run over the
       collected work-folder data to produce ALT and EPT/PSEPT outputs.
    8. **Cleanup** -- remove working directories and temporary FIFOs.
    9. **Completion check** -- verify every started process logged a
       successful finish.

Run types:

    * ``gul`` -- ground-up loss
    * ``il``  -- insured (direct) loss
    * ``ri``  -- reinsurance net loss
    * ``rl``  -- reinsurance gross loss
    * ``fc``  -- fully correlated ground-up loss
    * ``lb``  -- load-balanced stream (intermediary between GUL and FM)

Entry points:

    * :func:`genbash`     -- legacy single-call interface that writes a
      complete bash script.
    * :func:`bash_params` -- builds a parameter dict consumed by
      :func:`create_bash_analysis` and :func:`create_bash_outputs`.
"""

import contextlib
import io
import logging
import multiprocessing
import os
import random
import re
import shutil
import string
from collections import Counter
from functools import partial

import pandas as pd

from ..utils.defaults import (EVE_DEFAULT_SHUFFLE, EVE_FISHER_YATES,
                              EVE_NO_SHUFFLE, EVE_ROUND_ROBIN, EVE_STD_SHUFFLE,
                              KERNEL_N_FM_PER_LB, KERNEL_N_GUL_PER_LB,
                              KERNEL_ALLOC_GUL_DEFAULT,
                              KERNEL_ALLOC_IL_DEFAULT, KERNEL_ALLOC_RI_DEFAULT)
from ..utils.exceptions import OasisException

logger = logging.getLogger(__name__)


RUNTYPE_GROUNDUP_LOSS = 'gul'
RUNTYPE_LOAD_BALANCED_LOSS = 'lb'
RUNTYPE_INSURED_LOSS = 'il'
RUNTYPE_REINSURANCE_LOSS = 'ri'
RUNTYPE_REINSURANCE_GROSS_LOSS = 'rl'
RUNTYPE_FULL_CORRELATION = 'fc'

REINSURANCE_RUNTYPES = [
    RUNTYPE_REINSURANCE_LOSS,
    RUNTYPE_REINSURANCE_GROSS_LOSS
]
INTERMEDIATE_INURING_PRIORITY_PREFIX = 'IP'

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
        'executable': 'aalpy',
        'subfolder_flag': '-K',
        'csv_flag': '-a',
        'alct_flag': '-c',
        'alct_confidence_level': '-l',
        'skip_header_flag': '-H',
    }
}

ORD_ALT_MEANONLY_OUTPUT_SWITCHES = {
    "alt_meanonly": {
        'executable': 'aalpy',
        'subfolder_flag': '-K',
        'csv_flag': '-a',
        'skip_header_flag': '-H',
    }
}

ORD_PLT_OUTPUT_SWITCHES = {
    "plt_sample": {
        'table_name': 'splt',
        'kat_flag': '-S',
        'executable': 'pltpy',
        'csv_flag': '-s',
        'skip_header_flag': '-H'
    },
    "plt_quantile": {
        'table_name': 'qplt',
        'kat_flag': '-Q',
        'executable': 'pltpy',
        'csv_flag': '-q',
        'skip_header_flag': '-H'
    },
    "plt_moment": {
        'table_name': 'mplt',
        'kat_flag': '-M',
        'executable': 'pltpy',
        'csv_flag': '-m',
        'skip_header_flag': '-H'
    }
}

ORD_ELT_OUTPUT_SWITCHES = {
    "elt_quantile": {
        'table_name': 'qelt',
        'kat_flag': '-q',
        'executable': 'eltpy',
        'csv_flag': '-q',
        'skip_header_flag': '-H'
    },
    "elt_moment": {
        'table_name': 'melt',
        'kat_flag': '-m',
        'executable': 'eltpy',
        'csv_flag': '-m',
        'skip_header_flag': '-H'
    }
}

ORD_SELT_OUTPUT_SWITCH = {
    "elt_sample": {
        'table_name': 'selt',
        'kat_flag': '-s',
        'executable': 'eltpy',
        'csv_flag': '-s',
        'skip_header_flag': '-H'
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


TRAP_FUNC = """
touch $LOG_DIR/stderror.err
oasis_exec_monitor.sh $$ $LOG_DIR & pid0=$!

exit_handler(){
   exit_code=$?

   # disable handler
   trap - QUIT HUP INT KILL TERM ERR EXIT

   kill -9 $pid0 2> /dev/null
   if [ "$exit_code" -gt 0 ]; then
       # Error - run process clean up
       echo 'Kernel execution error - exitcode='$exit_code

       set +x
       group_pid=$(ps -p $$ -o pgid --no-headers)
       sess_pid=$(ps -p $$ -o sess --no-headers)
       script_pid=$$
       printf "Script PID:%d, GPID:%s, SPID:%d\n" $script_pid $group_pid $sess_pid >> $LOG_DIR/killout.txt

       ps -jf f -g $sess_pid > $LOG_DIR/subprocess_list
       PIDS_KILL=$(pgrep -a --pgroup $group_pid | awk \'BEGIN { FS = "[ \\t\\n]+" }{ if ($1 >= \'$script_pid\') print}\' | grep -v celery | egrep -v *\\\.log$  | egrep -v *startup.sh$ | sort -n -r)
       echo "$PIDS_KILL" >> $LOG_DIR/killout.txt
       kill -9 $(echo "$PIDS_KILL" | awk \'BEGIN { FS = "[ \\t\\n]+" }{ print $1 }\') 2>/dev/null
       exit $exit_code
   else
       # script successful
       exit 0
   fi
}
trap exit_handler QUIT HUP INT KILL TERM ERR EXIT"""


def get_check_function(custom_gulcalc_log_start=None, custom_gulcalc_log_finish=None):
    """Creates a bash function to check the logs to ensure same number of process started and finsished.

    Args:
        custom_gulcalc_log_start (str): Custom message printed to the logs when a process starts.
        custom_gulcalc_log_finish (str): Custom message printed to the logs when a process ends.
    """
    check_function = """
check_complete(){
    set +e
    proc_list="evepy modelpy gulpy fmpy gulmc summarypy plapy katpy eltpy pltpy aalpy lecpy"
    has_error=0
    for p in $proc_list; do
        started=$(find log -name "${p}_[0-9]*.log" | wc -l)
        finished=$(find log -name "${p}_[0-9]*.log" -exec grep -l "finish" {} + | wc -l)
        if [ "$finished" -lt "$started" ]; then
            echo "[ERROR] $p - $((started-finished)) processes lost"
            has_error=1
        elif [ "$started" -gt 0 ]; then
            echo "[OK] $p"
        fi
    done
"""
    # Add in check for custom gulcalc if settings are provided
    if custom_gulcalc_log_start and custom_gulcalc_log_finish:
        check_function += f"""
    started=$( grep "{custom_gulcalc_log_start}" log/gul_stderror.err | wc -l)
    finished=$( grep "{custom_gulcalc_log_finish}" log/gul_stderror.err | wc -l)
    if [ "$finished" -lt "$started" ]; then
        echo "[ERROR] gulcalc - $((started-finished)) processes lost"
        has_error=1
    elif [ "$started" -gt 0 ]; then
        echo "[OK] gulcalc"
    fi
"""

    check_function += """    if [ "$has_error" -ne 0 ]; then
        false # raise non-zero exit code
    else
        echo 'Run Completed'
    fi
}"""
    return check_function


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
    exec   > >(tee -ia $LOG_DIR/bash.log)
    exec  2> >(tee -ia $LOG_DIR/bash.log >& 2)
    exec 19> $LOG_DIR/bash.log
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


def get_modelcmd(server=False, peril_filter=[]) -> str:
    """
    Gets the construct model command line argument for the bash script.

    Args:
        server: (bool) if set then enable 'TCP' ipc server/client mode
        peril_filter: (list) list of perils to include (all included if empty)

    """
    py_cmd = 'modelpy'
    if server is True:
        py_cmd = f'{py_cmd} --data-server'

    if peril_filter:
        py_cmd = f"{py_cmd} --peril-filter {' '.join(peril_filter)}"
    return py_cmd


def get_gulcmd(gulmc, gul_random_generator, gulmc_effective_damageability, gulmc_vuln_cache_size, modelpy_server, peril_filter, model_df_engine='oasis_data_manager.df_reader.reader.OasisPandasReader', dynamic_footprint=False):
    """Get the ground-up loss calculation command.

    Args:
        gulmc (bool): if True, return the combined (model+ground up) command name, else use 'modelpy | gulpy' .

    Returns:
        str: the ground-up loss calculation command
    """
    if gulmc:
        cmd = f"gulmc --random-generator={gul_random_generator} {'--data-server' * modelpy_server} --model-df-engine=\'{model_df_engine}\'"

        if peril_filter:
            cmd += f" --peril-filter {' '.join(peril_filter)}"

        if gulmc_effective_damageability:
            cmd += " --effective-damageability"

        if gulmc_vuln_cache_size:
            cmd += f" --vuln-cache-size {gulmc_vuln_cache_size}"

        if dynamic_footprint:
            cmd += " --dynamic-footprint True"
    else:
        cmd = f'gulpy --random-generator={gul_random_generator}'

    return cmd


def get_fmcmd(fmpy_low_memory=False, fmpy_sort_output=False):
    """Build the fmpy (financial module) command string.

    Args:
        fmpy_low_memory (bool): If True, append the ``-l`` flag to enable
            low-memory mode.
        fmpy_sort_output (bool): If True, append the ``--sort-output`` flag
            so that output records are sorted.

    Returns:
        str: The assembled ``fmpy`` command with any requested flags.
    """
    cmd = 'fmpy'
    if fmpy_low_memory:
        cmd += ' -l'
    if fmpy_sort_output:
        cmd += ' --sort-output'
    return cmd


def print_command(command_file, cmd):
    """
    Writes the supplied command to the end of the generated script

    :param command_file: File to append command to.
    :param cmd: The command to append
    """
    with io.open(command_file, "a", encoding='utf-8') as myfile:
        myfile.writelines(cmd + "\n")


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
    stderr_guard=True,
    inuring_priority=None,
    join_summary_info=False,
):
    """Write post-wait aggregation commands (aalpy, lecpy) to the bash script.

    These commands run *after* all per-process ``wait`` calls have completed,
    operating on the binary work-folder data produced by ``tee`` during the
    main pipeline.  Specifically this handles:

    * **PALT** (``aalpy``) -- Period Average Loss Table output.
    * **ALT mean-only** (``aalpy``) -- Average Loss Table (mean only) output.
    * **EPT / PSEPT** (``lecpy``) -- Exceedance Probability Table and
      Per-Sample Exceedance Probability Table outputs.

    Args:
        runtype (str): The run type identifier (e.g. ``'gul'``, ``'il'``,
            ``'ri'``, ``'rl'``).
        analysis_settings (dict): The full analysis settings dictionary.
        filename (str): Path to the bash script being generated.
        process_counter (dict): Mutable counter dict tracking background PIDs
            (keys ``lpid_monitor_count``, etc.).
        work_sub_dir (str): Relative path prefix for work sub-directories.
        output_dir (str): Directory where final output files are written.
        stderr_guard (bool): If True, wrap commands in a sub-shell that
            redirects stderr to the log.
        inuring_priority (str or None): Inuring priority label to embed in
            file names, or None / empty for the final priority.
        join_summary_info (bool): If True, append ``join-summary-info``
            commands to enrich outputs with summary metadata.
    """
    if '{}_summaries'.format(runtype) not in analysis_settings:
        return

    if not inuring_priority:
        inuring_priority = ''

    for summary in analysis_settings['{}_summaries'.format(runtype)]:
        if "id" in summary:
            summary_set = summary['id']

            # ORD - PALT
            if ord_enabled(summary, ORD_ALT_OUTPUT_SWITCHES):
                aal_executable = ORD_ALT_OUTPUT_SWITCHES["alt_period"]["executable"]
                aal_subfolder_flag = ORD_ALT_OUTPUT_SWITCHES["alt_period"]["subfolder_flag"]
                cmd = f"{aal_executable} {aal_subfolder_flag}{work_sub_dir}{runtype}_{inuring_priority}S{summary_set}_summary_palt"

                palt_outfile_stem = f"{output_dir}{runtype}_{inuring_priority}S{summary_set}_palt"
                alct_outfile_stem = f"{output_dir}{runtype}_{inuring_priority}S{summary_set}_alct"

                outfile_ext = "csv"
                if summary.get('ord_output', {}).get('parquet_format'):
                    outfile_ext = "parquet"

                if summary.get('ord_output', {}).get('alct_convergence'):
                    aal_alct_flag = ORD_ALT_OUTPUT_SWITCHES["alt_period"]["alct_flag"]
                    cmd = f"{cmd} {aal_alct_flag} {alct_outfile_stem}.{outfile_ext}"
                    if summary.get('ord_output', {}).get('alct_confidence'):
                        aal_alct_confidence_level = ORD_ALT_OUTPUT_SWITCHES["alt_period"]["alct_confidence_level"]
                        cmd = f"{cmd} {aal_alct_confidence_level} {summary.get('ord_output', {}).get('alct_confidence')}"

                aal_csv_flag = ORD_ALT_OUTPUT_SWITCHES["alt_period"]["csv_flag"]
                if outfile_ext == 'parquet':
                    cmd = f"{cmd} -E parquet {aal_csv_flag} {palt_outfile_stem}.parquet"
                else:
                    cmd = f"{cmd} {aal_csv_flag} {palt_outfile_stem}.csv"

                process_counter['lpid_monitor_count'] += 1
                if stderr_guard:
                    cmd = '( {} ) 2>> $LOG_DIR/stderror.err & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                else:
                    cmd = '{} & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                print_command(filename, cmd)

                if join_summary_info or analysis_settings.get("join_summary_info", False):
                    summary_info_filename = f'{output_dir}{runtype}_S{summary_set}_summary-info.{outfile_ext}'
                    cmd = f'join-summary-info -s {summary_info_filename} -d {palt_outfile_stem}.{outfile_ext} -o {palt_outfile_stem}.{outfile_ext}'
                    print_command(filename, cmd)
                    if summary.get('ord_output', {}).get('alct_convergence'):
                        cmd = f'join-summary-info -s {summary_info_filename} -d {alct_outfile_stem}.{outfile_ext} -o {alct_outfile_stem}.{outfile_ext}'
                        print_command(filename, cmd)

            # ORD - aalcalcmeanonly
            if ord_enabled(summary, ORD_ALT_MEANONLY_OUTPUT_SWITCHES):
                aal_executable = ORD_ALT_MEANONLY_OUTPUT_SWITCHES["alt_meanonly"]["executable"]
                aal_subfolder_flag = ORD_ALT_MEANONLY_OUTPUT_SWITCHES["alt_meanonly"]["subfolder_flag"]
                cmd = f"{aal_executable} {aal_subfolder_flag}{work_sub_dir}{runtype}_{inuring_priority}S{summary_set}_summary_altmeanonly"
                altmeanonly_outfile_stem = f"{output_dir}{runtype}_{inuring_priority}S{summary_set}_altmeanonly"

                outfile_ext = 'csv'
                aal_csv_flag = ORD_ALT_MEANONLY_OUTPUT_SWITCHES["alt_meanonly"]["csv_flag"]
                if summary.get('ord_output', {}).get('parquet_format'):
                    cmd = f"{cmd} -E parquet {aal_csv_flag} {altmeanonly_outfile_stem}.cparquetsv"
                    outfile_ext = 'parquet'
                else:
                    cmd = f"{cmd} {aal_csv_flag} {altmeanonly_outfile_stem}.csv"

                process_counter['lpid_monitor_count'] += 1
                if stderr_guard:
                    cmd = '( {} ) 2>> $LOG_DIR/stderror.err & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                else:
                    cmd = '{} & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                print_command(filename, cmd)

                if join_summary_info or analysis_settings.get("join_summary_info", False):
                    summary_info_filename = f'{output_dir}{runtype}_S{summary_set}_summary-info.{outfile_ext}'
                    cmd = f'join-summary-info -s {summary_info_filename} -d {altmeanonly_outfile_stem}.{outfile_ext} -o {altmeanonly_outfile_stem}.{outfile_ext}'
                    print_command(filename, cmd)

            # ORD - PSEPT,EPT
            if ord_enabled(summary, ORD_LECCALC):

                ord_outputs = summary.get('ord_output', {})
                ept_output = False
                psept_output = False
                lec_executable = "lecpy"

                cmd = f"{lec_executable} {'-r' if ord_outputs.get('return_period_file') else ''}"
                cmd = f"{cmd} -K{work_sub_dir}{runtype}_{inuring_priority}S{summary_set}_summaryleccalc"

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
                    cmd = f"{cmd} -E parquet"
                    outfile_ext = 'parquet'

                ept_filename = '{}{}_{}S{}_ept.{}'.format(
                    output_dir, runtype, inuring_priority,
                    summary_set, outfile_ext
                )
                psept_filename = '{}{}_{}S{}_psept.{}'.format(
                    output_dir, runtype, inuring_priority,
                    summary_set, outfile_ext
                )

                if ept_output:
                    cmd = '{} {} {}'.format(
                        cmd, ept_output_flag, ept_filename
                    )

                if psept_output:
                    cmd = '{} {} {}'.format(
                        cmd, psept_output_flag, psept_filename
                    )

                if stderr_guard:
                    cmd = '( {} ) 2>> $LOG_DIR/stderror.err & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                else:
                    cmd = '{} & lpid{}=$!'.format(cmd, process_counter['lpid_monitor_count'])
                print_command(filename, cmd)

                if join_summary_info or analysis_settings.get("join_summary_info", False):
                    summary_info_filename = f'{output_dir}{runtype}_S{summary_set}_summary-info.{outfile_ext}'
                    cmd = f'join-summary-info -s {summary_info_filename} -d {ept_filename} -o {ept_filename}'
                    print_command(filename, cmd)
                    cmd = f'join-summary-info -s {summary_info_filename} -d {psept_filename} -o {psept_filename}'
                    print_command(filename, cmd)


def get_fifo_name(fifo_dir, producer, producer_id, consumer=''):
    """Build the standardised path for a named pipe (FIFO).

    The naming convention is ``<fifo_dir><producer>_<consumer>_P<id>`` when a
    consumer is specified, or ``<fifo_dir><producer>_P<id>`` otherwise.

    Args:
        fifo_dir (str): Base directory for FIFOs (e.g. ``'fifo/'``).
        producer (str): Run type that produces the stream (e.g. ``'gul'``).
        producer_id (int): Process number used as a suffix (``P<id>``).
        consumer (str): Optional consumer identifier appended between the
            producer and the process id.

    Returns:
        str: The fully-qualified FIFO path.
    """
    if consumer:
        return f'{fifo_dir}{producer}_{consumer}_P{producer_id}'
    else:
        return f'{fifo_dir}{producer}_P{producer_id}'


def do_fifo_exec(producer, producer_id, filename, fifo_dir, action='mkfifo', consumer=''):
    """Write a single FIFO create or remove command to the bash script.

    Args:
        producer (str): Run type that produces the stream.
        producer_id (int): Process number for the FIFO name.
        filename (str): Path to the bash script being generated.
        fifo_dir (str): Base directory for FIFOs.
        action (str): Shell command to execute (``'mkfifo'`` to create,
            ``'rm'`` to remove).
        consumer (str): Optional consumer identifier for the FIFO name.
    """
    print_command(filename, f'{action} {get_fifo_name(fifo_dir, producer, producer_id, consumer)}')


def do_fifos_exec(runtype, max_process_id, filename, fifo_dir, process_number=None, action='mkfifo', consumer=''):
    """Write FIFO create/remove commands for every process in the range.

    Iterates over all process IDs and emits one ``mkfifo`` (or ``rm``)
    command per process for the given run type.

    Args:
        runtype (str): The run type identifier (e.g. ``'gul'``, ``'il'``).
        max_process_id (int): Upper bound of process IDs.
        filename (str): Path to the bash script being generated.
        fifo_dir (str): Base directory for FIFOs.
        process_number (int or None): If set, only that single process ID
            is used instead of the full range.
        action (str): Shell command (``'mkfifo'`` or ``'rm'``).
        consumer (str): Optional consumer identifier for FIFO names.
    """
    for process_id in process_range(max_process_id, process_number):
        do_fifo_exec(runtype, process_id, filename, fifo_dir, action, consumer)
    print_command(filename, '')


def do_fifos_exec_full_correlation(
        runtype, max_process_id, filename, fifo_dir, process_number=None, action='mkfifo'):
    """Write FIFO create/remove commands for full-correlation sumcalc and fmcalc pipes.

    For each process ID two FIFOs are created: one for the summarycalc input
    (``<runtype>_sumcalc_P<id>``) and one for the fmcalc input
    (``<runtype>_fmcalc_P<id>``).

    Args:
        runtype (str): The run type identifier (typically ``'gul'``).
        max_process_id (int): Upper bound of process IDs.
        filename (str): Path to the bash script being generated.
        fifo_dir (str): Base directory for FIFOs.
        process_number (int or None): If set, restrict to a single process.
        action (str): Shell command (``'mkfifo'`` or ``'rm'``).
    """
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


def do_fifos_calc(runtype, analysis_settings, max_process_id, filename, fifo_dir='fifo/', process_number=None, consumer_prefix=None, action='mkfifo'):
    """Write FIFO create/remove commands for all summary and ORD output pipes.

    For every (process, summary) pair this creates:

    * A summary FIFO (and its ``.idx`` companion when leccalc or ALT output
      is enabled).
    * One FIFO per active ORD output type (plt, elt, selt).

    Args:
        runtype (str): The run type identifier.
        analysis_settings (dict): The full analysis settings dictionary.
        max_process_id (int): Upper bound of process IDs.
        filename (str): Path to the bash script being generated.
        fifo_dir (str): Base directory for FIFOs.
        process_number (int or None): If set, restrict to a single process.
        consumer_prefix (str or None): Optional prefix prepended to consumer
            names (used for inuring priority labelling).
        action (str): Shell command (``'mkfifo'`` or ``'rm'``).
    """

    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    if not consumer_prefix:
        consumer_prefix = ''

    for process_id in process_range(max_process_id, process_number):
        for summary in summaries:
            if 'id' in summary:
                summary_set = summary['id']
                do_fifo_exec(runtype, process_id, filename, fifo_dir, action, f'{consumer_prefix}S{summary_set}_summary')
                if ord_enabled(summary, ORD_LECCALC) or ord_enabled(summary, ORD_ALT_OUTPUT_SWITCHES):
                    idx_fifo = get_fifo_name(fifo_dir, runtype, process_id, f'{consumer_prefix}S{summary_set}_summary')
                    idx_fifo += '.idx'
                    print_command(filename, f'mkfifo {idx_fifo}')

                for ord_type, output_switch in OUTPUT_SWITCHES.items():
                    for ord_table in output_switch.keys():
                        if summary.get('ord_output', {}).get(ord_table):
                            do_fifo_exec(runtype, process_id, filename, fifo_dir, action, f'{consumer_prefix}S{summary_set}_{ord_type}')
                            break

        print_command(filename, '')


def create_workfolders(
    runtype,
    analysis_settings,
    filename,
    work_dir='work/',
    inuring_priority=None
):
    """Write ``mkdir -p`` commands for work sub-directories needed by summary consumers.

    Creates directories that ``tee`` will write binary summary data into for
    subsequent aggregation by ``lecpy`` (leccalc) and ``aalpy`` (aalcalc).

    Args:
        runtype (str): The run type identifier (e.g. ``'gul'``, ``'il'``).
        analysis_settings (dict): The full analysis settings dictionary.
        filename (str): Path to the bash script being generated.
        work_dir (str): Root working directory (e.g. ``'work/'``).
        inuring_priority (str or None): Inuring priority label to embed in
            directory names, or None / empty for the final priority.
    """

    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    if not inuring_priority:
        inuring_priority = ''

    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']

            # EDIT: leccalc and ordleccalc share the same summarycalc binary data
            # only create the workfolders once if either option is selected
            if ord_enabled(summary, ORD_LECCALC):
                print_command(
                    filename,
                    'mkdir -p {}{}_{}S{}_summaryleccalc'.format(work_dir, runtype, inuring_priority, summary_set)
                )

            if summary.get('ord_output', {}).get('alt_period'):
                print_command(
                    filename,
                    'mkdir -p {}{}_{}S{}_summary_palt'.format(work_dir, runtype, inuring_priority, summary_set)
                )

            if summary.get('ord_output', {}).get('alt_meanonly'):
                print_command(
                    filename,
                    'mkdir -p {}{}_{}S{}_summary_altmeanonly'.format(work_dir, runtype, inuring_priority, summary_set)
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
    inuring_priority=None,
    join_summary_info=False,
):
    """Write ``katpy`` concatenation commands to merge per-process ORD files.

    After the main pipeline ``wait`` has completed, per-process binary output
    files sit in the kat work directory.  ``katpy`` reads them all and
    produces a single sorted (or unsorted) output file per summary/ORD-table
    combination.

    Args:
        runtype (str): The run type identifier.
        analysis_settings (dict): The full analysis settings dictionary.
        max_process_id (int): Upper bound of process IDs.
        filename (str): Path to the bash script being generated.
        process_counter (dict): Mutable counter dict tracking background PIDs
            (key ``kpid_monitor_count``).
        work_dir (str): Kat work directory containing per-process files.
        output_dir (str): Directory for final concatenated output files.
        sort_by_event (bool): If True, ``katpy`` sorts output by event id.
        process_number (int or None): If set, restrict to a single process.
        inuring_priority (str or None): Inuring priority label for file names.
        join_summary_info (bool): If True, append ``join-summary-info``
            commands after each concatenation.

    Returns:
        bool: True if any ``katpy`` commands were emitted, False otherwise.
    """
    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return False

    if not inuring_priority:
        inuring_priority = ''

    anykats = False
    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']
            for ord_type, output_switch in OUTPUT_SWITCHES.items():
                for ord_table, v in output_switch.items():
                    if summary.get('ord_output', {}).get(ord_table):

                        anykats = True

                        cmd = f'katpy {v["kat_flag"]}' if sort_by_event else f'katpy -u {v["kat_flag"]}'
                        outfile_flag = '-o'
                        outfile_ext = 'csv'

                        cmd = f'{cmd} -f bin -i'

                        if summary.get('ord_output', {}).get('parquet_format'):
                            outfile_ext = 'parquet'

                        for process_id in process_range(max_process_id, process_number):
                            cmd = f'{cmd} {work_dir}{runtype}_{inuring_priority}S{summary_set}_{ord_table}_P{process_id}'

                        process_counter['kpid_monitor_count'] += 1
                        csv_outfile = f'{output_dir}{runtype}_{inuring_priority}S{summary_set}_{v["table_name"]}.{outfile_ext}'
                        cmd = f'{cmd} {outfile_flag} {csv_outfile}'
                        cmd = f'{cmd} & kpid{process_counter["kpid_monitor_count"]}=$!'
                        print_command(filename, cmd)

                        if join_summary_info or analysis_settings.get("join_summary_info", False):
                            summary_info_filename = f'{output_dir}{runtype}_S{summary_set}_summary-info.{outfile_ext}'
                            cmd = f'join-summary-info -s {summary_info_filename} -d {csv_outfile} -o {csv_outfile}'
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
    gul_full_correlation=False,
    inuring_priority=None,
):
    """Write a ``summarypy`` command for one process to the bash script.

    Reads the per-process run-type FIFO and fans the stream out to one
    summary FIFO per summary set defined in the analysis settings.

    Args:
        runtype (str): The run type identifier (e.g. ``'gul'``, ``'il'``).
        analysis_settings (dict): The full analysis settings dictionary.
        process_id (int): The process number for this command.
        filename (str): Path to the bash script being generated.
        fifo_dir (str): Base directory for FIFOs.
        stderr_guard (bool): If True, wrap the command in a sub-shell that
            redirects stderr to the log.
        num_reinsurance_iterations (int): Total number of reinsurance
            iterations (used for directory switching).
        gul_full_correlation (bool): If True, read from the
            ``_sumcalc`` FIFO variant used by the full-correlation path.
        inuring_priority (dict or None): Inuring priority dict with keys
            ``'text'`` and ``'level'``, or None for non-reinsurance runs.
    """

    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    if process_id == 1:
        print_command(filename, '')

    if runtype == RUNTYPE_REINSURANCE_GROSS_LOSS:
        summarycalc_switch = '-t ri'
    else:
        summarycalc_switch = f'-t {runtype}'

    summarycalc_directory_switch = ""
    inuring_priority_text = ''   # Only relevant for reinsurance
    if runtype == RUNTYPE_REINSURANCE_LOSS or runtype == RUNTYPE_REINSURANCE_GROSS_LOSS:
        if inuring_priority.get('level'):
            summarycalc_directory_switch = f"-p {os.path.join('input', 'RI_' + str(inuring_priority['level']))}"
            # Text field for final inuring priority is empty string
            inuring_priority_text = inuring_priority['text']

    input_filename_component = ''
    if gul_full_correlation:
        input_filename_component = '_sumcalc'

    # Use -m flag to create summary index files
    cmd = 'summarypy'
    cmd = f'{cmd} -m {summarycalc_switch} {summarycalc_directory_switch}'
    for summary in summaries:
        if 'id' in summary:
            summary_set = summary['id']
            cmd = '{0} -{1} {4}{2}_{5}S{1}_summary_P{3}'.format(
                cmd, summary_set, runtype, process_id, fifo_dir,
                inuring_priority_text
            )

    cmd = '{0} < {1}{2}{3}_{5}P{4}'.format(
        cmd, fifo_dir, runtype, input_filename_component, process_id,
        inuring_priority_text
    )
    cmd = '( {0} ) 2>> $LOG_DIR/stderror.err  &'.format(cmd) if stderr_guard else '{0} &'.format(cmd)      # Wrap in subshell and pipe stderr to file
    print_command(filename, cmd)


def do_tees(
    runtype,
    analysis_settings,
    process_id,
    filename,
    process_counter,
    fifo_dir='fifo/',
    work_dir='work/',
    inuring_priority=None
):
    """Write ``tee`` commands that fan each summary stream to ORD and work FIFOs.

    For each summary set, a ``tee`` reads from the summary FIFO and
    duplicates the stream to:

    * ORD output FIFOs (for ``eltpy``/``pltpy`` consumers).
    * Work-folder binary files (for ``aalpy``/``lecpy`` post-wait processing).

    When leccalc or ALT output is enabled, a companion ``.idx`` tee is also
    emitted.

    Args:
        runtype (str): The run type identifier.
        analysis_settings (dict): The full analysis settings dictionary.
        process_id (int): The process number.
        filename (str): Path to the bash script being generated.
        process_counter (dict): Mutable counter dict tracking background PIDs
            (key ``pid_monitor_count``).
        fifo_dir (str): Base directory for FIFOs.
        work_dir (str): Root working directory.
        inuring_priority (str or None): Inuring priority label for file and
            FIFO names, or None / empty for the final priority.
    """

    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    if not inuring_priority:
        inuring_priority = ''

    if process_id == 1:
        print_command(filename, '')

    for summary in summaries:
        if 'id' in summary:
            process_counter['pid_monitor_count'] += 1
            summary_set = summary['id']

            cmd = f'tee < {get_fifo_name(fifo_dir, runtype, process_id, f"{inuring_priority}S{summary_set}_summary")}'
            if ord_enabled(summary, ORD_LECCALC) or ord_enabled(summary, ORD_ALT_OUTPUT_SWITCHES):
                cmd_idx = cmd + '.idx'

            for ord_type, output_switch in OUTPUT_SWITCHES.items():
                for ord_table in output_switch.keys():
                    if summary.get('ord_output', {}).get(ord_table):
                        cmd = f'{cmd} {get_fifo_name(fifo_dir, runtype, process_id, f"{inuring_priority}S{summary_set}_{ord_type}")}'
                        break

            if summary.get('ord_output', {}).get('alt_period'):
                aalcalc_ord_out = f'{work_dir}{runtype}_{inuring_priority}S{summary_set}_summary_palt/P{process_id}'
                cmd = f'{cmd} {aalcalc_ord_out}.bin'
                cmd_idx = f'{cmd_idx} {aalcalc_ord_out}.idx'

            if summary.get('ord_output', {}).get('alt_meanonly'):
                aalcalcmeanonly_ord_out = f'{work_dir}{runtype}_{inuring_priority}S{summary_set}_summary_altmeanonly/P{process_id}'
                cmd = f'{cmd} {aalcalcmeanonly_ord_out}.bin'

            # leccalc and ordleccalc share the same summarycalc binary data
            # only create the workfolders once if either option is selected
            if ord_enabled(summary, ORD_LECCALC):
                leccalc_out = f'{work_dir}{runtype}_{inuring_priority}S{summary_set}_summaryleccalc/P{process_id}'
                cmd = f'{cmd} {leccalc_out}.bin'
                cmd_idx = f'{cmd_idx} {leccalc_out}.idx'

            cmd = '{} > /dev/null & pid{}=$!'.format(cmd, process_counter['pid_monitor_count'])
            print_command(filename, cmd)
            if ord_enabled(summary, ORD_LECCALC) or ord_enabled(summary, ORD_ALT_OUTPUT_SWITCHES):
                process_counter['pid_monitor_count'] += 1
                cmd_idx = '{} > /dev/null & pid{}=$!'.format(cmd_idx, process_counter['pid_monitor_count'])
                print_command(filename, cmd_idx)


def do_tees_fc_sumcalc_fmcalc(process_id, filename, correlated_output_stems):
    """Write a ``tee`` command that splits the full-correlation GUL output.

    Duplicates the correlated GUL stream into two FIFOs: one for the
    summarycalc path and one for the fmcalc (financial module) path.

    Args:
        process_id (int): The process number.
        filename (str): Path to the bash script being generated.
        correlated_output_stems (dict): FIFO path stems returned by
            :func:`get_correlated_output_stems`.
    """

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
    """Return a dict of FIFO path stems for the full-correlation pipeline.

    The returned dict contains three keys:

    * ``'gulcalc_output'`` -- stem for the GUL output FIFO.
    * ``'fmcalc_input'``   -- stem for the fmcalc input FIFO.
    * ``'sumcalc_input'``  -- stem for the summarycalc input FIFO.

    Each value is a string ending with ``_P`` so that a process ID can be
    appended directly.

    Args:
        fifo_dir (str): Base directory for FIFOs (e.g. ``'fifo/'``).

    Returns:
        dict: Mapping of stem names to FIFO path prefixes.
    """

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


def do_ord(
    runtype,
    analysis_settings,
    process_id,
    filename,
    process_counter,
    fifo_dir='fifo/',
    work_dir='work/',
    stderr_guard=True,
    inuring_priority=None,
):
    """Write ORD output consumer commands (``eltpy``, ``pltpy``) for one process.

    For each summary set and each active ORD output type, reads from the
    corresponding FIFO and writes per-process binary output files into the
    kat work directory for later concatenation by ``katpy``.

    Args:
        runtype (str): The run type identifier.
        analysis_settings (dict): The full analysis settings dictionary.
        process_id (int): The process number.
        filename (str): Path to the bash script being generated.
        process_counter (dict): Mutable counter dict tracking background PIDs.
        fifo_dir (str): Base directory for FIFOs.
        work_dir (str): Root working directory (kat sub-directory is used).
        stderr_guard (bool): If True, wrap commands in a sub-shell that
            redirects stderr to the log.
        inuring_priority (str or None): Inuring priority label for file and
            FIFO names.
    """

    summaries = analysis_settings.get('{}_summaries'.format(runtype))
    if not summaries:
        return

    if not inuring_priority:
        inuring_priority = ''

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
                            cmd += f' {flag_proc["skip_header_flag"]}'
                            skip_line = False

                        if summary.get('ord_output', {}).get('parquet_format'):
                            cmd += f' {flag_proc["csv_flag"]}'
                        else:
                            cmd += f' {flag_proc["csv_flag"]}'

                        fifo_out_name = get_fifo_name(f'{work_dir}kat/', runtype, process_id, f'{inuring_priority}S{summary_set}_{ord_table}')
                        cmd = f'{cmd} {fifo_out_name}'

                if cmd:
                    fifo_in_name = get_fifo_name(fifo_dir, runtype, process_id, f'{inuring_priority}S{summary_set}_{ord_type}')
                    cmd = f'{cmd} < {fifo_in_name}'
                    process_counter['pid_monitor_count'] += 1

                    # Add binary output flag for ELTpy and PLTpy, will be converted to csv during kats
                    cmd = f'{flag_proc["executable"]} -E bin {cmd}'

                    if stderr_guard:
                        cmd = f'( {cmd} ) 2>> $LOG_DIR/stderror.err & pid{process_counter["pid_monitor_count"]}=$!'
                    else:
                        cmd = f'{cmd} & pid{process_counter["pid_monitor_count"]}=$!'

                    print_command(filename, cmd)


def get_ri_inuring_priorities(analysis_settings, num_reinsurance_iterations):
    """Build the list of inuring priority dicts for reinsurance net loss.

    Returns intermediate inuring priorities specified in the analysis settings
    plus the final priority (the full reinsurance iteration count).

    Args:
        analysis_settings (dict): The full analysis settings dictionary,
            checked for the ``'ri_inuring_priorities'`` key.
        num_reinsurance_iterations (int): Total number of reinsurance
            iterations.

    Returns:
        list[dict]: Each dict has ``'text'`` (file-name prefix) and
        ``'level'`` (iteration number) keys.
    """
    intermediate_inuring_priorities = set(analysis_settings.get('ri_inuring_priorities', []))
    ri_inuring_priorities = [
        {
            'text': INTERMEDIATE_INURING_PRIORITY_PREFIX + str(inuring_priority) + '_',
            'level': inuring_priority
        } for inuring_priority in intermediate_inuring_priorities if inuring_priority < num_reinsurance_iterations
    ]
    ri_inuring_priorities.append({'text': '', 'level': num_reinsurance_iterations})   # Final inuring priority

    return ri_inuring_priorities


def get_rl_inuring_priorities(num_reinsurance_iterations):
    """Build the list of inuring priority dicts for reinsurance gross loss.

    Unlike :func:`get_ri_inuring_priorities`, every iteration from 1 to
    ``num_reinsurance_iterations`` is included (there is no "final" entry).

    Args:
        num_reinsurance_iterations (int): Total number of reinsurance
            iterations.

    Returns:
        list[dict]: Each dict has ``'text'`` (file-name prefix) and
        ``'level'`` (iteration number) keys.
    """
    rl_inuring_priorities = [
        {
            'text': INTERMEDIATE_INURING_PRIORITY_PREFIX + str(inuring_priority) + '_',
            'level': inuring_priority
        } for inuring_priority in range(1, num_reinsurance_iterations + 1)
    ]

    return rl_inuring_priorities


def rl(
    analysis_settings,
    max_process_id,
    filename,
    process_counter,
    num_reinsurance_iterations,
    fifo_dir='fifo/',
    work_dir='work/',
    stderr_guard=True,
    process_number=None
):
    """Write all reinsurance gross loss (RL) consumer commands.

    Iterates over each RL inuring priority and emits ``do_ord``, ``do_tees``,
    and ``do_summarycalcs`` commands for every process.

    Args:
        analysis_settings (dict): The full analysis settings dictionary.
        max_process_id (int): Upper bound of process IDs.
        filename (str): Path to the bash script being generated.
        process_counter (dict): Mutable counter dict tracking background PIDs.
        num_reinsurance_iterations (int): Total number of reinsurance
            iterations.
        fifo_dir (str): Base directory for FIFOs.
        work_dir (str): Root working directory.
        stderr_guard (bool): If True, wrap commands in a sub-shell that
            redirects stderr to the log.
        process_number (int or None): If set, restrict to a single process.
    """

    for inuring_priority in get_rl_inuring_priorities(num_reinsurance_iterations):
        for process_id in process_range(max_process_id, process_number):
            do_ord(
                RUNTYPE_REINSURANCE_GROSS_LOSS, analysis_settings, process_id,
                filename, process_counter, fifo_dir, work_dir, stderr_guard,
                inuring_priority=inuring_priority['text']
            )

        for process_id in process_range(max_process_id, process_number):
            do_tees(
                RUNTYPE_REINSURANCE_GROSS_LOSS, analysis_settings, process_id,
                filename, process_counter, fifo_dir, work_dir,
                inuring_priority=inuring_priority['text']
            )

        for process_id in process_range(max_process_id, process_number):
            do_summarycalcs(
                runtype=RUNTYPE_REINSURANCE_GROSS_LOSS,
                analysis_settings=analysis_settings,
                process_id=process_id,
                filename=filename,
                fifo_dir=fifo_dir,
                stderr_guard=stderr_guard,
                num_reinsurance_iterations=num_reinsurance_iterations,
                inuring_priority=inuring_priority
            )


def ri(
    analysis_settings,
    max_process_id,
    filename,
    process_counter,
    num_reinsurance_iterations,
    fifo_dir='fifo/',
    work_dir='work/',
    stderr_guard=True,
    process_number=None
):
    """Write all reinsurance net loss (RI) consumer commands.

    Iterates over each RI inuring priority and emits ``do_ord``, ``do_tees``,
    and ``do_summarycalcs`` commands for every process.

    Args:
        analysis_settings (dict): The full analysis settings dictionary.
        max_process_id (int): Upper bound of process IDs.
        filename (str): Path to the bash script being generated.
        process_counter (dict): Mutable counter dict tracking background PIDs.
        num_reinsurance_iterations (int): Total number of reinsurance
            iterations.
        fifo_dir (str): Base directory for FIFOs.
        work_dir (str): Root working directory.
        stderr_guard (bool): If True, wrap commands in a sub-shell that
            redirects stderr to the log.
        process_number (int or None): If set, restrict to a single process.
    """

    for inuring_priority in get_ri_inuring_priorities(analysis_settings, num_reinsurance_iterations):

        for process_id in process_range(max_process_id, process_number):
            do_ord(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_id,
                filename, process_counter, fifo_dir, work_dir, stderr_guard,
                inuring_priority=inuring_priority['text']
            )

        for process_id in process_range(max_process_id, process_number):
            do_tees(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_id,
                filename, process_counter, fifo_dir, work_dir,
                inuring_priority=inuring_priority['text']
            )

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
                inuring_priority=inuring_priority
            )


def il(analysis_settings, max_process_id, filename, process_counter, fifo_dir='fifo/', work_dir='work/', stderr_guard=True, process_number=None):
    """Write all insured loss (IL) consumer commands.

    Emits ``do_ord``, ``do_tees``, and ``do_summarycalcs`` commands for every
    process in the range.

    Args:
        analysis_settings (dict): The full analysis settings dictionary.
        max_process_id (int): Upper bound of process IDs.
        filename (str): Path to the bash script being generated.
        process_counter (dict): Mutable counter dict tracking background PIDs.
        fifo_dir (str): Base directory for FIFOs.
        work_dir (str): Root working directory.
        stderr_guard (bool): If True, wrap commands in a sub-shell that
            redirects stderr to the log.
        process_number (int or None): If set, restrict to a single process.
    """
    for process_id in process_range(max_process_id, process_number):
        do_ord(RUNTYPE_INSURED_LOSS, analysis_settings, process_id, filename,
               process_counter, fifo_dir, work_dir, stderr_guard)

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
    stderr_guard=True,
    process_number=None,
):
    """Write all ground-up loss (GUL) consumer commands.

    Emits ``do_ord``, ``do_tees``, and ``do_summarycalcs`` commands for every
    process in the range.

    Args:
        analysis_settings (dict): The full analysis settings dictionary.
        max_process_id (int): Upper bound of process IDs.
        filename (str): Path to the bash script being generated.
        process_counter (dict): Mutable counter dict tracking background PIDs.
        fifo_dir (str): Base directory for FIFOs.
        work_dir (str): Root working directory.
        stderr_guard (bool): If True, wrap commands in a sub-shell that
            redirects stderr to the log.
        process_number (int or None): If set, restrict to a single process.
    """

    for process_id in process_range(max_process_id, process_number):
        do_ord(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename,
               process_counter, fifo_dir, work_dir, stderr_guard)

    for process_id in process_range(max_process_id, process_number):
        do_tees(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_id, filename, process_counter, fifo_dir, work_dir)

    for process_id in process_range(max_process_id, process_number):
        do_summarycalcs(
            runtype=RUNTYPE_GROUNDUP_LOSS,
            analysis_settings=analysis_settings,
            process_id=process_id,
            filename=filename,
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
    stderr_guard=None,
    process_number=None,
):
    """Write GUL consumer commands for the full-correlation path.

    Similar to :func:`do_gul` but operates on the ``full_correlation/``
    sub-directories and only emits ``do_tees`` (no ``do_ord`` or
    ``do_summarycalcs``).

    Args:
        analysis_settings (dict): The full analysis settings dictionary.
        max_process_id (int): Upper bound of process IDs.
        filename (str): Path to the bash script being generated.
        process_counter (dict): Mutable counter dict tracking background PIDs.
        fifo_dir (str): FIFO directory for full-correlation pipes.
        work_dir (str): Work directory for full-correlation outputs.
        stderr_guard: Unused (kept for interface consistency).
        process_number (int or None): If set, restrict to a single process.
    """

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


def get_getmodel_cmd(
        number_of_samples,
        gul_threshold,
        use_random_number_file,
        gul_alloc_rule,
        item_output,
        process_id,
        max_process_id,
        correlated_output,
        eve_shuffle_flag,
        modelpy_server=False,
        peril_filter=[],
        gulmc=False,
        gul_random_generator=1,
        gulmc_effective_damageability=False,
        gulmc_vuln_cache_size=200,
        model_df_engine='oasis_data_manager.df_reader.reader.OasisPandasReader',
        dynamic_footprint=False,
        **kwargs):
    """
    Gets the GUL pipeline command (gulpy/gulmc) for a single process
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
    :param model_df_engine: The engine to use when loading dataframes
    :type  model_df_engine: str
    :return: The generated getmodel command
    """
    # events
    cmd = f'evepy {eve_shuffle_flag}{process_id} {max_process_id} | '

    # ground up
    if gulmc is True:
        gulcmd = get_gulcmd(
            gulmc, gul_random_generator, gulmc_effective_damageability,
            gulmc_vuln_cache_size, modelpy_server, peril_filter, model_df_engine=model_df_engine,
            dynamic_footprint=dynamic_footprint
        )
        cmd += f'{gulcmd} -S{number_of_samples} -L{gul_threshold}'

    else:
        modelcmd = get_modelcmd(modelpy_server, peril_filter)
        gulcmd = get_gulcmd(gulmc, gul_random_generator, False, 0, False, 0, False, [], model_df_engine=model_df_engine)
        cmd += f'{modelcmd} | {gulcmd} -S{number_of_samples} -L{gul_threshold}'

    cmd = '{} -a{}'.format(cmd, gul_alloc_rule)
    cmd = '{} {}'.format(cmd, item_output)

    return cmd


def add_pid_to_shell_command(cmd, process_counter):
    """
    Add a variable to the end of a command in order to track the ID of the process executing it.
    Each time this function is called, the counter `process_counter` is incremented.

    Args:
        cmd (str): the command whose process ID is to be stored in a variable.
        process_counter (Counter or dict): the number of process IDs that are being tracked.

    Returns:
        cmd (str): the updated command string.
    """

    process_counter["pid_monitor_count"] += 1
    cmd = f'{cmd} pid{process_counter["pid_monitor_count"]}=$!'

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
    fmpy_low_memory=False,
    fmpy_sort_output=False,
    step_flag='',
    process_counter=None,
    ri_inuring_priorities=None,
    rl_inuring_priorities=None
):
    """
    Gets the fmpy command for the reinsurance stream
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
    :param ri_inuring_priorities: Inuring priorities where net output has been requested
    :type ri_inuring_priorities: dict
    :param rl_inuring_priorities: Inuring priorities where gross output has been requested
    :type rl_inuring_priorities: dict
    """
    if from_file:
        main_cmd = f'{get_fmcmd(fmpy_low_memory, fmpy_sort_output)} -a{il_alloc_rule}{step_flag} < {cmd}'
    else:
        main_cmd = f'{cmd} | {get_fmcmd(fmpy_low_memory, fmpy_sort_output)} -a{il_alloc_rule}{step_flag}'

    if il_output:
        main_cmd += f" | tee {get_fifo_name(fifo_dir, RUNTYPE_INSURED_LOSS, process_id)}"

    for i in range(1, num_reinsurance_iterations + 1):
        main_cmd += f" | {get_fmcmd(fmpy_low_memory, fmpy_sort_output)} -a{ri_alloc_rule} -p {os.path.join('input', 'RI_' + str(i))}"
        if rl_inuring_priorities:   # If rl output is requested then produce gross output at all inuring priorities
            main_cmd += f" -o {get_fifo_name(fifo_dir, RUNTYPE_REINSURANCE_GROSS_LOSS, process_id, consumer=rl_inuring_priorities[i].rstrip('_'))}"
        if i < num_reinsurance_iterations:   # Net output required to process next inuring priority
            main_cmd += ' -n -'
        if i in ri_inuring_priorities.keys():
            if i == num_reinsurance_iterations:   # Final inuring priority always produces net output if ri output requested
                ri_fifo_name = get_fifo_name(fifo_dir, RUNTYPE_REINSURANCE_LOSS, process_id)
                main_cmd += f" -n - > {ri_fifo_name}"
            else:
                main_cmd += f" | tee {get_fifo_name(fifo_dir, RUNTYPE_REINSURANCE_LOSS, process_id, consumer=ri_inuring_priorities[i].rstrip('_'))}"

    main_cmd = f'( {main_cmd} ) 2>> $LOG_DIR/stderror.err' if stderr_guard else f'{main_cmd}'
    main_cmd = f'( {main_cmd} ) &'

    if process_counter is not None:
        main_cmd = add_pid_to_shell_command(main_cmd, process_counter)

    return main_cmd


def get_main_cmd_il_stream(
    cmd,
    process_id,
    il_alloc_rule,
    fifo_dir='fifo/',
    stderr_guard=True,
    from_file=False,
    fmpy_low_memory=False,
    fmpy_sort_output=False,
    step_flag='',
    process_counter=None,
):
    """
    Gets the fmpy command for the insured losses stream
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
        main_cmd = f'{get_fmcmd(fmpy_low_memory, fmpy_sort_output)} -a{il_alloc_rule}{step_flag} < {cmd} > {il_fifo_name}'
    else:
        # need extra space at the end to pass test
        main_cmd = f'{cmd} | {get_fmcmd(fmpy_low_memory, fmpy_sort_output)} -a{il_alloc_rule}{step_flag} > {il_fifo_name} '

    main_cmd = f'( {main_cmd} ) 2>> $LOG_DIR/stderror.err' if stderr_guard else f'{main_cmd}'
    main_cmd = f'( {main_cmd} ) &'

    if process_counter is not None:
        main_cmd = add_pid_to_shell_command(main_cmd, process_counter)

    return main_cmd


def get_main_cmd_gul_stream(
    cmd,
    process_id,
    fifo_dir='fifo/',
    stderr_guard=True,
    consumer='',
    process_counter=None,
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
    main_cmd = f'( {main_cmd} ) 2>> $LOG_DIR/stderror.err' if stderr_guard else f'{main_cmd}'
    main_cmd = f'( {main_cmd} ) & '

    if process_counter is not None:
        main_cmd = add_pid_to_shell_command(main_cmd, process_counter)

    return main_cmd


def get_complex_model_cmd(custom_gulcalc_cmd, analysis_settings):
    """Return a custom GUL command function for complex (third-party) models.

    If ``custom_gulcalc_cmd`` is explicitly provided it must exist in
    ``PATH``; otherwise the function infers a command name from
    ``<supplier>_<model>_gulcalc`` and checks for its presence.

    Args:
        custom_gulcalc_cmd (str or None): Explicit custom GUL binary name,
            or None to attempt auto-detection.
        analysis_settings (dict): The full analysis settings dictionary
            (used to infer the binary name via ``model_supplier_id`` and
            ``model_name_id``).

    Returns:
        callable or None: A function with the same signature as
        :func:`get_getmodel_cmd` that builds the custom GUL command string,
        or None if no custom binary was found.

    Raises:
        OasisException: If ``custom_gulcalc_cmd`` is explicitly set but
            cannot be found on ``PATH``.
    """
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
            if item_output != '':
                cmd = '{} -i {}'.format(cmd, item_output)
            if stderr_guard:
                cmd = '({}) 2>> $LOG_DIR/gul_stderror.err'.format(cmd)

            return cmd
    else:
        custom_get_getmodel_cmd = None
    return custom_get_getmodel_cmd


def do_computes(outputs):
    """Execute a list of deferred compute operations.

    Each entry in *outputs* is a dict with keys ``'compute_fun'``,
    ``'compute_args'``, and ``'loss_type'``.  The function calls
    ``compute_fun(**compute_args)`` for every entry, preceded by a
    comment header identifying the loss type.

    Args:
        outputs (list[dict]): Deferred compute descriptors built up by the
            caller (e.g. :func:`create_bash_analysis`).
    """

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
    """Yield ``load_balancer`` commands that redistribute streams across FIFOs.

    Each load balancer reads from *num_in_per_lb* input FIFOs and writes to
    *num_out_per_lb* output FIFOs.  Process IDs are assigned sequentially
    across all load balancers.

    Args:
        num_lb (int): Number of load balancer processes.
        num_in_per_lb (int): Number of input FIFOs per load balancer.
        num_out_per_lb (int): Number of output FIFOs per load balancer.
        get_input_stream_name (callable): Function accepting ``producer_id``
            and returning the input FIFO path.
        get_output_stream_name (callable): Function accepting ``producer_id``
            and returning the output FIFO path.
        stderr_guard (bool): If True, wrap commands in a sub-shell that
            redirects stderr to the log.

    Yields:
        str: A shell command string for each load balancer process.
    """
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
        lb_main_cmd = f'( {lb_main_cmd} ) 2>> $LOG_DIR/stderror.err &' if stderr_guard else f'{lb_main_cmd} &'
        yield lb_main_cmd


def get_pla_cmd(pla, secondary_factor, uniform_factor):
    """
    Determine whether Post Loss Amplification should be implemented and issue
    plapy command.

    Args:
        pla (bool): flag to apply post loss amplification
        secondary_factor (float): secondary factor to apply to post loss
          amplification
        uniform_factor (float): uniform factor to apply across all losses

    Returns:
        pla_cmd (str): post loss amplification command
    """
    pla_cmd = ' | plapy' * pla
    if pla:
        if uniform_factor > 0:
            pla_cmd += f' -F {uniform_factor}'
        elif secondary_factor != 1:
            pla_cmd += f' -f {secondary_factor}'
    return pla_cmd


def bash_params(
    analysis_settings,
    max_process_id=-1,
    number_of_processes=-1,
    num_reinsurance_iterations=0,
    model_storage_json=None,
    fifo_tmp_dir=True,
    gul_alloc_rule=None,
    il_alloc_rule=None,
    ri_alloc_rule=None,
    num_gul_per_lb=None,
    num_fm_per_lb=None,
    stderr_guard=True,
    bash_trace=False,
    filename='run_kools.sh',
    _get_getmodel_cmd=None,
    custom_gulcalc_cmd=None,
    custom_gulcalc_log_start=None,
    custom_gulcalc_log_finish=None,
    custom_args={},
    fmpy_low_memory=False,
    fmpy_sort_output=False,
    event_shuffle=None,
    gulmc=True,
    gul_random_generator=1,
    gulmc_effective_damageability=False,
    gulmc_vuln_cache_size=200,

    # new options
    process_number=None,
    remove_working_files=True,
    model_run_dir='',
    model_py_server=False,
    join_summary_info=False,
    peril_filter=[],
    exposure_df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
    model_df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
    dynamic_footprint=False,
    **kwargs
):
    """Build the parameter dict consumed by :func:`create_bash_analysis` and :func:`create_bash_outputs`.

    Validates analysis settings (checking output/summary coherence, reinsurance
    iteration requirements), resolves default allocation and event-shuffle rules,
    sets up directory paths (FIFO, work, output), and determines the
    full-correlation and complex-model flags.

    Args:
        analysis_settings (dict): The full analysis settings dictionary.
        max_process_id (int): Number of parallel processes (defaults to CPU
            count when ``<= 0``).
        number_of_processes (int): Alias used elsewhere (defaults to CPU
            count when ``<= 0``).
        num_reinsurance_iterations (int): Number of reinsurance iterations.
        model_storage_json (dict or None): Optional model storage metadata.
        fifo_tmp_dir (bool): If True, create FIFOs under ``/tmp/``.
        gul_alloc_rule (int or None): GUL allocation rule override.
        il_alloc_rule (int or None): IL allocation rule override.
        ri_alloc_rule (int or None): RI allocation rule override.
        num_gul_per_lb (int or None): GUL streams per load balancer.
        num_fm_per_lb (int or None): FM streams per load balancer.
        stderr_guard (bool): Wrap commands with stderr redirection.
        bash_trace (bool): Enable bash ``-x`` tracing.
        filename (str): Output script filename.
        _get_getmodel_cmd (callable or None): Custom GUL command builder.
        custom_gulcalc_cmd (str or None): Explicit custom GUL binary name.
        custom_gulcalc_log_start (str or None): Custom log-start marker.
        custom_gulcalc_log_finish (str or None): Custom log-finish marker.
        custom_args (dict): Extra arguments forwarded to downstream functions.
        fmpy_low_memory (bool): Enable low-memory mode in ``fmpy``.
        fmpy_sort_output (bool): Sort ``fmpy`` output.
        event_shuffle (int or None): Event shuffle rule override.
        gulmc (bool): Use ``gulmc`` (Monte Carlo sampler).
        gul_random_generator (int): Random number generator selector.
        gulmc_effective_damageability (bool): Enable effective damageability.
        gulmc_vuln_cache_size (int): Vulnerability cache size for ``gulmc``.
        process_number (int or None): Single chunk number (distributed mode).
        remove_working_files (bool): Clean up work dirs after completion.
        model_run_dir (str): Root directory for the model run.
        model_py_server (bool): Launch a ``servedata`` server.
        join_summary_info (bool): Append summary metadata to outputs.
        peril_filter (list): Peril filter list.
        exposure_df_engine (str): DataFrame engine for exposure data.
        model_df_engine (str): DataFrame engine for model data.
        dynamic_footprint (bool): Enable dynamic footprint mode.

    Returns:
        dict: Parameter dictionary ready for unpacking into
        :func:`create_bash_analysis` and :func:`create_bash_outputs`.

    Raises:
        OasisException: If no valid output settings are found or an unknown
            event shuffle rule is specified.
    """

    bash_params = {}
    bash_params['max_process_id'] = max_process_id if max_process_id > 0 else multiprocessing.cpu_count()
    bash_params['number_of_processes'] = number_of_processes if number_of_processes > 0 else multiprocessing.cpu_count()
    bash_params['process_counter'] = Counter()
    bash_params['num_reinsurance_iterations'] = num_reinsurance_iterations
    bash_params['fifo_tmp_dir'] = fifo_tmp_dir
    bash_params['bash_trace'] = bash_trace
    bash_params['filename'] = filename
    bash_params['custom_args'] = custom_args
    bash_params['gulmc'] = gulmc
    bash_params['gul_random_generator'] = gul_random_generator
    bash_params['gulmc_effective_damageability'] = gulmc_effective_damageability
    bash_params['gulmc_vuln_cache_size'] = gulmc_vuln_cache_size
    bash_params['fmpy_low_memory'] = fmpy_low_memory
    bash_params['fmpy_sort_output'] = fmpy_sort_output
    bash_params['process_number'] = process_number
    bash_params['remove_working_files'] = remove_working_files
    bash_params['model_run_dir'] = model_run_dir

    if model_storage_json:
        bash_params['model_storage_json'] = model_storage_json

    bash_params['gul_threshold'] = analysis_settings.get('gul_threshold', 0)
    bash_params['number_of_samples'] = analysis_settings.get('number_of_samples', 0)
    bash_params["static_path"] = os.path.join(model_run_dir, "static/")

    bash_params["model_py_server"] = model_py_server
    bash_params['join_summary_info'] = join_summary_info
    bash_params["peril_filter"] = peril_filter

    # set complex model gulcalc command
    if not _get_getmodel_cmd and custom_gulcalc_cmd:
        bash_params['_get_getmodel_cmd'] = get_complex_model_cmd(custom_gulcalc_cmd, analysis_settings)
    else:
        bash_params['_get_getmodel_cmd'] = _get_getmodel_cmd

    # Set custom gulcalc log statment checks,
        bash_params['custom_gulcalc_log_start'] = custom_gulcalc_log_start or analysis_settings.get('model_custom_gulcalc_log_start')
        bash_params['custom_gulcalc_log_finish'] = custom_gulcalc_log_finish or analysis_settings.get('model_custom_gulcalc_log_finish')

    # Set fifo dirs
    if fifo_tmp_dir:
        bash_params['fifo_queue_dir'] = '/tmp/{}/fifo/'.format(''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)))
    else:
        bash_params['fifo_queue_dir'] = os.path.join(model_run_dir, 'fifo/')

    # set work dir
    if process_number:
        work_base_dir = f'{process_number}.work/'
    else:
        work_base_dir = 'work/'

    # set dirs
    bash_params['stderr_guard'] = stderr_guard
    bash_params['work_dir'] = os.path.join(model_run_dir, work_base_dir)
    bash_params['work_kat_dir'] = os.path.join(model_run_dir, os.path.join(work_base_dir, 'kat/'))
    bash_params['work_full_correlation_dir'] = os.path.join(model_run_dir, os.path.join(work_base_dir, 'full_correlation/'))
    bash_params['work_full_correlation_kat_dir'] = os.path.join(model_run_dir, os.path.join(work_base_dir, 'full_correlation/kat/'))
    bash_params['output_dir'] = os.path.join(model_run_dir, 'output/')
    bash_params['output_full_correlation_dir'] = os.path.join(model_run_dir, 'output/full_correlation/')
    bash_params['fifo_full_correlation_dir'] = os.path.join(bash_params['fifo_queue_dir'], 'full_correlation/')

    # Set default alloc/shuffle rules if missing
    bash_params['gul_alloc_rule'] = gul_alloc_rule if isinstance(gul_alloc_rule, int) else KERNEL_ALLOC_GUL_DEFAULT
    bash_params['il_alloc_rule'] = il_alloc_rule if isinstance(il_alloc_rule, int) else KERNEL_ALLOC_IL_DEFAULT
    bash_params['ri_alloc_rule'] = ri_alloc_rule if isinstance(ri_alloc_rule, int) else KERNEL_ALLOC_RI_DEFAULT
    bash_params['num_gul_per_lb'] = num_gul_per_lb if isinstance(num_gul_per_lb, int) else KERNEL_N_GUL_PER_LB
    bash_params['num_fm_per_lb'] = num_fm_per_lb if isinstance(num_fm_per_lb, int) else KERNEL_N_FM_PER_LB

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
        if _get_getmodel_cmd is None:
            full_correlation = analysis_settings['full_correlation']
            if full_correlation and gulmc:
                full_correlation = False
                logger.info("full_correlation has been disable as it isn't compatible with gulmc, see oasislmf correlation documentation.")
    bash_params['full_correlation'] = full_correlation

    # Output depends on being enabled AND having at least one summaries section
    # checking output settings coherence
    for mod in ['gul', 'il', 'ri', 'rl']:
        if analysis_settings.get(f'{mod}_output') and not analysis_settings.get(f'{mod}_summaries'):
            logger.warning(f'{mod}_output set to True but there is no {mod}_summaries')
            analysis_settings[f'{mod}_output'] = False

    # additional check for ri and rl, must have num_reinsurance_iterations
    if not num_reinsurance_iterations:
        for runtype in REINSURANCE_RUNTYPES:
            if analysis_settings.get(f'{runtype}_output', False):
                logger.warning(f'{runtype}_output set to True but there are no reinsurance layers')
                analysis_settings[f'{runtype}_output'] = False

    if not any(analysis_settings.get(f'{mod}_output') for mod in ['gul', 'il', 'ri', 'rl']):
        raise OasisException('No valid output settings')

    # Get perfecting values from 'analysis_settings' settings)
    bash_params['analysis_settings'] = analysis_settings
    bash_params['gul_output'] = analysis_settings.get('gul_output', False)
    bash_params['il_output'] = analysis_settings.get('il_output', False)
    bash_params['ri_output'] = analysis_settings.get('ri_output', False)
    bash_params['rl_output'] = analysis_settings.get('rl_output', False)
    bash_params['need_summary_fifo_for_gul'] = bash_params['gul_output'] and (
        bash_params['il_output'] or bash_params['ri_output'] or bash_params['rl_output']
    )
    bash_params['exposure_df_engine'] = exposure_df_engine
    bash_params['model_df_engine'] = model_df_engine
    bash_params['dynamic_footprint'] = dynamic_footprint

    return bash_params


@contextlib.contextmanager
def bash_wrapper(
    filename,
    bash_trace,
    stderr_guard,
    log_sub_dir=None,
    process_number=None,
    custom_gulcalc_log_start=None,
    custom_gulcalc_log_finish=None
):
    """Context manager that wraps the script body with header and footer boilerplate.

    On entry, writes the bash shebang, shell options (``set -euET``), log
    directory setup, optional bash tracing, the error-trap function, and the
    completion-check function.

    On exit (after the ``yield``), writes the footer: either a
    ``check_complete`` call (single-script mode) or a chunk-validation block
    that verifies no output files are empty (distributed-chunk mode).

    Args:
        filename (str): Path to the bash script being generated.
        bash_trace (bool): If True, enable bash ``-x`` tracing.
        stderr_guard (bool): If True, install the error trap and write the
            completion-check function.
        log_sub_dir (str or None): Sub-directory under ``log/`` for chunk
            mode logging.
        process_number (int or None): Chunk number when running in
            distributed mode.
        custom_gulcalc_log_start (str or None): Custom log-start marker for
            the completion check.
        custom_gulcalc_log_finish (str or None): Custom log-finish marker for
            the completion check.

    Yields:
        None: Control is yielded to the caller to write the script body.
    """
    # Header
    print_command(filename, '#!/bin/bash')
    print_command(filename, 'SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")')
    print_command(filename, '')
    print_command(filename, '# --- Script Init ---')
    print_command(filename, 'set -euET -o pipefail')
    print_command(filename, 'shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."')

    print_command(filename, '')
    if process_number:
        print_command(filename, f'LOG_DIR=log/{log_sub_dir}')
    else:
        print_command(filename, 'LOG_DIR=log')

    print_command(filename, 'mkdir -p $LOG_DIR')
    print_command(filename, 'rm -R -f $LOG_DIR/*')
    print_command(filename, '')

    # Trap func and logging
    if bash_trace:
        print_command(filename, BASH_TRACE)
    if stderr_guard:
        print_command(filename, TRAP_FUNC)
        print_command(filename, get_check_function(custom_gulcalc_log_start, custom_gulcalc_log_finish))

    # Script content
    yield

    # Script footer
    if stderr_guard and not process_number:
        # run process dropped check (single script run)
        print_command(filename, '')
        print_command(filename, 'check_complete')
    elif stderr_guard and process_number:
        # check stderror.err before exit (fallback check in case of short run)
        print_command(filename, 'if [ -s $LOG_DIR/stderror.err ]; then')
        print_command(
            filename, '    if grep -qvE "(^[[:space:]]*$|.+:[0-9]+: [A-Za-z]+Warning:|^[[:space:]]+warnings\\.warn)" $LOG_DIR/stderror.err; then')
        print_command(filename, '        echo "Error detected in $LOG_DIR/stderror.err"')
        print_command(filename, '        exit 1')
        print_command(filename, '    fi')
        print_command(filename, 'fi')
        # check for empty work bin files
        print_command(filename, f'CHUNK_BINS=(`find {process_number}.work -name \'P{process_number}.bin\' | sort -r`)')
        print_command(filename, 'echo " === Checking analysis output chunks === "')
        print_command(filename, 'for b in "${CHUNK_BINS[@]}"; do')
        print_command(filename, '    wc -c $b')
        print_command(filename, 'done')
        print_command(filename, '')
        print_command(filename, '# exit error if empty')
        print_command(filename, 'for b in "${CHUNK_BINS[@]}"; do')
        print_command(filename, '    if [ ! -s $b ]; then')
        print_command(filename, '        echo "Chunk output error: File \'$b\' is empty"')
        print_command(filename, '        exit 1')
        print_command(filename, '    fi')
        print_command(filename, 'done')
        print_command(filename, 'echo "Chunk output check [OK]"')


def create_bash_analysis(
    process_counter,
    max_process_id,
    num_reinsurance_iterations,
    fifo_tmp_dir,
    bash_trace,
    filename,
    _get_getmodel_cmd,
    custom_args,
    fmpy_low_memory,
    fmpy_sort_output,
    process_number,
    remove_working_files,
    model_run_dir,
    fifo_queue_dir,
    fifo_full_correlation_dir,
    stderr_guard,
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
    rl_output,
    need_summary_fifo_for_gul,
    analysis_settings,
    gulmc,
    gul_random_generator,
    gulmc_effective_damageability,
    gulmc_vuln_cache_size,
    model_py_server,
    peril_filter,
    model_df_engine='oasis_data_manager.df_reader.reader.OasisPandasReader',
    dynamic_footprint=False,
    **kwargs
):
    """Write the main analysis section of the bash script.

    This is the core function that assembles the calculation pipeline.  It:

    1. Sets up output, FIFO, and work directories.
    2. Creates FIFOs for every (process, summary, ORD-type) combination.
    3. Creates work-folder directories for post-wait aggregation.
    4. Writes consumer commands in reverse pipeline order (ORD consumers,
       tees, summarycalcs) so readers are ready before writers.
    5. Writes the main GUL pipeline commands (``evepy | modelpy | gulpy/gulmc``
       piped through ``fmpy`` for IL/RI), optionally with load balancers.
    6. Adds ``wait`` calls for all background processes.
    7. Cleans up FIFOs.

    All parameters are typically supplied by unpacking the dict returned from
    :func:`bash_params`.
    """
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
        print_command(filename, 'mkdir -p {}'.format(output_full_correlation_dir))

    print_command(filename, '')
    if not fifo_tmp_dir:

        if not process_number:
            print_command(filename, 'rm -R -f {}*'.format(fifo_queue_dir))
        else:
            print_command(
                filename, f"find {fifo_queue_dir} \( -name '*P{process_number}[^0-9]*' -o -name '*P{process_number}' \)" + " -exec rm -R -f {} +")

        if full_correlation:
            print_command(filename, 'mkdir -p {}'.format(fifo_full_correlation_dir))

    # if not process_number:
    print_command(filename, 'rm -R -f {}*'.format(work_dir))
    # else:
    #    print_command(filename, f"find {work_dir} \( -name '*P{process_number}[^0-9]*' -o -name '*P{process_number}' \)" + " -exec rm -R -f {} +")

    print_command(filename, 'mkdir -p {}'.format(work_kat_dir))
    if full_correlation:
        print_command(filename, 'mkdir -p {}'.format(work_full_correlation_dir))
        print_command(
            filename, 'mkdir -p {}'.format(work_full_correlation_kat_dir)
        )

    if model_py_server:
        print_command(filename, '# --- run data server ---')
        print_command(command_file=filename, cmd=f"servedata {kwargs['static_path']} {max_process_id} &")
        # print_command(command_file=filename, cmd="while ! nc -vz localhost 8080 < /dev/null > /dev/null 2>&1; do")
        # print_command(command_file=filename, cmd="  printf '.'")
        # print_command(command_file=filename, cmd="  sleep 2")
        # print_command(command_file=filename, cmd="done")

    print_command(filename, '')

    if il_output or ri_output or rl_output:
        print_command(
            filename, f'#{get_fmcmd()} -a{il_alloc_rule} --create-financial-structure-files'
        )
    if ri_output or rl_output:
        for i in range(1, num_reinsurance_iterations + 1):
            print_command(
                filename, f"#{get_fmcmd()} -a{ri_alloc_rule} --create-financial-structure-files -p {os.path.join('input', 'RI_' + str(i))}")

    # Create FIFOS under /tmp/* (Windows support)
    if fifo_tmp_dir:

        # workaround to match bash tests
        if not process_number:
            if fifo_queue_dir.endswith('fifo/'):
                print_command(filename, 'rm -R -f {}'.format(fifo_queue_dir[:-5]))
            else:
                print_command(filename, 'rm -R -f {}'.format(fifo_queue_dir))

        print_command(filename, 'mkdir -p {}'.format(fifo_queue_dir))
        if full_correlation:
            print_command(filename, 'mkdir -p {}'.format(fifo_full_correlation_dir))

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
        for inuring_priority in get_ri_inuring_priorities(analysis_settings, num_reinsurance_iterations):
            create_workfolders(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings, filename, work_dir,
                inuring_priority=inuring_priority['text']
            )
        if full_correlation:
            create_workfolders(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings,
                filename, work_full_correlation_dir
            )
    if rl_output:
        for inuring_priority in get_rl_inuring_priorities(num_reinsurance_iterations):
            create_workfolders(
                RUNTYPE_REINSURANCE_GROSS_LOSS, analysis_settings, filename,
                work_dir, inuring_priority=inuring_priority['text']
            )
    print_command(filename, '')

    # infer number of calc block and FIFO to create, (no load balancer for old stream option)
    if num_gul_per_lb and num_fm_per_lb and (il_output or ri_output):
        block_process_size = num_gul_per_lb + (num_fm_per_lb * (2 if ri_output else 1))
        num_lb = (max_process_id - 1) // block_process_size + 1
        num_gul_output = num_lb * num_gul_per_lb
        num_fm_output = num_lb * num_fm_per_lb
    else:
        num_lb = 0
        num_gul_output = num_fm_output = max_process_id

    # --- Create FIFOs ---
    # Build the list of FIFO directories (standard + full-correlation if enabled)
    fifo_dirs = [fifo_queue_dir]
    if full_correlation:
        fifo_dirs.append(fifo_full_correlation_dir)
        if (il_output or ri_output) and (gul_output or not num_lb):
            # create fifo for il or ri full correlation compute
            do_fifos_exec(RUNTYPE_GROUNDUP_LOSS, num_gul_output, filename, fifo_full_correlation_dir,
                          process_number, consumer=RUNTYPE_FULL_CORRELATION)

    for fifo_dir in fifo_dirs:
        # create fifos for Summarycalc
        if gul_output:
            do_fifos_exec(RUNTYPE_GROUNDUP_LOSS, num_gul_output, filename, fifo_dir, process_number)
            do_fifos_calc(RUNTYPE_GROUNDUP_LOSS, analysis_settings, num_gul_output, filename, fifo_dir, process_number)
        if il_output:
            do_fifos_exec(RUNTYPE_INSURED_LOSS, num_fm_output, filename, fifo_dir, process_number)
            do_fifos_calc(RUNTYPE_INSURED_LOSS, analysis_settings, num_fm_output, filename, fifo_dir, process_number)
        if ri_output:
            for inuring_priority in get_ri_inuring_priorities(analysis_settings, num_reinsurance_iterations):
                do_fifos_exec(
                    RUNTYPE_REINSURANCE_LOSS, num_fm_output, filename,
                    fifo_dir, process_number,
                    consumer=inuring_priority['text'].rstrip('_')
                )
                do_fifos_calc(
                    RUNTYPE_REINSURANCE_LOSS, analysis_settings, num_fm_output,
                    filename, fifo_dir, process_number,
                    consumer_prefix=inuring_priority['text']
                )
        if rl_output:
            for inuring_priority in get_rl_inuring_priorities(num_reinsurance_iterations):
                do_fifos_exec(
                    RUNTYPE_REINSURANCE_GROSS_LOSS, num_fm_output, filename,
                    fifo_dir, process_number,
                    consumer=inuring_priority['text'].rstrip('_')
                )
                do_fifos_calc(
                    RUNTYPE_REINSURANCE_GROSS_LOSS, analysis_settings,
                    num_fm_output, filename, fifo_dir, process_number,
                    consumer_prefix=inuring_priority['text']
                )

        # create fifos for Load balancer
        if num_lb:
            do_fifos_exec(RUNTYPE_GROUNDUP_LOSS, num_gul_output, filename, fifo_dir, process_number, consumer=RUNTYPE_LOAD_BALANCED_LOSS)
            do_fifos_exec(RUNTYPE_LOAD_BALANCED_LOSS, num_fm_output, filename, fifo_dir, process_number, consumer=RUNTYPE_INSURED_LOSS)

    print_command(filename, '')
    dirs = [(fifo_queue_dir, work_dir)]
    if full_correlation:
        dirs.append((fifo_full_correlation_dir, work_full_correlation_dir))

    # --- Build deferred consumer commands (ORD, tee, summarycalc) ---
    # These are written in reverse pipeline order so readers are ready
    # before writers start producing data.
    compute_outputs = []
    for (_fifo_dir, _work_dir) in dirs:
        if rl_output:
            rl_computes = {
                'loss_type': 'reinsurance gross',
                'compute_fun': rl,
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
            compute_outputs.append(rl_computes)

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
                    'stderr_guard': stderr_guard,
                    'process_number': process_number
                }
            }
            compute_outputs.append(gul_computes)

    do_computes(compute_outputs)

    print_command(filename, '')

    # --- Build main GUL pipeline commands ---
    # Accumulate (command, from_file) pairs per FIFO directory.
    # Each entry is either a getmodel pipeline command string or a FIFO path
    # to read from (when a load balancer is used).
    get_gul_stream_cmds = {}

    if kwargs.get("socket_server", False) and kwargs.get("analysis_pk", None) is None:
        print_command(filename, f"socket-server {kwargs['socket_server']} > /dev/null & spid=$!")
        print_command(filename, "trap 'kill -TERM -\"$spid\" 2>/dev/null' INT TERM")

    # WARNING: this probably wont work well with the load balancer (needs guard/ edit)
    # for gul_id in range(1, num_gul_output + 1):
    for gul_id in process_range(num_gul_output, process_number):
        getmodel_args = {
            'number_of_samples': number_of_samples,
            'gul_threshold': gul_threshold,
            'use_random_number_file': use_random_number_file,
            'gul_alloc_rule': gul_alloc_rule,
            'process_id': gul_id,
            'max_process_id': num_gul_output,
            'stderr_guard': stderr_guard,
            'eve_shuffle_flag': eve_shuffle_flag,
            'gulmc': gulmc,
            'gul_random_generator': gul_random_generator,
            'gulmc_effective_damageability': gulmc_effective_damageability,
            'gulmc_vuln_cache_size': gulmc_vuln_cache_size,
            'modelpy_server': model_py_server,
            'peril_filter': peril_filter,
            "model_df_engine": model_df_engine,
            'dynamic_footprint': dynamic_footprint
        }

        # Establish whether items to amplifications map file is present
        pla = os.path.isfile(
            os.path.join(os.getcwd(), 'input/amplifications.bin')
        )

        # GUL coverage
        gul_fifo_name = get_fifo_name(fifo_queue_dir, RUNTYPE_GROUNDUP_LOSS, gul_id)
        getmodel_args['item_output'] = ''
        getmodel_args['item_output'] = getmodel_args['item_output'] + get_pla_cmd(
            analysis_settings.get('pla', False),
            analysis_settings.get('pla_secondary_factor', 1),
            analysis_settings.get('pla_uniform_factor', 0)
        )
        if need_summary_fifo_for_gul:
            getmodel_args['item_output'] = '{} | tee {}'.format(getmodel_args['item_output'], gul_fifo_name)
        _get_getmodel_cmd = (_get_getmodel_cmd or get_getmodel_cmd)

        # gulcalc output file for fully correlated output
        if full_correlation:
            fc_gul_fifo_name = get_fifo_name(fifo_full_correlation_dir, RUNTYPE_GROUNDUP_LOSS, gul_id)
            if need_summary_fifo_for_gul:  # need both stream for summary and tream for il
                getmodel_args['correlated_output'] = get_fifo_name(fifo_full_correlation_dir, RUNTYPE_GROUNDUP_LOSS, gul_id,
                                                                   consumer=RUNTYPE_FULL_CORRELATION)
                if num_lb:
                    tee_output = get_fifo_name(fifo_full_correlation_dir, RUNTYPE_GROUNDUP_LOSS, gul_id,
                                               consumer=RUNTYPE_LOAD_BALANCED_LOSS)
                    tee_cmd = f"tee < {getmodel_args['correlated_output']} {fc_gul_fifo_name} > {tee_output} &"
                    print_command(filename, add_server_call(tee_cmd, kwargs.get("analysis_pk", None), kwargs.get("socket_server", False)))

                else:
                    tee_output = get_fifo_name(fifo_full_correlation_dir, RUNTYPE_GROUNDUP_LOSS, gul_id,
                                               consumer=RUNTYPE_INSURED_LOSS)
                    tee_cmd = f"tee < {getmodel_args['correlated_output']} {fc_gul_fifo_name} "
                    get_gul_stream_cmds.setdefault(fifo_full_correlation_dir, []).append((tee_cmd, False))

            elif gul_output:  # only gul direct correlated_output to summary
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

        if num_lb:  # print main_cmd_gul_stream, get_gul_stream_cmds will be updated after by the main lb block
            main_cmd_gul_stream = get_main_cmd_gul_stream(
                getmodel_cmd, gul_id, fifo_queue_dir, stderr_guard, RUNTYPE_LOAD_BALANCED_LOSS
            )
            print_command(filename, add_server_call(main_cmd_gul_stream, kwargs.get("analysis_pk", None), kwargs.get("socket_server", False)))
        else:
            get_gul_stream_cmds.setdefault(fifo_queue_dir, []).append((getmodel_cmd, False))

    if num_lb:  # create load balancer cmds
        for fifo_dir in fifo_dirs:
            get_gul_stream_cmds[fifo_dir] = [
                (get_fifo_name(fifo_dir, RUNTYPE_LOAD_BALANCED_LOSS, fm_id, RUNTYPE_INSURED_LOSS), True) for
                fm_id in range(1, num_fm_output + 1)]

            # print the load balancing command
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
                print_command(filename, add_server_call(lb_main_cmd, kwargs.get("analysis_pk", None), kwargs.get("socket_server", False)))

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

    # --- Route GUL streams to downstream pipelines (RI, IL, or GUL-only) ---
    for fifo_dir, gul_streams in get_gul_stream_cmds.items():
        for i, (getmodel_cmd, from_file) in enumerate(gul_streams):

            # THIS NEEDS EDIT - temp workaround for dist work chunk
            if process_number is not None:
                process_id = process_number
            else:
                process_id = i + 1
            #######################################################

            if ri_output or rl_output:
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
                    fmpy_low_memory,
                    fmpy_sort_output,
                    step_flag,
                    process_counter=process_counter,
                    ri_inuring_priorities={ip['level']: ip['text'] for ip in get_ri_inuring_priorities(
                        analysis_settings, num_reinsurance_iterations) if ip['level'] and ri_output},
                    rl_inuring_priorities={ip['level']: ip['text'] for ip in get_rl_inuring_priorities(num_reinsurance_iterations) if rl_output}
                )
                print_command(filename, add_server_call(main_cmd, kwargs.get("analysis_pk", None), kwargs.get("socket_server", False)))

            elif il_output:
                main_cmd = get_main_cmd_il_stream(
                    getmodel_cmd, process_id, il_alloc_rule, fifo_dir,
                    stderr_guard,
                    from_file,
                    fmpy_low_memory,
                    fmpy_sort_output,
                    step_flag,
                    process_counter=process_counter
                )
                print_command(filename, add_server_call(main_cmd, kwargs.get("analysis_pk", None), kwargs.get("socket_server", False)))

            else:
                main_cmd = get_main_cmd_gul_stream(
                    cmd=getmodel_cmd,
                    process_id=process_id,
                    fifo_dir=fifo_dir,
                    stderr_guard=stderr_guard,
                    process_counter=process_counter,
                )
                print_command(filename, add_server_call(main_cmd, kwargs.get("analysis_pk", None), kwargs.get("socket_server", False)))

    # --- Wait for all background pipeline processes ---
    print_command(filename, '')
    do_pwaits(filename, process_counter)
    if kwargs.get("socket_server", False) and kwargs.get("analysis_pk", None) is None:
        # Ensure killed if server doesnt end
        print_command(filename, 'kill -0 "$spid" 2>/dev/null && kill -9 "$spid"')


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
    rl_output,
    analysis_settings,
    num_reinsurance_iterations,
    num_gul_per_lb,
    num_fm_per_lb,
    max_process_id,
    work_kat_dir,
    kat_sort_by_event,
    work_full_correlation_kat_dir,
    join_summary_info,
    **kwargs
):
    """Write the output-aggregation and cleanup section of the bash script.

    This function runs *after* the main analysis ``wait`` and handles:

    1. **Kat concatenation** -- ``katpy`` merges per-process binary ORD files
       into single output files for each run type.
    2. **Post-wait aggregation** -- ``aalpy`` (PALT/ALT) and ``lecpy``
       (EPT/PSEPT) run over collected work-folder data.
    3. **Cleanup** -- removes working directories and temporary FIFOs.

    All parameters are typically supplied by unpacking the dict returned from
    :func:`bash_params`.
    """

    if max_process_id is not None:
        num_gul_per_lb = 0
        num_fm_per_lb = 0

    # infer number of calc block and FIFO to create, (no load balancer for old stream option)
    if num_gul_per_lb and num_fm_per_lb and (il_output or ri_output):
        block_process_size = num_gul_per_lb + (num_fm_per_lb * (2 if ri_output else 1))
        num_lb = (max_process_id - 1) // block_process_size + 1
        num_gul_output = num_lb * num_gul_per_lb
        num_fm_output = num_lb * num_fm_per_lb
    else:
        num_lb = 0
        num_gul_output = num_fm_output = max_process_id

    # Output Kats
    if rl_output:
        print_command(filename, '')
        print_command(filename, '# --- Do reinsurance gross loss kats ---')
        print_command(filename, '')
        for inuring_priority in get_rl_inuring_priorities(num_reinsurance_iterations):
            do_kats(
                RUNTYPE_REINSURANCE_GROSS_LOSS, analysis_settings,
                num_fm_output, filename, process_counter, work_kat_dir,
                output_dir, kat_sort_by_event,
                inuring_priority=inuring_priority['text'], join_summary_info=join_summary_info
            )

    if ri_output:
        print_command(filename, '')
        print_command(filename, '# --- Do reinsurance loss kats ---')
        print_command(filename, '')
        for inuring_priority in get_ri_inuring_priorities(analysis_settings, num_reinsurance_iterations):
            do_kats(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings, num_fm_output,
                filename, process_counter, work_kat_dir, output_dir, kat_sort_by_event,
                inuring_priority=inuring_priority['text'], join_summary_info=join_summary_info
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
                output_full_correlation_dir, kat_sort_by_event, join_summary_info=join_summary_info
            )

    if il_output:
        print_command(filename, '')
        print_command(filename, '# --- Do insured loss kats ---')
        print_command(filename, '')
        do_kats(
            RUNTYPE_INSURED_LOSS, analysis_settings, num_fm_output, filename,
            process_counter, work_kat_dir, output_dir, kat_sort_by_event, join_summary_info=join_summary_info
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
                output_full_correlation_dir, kat_sort_by_event, join_summary_info=join_summary_info
            )

    if gul_output:
        print_command(filename, '')
        print_command(filename, '# --- Do ground up loss kats ---')
        print_command(filename, '')
        do_kats(
            RUNTYPE_GROUNDUP_LOSS, analysis_settings, num_gul_output, filename,
            process_counter, work_kat_dir, output_dir, kat_sort_by_event, join_summary_info=join_summary_info
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
                output_full_correlation_dir, kat_sort_by_event, join_summary_info=join_summary_info
            )

    do_kwaits(filename, process_counter)

    # Output calcs
    print_command(filename, '')
    if rl_output:
        for inuring_priority in get_rl_inuring_priorities(num_reinsurance_iterations):
            do_post_wait_processing(
                RUNTYPE_REINSURANCE_GROSS_LOSS, analysis_settings, filename,
                process_counter, '', output_dir, stderr_guard,
                inuring_priority=inuring_priority['text'], join_summary_info=join_summary_info,
            )
    if ri_output:
        for inuring_priority in get_ri_inuring_priorities(analysis_settings, num_reinsurance_iterations):
            do_post_wait_processing(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings, filename,
                process_counter, '', output_dir, stderr_guard,
                inuring_priority=inuring_priority['text'], join_summary_info=join_summary_info,
            )
    if il_output:
        do_post_wait_processing(
            RUNTYPE_INSURED_LOSS, analysis_settings, filename, process_counter, '',
            output_dir, stderr_guard, join_summary_info=join_summary_info
        )
    if gul_output:
        do_post_wait_processing(
            RUNTYPE_GROUNDUP_LOSS, analysis_settings, filename, process_counter, '',
            output_dir, stderr_guard, join_summary_info=join_summary_info
        )

    if full_correlation:
        work_sub_dir = re.sub('^work/', '', work_full_correlation_dir)
        if ri_output:
            do_post_wait_processing(
                RUNTYPE_REINSURANCE_LOSS, analysis_settings, filename, process_counter,
                work_sub_dir, output_full_correlation_dir, stderr_guard, join_summary_info=join_summary_info,
            )
        if il_output:
            do_post_wait_processing(
                RUNTYPE_INSURED_LOSS, analysis_settings, filename, process_counter,
                work_sub_dir, output_full_correlation_dir, stderr_guard, join_summary_info=join_summary_info,
            )
        if gul_output:
            do_post_wait_processing(
                RUNTYPE_GROUNDUP_LOSS, analysis_settings, filename, process_counter,
                work_sub_dir, output_full_correlation_dir, stderr_guard, join_summary_info=join_summary_info,
            )

    do_awaits(filename, process_counter)  # waits for aalcalc
    do_lwaits(filename, process_counter)  # waits for leccalc

    if remove_working_files:
        print_command(filename, 'rm -R -f {}'.format(os.path.join(work_dir, '*')))

        if fifo_tmp_dir:
            # workaround to match bash tests
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
    bash_trace=False,
    filename='run_kools.sh',
    _get_getmodel_cmd=None,
    custom_gulcalc_log_start=None,
    custom_gulcalc_log_finish=None,
    custom_args={},
    fmpy_low_memory=False,
    fmpy_sort_output=False,
    event_shuffle=None,
    gulmc=True,
    gul_random_generator=1,
    gulmc_effective_damageability=False,
    gulmc_vuln_cache_size=200,
    model_py_server=False,
    peril_filter=[],
    join_summary_info=False,
    base_df_engine='oasis_data_manager.df_reader.reader.OasisPandasReader',
    model_df_engine=None,
    dynamic_footprint=False,
    analysis_pk=None,
    socket_server=None
):
    """
    Generates a bash script containing pytools calculation instructions for an
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

    :param base_df_engine: The engine to use when loading dataframes.
    :type  base_df_engine: str

    :param model_df_engine: The engine to use when loading model dataframes.
    :type  model_df_engine: str
    """
    model_df_engine = model_df_engine or base_df_engine

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
        bash_trace=bash_trace,
        filename=filename,
        _get_getmodel_cmd=_get_getmodel_cmd,
        custom_gulcalc_log_start=custom_gulcalc_log_start,
        custom_gulcalc_log_finish=custom_gulcalc_log_finish,
        custom_args=custom_args,
        fmpy_low_memory=fmpy_low_memory,
        fmpy_sort_output=fmpy_sort_output,
        event_shuffle=event_shuffle,
        gulmc=gulmc,
        gul_random_generator=gul_random_generator,
        gulmc_effective_damageability=gulmc_effective_damageability,
        gulmc_vuln_cache_size=gulmc_vuln_cache_size,
        model_py_server=model_py_server,
        peril_filter=peril_filter,
        join_summary_info=join_summary_info,
        model_df_engine=model_df_engine,
        dynamic_footprint=dynamic_footprint
    )

    # remove the file if it already exists
    if os.path.exists(filename):
        os.remove(filename)

    params['analysis_pk'] = analysis_pk
    params['socket_server'] = socket_server

    with bash_wrapper(
        filename,
        bash_trace,
        stderr_guard,
        custom_gulcalc_log_start=params['custom_gulcalc_log_start'],
        custom_gulcalc_log_finish=params['custom_gulcalc_log_finish'],
    ):
        create_bash_analysis(**params)
        create_bash_outputs(**params)


def add_server_call(call, analysis_pk=None, socket_server=False):
    """Inject WebSocket or socket-server flags into a GUL command string.

    If the environment variables ``OASIS_WEBSOCKET_URL`` and
    ``OASIS_WEBSOCKET_PORT`` are set and ``analysis_pk`` is provided, the
    ``gulmc``/``gulpy`` invocation within *call* is augmented with
    ``--socket-server`` and ``--analysis-pk`` flags.  Otherwise, only the
    ``--socket-server`` flag is added.

    Args:
        call (str): The full pipeline command string.
        analysis_pk (int or None): Analysis primary key for WebSocket mode.
        socket_server (bool): Default socket-server flag value.

    Returns:
        str: The (possibly modified) command string.
    """
    if '| gul' not in call:
        return call
    if all(item in os.environ for item in ['OASIS_WEBSOCKET_URL', 'OASIS_WEBSOCKET_PORT']) and analysis_pk is not None:
        return re.sub(r'(\bgulmc\b|\bgulpy\b)', rf"\1 --socket-server='True' --analysis-pk='{analysis_pk}'", call)
    return re.sub(r'(\bgulmc\b|\bgulpy\b)', rf"\1 --socket-server='{socket_server}'", call)
