import logging
import os
import shutil
import subprocess
import json
import re

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from .bash import (bash_wrapper, create_bash_analysis,
                   create_bash_outputs, genbash)


@oasis_log()
def run(analysis_settings,
        number_of_processes=-1,
        set_alloc_rule_gul=None,
        set_alloc_rule_il=None,
        set_alloc_rule_ri=None,
        run_debug=False,
        custom_gulcalc_cmd=None,
        custom_gulcalc_log_start=None,
        custom_gulcalc_log_finish=None,
        custom_get_getmodel_cmd=None,
        filename='run_ktools.sh',
        gul_legacy_stream=False,
        df_engine='oasis_data_manager.df_reader.reader.OasisPandasReader',
        model_df_engine=None,
        dynamic_footprint=False,
        **kwargs
        ):
    model_df_engine = model_df_engine or df_engine

    #  MOVED into bash_params #########################################
    #  keep here for the moment and refactor after testing
    #
    #  Example:
    #  from .bash import get_complex_model_cmd
    #  <var> = get_complex_model_cmd(custom_gulcalc_cmd, analysis_settings)
    #
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

    # TODO: should be integrated into bash.py
    if custom_gulcalc_cmd:
        if not custom_get_getmodel_cmd:
            def custom_get_getmodel_cmd(
                number_of_samples,
                gul_threshold,
                gul_legacy_stream,
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

    ###########################################################

    # Calls run_analysis + run_outputs in a single script
    genbash(
        number_of_processes,
        analysis_settings,
        gul_alloc_rule=set_alloc_rule_gul,
        il_alloc_rule=set_alloc_rule_il,
        ri_alloc_rule=set_alloc_rule_ri,
        gul_legacy_stream=gul_legacy_stream,
        bash_trace=run_debug,
        filename=filename,
        _get_getmodel_cmd=custom_get_getmodel_cmd,
        custom_gulcalc_log_start=custom_gulcalc_log_start,
        custom_gulcalc_log_finish=custom_gulcalc_log_finish,
        model_df_engine=model_df_engine,
        dynamic_footprint=dynamic_footprint,
        **kwargs,
    )
    bash_trace = subprocess.check_output(['bash', filename])
    logging.info(bash_trace.decode('utf-8'))


def rerun():
    """
    A function to find where an error was made and to rerun that part of the script without
    NumBa to give better error messages
    """
    try:
        with open("event_error.json", "r") as f:
            event_error = json.load(f).get("event_id")
    except FileNotFoundError:
        return

    env = os.environ.copy()
    env['NUMBA_DISABLE_JIT'] = "1"
    eve_cmd = f"printf 'event_id\n {event_error}\n' | evetobin"
    ktools_pipeline = ''

    with open("run_ktools.sh", "r") as bash_script:
        for line in bash_script:
            if "( ( eve" in line:
                ktools_pipeline = re.split(r'\||\)', line)
                break

    gul_cmd = [cmd.strip() for cmd in ktools_pipeline if cmd.strip().startswith(('gul'))].pop(0)
    fm_cmds = [cmd.strip() for cmd in ktools_pipeline if cmd.strip().startswith(('fm'))]

    pipe_output = "/tmp/il_P1"
    summary_output = "/tmp/il_S1_summary_P1"
    gul_output = f"{event_error}_gul.bin"

    gul_pipe = f"{eve_cmd} | {gul_cmd} -o {gul_output}"
    with open("gul_errors.log", "w") as error_log:
        subprocess.run(gul_pipe, shell=True, env=env, stderr=error_log)

    fm_input = gul_output
    for i in range(len(fm_cmds)):
        fm_cmd = re.sub(r"-\s*>\s*\S+", f"-o 64_ri{i+1}.bin", fm_cmds[i])
        fm_output = f"{event_error}_fm{i+1}.bin"
        fm_pipe = f"{fm_cmd} -o {fm_output} -i {fm_input}"
        with open("fm_errors.log", "a") as error_log:
            subprocess.run(fm_pipe, shell=True, env=env, stderr=error_log)
        fm_input = fm_output

    summary_pipe = f"summarypy -t il -m -1 {summary_output} < {fm_input}"
    with open("summary_errors.log", "w") as error_log:
        subprocess.run(summary_pipe, shell=True, env=env, stderr=error_log)


@oasis_log()
def run_analysis(**params):
    with bash_wrapper(params['filename'],
                      params['bash_trace'],
                      params['stderr_guard'],
                      log_sub_dir=params.get("process_number", None),
                      process_number=params.get("process_number", None)):
        create_bash_analysis(**params)

    bash_trace = subprocess.check_output(['bash', params['filename']]).decode('utf-8')
    logging.info(bash_trace)
    return params['fifo_queue_dir'], bash_trace


@oasis_log()
def run_outputs(**params):
    with bash_wrapper(params['filename'], params['bash_trace'], params['stderr_guard'], log_sub_dir='out'):
        create_bash_outputs(**params)
    bash_trace = subprocess.check_output(['bash', params['filename']]).decode('utf-8')
    logging.info(bash_trace)
    return bash_trace
