import glob
import logging
import os
import shutil
import subprocess
import json
import re
import time

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from .bash import (bash_wrapper, create_bash_analysis,
                   create_bash_outputs, genbash)
from .resource_monitor import MONITORED_TOOLS, ResourceMonitor


def _snapshot_log_dir(log_dir):
    """(path -> (size, mtime)) for every regular file under log_dir."""
    snapshot = {}
    for root, _dirs, files in os.walk(log_dir):
        for name in files:
            path = os.path.join(root, name)
            try:
                st = os.stat(path)
            except OSError:
                # file replaced/removed between listing and stat - not settled
                continue
            snapshot[path] = (st.st_size, st.st_mtime)
    return snapshot


def _wait_for_log_writers(log_dir, timeout=30, poll_interval=0.5, stable_seconds=10.0):
    """Block until files under log_dir stop changing, or timeout elapses.

    Only called once `_find_incomplete_pytool_logs` has already found something
    missing its "finish" marker - so this exists purely to give a genuinely
    still-running (but not yet finished) worker a chance to catch up, not as
    a blanket tax on every run. It does not raise on its own timeout: the
    caller always re-checks completeness afterward and raises with the
    specific tool/files still missing, which is a more useful error than a
    generic "timed out waiting" here.

    bash's `wait` only reaps the direct child PIDs it captured with `$!`.
    A pytool (e.g. gulmc, fmpy) that internally forks worker processes for
    parallel computation can leave those workers running past that point,
    still writing to their log files, since they are reparented rather than
    tracked by the script's `wait` calls. Without this check, callers that
    archive `log_dir` immediately after `run_analysis`/`run_outputs` returns
    can capture a snapshot with truncated log files.

    Settled means every file's (size, mtime) has been unchanged for at least
    `stable_seconds`. This needs no special permissions and works regardless
    of which host/process is writing, but a short window gives false
    positives whenever a writer has a quiet gap (e.g. it is still computing
    between log lines), so `stable_seconds` should comfortably exceed that.
    """
    log_dir = os.path.abspath(log_dir)
    logging.debug("Waiting for writers under %s to finish", log_dir)
    deadline = time.time() + timeout
    previous = _snapshot_log_dir(log_dir)
    stable_since = None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        time.sleep(poll_interval)
        current = _snapshot_log_dir(log_dir)

        if current == previous:
            stable_since = stable_since or time.time() - poll_interval
            if time.time() - stable_since >= stable_seconds:
                logging.debug("Files under %s settled after %d attempt(s)", log_dir, attempt)
                return
        else:
            stable_since = None

        logging.debug("Attempt %d: files under %s not yet settled", attempt, log_dir)
        previous = current
    logging.warning(
        "Timed out after %.1fs waiting for files under %s to stop changing",
        timeout, log_dir,
    )


def _find_incomplete_pytool_logs(log_dir):
    """Return {tool: [path, ...]} for every pytool log under log_dir missing its 'finish' marker.

    Python-side equivalent of bash's own `check_complete()` function. Empty
    dict means every log file found reached "finish" (see
    oasislmf/pytools/utils.py's `redirect_logging`, which writes 'finishing
    process' on a clean exit) - i.e. nothing here needs waiting or raising on.
    """
    lost = {}
    for tool in sorted(MONITORED_TOOLS):
        log_files = glob.glob(os.path.join(log_dir, f'{tool}_[0-9]*.log'))
        if not log_files:
            continue
        missing = []
        for path in log_files:
            try:
                with open(path) as f:
                    content = f.read()
            except OSError:
                missing.append(path)
                continue
            if 'finish' not in content:
                missing.append(path)
        if missing:
            lost[tool] = missing
    return lost


def _ensure_pytool_logs_complete(log_dir):
    """Check log_dir is complete; if not, wait for stragglers and check again.

    Cheap in the common case: if every pytool log already has its "finish"
    marker by the time the bash script's tracked process has exited, this
    returns immediately with no polling at all. Only when something is
    actually missing does it fall back to `_wait_for_log_writers` (to give a
    genuinely still-running, reparented worker a chance to catch up) and
    re-check - raising `OasisException`, naming exactly which tool/files are
    still incomplete, only if it's still missing after that.
    """
    lost = _find_incomplete_pytool_logs(log_dir)
    if not lost:
        return

    logging.warning(
        "Incomplete pytool logs found under %s before any wait: %s - waiting for stragglers to finish",
        log_dir, lost,
    )
    _wait_for_log_writers(log_dir)
    lost = _find_incomplete_pytool_logs(log_dir)
    if lost:
        summary = ", ".join(f"{tool} ({len(paths)} lost)" for tool, paths in lost.items())
        raise OasisException(
            "Incomplete pytool logs found under {}: {}. Details: {}".format(log_dir, summary, lost)
        )


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
        filename='run_kernel.sh',
        df_engine='oasis_data_manager.df_reader.reader.OasisPandasReader',
        model_df_engine=None,
        dynamic_footprint=False,
        resource_monitor_interval=1,
        log_level=None,
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
        bash_trace=run_debug,
        filename=filename,
        _get_getmodel_cmd=custom_get_getmodel_cmd,
        custom_gulcalc_log_start=custom_gulcalc_log_start,
        custom_gulcalc_log_finish=custom_gulcalc_log_finish,
        model_df_engine=model_df_engine,
        dynamic_footprint=dynamic_footprint,
        log_level=log_level,
        **kwargs,
    )
    monitor = ResourceMonitor(
        output_dir='log',
        poll_interval=resource_monitor_interval,
    )
    proc = subprocess.Popen(['bash', filename], stdout=subprocess.PIPE)
    monitor.start(proc.pid)
    stdout, _ = proc.communicate()
    monitor.stop()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, ['bash', filename], output=stdout)
    logging.info(stdout.decode('utf-8'))


def rerun():
    """A function to find where an error was made and to rerun that part of the script without
    NumBa to give better error messages
    """
    try:
        with open("event_error.json", "r") as f:
            event_error = json.load(f).get("event_id")
    except FileNotFoundError:
        return

    env = os.environ.copy()
    env['NUMBA_DISABLE_JIT'] = "1"
    eve_cmd = f"printf 'event_id\n {event_error}\n' | csvtobin eve"
    kernel_pipeline = ''

    with open("run_kernel.sh", "r") as bash_script:
        for line in bash_script:
            if "( ( eve" in line:
                kernel_pipeline = re.split(r'\||\)', line)
                break

    gul_cmd = [cmd.strip() for cmd in kernel_pipeline if cmd.strip().startswith(('gul'))].pop(0)
    fm_cmds = [cmd.strip() for cmd in kernel_pipeline if cmd.strip().startswith(('fm'))]

    pipe_output = "/tmp/il_P1"
    summary_output = "/tmp/il_S1_summary_P1"
    gul_output = f"{event_error}_gul.bin"

    gul_pipe = f"{eve_cmd} | {gul_cmd} -o {gul_output}"
    with open("gul_errors.log", "w") as error_log:
        subprocess.run(gul_pipe, shell=True, env=env, stderr=error_log)

    fm_input = gul_output
    for i in range(len(fm_cmds)):
        fm_cmd = re.sub(r"-\s*>\s*\S+", f"-o 64_ri{i + 1}.bin", fm_cmds[i])
        fm_output = f"{event_error}_fm{i + 1}.bin"
        fm_pipe = f"{fm_cmd} -o {fm_output} -i {fm_input}"
        with open("fm_errors.log", "a") as error_log:
            subprocess.run(fm_pipe, shell=True, env=env, stderr=error_log)
        fm_input = fm_output

    summary_pipe = f"summarypy -t il -m -1 {summary_output} < {fm_input}"
    with open("summary_errors.log", "w") as error_log:
        subprocess.run(summary_pipe, shell=True, env=env, stderr=error_log)


@oasis_log()
def run_analysis(**params):
    resource_monitor_interval = params.pop('resource_monitor_interval', 1.0)

    with bash_wrapper(params['filename'],
                      params['bash_trace'],
                      params['stderr_guard'],
                      log_sub_dir=params.get("process_number", None),
                      process_number=params.get("process_number", None),
                      run_check_complete=False):
        create_bash_analysis(**params)

    process_number = params.get('process_number')
    run_dir = os.path.dirname(params['filename'])
    log_root = os.path.join(run_dir, 'log')
    monitor_dir = os.path.join(log_root, str(process_number)) if process_number else log_root
    monitor = ResourceMonitor(output_dir=monitor_dir, poll_interval=resource_monitor_interval, generate_report=False)
    proc = subprocess.Popen(['bash', params['filename']], stdout=subprocess.PIPE)
    monitor.start(proc.pid)
    stdout, _ = proc.communicate()
    monitor.stop()
    logging.debug("run_analysis: bash script (pid=%s) exited with code %s, checking log completeness in %s",
                  proc.pid, proc.returncode, monitor_dir)
    check_start = time.time()

    # Check for bash errors
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, ['bash', params['filename']], output=stdout)

    # Check and wait for loggers to complete
    _ensure_pytool_logs_complete(monitor_dir)
    logging.debug("run_analysis: log completeness check for %s took %.2fs", monitor_dir, time.time() - check_start)

    bash_trace = stdout.decode('utf-8')
    logging.info(bash_trace)
    return params['fifo_queue_dir'], bash_trace


@oasis_log()
def run_outputs(**params):
    resource_monitor_interval = params.pop('resource_monitor_interval', 1.0)

    with bash_wrapper(params['filename'], params['bash_trace'], params['stderr_guard'],
                      log_sub_dir='out', run_check_complete=False):
        create_bash_outputs(**params)

    run_dir = os.path.dirname(params['filename'])
    log_root = os.path.join(run_dir, 'log')
    monitor = ResourceMonitor(output_dir=os.path.join(log_root, 'out'), poll_interval=resource_monitor_interval, log_root=log_root)
    proc = subprocess.Popen(['bash', params['filename']], stdout=subprocess.PIPE)
    monitor.start(proc.pid)
    stdout, _ = proc.communicate()
    monitor.stop()
    out_log_dir = os.path.join(log_root, 'out')
    logging.debug("run_outputs: bash script (pid=%s) exited with code %s, checking log completeness in %s",
                  proc.pid, proc.returncode, out_log_dir)
    check_start = time.time()

    # Check for bash errors
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, ['bash', params['filename']], output=stdout)

    # Check and wait for loggers to complete
    _ensure_pytool_logs_complete(out_log_dir)
    logging.debug("run_outputs: log completeness check for %s took %.2fs", out_log_dir, time.time() - check_start)

    bash_trace = stdout.decode('utf-8')
    logging.info(bash_trace)
    return bash_trace
