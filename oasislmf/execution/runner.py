import logging
import os
import shutil
import subprocess
import json
import re
import time

import psutil

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from .bash import (bash_wrapper, create_bash_analysis,
                   create_bash_outputs, genbash)
from .resource_monitor import ResourceMonitor


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


def _find_open_writers(log_dir):
    """Return [(pid, name, path), ...] for processes with a file open under log_dir.

    Returns (writers, fully_inspected). fully_inspected is False if any process
    could not be inspected (e.g. psutil.AccessDenied under a restricted
    container security context) - in that case an empty `writers` list does
    NOT mean nothing is writing, it means this signal is unreliable.

    Only same-UID processes are considered: any orphaned/reparented pytool
    worker still runs as the celery worker's own user, and skipping other
    users' processes (root-owned daemons etc.) avoids spurious AccessDenied
    noise from processes that were never a candidate writer in the first
    place - on a real host, scanning *every* process would otherwise trip
    the "degraded" fallback on essentially every run.
    """
    writers = []
    fully_inspected = True
    own_uid = os.getuid()
    for proc in psutil.process_iter(['pid', 'name', 'uids']):
        uids = proc.info.get('uids')
        if uids is not None and uids.real != own_uid:
            continue
        try:
            for f in proc.open_files():
                if f.path.startswith(log_dir):
                    writers.append((proc.pid, proc.info.get('name'), f.path))
                    break
        except psutil.AccessDenied:
            fully_inspected = False
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return writers, fully_inspected


def _wait_for_log_writers(log_dir, timeout=30, poll_interval=0.5, stable_checks=2, degraded_stable_seconds=10.0):
    """Block until nothing appears to still be writing under log_dir.

    Raises `OasisException` if `timeout` is reached without settling, rather
    than silently returning - a caller archiving `log_dir` right after this
    call must be able to trust that "it returned" means "it's actually done",
    the same way bash's own `check_complete` fails loudly instead of letting
    the script exit 0 with lost/incomplete process logs.

    bash's `wait` only reaps the direct child PIDs it captured with `$!`.
    A pytool (e.g. gulmc, fmpy) that internally forks worker processes for
    parallel computation can leave those workers running past that point,
    still writing to their log files, since they are reparented rather than
    tracked by the script's `wait` calls. Without this check, callers that
    archive `log_dir` immediately after `run_analysis`/`run_outputs` returns
    can capture a snapshot with truncated log files.

    Two signals are combined, since neither is reliable alone:
      - an open-file-handle scan (psutil) is precise and immune to a writer
        simply pausing between log lines, but can be silently defeated by
        `AccessDenied` under a restricted container/k8s security context;
      - file (size, mtime) stability needs no special permissions and works
        regardless of which host/process is writing, but on its own produces
        false positives whenever a writer has a quiet gap longer than the
        stability window (e.g. it is still computing between log lines).

    Settled requires BOTH: no inspectable process still has a file under
    log_dir open, AND the files have been stable across `stable_checks`
    consecutive polls. If any process couldn't be inspected during this
    wait, the psutil signal is no longer trusted for the rest of this call:
    a warning is logged once, and the required stability window is widened
    to `degraded_stable_seconds` (instead of `stable_checks` polls) before
    settling is declared, since a short window is not enough to be confident
    a writer we can no longer see isn't just between log lines.
    """
    log_dir = os.path.abspath(log_dir)
    logging.debug("Waiting for writers under %s to finish", log_dir)
    deadline = time.time() + timeout
    previous = _snapshot_log_dir(log_dir)
    stable_count = 0
    stable_since = None
    attempt = 0
    degraded = False
    writers, current = [], previous
    while time.time() < deadline:
        attempt += 1
        time.sleep(poll_interval)
        current = _snapshot_log_dir(log_dir)
        writers, fully_inspected = _find_open_writers(log_dir)
        if not fully_inspected and not degraded:
            logging.warning(
                "Could not inspect all processes while checking %s (AccessDenied) - "
                "widening required settle window to %.1fs of file-stability for this run",
                log_dir, degraded_stable_seconds,
            )
            degraded = True

        files_stable = current == previous
        if files_stable:
            stable_count += 1
            stable_since = stable_since or time.time() - poll_interval
        else:
            stable_count = 0
            stable_since = None

        if degraded:
            settled_long_enough = stable_since is not None and (time.time() - stable_since) >= degraded_stable_seconds
        else:
            settled_long_enough = stable_count >= stable_checks

        if not writers and files_stable and settled_long_enough:
            logging.debug("Writers under %s settled after %d attempt(s) (degraded=%s)", log_dir, attempt, degraded)
            return

        logging.debug(
            "Attempt %d: %s not yet settled (open_writers=%s, files_stable=%s, stable_count=%d, degraded=%s)",
            attempt, log_dir, writers, files_stable, stable_count, degraded,
        )
        previous = current
    raise OasisException(
        "Timed out after {:.1f}s waiting for writers under {} to finish (open_writers={}). "
        "Refusing to archive logs that may still be truncated.".format(timeout, log_dir, writers)
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
                      process_number=params.get("process_number", None)):
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
    logging.debug("run_analysis: bash script (pid=%s) exited with code %s, waiting on log writers in %s",
                  proc.pid, proc.returncode, monitor_dir)
    wait_start = time.time()
    _wait_for_log_writers(monitor_dir)
    logging.debug("run_analysis: log writer check for %s took %.2fs", monitor_dir, time.time() - wait_start)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, ['bash', params['filename']], output=stdout)
    bash_trace = stdout.decode('utf-8')
    logging.info(bash_trace)
    return params['fifo_queue_dir'], bash_trace


@oasis_log()
def run_outputs(**params):
    resource_monitor_interval = params.pop('resource_monitor_interval', 1.0)

    with bash_wrapper(params['filename'], params['bash_trace'], params['stderr_guard'], log_sub_dir='out'):
        create_bash_outputs(**params)

    run_dir = os.path.dirname(params['filename'])
    log_root = os.path.join(run_dir, 'log')
    monitor = ResourceMonitor(output_dir=os.path.join(log_root, 'out'), poll_interval=resource_monitor_interval, log_root=log_root)
    proc = subprocess.Popen(['bash', params['filename']], stdout=subprocess.PIPE)
    monitor.start(proc.pid)
    stdout, _ = proc.communicate()
    monitor.stop()
    out_log_dir = os.path.join(log_root, 'out')
    logging.debug("run_outputs: bash script (pid=%s) exited with code %s, waiting on log writers in %s",
                  proc.pid, proc.returncode, out_log_dir)
    wait_start = time.time()
    _wait_for_log_writers(out_log_dir)
    logging.debug("run_outputs: log writer check for %s took %.2fs", out_log_dir, time.time() - wait_start)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, ['bash', params['filename']], output=stdout)
    bash_trace = stdout.decode('utf-8')
    logging.info(bash_trace)
    return bash_trace
