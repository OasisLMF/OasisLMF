import glob
import logging
import os
import shlex
import shutil
import subprocess
import json
import re
import time

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.path import setcwd
from .bash import (bash_wrapper, check_custom_gulcalc_can_read_inputs, create_bash_analysis,
                   create_bash_outputs, genbash, genbash_params, get_complex_model_cmd)
from .orchestrator.build import Run, build_analysis, build_outputs, unsupported_reason
from .orchestrator.executor import execute
from .resource_monitor import MONITORED_TOOLS, ResourceMonitor

logger = logging.getLogger(__name__)

ORCHESTRATOR_FALLBACK_WARNING = "The orchestrator cannot run this analysis (%s): running the kernel with the generated bash script instead"


def manifest_path(script_path):
    """Path of the orchestrator manifest written in place of the bash script ``script_path``."""
    return os.path.splitext(script_path)[0] + '.json'


def _orchestrator_cmd(script_path):
    return ['oasislmf-orchestrator', manifest_path(script_path)]


def _concat_runs(*runs):
    """One :class:`Run` executing the stages of ``runs`` in order, as a single bash script does."""
    combined = Run()
    for run_ in runs:
        combined.stages += run_.stages
        combined.background += run_.background
        combined.env.update(run_.env)
        combined.chunk_check = combined.chunk_check or run_.chunk_check
    return combined


def _run_orchestrated(script_path, log_dir, stderr_guard, describe, monitor):
    """Describe and execute a run in the directory of ``script_path``, where bash would ``cd``.

    No script is written (a stale one at ``script_path`` is removed); the manifest is written next
    to where it would be. ``describe`` is called in that directory, since the builders probe it.

    Returns:
        str: the executor's run log.

    Raises:
        subprocess.CalledProcessError: the run failed (see :func:`.orchestrator.executor.execute`).
    """
    script_path = os.path.abspath(script_path)
    if os.path.exists(script_path):
        os.remove(script_path)
    with setcwd(os.path.dirname(script_path)):
        run_ = describe()
        monitor.start(os.getpid())
        try:
            return execute(run_, log_dir, manifest_path=manifest_path(script_path), stderr_guard=stderr_guard)
        finally:
            monitor.stop()


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


def _find_incomplete_custom_gulcalc(log_dir, log_start, log_finish):
    """Return how many custom gulcalc processes logged `log_start` but not `log_finish`.

    Custom gulcalc binaries write to `gul_stderror.err` in log_dir rather than
    to a pytool log, so they are counted by line markers instead, matched as
    plain substrings. The file holds raw stderr from a third-party
    binary, so bytes that are not valid UTF-8 are replaced rather than failing
    the check. Returns 0 when either marker is unset or the file does not exist.
    """
    path = os.path.join(log_dir, 'gul_stderror.err')
    if not (log_start and log_finish and os.path.isfile(path)):
        return 0
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        logging.warning("Could not read %s to count custom gulcalc processes", path)
        return 0
    started = sum(log_start in line for line in lines)
    finished = sum(log_finish in line for line in lines)
    return max(started - finished, 0)


def _find_incomplete_pytool_logs(log_dir, custom_gulcalc_log_start=None, custom_gulcalc_log_finish=None):
    """Return {tool: [path, ...]} for every pytool log under log_dir missing its 'finish' marker.

    Empty dict means every log file found reached "finish" (see
    oasislmf/pytools/utils.py's `redirect_logging`, which writes 'finishing
    process' on a clean exit) - i.e. nothing here needs waiting or raising on.
    When both custom gulcalc markers are given, lost custom gulcalc processes
    are reported under 'gulcalc', with `gul_stderror.err` listed once per lost
    process.
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
    lost_gulcalc = _find_incomplete_custom_gulcalc(log_dir, custom_gulcalc_log_start, custom_gulcalc_log_finish)
    if lost_gulcalc:
        lost['gulcalc'] = [os.path.join(log_dir, 'gul_stderror.err')] * lost_gulcalc
    return lost


def _ensure_pytool_logs_complete(log_dir, custom_gulcalc_log_start=None, custom_gulcalc_log_finish=None):
    """Check log_dir is complete; if not, wait for stragglers and check again.

    Cheap in the common case: if every pytool log already has its "finish"
    marker by the time the bash script's tracked process has exited, this
    returns immediately with no polling at all. Only when something is
    actually missing does it fall back to `_wait_for_log_writers` (to give a
    genuinely still-running, reparented worker a chance to catch up) and
    re-check - raising `OasisException`, naming exactly which tool/files are
    still incomplete, only if it's still missing after that.

    Args:
        log_dir (str): Directory holding the script's pytool logs.
        custom_gulcalc_log_start (str or None): Line a custom gulcalc binary
            writes to `gul_stderror.err` when it starts.
        custom_gulcalc_log_finish (str or None): Line a custom gulcalc binary
            writes to `gul_stderror.err` when it finishes.
    """
    lost = _find_incomplete_pytool_logs(log_dir, custom_gulcalc_log_start, custom_gulcalc_log_finish)
    if not lost:
        return

    logging.warning(
        "Incomplete pytool logs found under %s before any wait: %s - waiting for stragglers to finish",
        log_dir, lost,
    )
    _wait_for_log_writers(log_dir)
    lost = _find_incomplete_pytool_logs(log_dir, custom_gulcalc_log_start, custom_gulcalc_log_finish)
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
        custom_gulcalc_supports_packing=False,
        filename='run_kernel.sh',
        df_engine='oasis_data_manager.df_reader.reader.OasisPandasReader',
        model_df_engine=None,
        dynamic_footprint=False,
        resource_monitor_interval=1,
        disable_resource_report=False,
        log_level=None,
        orchestrator=False,
        **kwargs
        ):
    """Run the whole kernel (analysis then outputs) for ``analysis_settings`` in the script's directory.

    By default a bash script is generated at ``filename`` and executed. With ``orchestrator`` set,
    the same processes are run by :mod:`oasislmf.execution.orchestrator` instead and a JSON manifest
    is written in place of the script, unless the run is one the orchestrator does not support (a
    caller-supplied ``custom_get_getmodel_cmd``, or a reason from
    :func:`~.orchestrator.build.unsupported_reason`), in which case a warning is logged and bash is used.

    Raises:
        subprocess.CalledProcessError: the kernel failed, or its pytool logs are incomplete.
    """
    model_df_engine = model_df_engine or df_engine
    caller_get_getmodel_cmd = custom_get_getmodel_cmd

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

    check_custom_gulcalc_can_read_inputs(
        'input', custom_gulcalc_cmd or (custom_get_getmodel_cmd and 'custom_get_getmodel_cmd'),
        custom_gulcalc_supports_packing)

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

    monitor = ResourceMonitor(
        output_dir='log',
        poll_interval=resource_monitor_interval,
        enabled=not disable_resource_report,
    )
    orchestrator_params = None
    if orchestrator:
        if caller_get_getmodel_cmd is not None:
            reason = 'a custom_get_getmodel_cmd callable was supplied; its shell command cannot be translated into an argument list'
        else:
            orchestrator_params = genbash_params(
                number_of_processes,
                analysis_settings,
                gul_alloc_rule=set_alloc_rule_gul,
                il_alloc_rule=set_alloc_rule_il,
                ri_alloc_rule=set_alloc_rule_ri,
                bash_trace=run_debug,
                filename=filename,
                _get_getmodel_cmd=get_complex_model_cmd(custom_gulcalc_cmd, analysis_settings) if custom_gulcalc_cmd else None,
                custom_gulcalc_log_start=custom_gulcalc_log_start,
                custom_gulcalc_log_finish=custom_gulcalc_log_finish,
                model_df_engine=model_df_engine,
                dynamic_footprint=dynamic_footprint,
                log_level=log_level,
                **kwargs,
            )
            orchestrator_params['custom_gulcalc_cmd'] = custom_gulcalc_cmd
            reason = unsupported_reason(orchestrator_params)
        if reason:
            logger.warning(ORCHESTRATOR_FALLBACK_WARNING, reason)
            orchestrator_params = None

    if orchestrator_params is not None:
        cmd = _orchestrator_cmd(filename)
        stdout = _run_orchestrated(
            filename, 'log', orchestrator_params['stderr_guard'],
            lambda: _concat_runs(build_analysis(orchestrator_params), build_outputs(orchestrator_params)),
            monitor,
        ).encode('utf-8')
    else:
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
        cmd = ['bash', filename]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        monitor.start(proc.pid)
        stdout, _ = proc.communicate()
        monitor.stop()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout)
    logging.info(stdout.decode('utf-8'))
    try:
        _ensure_pytool_logs_complete(
            os.path.join(os.path.dirname(os.path.abspath(filename)), 'log'),
            custom_gulcalc_log_start or analysis_settings.get('model_custom_gulcalc_log_start'),
            custom_gulcalc_log_finish or analysis_settings.get('model_custom_gulcalc_log_finish'),
        )
    except OasisException as e:
        # Callers such as GenerateLosses dump the run logs and call rerun() on CalledProcessError.
        raise subprocess.CalledProcessError(
            1, cmd, output=stdout + f'\n[ERROR] {e}\n'.encode('utf-8')
        ) from e


# matches a trailing output redirect (e.g. `> /path/to/fifo` or `2>> log/err`)
STALE_REDIRECT_RE = re.compile(r'\s*\d*>>?\s*\S+$')


RERUN_SCRIPT = 'run_kernel.sh'
STREAM_PLACEHOLDER_RE = re.compile(r'^<stream:[^>]*>$')


def _use_kernel_manifest(script, manifest):
    """True when the orchestrator manifest, not the bash script, records the last run."""
    if not os.path.exists(manifest):
        return False
    return not os.path.exists(script) or os.path.getmtime(manifest) >= os.path.getmtime(script)


def _command_without_streams(argv):
    """Shell command for ``argv`` without its stream placeholders (and the option that introduces each)."""
    kept = []
    for arg in argv:
        if STREAM_PLACEHOLDER_RE.match(arg):
            if kept and kept[-1].startswith('-'):
                kept.pop()
            continue
        kept.append(arg)
    return shlex.join(kept)


def _kernel_cmds_from_manifest(path):
    """The gul command and fmpy commands of the first partition recorded in an orchestrator manifest.

    The first process of the analysis stage whose name starts with ``gul`` is the first partition's
    ground-up process; its fmpy processes are found by following its stdout from stream to stream.

    Returns:
        tuple: (gul command, [fmpy commands in pipeline order]) as shell command strings.
    """
    with open(path, encoding='utf-8') as f:
        processes = json.load(f)['stages'][0]['processes']
    readers = {process['stdin']: process for process in processes if STREAM_PLACEHOLDER_RE.match(process['stdin'] or '')}
    gul = next(process for process in processes if process['name'].startswith('gul'))
    fm_cmds = []
    current = gul
    while current['stdout'] in readers:
        current = readers[current['stdout']]
        if current['name'].startswith('fm'):
            fm_cmds.append(_command_without_streams(current['argv']))
    return _command_without_streams(gul['argv']), fm_cmds


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

    if _use_kernel_manifest(RERUN_SCRIPT, manifest_path(RERUN_SCRIPT)):
        gul_cmd, fm_cmds = _kernel_cmds_from_manifest(manifest_path(RERUN_SCRIPT))
    else:
        kernel_pipeline = ''
        with open(RERUN_SCRIPT, "r") as bash_script:
            for line in bash_script:
                if "( ( eve" in line:
                    kernel_pipeline = re.split(r'\||\)', line)
                    break

        gul_cmd = [cmd.strip() for cmd in kernel_pipeline if cmd.strip().startswith(('gul'))].pop(0)
        fm_cmds = [cmd.strip() for cmd in kernel_pipeline if cmd.strip().startswith(('fm'))]

    # strip a trailing output redirect from the extracted command (e.g. to a fifo whose
    # reader has already exited in the main run) before pointing it at our own output file
    gul_cmd = STALE_REDIRECT_RE.sub('', gul_cmd).strip()

    pipe_output = "/tmp/il_P1"
    summary_output = "/tmp/il_S1_summary_P1"
    gul_output = f"{event_error}_gul.bin"

    gul_pipe = f"{eve_cmd} | {gul_cmd} -o {gul_output}"
    with open("gul_errors.log", "w") as error_log:
        subprocess.run(gul_pipe, shell=True, env=env, stderr=error_log)

    fm_input = gul_output
    for i in range(len(fm_cmds)):
        fm_cmd = STALE_REDIRECT_RE.sub('', fm_cmds[i]).strip()
        fm_output = f"{event_error}_fm{i + 1}.bin"
        fm_pipe = f"{fm_cmd} -o {fm_output} -i {fm_input}"
        with open("fm_errors.log", "a") as error_log:
            subprocess.run(fm_pipe, shell=True, env=env, stderr=error_log)
        fm_input = fm_output

    summary_pipe = f"summarypy -t il -m -1 {summary_output} < {fm_input}"
    with open("summary_errors.log", "w") as error_log:
        subprocess.run(summary_pipe, shell=True, env=env, stderr=error_log)


def _orchestrator_usable(params):
    """True when ``params['orchestrator']`` asks for the orchestrator and it supports the run; warns otherwise."""
    if not params.get('orchestrator'):
        return False
    reason = unsupported_reason(params)
    if reason:
        logger.warning(ORCHESTRATOR_FALLBACK_WARNING, reason)
        return False
    return True


@oasis_log()
def run_analysis(**params):
    """Run the analysis of one chunk (``process_number``) or of all chunks, from ``bash_params``.

    With ``params['orchestrator']`` set and a run the orchestrator supports, the processes are run
    by :mod:`oasislmf.execution.orchestrator` and a JSON manifest replaces the script at
    ``params['filename']``; otherwise a bash script is generated and executed.

    Returns:
        tuple: ``(params['fifo_queue_dir'], run log text)``.

    Raises:
        subprocess.CalledProcessError: the kernel failed.
    """
    resource_monitor_interval = params.pop('resource_monitor_interval', 1.0)
    disable_resource_report = params.pop('disable_resource_report', False)
    use_orchestrator = _orchestrator_usable(params)
    params.pop('orchestrator', None)

    process_number = params.get('process_number')
    run_dir = os.path.dirname(params['filename'])
    log_root = os.path.join(run_dir, 'log')
    monitor_dir = os.path.join(log_root, str(process_number)) if process_number else log_root
    monitor = ResourceMonitor(output_dir=monitor_dir, poll_interval=resource_monitor_interval, generate_report=False,
                              enabled=not disable_resource_report)

    if use_orchestrator:
        log_dir = os.path.join('log', str(process_number)) if process_number else 'log'
        bash_trace = _run_orchestrated(params['filename'], log_dir, params['stderr_guard'], lambda: build_analysis(params), monitor)
        logging.debug("run_analysis: orchestrator run finished, checking log completeness in %s", monitor_dir)
    else:
        with bash_wrapper(params['filename'],
                          params['bash_trace'],
                          params['stderr_guard'],
                          log_sub_dir=params.get("process_number", None),
                          process_number=params.get("process_number", None)):
            create_bash_analysis(**params)

        proc = subprocess.Popen(['bash', params['filename']], stdout=subprocess.PIPE)
        monitor.start(proc.pid)
        stdout, _ = proc.communicate()
        monitor.stop()
        logging.debug("run_analysis: bash script (pid=%s) exited with code %s, checking log completeness in %s",
                      proc.pid, proc.returncode, monitor_dir)

        # Check for bash errors
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, ['bash', params['filename']], output=stdout)

        bash_trace = stdout.decode('utf-8')
    check_start = time.time()
    logging.info(bash_trace)

    # Check and wait for loggers to complete
    _ensure_pytool_logs_complete(
        monitor_dir,
        params.get('custom_gulcalc_log_start'),
        params.get('custom_gulcalc_log_finish'),
    )
    logging.debug("run_analysis: log completeness check for %s took %.2fs", monitor_dir, time.time() - check_start)
    return params['fifo_queue_dir'], bash_trace


@oasis_log()
def run_outputs(**params):
    """Run the output stage (katpy, aalpy/lecpy, join-summary-info, cleanup) from ``bash_params``.

    With ``params['orchestrator']`` set, the processes are run by :mod:`oasislmf.execution.orchestrator`
    and a JSON manifest replaces the script at ``params['filename']``; otherwise a bash script is
    generated and executed.

    Returns:
        str: the run log text.

    Raises:
        subprocess.CalledProcessError: the kernel failed.
    """
    resource_monitor_interval = params.pop('resource_monitor_interval', 1.0)
    disable_resource_report = params.pop('disable_resource_report', False)
    use_orchestrator = _orchestrator_usable(params)
    params.pop('orchestrator', None)

    run_dir = os.path.dirname(params['filename'])
    log_root = os.path.join(run_dir, 'log')
    out_log_dir = os.path.join(log_root, 'out')
    monitor = ResourceMonitor(output_dir=out_log_dir, poll_interval=resource_monitor_interval,
                              log_root=log_root, enabled=not disable_resource_report)

    if use_orchestrator:
        bash_trace = _run_orchestrated(params['filename'], os.path.join('log', 'out'), params['stderr_guard'],
                                       lambda: build_outputs(params), monitor)
        logging.debug("run_outputs: orchestrator run finished, checking log completeness in %s", out_log_dir)
    else:
        with bash_wrapper(params['filename'], params['bash_trace'], params['stderr_guard'],
                          log_sub_dir='out'):
            create_bash_outputs(**params)

        proc = subprocess.Popen(['bash', params['filename']], stdout=subprocess.PIPE)
        monitor.start(proc.pid)
        stdout, _ = proc.communicate()
        monitor.stop()
        logging.debug("run_outputs: bash script (pid=%s) exited with code %s, checking log completeness in %s",
                      proc.pid, proc.returncode, out_log_dir)

        # Check for bash errors
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, ['bash', params['filename']], output=stdout)

        bash_trace = stdout.decode('utf-8')
    check_start = time.time()
    logging.info(bash_trace)

    # Check and wait for loggers to complete
    _ensure_pytool_logs_complete(out_log_dir)
    logging.debug("run_outputs: log completeness check for %s took %.2fs", out_log_dir, time.time() - check_start)
    return bash_trace
