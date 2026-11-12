"""Describe a kernel run from ``bash_params`` as stages of processes connected by streams.

The builders mirror :func:`oasislmf.execution.bash.create_bash_analysis` and
:func:`oasislmf.execution.bash.create_bash_outputs`: they produce the same processes with the same
arguments, but as argument lists connected by :class:`~.model.Stream` objects instead of a shell
script with named pipes. Nothing is started and nothing is written to disk; the only filesystem
access is the two generation-time probes bash also makes (``input/fm_profile.csv`` for the fmpy
``-S`` flag, and ``analysis_settings.json`` made absolute for a custom gulcalc binary).

Both builders return a :class:`Run`:

* ``stages``: the stages to run one after the other. Each bash ``exec_wait`` group is one stage;
  stages bash would not emit (no process to wait for) are left out, except a final cleanup stage
  that has only ``remove_paths``.
* ``background``: processes that do not exit by themselves (the local progress ``socket-server``).
  The executor starts them before the first stage and kills them after the last stage of the run,
  as bash does with ``spid``. Always empty for :func:`build_outputs`.
* ``env``: environment variables to export to every process of the run.
* ``chunk_check``: what bash's footer checks after a chunk (``process_number``) analysis, or None.

Stream naming: a stream standing for a bash FIFO is named after the FIFO basename (``gul_P1``,
``il_S1_summary_P1``, ``ri_IP1_P1``...). FIFOs of the full-correlation directory share basenames
with the main ones, so their streams carry :data:`FULL_CORRELATION_STREAM_PREFIX`. Streams that
replace an anonymous shell pipe (``a | b``) are named with :data:`PIPE_STREAM_PREFIX`.
"""

import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ...pytools.common.run_types import (RUNTYPE_GROUNDUP_LOSS, RUNTYPE_INSURED_LOSS, RUNTYPE_REINSURANCE_GROSS_LOSS,
                                         RUNTYPE_REINSURANCE_LOSS)
from ...utils.exceptions import OasisException
from ..bash import (ORD_ALT_MEANONLY_OUTPUT_SWITCHES, ORD_ALT_OUTPUT_SWITCHES, ORD_EPT_OUTPUT_SWITCHES, ORD_LECCALC,
                    ORD_PSEPT_OUTPUT_SWITCHES, OUTPUT_SWITCHES, get_complex_model_cmd, get_fifo_name, get_fmcmd, get_gulcmd,
                    get_modelcmd, get_pla_cmd, get_ri_inuring_priorities, get_rl_inuring_priorities, ord_enabled, process_range)
from .model import GUL_STDERR_LOG, STDERR_LOG, Process, Stage, Stream, StreamPath

PIPE_STREAM_PREFIX = 'pipe_'
FULL_CORRELATION_STREAM_PREFIX = 'full_correlation_'
OUTPUT_DIR_TO_CLEAN = 'output'
TEE_TOOL = 'teepy'
TEE_STDOUT = '-'

# Lines of stderror.err that bash's chunk footer tolerates (blank lines and Python warnings).
IGNORABLE_STDERR_LINE = r'^\s*$|.+:[0-9]+: [A-Za-z]+Warning:|^\s+warnings\.warn'

_BUILTIN_CUSTOM_GULCALC_QUALNAME = f'{get_complex_model_cmd.__name__}.<locals>.custom_get_getmodel_cmd'


@dataclass
class ChunkCheck:
    """Checks bash runs at the end of a chunk analysis script (``bash_wrapper`` footer).

    Attributes:
        stderr_log: log file name (under the run's log directory) that must contain no line
            other than those matching ``ignorable_stderr_line``.
        ignorable_stderr_line: regular expression (``re.search``, per line) for tolerated lines.
        non_empty_glob: recursive glob (``glob.glob(..., recursive=True)``) of files that must
            all be non-empty.
    """
    stderr_log: str
    ignorable_stderr_line: str
    non_empty_glob: str


@dataclass
class Run:
    """A run description: see the module docstring."""
    stages: List[Stage] = field(default_factory=list)
    background: List[Process] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    chunk_check: Optional[ChunkCheck] = None


class _Pipes:
    """Hands out uniquely named streams for anonymous shell pipes within one stage."""

    def __init__(self):
        self.count = 0

    def new(self):
        self.count += 1
        return Stream(f'{PIPE_STREAM_PREFIX}{self.count}')

    def chain(self, processes):
        """Connect each process's stdout to the next one's stdin, like ``a | b | c``."""
        for writer, reader in zip(processes, processes[1:]):
            stream = self.new()
            writer.stdout = stream
            reader.stdin = stream
        return processes


def _merge(params, kwargs):
    merged = dict(params or {})
    merged.update(kwargs)
    return merged


def _fifo(prefix, producer, producer_id, consumer=''):
    return Stream(prefix + get_fifo_name('', producer, producer_id, consumer))


def _split(fragment):
    """Tokenise a command fragment produced by a bash.py helper (plain words, no shell operators)."""
    return shlex.split(fragment)


def _tee(outputs, stdin=None, stderr_log=None):
    return Process(TEE_TOOL, [TEE_TOOL, *outputs], stdin=stdin, stderr_log=stderr_log)


def _is_builtin_custom_gulcalc(get_getmodel_cmd):
    return getattr(get_getmodel_cmd, '__qualname__', '') == _BUILTIN_CUSTOM_GULCALC_QUALNAME


def _load_balancer_active(p):
    lb_sizes_set = p.get('num_gul_per_lb') and p.get('num_fm_per_lb')
    return bool(p.get('process_number') is None and lb_sizes_set and (p.get('il_output') or p.get('ri_output')))


def unsupported_reason(params=None, **kwargs):
    """Why the orchestrator cannot run this configuration, or None when it can.

    Args:
        params (dict): the dict returned by ``bash_params`` (extra keys are ignored).
        **kwargs: same, as keyword arguments; they override ``params``.

    Returns:
        str or None: a human-readable reason, or None if :func:`build_analysis` supports the run.
    """
    p = _merge(params, kwargs)
    if _load_balancer_active(p):
        return (f"the load balancer is enabled (num_gul_per_lb={p.get('num_gul_per_lb')}, num_fm_per_lb={p.get('num_fm_per_lb')}) "
                "and is not supported by the orchestrator")
    if p.get('summarypy_low_memory'):
        return 'summarypy_low_memory is not supported by the orchestrator'
    if p.get('model_py_server'):
        return 'model_py_server (servedata) is not supported by the orchestrator'
    get_getmodel_cmd = p.get('_get_getmodel_cmd')
    if get_getmodel_cmd is not None:
        if not _is_builtin_custom_gulcalc(get_getmodel_cmd):
            return 'a custom _get_getmodel_cmd callable is set; its shell command cannot be translated into an argument list'
        if not p.get('custom_gulcalc_cmd'):
            return 'a custom gulcalc binary is used but its name was not given in custom_gulcalc_cmd'
    if p.get('full_correlation'):
        return ('full_correlation is set, but the built-in gulpy has no correlated-output option, so the full-correlation '
                'streams bash wires up would have no writer (the bash script blocks on them too)')
    return None


def _has_step_policies():
    try:
        pd.read_csv(os.path.join(os.getcwd(), 'input/fm_profile.csv'))['step_id']
    except (OSError, FileNotFoundError, KeyError):
        return False
    return True


def _server_flags(p):
    """The progress-server flags bash's ``add_server_call`` injects after gulmc/gulpy."""
    has_websocket = all(item in os.environ for item in ['OASIS_WEBSOCKET_URL', 'OASIS_WEBSOCKET_PORT'])
    has_http = 'OASIS_ANALYSIS_STATUS_URL' in os.environ
    analysis_pk = p.get('analysis_pk')
    socket_server_port = p.get('socket_server_port')
    if (has_websocket or has_http) and analysis_pk is not None:
        return ['--socket-server=True', f'--analysis-pk={analysis_pk}']
    if socket_server_port is not None:
        return [f'--socket-server={socket_server_port}']
    return []


def _summaries(analysis_settings, runtype):
    return analysis_settings.get(f'{runtype}_summaries') or []


def _ord_processes(runtype, analysis_settings, process_id, prefix, work_dir, stderr_log, inuring_priority=''):
    """Mirror of ``do_ord``: eltpy/pltpy readers of the per-ORD-type streams."""
    processes = []
    for summary in _summaries(analysis_settings, runtype):
        if 'id' not in summary:
            continue
        summary_set = summary['id']
        for ord_type, output_switch in OUTPUT_SWITCHES.items():
            args, executable = [], None
            for ord_table, flag_proc in output_switch.items():
                if summary.get('ord_output', {}).get(ord_table):
                    if process_id != 1 and not args:
                        args.append(flag_proc['skip_header_flag'])
                    executable = flag_proc['executable']
                    args += [flag_proc['csv_flag'],
                             get_fifo_name(f'{work_dir}kat/', runtype, process_id, f'{inuring_priority}S{summary_set}_{ord_table}')]
            if args:
                processes.append(Process(
                    executable, [executable, '-E', 'bin', *args],
                    stdin=_fifo(prefix, runtype, process_id, f'{inuring_priority}S{summary_set}_{ord_type}'),
                    stderr_log=stderr_log,
                ))
    return processes


def _tee_processes(runtype, analysis_settings, process_id, prefix, work_dir, inuring_priority=''):
    """Mirror of ``do_tees``: fan each summary stream out to ORD streams and work files (unguarded in bash)."""
    processes = []
    for summary in _summaries(analysis_settings, runtype):
        if 'id' not in summary:
            continue
        summary_set = summary['id']
        ord_output = summary.get('ord_output', {})
        outputs = []
        for ord_type, output_switch in OUTPUT_SWITCHES.items():
            if any(ord_output.get(ord_table) for ord_table in output_switch):
                outputs.append(StreamPath(_fifo(prefix, runtype, process_id, f'{inuring_priority}S{summary_set}_{ord_type}')))
        if ord_output.get('alt_period'):
            outputs.append(f'{work_dir}{runtype}_{inuring_priority}S{summary_set}_summary_palt/P{process_id}.bin')
        if ord_output.get('alt_meanonly'):
            outputs.append(f'{work_dir}{runtype}_{inuring_priority}S{summary_set}_summary_altmeanonly/P{process_id}.bin')
        if ord_enabled(summary, ORD_LECCALC):
            outputs.append(f'{work_dir}{runtype}_{inuring_priority}S{summary_set}_summaryleccalc/P{process_id}.bin')
        # bash's `tee < x > /dev/null` with no file argument only drains x: '-' to a discarded stdout does the same
        processes.append(_tee(outputs or [TEE_STDOUT],
                              stdin=_fifo(prefix, runtype, process_id, f'{inuring_priority}S{summary_set}_summary')))
    return processes


def _summary_processes(runtype, analysis_settings, process_id, prefix, stderr_log, inuring_priority=None):
    """Mirror of ``do_summarycalcs``: one summarypy per process."""
    summaries = _summaries(analysis_settings, runtype)
    if not summaries:
        return []
    argv = ['summarypy', '-t', 'ri' if runtype == RUNTYPE_REINSURANCE_GROSS_LOSS else runtype]
    inuring_priority_text = ''
    if runtype in (RUNTYPE_REINSURANCE_LOSS, RUNTYPE_REINSURANCE_GROSS_LOSS) and inuring_priority.get('level'):
        argv += ['-p', os.path.join('input', f"RI_{inuring_priority['level']}")]
        inuring_priority_text = inuring_priority['text']
    for summary in summaries:
        if 'id' in summary:
            argv += [f"-{summary['id']}",
                     StreamPath(_fifo(prefix, runtype, process_id, f"{inuring_priority_text}S{summary['id']}_summary"))]
    return [Process('summarypy', argv, stdin=_fifo(prefix, runtype, process_id, inuring_priority_text.rstrip('_')),
                    stderr_log=stderr_log)]


def _consumers(runtype, analysis_settings, process_ids, prefix, work_dir, stderr_log, inuring_priority=None):
    """ORD readers, tees then summarypy for every process, in bash's order (readers first)."""
    text = inuring_priority['text'] if inuring_priority else ''
    processes = []
    for process_id in process_ids:
        processes += _ord_processes(runtype, analysis_settings, process_id, prefix, work_dir, stderr_log, text)
    for process_id in process_ids:
        processes += _tee_processes(runtype, analysis_settings, process_id, prefix, work_dir, text)
    for process_id in process_ids:
        processes += _summary_processes(runtype, analysis_settings, process_id, prefix, stderr_log, inuring_priority)
    return processes


def _workfolders(runtype, analysis_settings, work_dir, inuring_priority=''):
    """Mirror of ``create_workfolders``."""
    dirs = []
    for summary in _summaries(analysis_settings, runtype):
        if 'id' not in summary:
            continue
        summary_set = summary['id']
        if ord_enabled(summary, ORD_LECCALC):
            dirs.append(f'{work_dir}{runtype}_{inuring_priority}S{summary_set}_summaryleccalc')
        if summary.get('ord_output', {}).get('alt_period'):
            dirs.append(f'{work_dir}{runtype}_{inuring_priority}S{summary_set}_summary_palt')
        if summary.get('ord_output', {}).get('alt_meanonly'):
            dirs.append(f'{work_dir}{runtype}_{inuring_priority}S{summary_set}_summary_altmeanonly')
    return dirs


def _getmodel_processes(p, gul_id, max_process_id, stderr_log):
    """Processes of the bash getmodel command (``evepy | [modelpy |] gul... [| plapy] [| tee gul fifo]``), unconnected."""
    analysis_settings = p['analysis_settings']
    args = {
        'number_of_samples': p['number_of_samples'],
        'gul_threshold': p['gul_threshold'],
        'gul_alloc_rule': p['gul_alloc_rule'],
        'process_id': gul_id,
        'max_process_id': max_process_id,
        'eve_shuffle_flag': p['eve_shuffle_flag'],
        'gulmc': p['gulmc'],
        'gul_random_generator': p['gul_random_generator'],
        'gulmc_effective_damageability': p['gulmc_effective_damageability'],
        'gulmc_vuln_cache_size': p['gulmc_vuln_cache_size'],
        'modelpy_server': p['model_py_server'],
        'peril_filter': p['peril_filter'],
        'model_df_engine': p.get('model_df_engine', 'oasis_data_manager.df_reader.reader.OasisPandasReader'),
        'dynamic_footprint': p.get('dynamic_footprint', False),
    }
    args.update(p.get('custom_args') or {})

    item_processes = []
    pla_fragment = get_pla_cmd(analysis_settings.get('pla', False), analysis_settings.get('pla_secondary_factor', 1),
                               analysis_settings.get('pla_uniform_factor', 0))
    if pla_fragment:
        item_processes.append(Process('plapy', _split(pla_fragment.replace('|', '', 1))))
    if p['need_summary_fifo_for_gul']:
        item_processes.append(_tee([StreamPath(_fifo('', RUNTYPE_GROUNDUP_LOSS, gul_id)), TEE_STDOUT]))

    custom_gulcalc_cmd = p.get('custom_gulcalc_cmd') if p.get('_get_getmodel_cmd') is not None else None
    if custom_gulcalc_cmd:
        # the whole group, item processes included, logs to gul_stderror.err as in bash
        argv = [custom_gulcalc_cmd, '-e', str(args['process_id']), str(args['max_process_id']),
                '-a', os.path.abspath('analysis_settings.json'), '-p', 'input']
        processes = [Process(custom_gulcalc_cmd, argv)] + item_processes
        for process in processes:
            process.stderr_log = GUL_STDERR_LOG if p['stderr_guard'] else None
        return processes

    processes = [Process('evepy', ['evepy', *args['eve_shuffle_flag'].split(), str(args['process_id']), str(args['max_process_id'])])]
    if args['gulmc'] is True:
        gul_argv = _split(get_gulcmd(args['gulmc'], args['gul_random_generator'], args['gulmc_effective_damageability'],
                                     args['gulmc_vuln_cache_size'], args['modelpy_server'], args['peril_filter'],
                                     model_df_engine=args['model_df_engine'], dynamic_footprint=args['dynamic_footprint']))
    else:
        processes.append(Process('modelpy', _split(get_modelcmd(args['modelpy_server'], args['peril_filter']))))
        gul_argv = _split(get_gulcmd(args['gulmc'], args['gul_random_generator'], False, 0, False, 0,
                                     model_df_engine=args['model_df_engine']))
    gul_argv = gul_argv[:1] + _server_flags(p) + gul_argv[1:]
    gul_argv += [f"-S{args['number_of_samples']}", f"-L{args['gul_threshold']}", f"-a{args['gul_alloc_rule']}"]
    processes.append(Process(gul_argv[0], gul_argv))
    processes += item_processes
    for process in processes:
        process.stderr_log = stderr_log
    return processes


def _fmpy_argv(p, *flags):
    return _split(get_fmcmd(p['fmpy_low_memory'], p['fmpy_sort_output'])) + list(flags)


def _ri_pipeline(p, head, source, process_id, prefix, step_flags, stderr_log, pipes):
    """Mirror of ``get_main_cmd_ri_stream``."""
    analysis_settings = p['analysis_settings']
    num_ri = p['num_reinsurance_iterations']
    ri_inuring_priorities = {ip['level']: ip['text'] for ip in get_ri_inuring_priorities(analysis_settings, num_ri)
                             if ip['level'] and p['ri_output']}
    rl_inuring_priorities = {ip['level']: ip['text'] for ip in get_rl_inuring_priorities(num_ri) if p['rl_output']}

    processes = list(head)
    processes.append(Process('fmpy', _fmpy_argv(p, f"-a{p['il_alloc_rule']}", *step_flags), stdin=source, stderr_log=stderr_log))
    if p['il_output']:
        processes.append(_tee([StreamPath(_fifo(prefix, RUNTYPE_INSURED_LOSS, process_id)), TEE_STDOUT], stderr_log=stderr_log))

    final_stdout = None
    for i in range(1, num_ri + 1):
        argv = _fmpy_argv(p, f"-a{p['ri_alloc_rule']}", '-p', os.path.join('input', f'RI_{i}'))
        if rl_inuring_priorities:
            argv += ['-o', StreamPath(_fifo(prefix, RUNTYPE_REINSURANCE_GROSS_LOSS, process_id, rl_inuring_priorities[i].rstrip('_')))]
        if i < num_ri:
            argv += ['-n', '-']
        tee = None
        if i in ri_inuring_priorities:
            if i == num_ri:
                argv += ['-n', '-']
                final_stdout = _fifo(prefix, RUNTYPE_REINSURANCE_LOSS, process_id)
            else:
                tee = _tee([StreamPath(_fifo(prefix, RUNTYPE_REINSURANCE_LOSS, process_id, ri_inuring_priorities[i].rstrip('_'))),
                            TEE_STDOUT], stderr_log=stderr_log)
        processes.append(Process('fmpy', argv, stderr_log=stderr_log))
        if tee is not None:
            processes.append(tee)

    pipes.chain(processes)
    processes[-1].stdout = final_stdout
    return processes


def _il_pipeline(p, head, source, process_id, prefix, step_flags, stderr_log, pipes):
    """Mirror of ``get_main_cmd_il_stream``."""
    processes = list(head) + [Process('fmpy', _fmpy_argv(p, f"-a{p['il_alloc_rule']}", *step_flags), stdin=source, stderr_log=stderr_log)]
    pipes.chain(processes)
    processes[-1].stdout = _fifo(prefix, RUNTYPE_INSURED_LOSS, process_id)
    return processes


def _gul_pipeline(head, process_id, prefix, pipes):
    """Mirror of ``get_main_cmd_gul_stream``."""
    processes = pipes.chain(list(head))
    processes[-1].stdout = _fifo(prefix, RUNTYPE_GROUNDUP_LOSS, process_id)
    return processes


def describe_analysis(params=None, **kwargs):
    """Build the analysis :class:`Run` without the support check or stream validation.

    :func:`build_analysis` is the entry point; this exists so a configuration bash can generate
    but the orchestrator cannot run (full correlation) can still be compared with the bash script.
    """
    p = _merge(params, kwargs)
    analysis_settings = p['analysis_settings']
    process_number = p.get('process_number')
    stderr_guard = p['stderr_guard']
    stderr_log = STDERR_LOG if stderr_guard else None
    full_correlation = p['full_correlation']
    gul_output, il_output, ri_output, rl_output = p['gul_output'], p['il_output'], p['ri_output'], p['rl_output']
    num_ri = p['num_reinsurance_iterations']
    max_process_id = p['max_process_id']
    work_dir = p['work_dir']
    work_fc_dir = p['work_full_correlation_dir']
    process_ids = process_range(max_process_id, process_number)

    make_dirs = []
    if full_correlation:
        make_dirs.append(p['output_full_correlation_dir'])
    make_dirs.append(p['work_kat_dir'])
    if full_correlation:
        make_dirs += [work_fc_dir, p['work_full_correlation_kat_dir']]
    if gul_output:
        make_dirs += _workfolders(RUNTYPE_GROUNDUP_LOSS, analysis_settings, work_dir)
        if full_correlation:
            make_dirs += _workfolders(RUNTYPE_GROUNDUP_LOSS, analysis_settings, work_fc_dir)
    if il_output:
        make_dirs += _workfolders(RUNTYPE_INSURED_LOSS, analysis_settings, work_dir)
        if full_correlation:
            make_dirs += _workfolders(RUNTYPE_INSURED_LOSS, analysis_settings, work_fc_dir)
    if ri_output:
        for inuring_priority in get_ri_inuring_priorities(analysis_settings, num_ri):
            make_dirs += _workfolders(RUNTYPE_REINSURANCE_LOSS, analysis_settings, work_dir, inuring_priority['text'])
        if full_correlation:
            make_dirs += _workfolders(RUNTYPE_REINSURANCE_LOSS, analysis_settings, work_fc_dir)
    if rl_output:
        for inuring_priority in get_rl_inuring_priorities(num_ri):
            make_dirs += _workfolders(RUNTYPE_REINSURANCE_GROSS_LOSS, analysis_settings, work_dir, inuring_priority['text'])

    processes = []
    dirs = [('', work_dir)] + ([(FULL_CORRELATION_STREAM_PREFIX, work_fc_dir)] if full_correlation else [])
    for prefix, consumer_work_dir in dirs:
        if rl_output:
            for inuring_priority in get_rl_inuring_priorities(num_ri):
                processes += _consumers(RUNTYPE_REINSURANCE_GROSS_LOSS, analysis_settings, process_ids, prefix, consumer_work_dir,
                                        stderr_log, inuring_priority)
        if ri_output:
            for inuring_priority in get_ri_inuring_priorities(analysis_settings, num_ri):
                processes += _consumers(RUNTYPE_REINSURANCE_LOSS, analysis_settings, process_ids, prefix, consumer_work_dir,
                                        stderr_log, inuring_priority)
        if il_output:
            processes += _consumers(RUNTYPE_INSURED_LOSS, analysis_settings, process_ids, prefix, consumer_work_dir, stderr_log)
        if gul_output:
            processes += _consumers(RUNTYPE_GROUNDUP_LOSS, analysis_settings, process_ids, prefix, consumer_work_dir, stderr_log)

    background = []
    if p.get('socket_server_size') and p.get('analysis_pk') is None:
        argv = ['socket-server', str(p['socket_server_size'])]
        if p.get('socket_server_port') is not None:
            argv.append(str(p['socket_server_port']))
        background.append(Process('socket-server', argv, stderr_log=None))

    # Each entry is (head processes, source stream): the getmodel processes, or a stream fmpy reads directly.
    gul_streams = {}
    for gul_id in process_ids:
        if full_correlation:
            fc_gul = _fifo(FULL_CORRELATION_STREAM_PREFIX, RUNTYPE_GROUNDUP_LOSS, gul_id)
            correlated = _fifo(FULL_CORRELATION_STREAM_PREFIX, RUNTYPE_GROUNDUP_LOSS, gul_id, 'fc')
            if p['need_summary_fifo_for_gul']:
                gul_streams.setdefault(FULL_CORRELATION_STREAM_PREFIX, []).append(
                    ([_tee([StreamPath(fc_gul), TEE_STDOUT], stdin=correlated, stderr_log=stderr_log)], None))
            elif not gul_output:
                gul_streams.setdefault(FULL_CORRELATION_STREAM_PREFIX, []).append(([], correlated))
        gul_streams.setdefault('', []).append((_getmodel_processes(p, gul_id, max_process_id, stderr_log), None))

    step_flags = ['-S'] if _has_step_policies() else []
    pipes = _Pipes()
    for prefix, entries in gul_streams.items():
        for i, (head, source) in enumerate(entries):
            process_id = process_number if process_number is not None else i + 1
            if ri_output or rl_output:
                pipeline = _ri_pipeline(p, head, source, process_id, prefix, step_flags, stderr_log, pipes)
            elif il_output:
                pipeline = _il_pipeline(p, head, source, process_id, prefix, step_flags, stderr_log, pipes)
            else:
                pipeline = _gul_pipeline(head, process_id, prefix, pipes)
            processes += reversed(pipeline)

    stage = Stage(name='analysis', processes=processes, make_dirs=make_dirs, remove_paths=[f'{work_dir}*'],
                  clean_output_dir=OUTPUT_DIR_TO_CLEAN)

    env = {}
    if p.get('log_level') is not None:
        env['OASIS_PYTOOLS_LOG_LEVEL'] = str(p['log_level'])

    chunk_check = None
    if stderr_guard and process_number:
        chunk_check = ChunkCheck(stderr_log=STDERR_LOG, ignorable_stderr_line=IGNORABLE_STDERR_LINE,
                                 non_empty_glob=f'{process_number}.work/**/P{process_number}.bin')
    return Run(stages=[stage], background=background, env=env, chunk_check=chunk_check)


def build_analysis(params=None, **kwargs):
    """Describe the analysis run that ``create_bash_analysis`` would script.

    Args:
        params (dict): the dict returned by ``bash_params``, plus optionally ``analysis_pk``,
            ``socket_server_size``, ``socket_server_port`` and ``custom_gulcalc_cmd``.
            Extra keys are ignored.
        **kwargs: same, as keyword arguments; they override ``params``.

    Returns:
        Run: a single ``analysis`` stage, the optional local socket-server as a background
        process, the environment and the chunk-mode check.

    Raises:
        OasisException: if :func:`unsupported_reason` returns a reason.
    """
    p = _merge(params, kwargs)
    reason = unsupported_reason(p)
    if reason:
        raise OasisException(f'Orchestrator cannot run this analysis: {reason}')
    run = describe_analysis(p)
    for stage in run.stages:
        stage.validate()
    return run


def _join_summary_info(summary_info_filename, filename):
    return Process('join-summary-info', ['join-summary-info', '-s', summary_info_filename, '-d', filename, '-o', filename],
                   stderr_log=None)


def _join_enabled(analysis_settings, join_summary_info):
    return bool(join_summary_info or analysis_settings.get('join_summary_info', False))


def _kat_processes(runtype, analysis_settings, max_process_id, work_dir, output_dir, sort_by_event, join, inuring_priority=''):
    """Mirror of ``do_kats``: returns (katpy processes, join-summary-info processes)."""
    kats, joins = [], []
    for summary in _summaries(analysis_settings, runtype):
        if 'id' not in summary:
            continue
        summary_set = summary['id']
        for output_switch in OUTPUT_SWITCHES.values():
            for ord_table, v in output_switch.items():
                if not summary.get('ord_output', {}).get(ord_table):
                    continue
                argv = ['katpy', v['kat_flag']] if sort_by_event else ['katpy', '-u', v['kat_flag']]
                argv += ['-f', 'bin', '-i']
                argv += [f'{work_dir}{runtype}_{inuring_priority}S{summary_set}_{ord_table}_P{process_id}'
                         for process_id in process_range(max_process_id)]
                outfile_ext = 'parquet' if summary.get('ord_output', {}).get('parquet_format') else 'csv'
                outfile = f"{output_dir}{runtype}_{inuring_priority}S{summary_set}_{v['table_name']}.{outfile_ext}"
                kats.append(Process('katpy', argv + ['-o', outfile], stderr_log=None))
                if join:
                    joins.append(_join_summary_info(f'{output_dir}{runtype}_S{summary_set}_summary-info.{outfile_ext}', outfile))
    return kats, joins


def _post_wait_processes(runtype, analysis_settings, work_sub_dir, output_dir, stderr_log, join, inuring_priority=''):
    """Mirror of ``do_post_wait_processing``: returns (aalpy/lecpy processes, join-summary-info processes)."""
    processes, joins = [], []
    for summary in _summaries(analysis_settings, runtype):
        if 'id' not in summary:
            continue
        summary_set = summary['id']
        ord_output = summary.get('ord_output', {})
        outfile_ext = 'parquet' if ord_output.get('parquet_format') else 'csv'
        summary_info = f'{output_dir}{runtype}_S{summary_set}_summary-info.{outfile_ext}'

        if ord_enabled(summary, ORD_ALT_OUTPUT_SWITCHES):
            switches = ORD_ALT_OUTPUT_SWITCHES['alt_period']
            palt = f'{output_dir}{runtype}_{inuring_priority}S{summary_set}_palt.{outfile_ext}'
            alct = f'{output_dir}{runtype}_{inuring_priority}S{summary_set}_alct.{outfile_ext}'
            argv = [switches['executable'], f"{switches['subfolder_flag']}{work_sub_dir}{runtype}_{inuring_priority}S{summary_set}_summary_palt"]
            if ord_output.get('alct_convergence'):
                argv += [switches['alct_flag'], alct]
                if ord_output.get('alct_confidence'):
                    argv += [switches['alct_confidence_level'], str(ord_output.get('alct_confidence'))]
            if outfile_ext == 'parquet':
                argv += ['-E', 'parquet']
            argv += [switches['csv_flag'], palt]
            processes.append(Process(switches['executable'], argv, stderr_log=stderr_log))
            if join:
                joins.append(_join_summary_info(summary_info, palt))
                if ord_output.get('alct_convergence'):
                    joins.append(_join_summary_info(summary_info, alct))

        if ord_enabled(summary, ORD_ALT_MEANONLY_OUTPUT_SWITCHES):
            switches = ORD_ALT_MEANONLY_OUTPUT_SWITCHES['alt_meanonly']
            altmeanonly = f'{output_dir}{runtype}_{inuring_priority}S{summary_set}_altmeanonly.{outfile_ext}'
            argv = [switches['executable'],
                    f"{switches['subfolder_flag']}{work_sub_dir}{runtype}_{inuring_priority}S{summary_set}_summary_altmeanonly"]
            if outfile_ext == 'parquet':
                argv += ['-E', 'parquet']
            argv += [switches['csv_flag'], altmeanonly]
            processes.append(Process(switches['executable'], argv, stderr_log=stderr_log))
            if join:
                joins.append(_join_summary_info(summary_info, altmeanonly))

        if ord_enabled(summary, ORD_LECCALC):
            argv = ['lecpy']
            if ord_output.get('return_period_file'):
                argv.append('-r')
            argv.append(f'-K{work_sub_dir}{runtype}_{inuring_priority}S{summary_set}_summaryleccalc')
            ept_output = psept_output = False
            for option, active in sorted(ord_output.items()):
                if active and option in ORD_EPT_OUTPUT_SWITCHES:
                    argv.append(ORD_EPT_OUTPUT_SWITCHES[option])
                    ept_output = True
                if active and option in ORD_PSEPT_OUTPUT_SWITCHES:
                    argv.append(ORD_PSEPT_OUTPUT_SWITCHES[option])
                    psept_output = True
            if outfile_ext == 'parquet':
                argv += ['-E', 'parquet']
            ept = f'{output_dir}{runtype}_{inuring_priority}S{summary_set}_ept.{outfile_ext}'
            psept = f'{output_dir}{runtype}_{inuring_priority}S{summary_set}_psept.{outfile_ext}'
            if ept_output:
                argv += ['-O', ept]
            if psept_output:
                argv += ['-o', psept]
            processes.append(Process('lecpy', argv, stderr_log=stderr_log))
            if join:
                if ept_output:
                    joins.append(_join_summary_info(summary_info, ept))
                if psept_output:
                    joins.append(_join_summary_info(summary_info, psept))
    return processes, joins


def build_outputs(params=None, **kwargs):
    """Describe the output run that ``create_bash_outputs`` would script.

    The stages are, each only when it has processes: ``kats`` (katpy), ``kats-join-summary-info``,
    ``aal-lec`` (aalpy/lecpy), ``aal-lec-join-summary-info``; then ``cleanup`` (work directory
    removal) when ``remove_working_files`` is set. The load balancer never applies here (bash
    forces it off), so no support check is made.

    Args:
        params (dict): the dict returned by ``bash_params``. Extra keys are ignored.
        **kwargs: same, as keyword arguments; they override ``params``.

    Returns:
        Run: the output stages, with no background process and no environment.
    """
    p = _merge(params, kwargs)
    analysis_settings = p['analysis_settings']
    stderr_log = STDERR_LOG if p['stderr_guard'] else None
    join = _join_enabled(analysis_settings, p['join_summary_info'])
    full_correlation = p['full_correlation']
    gul_output, il_output, ri_output, rl_output = p['gul_output'], p['il_output'], p['ri_output'], p['rl_output']
    num_ri = p['num_reinsurance_iterations']
    max_process_id = p['max_process_id']
    sort_by_event = p['kat_sort_by_event']
    output_dir, output_fc_dir = p['output_dir'], p['output_full_correlation_dir']
    work_kat_dir, work_fc_kat_dir = p['work_kat_dir'], p['work_full_correlation_kat_dir']

    kat_calls = []
    if rl_output:
        kat_calls += [(RUNTYPE_REINSURANCE_GROSS_LOSS, work_kat_dir, output_dir, ip['text']) for ip in get_rl_inuring_priorities(num_ri)]
    if ri_output:
        kat_calls += [(RUNTYPE_REINSURANCE_LOSS, work_kat_dir, output_dir, ip['text'])
                      for ip in get_ri_inuring_priorities(analysis_settings, num_ri)]
        if full_correlation:
            kat_calls.append((RUNTYPE_REINSURANCE_LOSS, work_fc_kat_dir, output_fc_dir, ''))
    if il_output:
        kat_calls.append((RUNTYPE_INSURED_LOSS, work_kat_dir, output_dir, ''))
        if full_correlation:
            kat_calls.append((RUNTYPE_INSURED_LOSS, work_fc_kat_dir, output_fc_dir, ''))
    if gul_output:
        kat_calls.append((RUNTYPE_GROUNDUP_LOSS, work_kat_dir, output_dir, ''))
        if full_correlation:
            kat_calls.append((RUNTYPE_GROUNDUP_LOSS, work_fc_kat_dir, output_fc_dir, ''))

    kats, kat_joins = [], []
    for runtype, kat_dir, out_dir, inuring_priority in kat_calls:
        procs, joins = _kat_processes(runtype, analysis_settings, max_process_id, kat_dir, out_dir, sort_by_event, join, inuring_priority)
        kats += procs
        kat_joins += joins

    post_calls = []
    if rl_output:
        post_calls += [(RUNTYPE_REINSURANCE_GROSS_LOSS, '', output_dir, ip['text']) for ip in get_rl_inuring_priorities(num_ri)]
    if ri_output:
        post_calls += [(RUNTYPE_REINSURANCE_LOSS, '', output_dir, ip['text']) for ip in get_ri_inuring_priorities(analysis_settings, num_ri)]
    if il_output:
        post_calls.append((RUNTYPE_INSURED_LOSS, '', output_dir, ''))
    if gul_output:
        post_calls.append((RUNTYPE_GROUNDUP_LOSS, '', output_dir, ''))
    if full_correlation:
        work_sub_dir = re.sub('^work/', '', p['work_full_correlation_dir'])
        for runtype, enabled in ((RUNTYPE_REINSURANCE_LOSS, ri_output), (RUNTYPE_INSURED_LOSS, il_output), (RUNTYPE_GROUNDUP_LOSS, gul_output)):
            if enabled:
                post_calls.append((runtype, work_sub_dir, output_fc_dir, ''))

    post, post_joins = [], []
    for runtype, work_sub_dir, out_dir, inuring_priority in post_calls:
        procs, joins = _post_wait_processes(runtype, analysis_settings, work_sub_dir, out_dir, stderr_log, join, inuring_priority)
        post += procs
        post_joins += joins

    stages = [Stage(name=name, processes=procs) for name, procs in (
        ('kats', kats), ('kats-join-summary-info', kat_joins), ('aal-lec', post), ('aal-lec-join-summary-info', post_joins)) if procs]
    if p['remove_working_files']:
        stages.append(Stage(name='cleanup', remove_paths=[os.path.join(p['work_dir'], '*')]))
    for stage in stages:
        stage.validate()
    return Run(stages=stages)
