"""Equivalence of the orchestrator run description with the bash script generated from the same params."""

import json
import os
import shlex
from collections import Counter
from pathlib import Path

import pytest

from oasislmf.execution import bash
from oasislmf.execution.bash import bash_params, bash_wrapper, create_bash_analysis, create_bash_outputs
from oasislmf.execution.orchestrator.build import (FULL_CORRELATION_STREAM_PREFIX, PIPE_STREAM_PREFIX, TEE_STDOUT, TEE_TOOL, Run,
                                                   build_analysis, build_outputs, describe_analysis, unsupported_reason)
from oasislmf.execution.orchestrator.model import GUL_STDERR_LOG, STDERR_LOG, Stream, StreamPath
from oasislmf.utils.exceptions import OasisException

SETTINGS_DIR = Path(__file__).parent / 'input_json_settings'
SETTINGS_NAMES = sorted(p.stem for p in SETTINGS_DIR.glob('*.json'))
PARTITIONS = [1, 8]
PIPE = 'PIPE'
FAKE_GULCALC = 'fake_supplier_gulcalc'

BASH_SKIP_PREFIXES = ('#', 'mkfifo ', 'check_fifos', '    ', 'kill -0 ')
SHELL_OPERATORS = '();<>|&'


def load_settings(name):
    with open(SETTINGS_DIR / f'{name}.json', encoding='utf-8') as f:
        return json.load(f)['analysis_settings']


def ri_settings():
    """Settings exercising intermediate inuring priorities, RI and RL outputs together."""
    settings = load_settings('ri_only_no_ord_leccalc_3')
    settings['ri_inuring_priorities'] = [1]
    settings['ri_summaries'] = [{'id': 1, 'ord_output': {'elt_sample': True, 'plt_sample': True, 'ept_full_uncertainty_aep': True}},
                                {'id': 2}]
    settings['rl_summaries'] = [{'id': 1, 'ord_output': {'elt_sample': True, 'alt_period': True}}]
    settings['il_summaries'] = [{'id': 1}]
    return settings


def make_params(settings, partitions, **overrides):
    kwargs = dict(
        max_process_id=partitions, analysis_settings=settings, fifo_tmp_dir=False, stderr_guard=True,
        gul_alloc_rule=1, il_alloc_rule=2, ri_alloc_rule=3, num_gul_per_lb=0, num_fm_per_lb=0, event_shuffle=1,
        summarypy_low_memory=False,
    )
    kwargs.update(overrides)
    params = bash_params(**kwargs)
    params.update(analysis_pk=None, socket_server_size=None, socket_server_port=None)
    return params


def bash_script(params, path, analysis=True, outputs=True):
    if path.exists():
        path.unlink()
    params = {**params, 'filename': str(path)}
    with bash_wrapper(str(path), False, params['stderr_guard']):
        if analysis:
            create_bash_analysis(**params)
        if outputs:
            create_bash_outputs(**params)
    return path.read_text()


def tokens(line):
    lexer = shlex.shlex(line, posix=True, punctuation_chars=SHELL_OPERATORS)
    lexer.whitespace_split = True
    return list(lexer)


def parse_command_line(line):
    """Split one bash command line into pipeline segments, as (argv, stdin, stdout, stderr_log) tuples."""
    segments, groups = [], []
    toks = tokens(line)
    current = None

    def new_segment(piped):
        segment = {'argv': [], 'stdin': PIPE if piped else None, 'stdout': None, 'log': None}
        segments.append(segment)
        return segment

    current = new_segment(False)
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok == '(':
            groups.append(len(segments) - 1)
        elif tok == ')':
            start = groups.pop()
            if toks[i + 1:i + 3] == ['2', '>>']:
                for segment in segments[start:]:
                    if segment['log'] is None:
                        segment['log'] = os.path.basename(toks[i + 3])
                i += 3
        elif tok == '|':
            current['stdout'] = PIPE
            current = new_segment(True)
        elif tok == '<':
            current['stdin'] = toks[i + 1]
            i += 1
        elif tok == '>':
            current['stdout'] = None if toks[i + 1] == '/dev/null' else toks[i + 1]
            i += 1
        elif tok == '&':
            break
        else:
            current['argv'].append(tok)
        i += 1
    return [(tuple(s['argv']), s['stdin'], s['stdout'], s['log']) for s in segments]


def parse_bash(script, fifo_dir):
    """Group the bash script into exec_wait-delimited stages of normalised commands and setup operations."""
    lines = script.splitlines()
    body = lines[max(i for i, line in enumerate(lines) if line == '}') + 1:]
    groups, background, env = [], [], {}
    current = {'commands': [], 'remove': [], 'mkdir': [], 'clean': None}

    def close():
        nonlocal current
        groups.append(current)
        current = {'commands': [], 'remove': [], 'mkdir': [], 'clean': None}

    for line in body:
        if not line.strip() or line.startswith(BASH_SKIP_PREFIXES):
            continue
        if line.startswith('exec_wait'):
            close()
        elif line.startswith('rm -R -f '):
            path = line[len('rm -R -f '):]
            if not path.startswith(fifo_dir):
                current['remove'].append(path)
        elif line.startswith('mkdir -p '):
            path = line[len('mkdir -p '):]
            if not path.startswith(fifo_dir):
                current['mkdir'].append(path)
        elif line.startswith('find '):
            if line.startswith('find output '):
                current['clean'] = 'output'
        elif line.startswith('export '):
            key, value = line[len('export '):].split('=', 1)
            env[key] = value
        elif line.startswith('socket-server '):
            background += parse_command_line(line)
        else:
            current['commands'] += parse_command_line(line)
    if current['commands'] or current['remove'] or current['mkdir'] or current['clean']:
        close()
    return groups, background, env


def stream_path(stream, params):
    if stream.name.startswith(PIPE_STREAM_PREFIX):
        return PIPE
    if stream.name.startswith(FULL_CORRELATION_STREAM_PREFIX):
        return params['fifo_full_correlation_dir'] + stream.name[len(FULL_CORRELATION_STREAM_PREFIX):]
    return params['fifo_queue_dir'] + stream.name


def endpoint(value, params):
    return stream_path(value, params) if isinstance(value, Stream) else value


def normalise(process, params):
    argv = [stream_path(arg.stream, params) if isinstance(arg, StreamPath) else arg for arg in process.argv]
    if argv[0] == TEE_TOOL:
        argv = ['tee'] + [arg for arg in argv[1:] if arg != TEE_STDOUT]
    return tuple(argv), endpoint(process.stdin, params), endpoint(process.stdout, params), process.stderr_log


def describe_run(run, params):
    groups = [{
        'commands': [normalise(process, params) for process in stage.processes],
        'remove': stage.remove_paths,
        'mkdir': stage.make_dirs,
        'clean': stage.clean_output_dir,
    } for stage in run.stages]
    return groups, [normalise(process, params) for process in run.background], run.env


def fifo_ends(commands, fifo_dir):
    """Map each FIFO path to (writer command, reader command)."""
    ends = {}
    for command in commands:
        argv, stdin, stdout, _ = command
        for path in list(argv) + [stdout]:
            if isinstance(path, str) and path.startswith(fifo_dir):
                ends.setdefault(path, [None, None])[0] = command
        if isinstance(stdin, str) and stdin.startswith(fifo_dir):
            ends.setdefault(stdin, [None, None])[1] = command
    return ends


def assert_same_run(bash_text, runs, params):
    bash_groups, bash_background, bash_env = parse_bash(bash_text, params['fifo_queue_dir'])
    orch_groups, orch_background, orch_env = [], [], {}
    for run in runs:
        groups, background, env = describe_run(run, params)
        orch_groups += groups
        orch_background += background
        orch_env.update(env)

    assert len(bash_groups) == len(orch_groups), (len(bash_groups), len(orch_groups))
    for bash_group, orch_group in zip(bash_groups, orch_groups):
        missing = Counter(bash_group['commands']) - Counter(orch_group['commands'])
        extra = Counter(orch_group['commands']) - Counter(bash_group['commands'])
        assert not missing and not extra, f'missing from orchestrator: {list(missing)}\nextra in orchestrator: {list(extra)}'
        assert bash_group['remove'] == orch_group['remove']
        assert bash_group['mkdir'] == orch_group['mkdir']
        assert bash_group['clean'] == orch_group['clean']
        assert fifo_ends(bash_group['commands'], params['fifo_queue_dir']) == fifo_ends(orch_group['commands'], params['fifo_queue_dir'])
    assert Counter(bash_background) == Counter(orch_background)
    assert bash_env == orch_env


def check_equivalence(params, tmp_path, process_number=None):
    """Compare bash's analysis + outputs scripts with the orchestrator description of the same params."""
    if process_number is not None:
        params = {**params, 'process_number': process_number}
    reason = unsupported_reason(params)
    if reason and not params['full_correlation']:
        pytest.skip(f'unsupported by the orchestrator: {reason}')
    if reason:
        with pytest.raises(OasisException):
            build_analysis(params)
        analysis = describe_analysis(params)
    else:
        analysis = build_analysis(params)
    if process_number is None:
        text = bash_script(params, tmp_path / 'run.sh')
        assert_same_run(text, [analysis, build_outputs(params)], params)
    else:
        text = bash_script(params, tmp_path / f'run.{process_number}.sh', outputs=False)
        assert_same_run(text, [analysis], params)
    return analysis


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Run in an empty directory so bash's input/ probes see the same thing as the builder."""
    for var in ('OASIS_WEBSOCKET_URL', 'OASIS_WEBSOCKET_PORT', 'OASIS_ANALYSIS_STATUS_URL'):
        monkeypatch.delenv(var, raising=False)
    work = tmp_path / 'cwd'
    work.mkdir()
    monkeypatch.chdir(work)


def _ri_iterations(name):
    settings = load_settings(name)
    return [0, 2] if settings.get('ri_output') or settings.get('rl_output') else [0]


EQUIVALENCE_CASES = [(name, partitions, ri) for name in SETTINGS_NAMES for partitions in PARTITIONS for ri in _ri_iterations(name)]


@pytest.mark.parametrize('gulmc', [True, False], ids=['gulmc', 'gulpy'])
@pytest.mark.parametrize('name,partitions,ri_iterations', EQUIVALENCE_CASES)
def test_equivalent_to_bash(name, partitions, ri_iterations, gulmc, tmp_path):
    params = make_params(load_settings(name), partitions, num_reinsurance_iterations=ri_iterations, gulmc=gulmc)
    check_equivalence(params, tmp_path)


@pytest.mark.parametrize('name,partitions,ri_iterations', EQUIVALENCE_CASES)
def test_chunk_mode_equivalent_to_bash(name, partitions, ri_iterations, tmp_path):
    params = make_params(load_settings(name), partitions, num_reinsurance_iterations=ri_iterations)
    for process_number in range(1, partitions + 1):
        run = check_equivalence(params, tmp_path, process_number=process_number)
        assert run.chunk_check is not None
        assert run.chunk_check.non_empty_glob == f'{process_number}.work/**/P{process_number}.bin'
        assert run.chunk_check.stderr_log == STDERR_LOG


def test_chunk_mode_work_dir_from_bash_params(tmp_path):
    params = make_params(load_settings('gul_il_ord_palt_output'), 4, process_number=3)
    run = check_equivalence(params, tmp_path, process_number=3)
    assert run.stages[0].remove_paths == ['3.work/*']
    assert '3.work/kat/' in run.stages[0].make_dirs
    assert {p.argv[-2:] == ['3', '4'] for p in run.stages[0].processes if p.name == 'evepy'} == {True}


@pytest.mark.parametrize('partitions', PARTITIONS)
@pytest.mark.parametrize('ri_iterations', [1, 3])
@pytest.mark.parametrize('stderr_guard', [True, False])
def test_inuring_priorities_equivalent_to_bash(partitions, ri_iterations, stderr_guard, tmp_path):
    params = make_params(ri_settings(), partitions, num_reinsurance_iterations=ri_iterations, stderr_guard=stderr_guard)
    run = check_equivalence(params, tmp_path)
    if not stderr_guard:
        assert {p.stderr_log for p in run.stages[0].processes} == {None}


def test_rl_only_final_fmpy_has_no_stdout(tmp_path):
    settings = ri_settings()
    settings['ri_output'] = False
    params = make_params(settings, 1, num_reinsurance_iterations=2)
    run = check_equivalence(params, tmp_path)
    fmpys = [p for p in run.stages[0].processes if p.name == 'fmpy']
    final = next(p for p in fmpys if 'input/RI_2' in p.argv)
    assert final.stdout is None and '-n' not in final.argv


def test_full_correlation_gulpy_is_described_but_unsupported(tmp_path):
    params = make_params(load_settings('all_outputs_1_fc'), 2, gulmc=False)
    assert params['full_correlation']
    assert 'full_correlation' in unsupported_reason(params)
    run = check_equivalence(params, tmp_path)
    with pytest.raises(ValueError, match='full_correlation_gul_fc_P1'):
        run.stages[0].validate()
    assert 'output/full_correlation/' in run.stages[0].make_dirs


def test_gulpy_pipeline(tmp_path):
    params = make_params(load_settings('gul_il_ord_palt_output'), 2, gulmc=False)
    run = check_equivalence(params, tmp_path)
    names = [p.name for p in run.stages[0].processes]
    assert 'modelpy' in names and 'gulpy' in names and 'gulmc' not in names


def test_pla_and_peril_filter(tmp_path):
    settings = load_settings('gul_ord_ept_2_output')
    settings['pla_uniform_factor'] = 0
    settings['pla_secondary_factor'] = 0.5
    params = make_params(settings, 2, peril_filter=['WTC', 'WSS'], gulmc_effective_damageability=True)
    run = check_equivalence(params, tmp_path)
    plapy = next(p for p in run.stages[0].processes if p.name == 'plapy')
    assert plapy.argv == ['plapy', '-f', '0.5']
    gulmc = next(p for p in run.stages[0].processes if p.name == 'gulmc')
    assert '--model-df-engine=oasis_data_manager.df_reader.reader.OasisPandasReader' in gulmc.argv
    assert gulmc.argv[gulmc.argv.index('--peril-filter') + 1:][:2] == ['WTC', 'WSS']


@pytest.mark.parametrize('event_shuffle', [0, 1, 2, 3])
def test_event_shuffle_flags(event_shuffle, tmp_path):
    params = make_params(load_settings('gul_summarycalc_1_output'), 2, event_shuffle=event_shuffle)
    run = check_equivalence(params, tmp_path)
    evepy = next(p for p in run.stages[0].processes if p.name == 'evepy')
    assert all(arg and ' ' not in arg for arg in evepy.argv)


def test_step_policies_probe(tmp_path):
    Path('input').mkdir()
    Path('input/fm_profile.csv').write_text('profile_id,calcrule_id,step_id\n1,1,1\n')
    params = make_params(load_settings('gul_il_ord_palt_output'), 2)
    run = check_equivalence(params, tmp_path)
    assert all('-S' in p.argv for p in run.stages[0].processes if p.name == 'fmpy')


def test_join_summary_info_and_parquet(tmp_path):
    params = make_params(load_settings('all_outputs_1_parquet'), 2, join_summary_info=True, num_reinsurance_iterations=1)
    check_equivalence(params, tmp_path)


def test_log_level_env(tmp_path):
    params = make_params(load_settings('gul_summarycalc_1_output'), 1, log_level=10)
    run = check_equivalence(params, tmp_path)
    assert run.env == {'OASIS_PYTOOLS_LOG_LEVEL': '10'}


@pytest.mark.parametrize('port', [None, 8765])
def test_local_socket_server(port, tmp_path):
    params = make_params(load_settings('gul_il_ord_palt_output'), 2)
    params.update(socket_server_size=1000, socket_server_port=port)
    run = check_equivalence(params, tmp_path)
    assert [p.argv for p in run.background] == [['socket-server', '1000'] + ([str(port)] if port else [])]
    assert run.background[0].stderr_log is None
    gulmc = next(p for p in run.stages[0].processes if p.name == 'gulmc')
    assert (f'--socket-server={port}' in gulmc.argv) == (port is not None)


def test_platform_progress_flags(monkeypatch, tmp_path):
    monkeypatch.setenv('OASIS_ANALYSIS_STATUS_URL', 'http://example.invalid')
    params = make_params(load_settings('gul_il_ord_palt_output'), 2, gulmc=False)
    params.update(analysis_pk=42, socket_server_size=1000, socket_server_port=8765)
    run = check_equivalence(params, tmp_path)
    assert run.background == []
    gulpy = next(p for p in run.stages[0].processes if p.name == 'gulpy')
    assert gulpy.argv[1:3] == ['--socket-server=True', '--analysis-pk=42']


@pytest.mark.parametrize('name', ['custom_gul_sample_1_output', 'gul_il_ord_palt_output', 'gul_ord_ept_2_output', 'il_only_summarycalc_1_output'])
@pytest.mark.parametrize('partitions', PARTITIONS)
def test_custom_gulcalc_binary(name, partitions, monkeypatch, tmp_path):
    real_which = bash.shutil.which
    monkeypatch.setattr(bash.shutil, 'which', lambda cmd, *a, **k: f'/usr/bin/{cmd}' if cmd == FAKE_GULCALC else real_which(cmd, *a, **k))
    params = make_params(load_settings(name), partitions, custom_gulcalc_cmd=FAKE_GULCALC, gulmc=False)
    params['custom_gulcalc_cmd'] = FAKE_GULCALC
    assert unsupported_reason(params) is None
    run = check_equivalence(params, tmp_path)
    processes = run.stages[0].processes
    binary = [p for p in processes if p.name == FAKE_GULCALC]
    assert len(binary) == partitions
    assert all(p.stderr_log == GUL_STDERR_LOG for p in binary)
    assert binary[0].argv[5] == os.path.abspath('analysis_settings.json')
    assert not any(p.name in ('evepy', 'gulmc', 'gulpy', 'modelpy') for p in processes)
    head_feeds_more = params['need_summary_fifo_for_gul'] or params['analysis_settings'].get('pla')
    assert (binary[0].argv[-1] == '-i') == bool(head_feeds_more)


def test_custom_gulcalc_without_name_is_unsupported(monkeypatch):
    monkeypatch.setattr(bash.shutil, 'which', lambda cmd, *a, **k: f'/usr/bin/{cmd}')
    params = make_params(load_settings('gul_summarycalc_1_output'), 1, custom_gulcalc_cmd=FAKE_GULCALC)
    assert 'custom_gulcalc_cmd' in unsupported_reason(params)
    with pytest.raises(OasisException):
        build_analysis(params)


def test_unsupported_reason():
    def params(**overrides):
        return make_params(load_settings('gul_il_ord_palt_output'), 4, **overrides)

    assert unsupported_reason(params()) is None
    assert unsupported_reason(**params()) is None
    assert 'load balancer' in unsupported_reason(params(num_gul_per_lb=2, num_fm_per_lb=2))
    assert unsupported_reason(params(num_gul_per_lb=2, num_fm_per_lb=2), process_number=1) is None
    assert unsupported_reason(params(num_gul_per_lb=2, num_fm_per_lb=0)) is None
    gul_only = make_params(load_settings('gul_summarycalc_1_output'), 4, num_gul_per_lb=2, num_fm_per_lb=2)
    assert unsupported_reason(gul_only) is None
    assert 'summarypy_low_memory' in unsupported_reason(params(summarypy_low_memory=True))
    assert 'model_py_server' in unsupported_reason(params(model_py_server=True))
    assert '_get_getmodel_cmd' in unsupported_reason(params(_get_getmodel_cmd=lambda **kwargs: 'my_gulcalc'))
    with pytest.raises(OasisException, match='_get_getmodel_cmd'):
        build_analysis(params(_get_getmodel_cmd=lambda **kwargs: 'my_gulcalc'))
    assert 'full_correlation' in unsupported_reason(make_params(load_settings('all_outputs_1_fc'), 1, gulmc=False))
    assert unsupported_reason(make_params(load_settings('all_outputs_1_fc'), 1, gulmc=True)) is None


def test_build_accepts_kwargs_and_extra_keys():
    params = make_params(load_settings('gul_il_ord_palt_output'), 2)
    params['unrelated_key'] = object()
    assert isinstance(build_analysis(**params), Run)
    assert isinstance(build_outputs(params, stderr_guard=False), Run)


def test_readers_start_before_writers():
    params = make_params(ri_settings(), 2, num_reinsurance_iterations=3)
    stage = build_analysis(params).stages[0]
    position = {id(p): i for i, p in enumerate(stage.processes)}
    for stream in stage.streams():
        writer = next(p for p in stage.processes if stream in p.streams_written())
        reader = next(p for p in stage.processes if stream in p.streams_read())
        assert position[id(reader)] < position[id(writer)], stream.name


def test_no_shell_syntax_in_argv():
    params = make_params(load_settings('all_outputs_1_csv'), 2, num_reinsurance_iterations=2)
    for run in (build_analysis(params), build_outputs(params)):
        for stage in run.stages:
            for process in stage.processes:
                for arg in process.argv:
                    if isinstance(arg, str):
                        assert arg and not set(arg) & set('|<>&()\'"$ '), (process.argv, arg)


def stage_names(run):
    return [stage.name for stage in run.stages]


def test_output_stages_full():
    params = make_params(load_settings('all_outputs_1_csv'), 2, join_summary_info=True)
    run = build_outputs(params)
    assert stage_names(run) == ['kats', 'kats-join-summary-info', 'aal-lec', 'aal-lec-join-summary-info', 'cleanup']
    assert {p.name for p in run.stages[0].processes} == {'katpy'}
    assert {p.name for p in run.stages[2].processes} == {'aalpy', 'lecpy'}
    assert {p.name for p in run.stages[1].processes} == {p.name for p in run.stages[3].processes} == {'join-summary-info'}
    assert run.stages[-1].processes == [] and run.stages[-1].remove_paths == ['work/*']
    assert run.background == [] and run.env == {} and run.chunk_check is None
    assert all(p.stderr_log is None for p in run.stages[0].processes + run.stages[1].processes)
    assert all(p.stderr_log == STDERR_LOG for p in run.stages[2].processes)


def test_output_stages_without_join_or_cleanup():
    params = make_params(load_settings('all_outputs_1_csv'), 2, remove_working_files=False)
    assert stage_names(build_outputs(params)) == ['kats', 'aal-lec']


def test_output_stages_summarycalc_only():
    params = make_params(load_settings('gul_summarycalc_1_output'), 2)
    assert stage_names(build_outputs(params)) == ['cleanup']


def test_output_stages_lec_only():
    params = make_params(load_settings('gul_ord_ept_1_output'), 2)
    assert stage_names(build_outputs(params)) == ['aal-lec', 'cleanup']


def test_analysis_stage_setup():
    params = make_params(load_settings('gul_il_ord_palt_output'), 2)
    run = build_analysis(params)
    assert stage_names(run) == ['analysis']
    stage = run.stages[0]
    assert stage.remove_paths == ['work/*']
    assert stage.make_dirs[0] == 'work/kat/'
    assert stage.clean_output_dir == 'output'
    assert run.chunk_check is None
    summary_tees = [p for p in stage.processes if p.name == TEE_TOOL and isinstance(p.stdin, Stream)
                    and p.stdin.name.endswith('_summary_P1')]
    assert summary_tees and all(p.stderr_log is None and p.stdout is None for p in summary_tees)
