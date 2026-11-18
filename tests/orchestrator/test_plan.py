"""The planner against genbash: the plan of a bash_params dict renders to exactly the commands of run_kernel.sh."""
import copy
import glob
import json
import os
import re
import shlex
import sys
from collections import Counter

import numpy as np
import pytest

from oasislmf.execution import bash
from oasislmf.execution.orchestrator import inline, pool
from oasislmf.execution.orchestrator.chain import ChainSpec, spec_for_taps
from oasislmf.execution.orchestrator.chunks import partition_events, read_events
from oasislmf.execution.orchestrator.plan import (OutputJob, Plan, Unsupported, plan_from_bash_params,
                                                  render_chunk_commands)
from oasislmf.pytools.eve.manager import event_id_dtype, partition_events__random
from oasislmf.pytools.eve.manager import run as evepy_run
from oasislmf.utils.defaults import EVE_FISHER_YATES, EVE_NO_SHUFFLE, EVE_ROUND_ROBIN, EVE_STD_SHUFFLE
from tests.orchestrator.tools.compare_outputs import compare_trees_bytes

from .test_chain import TREE_FACTORIES, ri_path
from .test_pool import TREE_FACTORY_STRINGS

ALL_ORD = {
    'plt_sample': True, 'plt_quantile': True, 'plt_moment': True,
    'elt_sample': True, 'elt_quantile': True, 'elt_moment': True,
    'alt_period': True, 'alt_meanonly': True, 'alct_convergence': True, 'alct_confidence': 0.9,
    'ept_full_uncertainty_aep': True, 'ept_full_uncertainty_oep': True, 'ept_mean_sample_aep': True,
    'ept_mean_sample_oep': True, 'ept_per_sample_mean_aep': True, 'ept_per_sample_mean_oep': True,
    'psept_aep': True, 'psept_oep': True, 'return_period_file': True,
}

CASES = {
    'gul_all_ord': (
        {'number_of_samples': 10, 'gul_threshold': 0, 'gul_output': True, 'gul_summaries': [{'id': 1, 'ord_output': ALL_ORD}]},
        {'num_reinsurance_iterations': 0},
    ),
    'gul_il_ri_low_memory': (
        {'number_of_samples': 4, 'gul_threshold': 0, 'gul_output': True, 'il_output': True, 'ri_output': True,
         'gul_summaries': [{'id': 1, 'ord_output': ALL_ORD}, {'id': 2, 'ord_output': {'elt_sample': True}}],
         'il_summaries': [{'id': 1, 'ord_output': {'alt_period': True}}, {'id': 2, 'ord_output': {'psept_aep': True, 'alt_meanonly': True}}],
         'ri_summaries': [{'id': 1, 'ord_output': {'ept_mean_sample_oep': True, 'alt_period': True}}]},
        {'num_reinsurance_iterations': 1, 'summarypy_low_memory': True},
    ),
    'gul_il_two_sets_parquet': (
        {'number_of_samples': 3, 'gul_threshold': 0.5, 'gul_output': True, 'il_output': True,
         'gul_summaries': [{'id': 1, 'ord_output': {'elt_sample': True, 'ept_mean_sample_oep': True}},
                           {'id': 2, 'ord_output': {'plt_moment': True, 'alt_period': True, 'parquet_format': True}}],
         'il_summaries': [{'id': 1, 'ord_output': {'elt_moment': True, 'elt_quantile': True}},
                          {'id': 3, 'ord_output': {'alt_meanonly': True, 'psept_aep': True, 'parquet_format': True}}]},
        {'num_reinsurance_iterations': 0, 'gul_alloc_rule': 1, 'il_alloc_rule': 3},
    ),
    'gul_il_ri_rl_intermediate': (
        {'number_of_samples': 10, 'gul_threshold': 0, 'gul_output': True, 'il_output': True, 'ri_output': True, 'rl_output': True,
         'ri_inuring_priorities': [1],
         'gul_summaries': [{'id': 1, 'ord_output': {'plt_sample': True, 'elt_quantile': True, 'elt_sample': True, 'alt_period': True,
                                                    'ept_full_uncertainty_aep': True, 'psept_oep': True}}],
         'il_summaries': [{'id': 1, 'ord_output': {'elt_moment': True}}, {'id': 2, 'ord_output': {'alt_meanonly': True}}],
         'ri_summaries': [{'id': 1, 'ord_output': {'elt_moment': True, 'plt_quantile': True}}],
         'rl_summaries': [{'id': 1, 'ord_output': {'alt_period': True}}]},
        {'num_reinsurance_iterations': 2},
    ),
    'il_ri_options': (
        {'number_of_samples': 5, 'gul_threshold': 1e-3, 'il_output': True, 'ri_output': True,
         'il_summaries': [{'id': 1, 'ord_output': {'elt_sample': True, 'ept_full_uncertainty_oep': True}}],
         'ri_summaries': [{'id': 1, 'ord_output': ALL_ORD}, {'id': 2, 'ord_output': {'plt_sample': True}}]},
        {'num_reinsurance_iterations': 3, 'fmpy_low_memory': True, 'fmpy_sort_output': True, 'event_shuffle': EVE_NO_SHUFFLE,
         'peril_filter': ['WTC', 'WSS'], 'gulmc_effective_damageability': True, 'gulmc_vuln_cache_size': 50,
         'dynamic_footprint': True, 'gul_random_generator': 1, 'ri_alloc_rule': 2},
    ),
    'ri_only_one_layer': (
        {'number_of_samples': 2, 'gul_threshold': 0, 'ri_output': True, 'ri_summaries': [{'id': 1, 'ord_output': {'elt_sample': True}}]},
        {'num_reinsurance_iterations': 1, 'event_shuffle': EVE_FISHER_YATES},
    ),
    'rl_only': (
        {'number_of_samples': 2, 'gul_threshold': 0, 'rl_output': True, 'rl_summaries': [{'id': 1, 'ord_output': {'elt_moment': True}}]},
        {'num_reinsurance_iterations': 2},
    ),
    'il_only': (
        {'number_of_samples': 2, 'gul_threshold': 0, 'il_output': True, 'il_summaries': [{'id': 1, 'ord_output': {'plt_sample': True}}]},
        {'num_reinsurance_iterations': 0},
    ),
    'pla_default': (
        {'number_of_samples': 2, 'gul_threshold': 0, 'gul_output': True, 'il_output': True, 'pla': True,
         'gul_summaries': [{'id': 1, 'ord_output': {'elt_sample': True}}], 'il_summaries': [{'id': 1, 'ord_output': {'plt_sample': True}}]},
        {'num_reinsurance_iterations': 0},
    ),
    'pla_secondary_factor': (
        {'number_of_samples': 2, 'gul_threshold': 0, 'il_output': True, 'ri_output': True, 'pla': True, 'pla_secondary_factor': 0.5,
         'il_summaries': [{'id': 1, 'ord_output': {'elt_sample': True}}], 'ri_summaries': [{'id': 1, 'ord_output': {'elt_sample': True}}]},
        {'num_reinsurance_iterations': 1},
    ),
    'pla_uniform_factor': (
        {'number_of_samples': 2, 'gul_threshold': 0, 'gul_output': True, 'pla': True, 'pla_uniform_factor': 1.5, 'pla_secondary_factor': 0.5,
         'gul_summaries': [{'id': 1, 'ord_output': {'elt_sample': True}}]},
        {'num_reinsurance_iterations': 0},
    ),
}

DROP = {'(', ')', '2>>', '$LOG_DIR/stderror.err', '&'}
KINDS = ('evepy', 'summarypy', 'tee', 'eltpy', 'pltpy', 'katpy', 'aalpy', 'lecpy')


def normalise(argv, fifo_dir='fifo/'):
    """Canonical tokens of a command: no stderr guard or pid bookkeeping, the fifo dir as 'fifo/', -L as a float."""
    out = []
    for token in argv:
        if token in DROP or re.fullmatch(r'\w+=\$!', token):
            continue
        if token.startswith(fifo_dir):
            token = 'fifo/' + token[len(fifo_dir):]
        if re.fullmatch(r'-L[-+.\deE]+', token):
            token = f'-L{float(token[2:])!r}'
        out.append(token)
    if out[:1] == ['tee'] and out[-2:] == ['>', '/dev/null']:
        out = out[:-2]
    return tuple(out)


def split_words(line):
    """Shell words of a script line, backslashes kept: genbash joins paths with os.sep, a backslash on Windows."""
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ''
    return list(lexer)


def script_commands(script, fifo_dir='fifo/'):
    """The pipeline, summary, ORD and output commands of a generated script, normalised, as a Counter."""
    found = Counter()
    with open(script) as f:
        for line in f:
            line = line.strip()
            words = line.lstrip('( ').split(maxsplit=1)
            if not words or words[0] not in KINDS:
                continue
            argv = normalise(split_words(line), fifo_dir)
            if argv and argv[0] in KINDS:
                found[argv] += 1
    return found


def plan_commands(plan, fifo_dir='fifo/'):
    """The same commands, rendered from a plan."""
    found = Counter()
    for k in range(1, plan.n_chunks + 1):
        for argv in render_chunk_commands(plan.spec, k, plan.n_chunks, plan.event_shuffle, fifo_dir):
            found[normalise(argv, fifo_dir)] += 1
    for job in plan.output_jobs:
        found[normalise(job.command(), fifo_dir)] += 1
    return found


def generate(tmp_path, settings, options, n_chunks, stderr_guard=True, step_policies=False):
    """Write run_kernel.sh with genbash and build bash_params with the same arguments, in a run directory."""
    run_dir = tmp_path / 'run'
    (run_dir / 'input').mkdir(parents=True)
    if step_policies:
        (run_dir / 'input' / 'fm_profile.csv').write_text('profile_id,calcrule_id,step_id\n1,12,1\n')
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        kwargs = dict(max_process_id=n_chunks, fifo_tmp_dir=False, stderr_guard=stderr_guard, **options)
        bash.genbash(analysis_settings=copy.deepcopy(settings), filename='run_kernel.sh', **kwargs)
        params = bash.bash_params(analysis_settings=copy.deepcopy(settings), filename='run_kernel.sh', **kwargs)
    finally:
        os.chdir(cwd)
    return run_dir, params


@pytest.mark.parametrize('stderr_guard', [True, False])
@pytest.mark.parametrize('n_chunks', [1, 3])
@pytest.mark.parametrize('case', sorted(CASES))
def test_plan_renders_exactly_the_genbash_commands(tmp_path, case, n_chunks, stderr_guard):
    settings, options = CASES[case]
    run_dir, params = generate(tmp_path, settings, options, n_chunks, stderr_guard, step_policies=case == 'il_ri_options')
    plan = plan_from_bash_params(params, run_dir=str(run_dir))
    expected = script_commands(run_dir / 'run_kernel.sh')
    got = plan_commands(plan)
    assert got == expected, (sorted(set(got) - set(expected)), sorted(set(expected) - set(got)))
    assert plan.n_chunks == n_chunks


@pytest.mark.parametrize('k', [1, 3])
@pytest.mark.parametrize('case', ['gul_il_ri_low_memory', 'gul_il_ri_rl_intermediate', 'il_ri_options'])
def test_plan_of_a_distributed_chunk_renders_the_chunk_script_commands(tmp_path, case, k):
    """generate-losses-chunk k of 3: the plan's chunk k is the analysis script bash writes for process_number k."""
    settings, options = CASES[case]
    run_dir = tmp_path / 'run'
    (run_dir / 'input').mkdir(parents=True)
    if case == 'il_ri_options':
        (run_dir / 'input' / 'fm_profile.csv').write_text('profile_id,calcrule_id,step_id\n1,12,1\n')
    script = run_dir / f'{k}.run_analysis.sh'
    params = bash.bash_params(analysis_settings=copy.deepcopy(settings), max_process_id=3, process_number=k, fifo_tmp_dir=False,
                              filename=str(script), **options)
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        with bash.bash_wrapper(str(script), False, True, log_sub_dir=k, process_number=k):
            bash.create_bash_analysis(**copy.deepcopy(params))
    finally:
        os.chdir(cwd)
    plan = plan_from_bash_params(params, run_dir=str(run_dir))
    assert plan.chunk_numbers == [k]
    got = Counter(normalise(argv) for argv in render_chunk_commands(plan.spec, k, plan.n_chunks, plan.event_shuffle))
    expected = script_commands(script)
    assert got == expected, (sorted(set(got) - set(expected)), sorted(set(expected) - set(got)))
    assert all(path.startswith(f'{k}.work/') for path in plan.chunk_paths(k))


def test_plan_from_bash_params_json_round_trip(tmp_path):
    settings, options = CASES['gul_il_ri_rl_intermediate']
    run_dir, params = generate(tmp_path, settings, options, 2)
    as_json = json.loads(json.dumps(params, default=lambda v: None if callable(v) else str(v)))
    plan = plan_from_bash_params(as_json, run_dir=str(run_dir))
    assert plan == plan_from_bash_params(params, run_dir=str(run_dir))
    restored = Plan.from_json(plan.to_json())
    assert restored == plan
    assert plan_commands(restored) == script_commands(run_dir / 'run_kernel.sh')
    assert ChainSpec.from_json(plan.spec.to_json()) == plan.spec


def test_plan_tree_shape(tmp_path):
    settings, options = CASES['gul_il_ri_rl_intermediate']
    run_dir, params = generate(tmp_path, settings, options, 2)
    spec = plan_from_bash_params(params, run_dir=str(run_dir)).spec
    assert spec.run_dir == str(run_dir)
    assert spec.gul == {'sample_size': 10, 'loss_threshold': 0.0, 'alloc_rule': 0, 'debug': 0, 'random_generator': 2, 'peril_filter': [],
                        'effective_damageability': False, 'max_cached_vuln_cdf_size_MB': 200,
                        'model_df_engine': 'oasis_data_manager.df_reader.reader.OasisPandasReader', 'dynamic_footprint': False}
    assert [(s['name'], s['static_path'], s['allocation_rule'], s['gross'], s['net'], s['feeds_next']) for s in spec.fm] == [
        ('il', 'input', 2, True, False, 'gross'), ('ri', ri_path(1), 3, True, True, 'net'), ('ri', ri_path(2), 3, True, True, None)]
    assert [(n['runtype'], n['inuring_priority'], n['source']['step'], n['source']['output'], n['stage']['run_type'],
             n['stage']['static_path'], n['stage']['summary_set_ids']) for n in spec.summaries] == [
        ('gul', '', None, None, 'gul', 'input', [1]),
        ('il', '', 0, 'gross', 'il', 'input', [1, 2]),
        ('ri', 'IP1_', 1, 'net', 'ri', ri_path(1), [1]),
        ('ri', '', 2, 'net', 'ri', ri_path(2), [1]),
        ('rl', 'IP1_', 1, 'gross', 'ri', ri_path(1), [1]),
        ('rl', 'IP2_', 2, 'gross', 'ri', ri_path(2), [1]),
    ]
    gul_set = spec.summaries[0]['sets'][0]
    assert gul_set['outputs'] == [
        {'kind': 'plt', 'ord_type': 'plt_ord', 'kwargs': {'splt': True, 'output_format': 'bin', 'noheader': False},
         'paths': {'splt': 'work/kat/gul_S1_plt_sample_P{k}'}},
        {'kind': 'elt', 'ord_type': 'elt_ord', 'kwargs': {'qelt': True, 'output_format': 'bin', 'noheader': False},
         'paths': {'qelt': 'work/kat/gul_S1_elt_quantile_P{k}'}},
        {'kind': 'elt', 'ord_type': 'selt_ord', 'kwargs': {'selt': True, 'output_format': 'bin', 'noheader': False},
         'paths': {'selt': 'work/kat/gul_S1_elt_sample_P{k}'}},
        {'kind': 'file', 'path': 'work/gul_S1_summary_palt/P{k}.bin'},
        {'kind': 'file', 'path': 'work/gul_S1_summaryleccalc/P{k}.bin'},
    ]


def test_plan_output_jobs_kwargs(tmp_path):
    settings, options = CASES['gul_all_ord']
    run_dir, params = generate(tmp_path, settings, options, 2)
    jobs = {(job.tool, job.kwargs.get('out') or job.kwargs.get('aal') or job.kwargs.get('ept')): job.kwargs
            for job in plan_from_bash_params(params, run_dir=str(run_dir)).output_jobs}
    assert jobs[('katpy', 'output/gul_S1_qplt.csv')] == {
        'out': 'output/gul_S1_qplt.csv', 'file_type': 'bin', 'qplt': True, 'unsorted': False,
        'files_in': ['work/kat/gul_S1_plt_quantile_P1', 'work/kat/gul_S1_plt_quantile_P2']}
    assert jobs[('aalpy', 'output/gul_S1_palt.csv')] == {
        'subfolder': 'gul_S1_summary_palt', 'aal': 'output/gul_S1_palt.csv', 'alct': 'output/gul_S1_alct.csv', 'confidence': 0.9,
        'ext': 'csv'}
    assert jobs[('aalpy', 'output/gul_S1_altmeanonly.csv')] == {
        'subfolder': 'gul_S1_summary_altmeanonly', 'aal': 'output/gul_S1_altmeanonly.csv', 'ext': 'csv'}
    assert jobs[('lecpy', 'output/gul_S1_ept.csv')] == {
        'subfolder': 'gul_S1_summaryleccalc', 'use_return_period': True, 'ext': 'csv', 'ept': 'output/gul_S1_ept.csv',
        'psept': 'output/gul_S1_psept.csv', 'agg_full_uncertainty': True, 'occ_full_uncertainty': True, 'agg_sample_mean': True,
        'occ_sample_mean': True, 'agg_wheatsheaf_mean': True, 'occ_wheatsheaf_mean': True, 'agg_wheatsheaf': True,
        'occ_wheatsheaf': True}
    assert len(jobs) == 6 + 2 + 1
    assert OutputJob('katpy', {}).entry_point == 'oasislmf.pytools.kat.manager:main'


def test_spec_for_taps_switches_on_the_outputs_taps_read(tmp_path):
    settings, options = CASES['rl_only']
    run_dir, params = generate(tmp_path, settings, options, 2)
    spec = plan_from_bash_params(params, run_dir=str(run_dir)).spec
    assert [(s['gross'], s['net']) for s in spec.fm] == [(True, False), (True, True), (True, False)]
    assert [(s['gross'], s['net']) for s in spec_for_taps(spec, ['gul', 'il', 'ri']).fm] == [(True, False), (True, True), (True, True)]
    assert spec_for_taps(spec, ['gul']).to_json() == spec.to_json()
    assert spec.fm[2]['net'] is False
    without_flags = ChainSpec(run_dir='.', fm=[{'name': 'il', 'allocation_rule': 2, 'feeds_next': None}])
    assert spec_for_taps(without_flags, ['il', 'ri']).fm == without_flags.fm


@pytest.mark.parametrize('executor', ['inline', 'pool'])
def test_planned_run_with_ri_tap_and_rl_output_only(tmp_path, executor):
    settings, options = CASES['rl_only']
    run_dir, params = generate(tmp_path, settings, options, 2)
    plan = plan_from_bash_params(params, run_dir=str(run_dir))
    chunks = plan.make_chunks(np.arange(1, 11, dtype=np.int32))
    factories = TREE_FACTORIES if executor == 'inline' else TREE_FACTORY_STRINGS
    module = inline if executor == 'inline' else pool
    module.run_chunks(plan.spec, chunks, tmp_path / 'out', ['ri'], 2, factories=factories)
    assert sorted(os.listdir(tmp_path / 'out')) == ['ri_P1.bin', 'ri_P2.bin']
    assert all(os.path.getsize(tmp_path / 'out' / name) > 0 for name in os.listdir(tmp_path / 'out'))


@pytest.mark.parametrize('n_events', [1, 2, 7, 100, 1001])
@pytest.mark.parametrize('n_chunks', [1, 2, 3, 12, 150])
def test_fisher_yates_chunks_equal_evepy_per_partition(n_events, n_chunks):
    events = np.arange(3, 3 + 2 * n_events, 2, dtype=event_id_dtype)
    chunks = Plan(spec=ChainSpec(run_dir='.'), output_jobs=[], event_shuffle=EVE_FISHER_YATES).make_chunks(events, n_chunks)
    assert len(chunks) == n_chunks
    for k, chunk in enumerate(chunks, 1):
        expected = list(partition_events__random(events.copy(), k, n_chunks))
        assert chunk.dtype == event_id_dtype
        assert chunk.tolist() == [int(e) for e in expected], k
    assert sorted(np.concatenate(chunks).tolist()) == events.tolist()


def test_plan_chunk_paths_and_work_dirs(tmp_path):
    settings, options = CASES['gul_il_two_sets_parquet']
    run_dir, params = generate(tmp_path, settings, options, 2)
    plan = plan_from_bash_params(params, run_dir=str(run_dir))
    assert sorted(plan.chunk_paths(2)) == sorted([
        'work/kat/gul_S1_elt_sample_P2', 'work/kat/gul_S2_plt_moment_P2', 'work/gul_S1_summaryleccalc/P2.bin',
        'work/gul_S2_summary_palt/P2.bin', 'work/kat/il_S1_elt_quantile_P2', 'work/kat/il_S1_elt_moment_P2',
        'work/il_S3_summary_altmeanonly/P2.bin', 'work/il_S3_summaryleccalc/P2.bin'])
    assert plan.work_dirs() == ['work/gul_S1_summaryleccalc', 'work/gul_S2_summary_palt', 'work/il_S3_summary_altmeanonly',
                                'work/il_S3_summaryleccalc', 'work/kat']


@pytest.mark.parametrize('shuffle', [EVE_NO_SHUFFLE, EVE_ROUND_ROBIN, EVE_FISHER_YATES, EVE_STD_SHUFFLE])
@pytest.mark.parametrize('n_chunks', [1, 4, 7])
def test_plan_chunks_match_evepy(tmp_path, shuffle, n_chunks):
    events = np.array([5, 3, 9, 1, 12, 7, 2, 30, 31, 8, 4], dtype=np.int32)
    events_file = tmp_path / 'events.bin'
    events.tofile(events_file)
    plan = Plan(spec=ChainSpec(run_dir=str(tmp_path)), n_chunks=n_chunks, event_shuffle=shuffle)
    chunks = plan.make_chunks(events)
    flags = {EVE_NO_SHUFFLE: {'no_shuffle': True}, EVE_FISHER_YATES: {'randomise': True},
             EVE_STD_SHUFFLE: {'randomise_builtin': True}, EVE_ROUND_ROBIN: {}}[shuffle]
    for k in range(1, n_chunks + 1):
        out = tmp_path / f'P{k}.bin'
        evepy_run(str(events_file), k, n_chunks, output_file=str(out), **flags)
        assert chunks[k - 1].tolist() == np.fromfile(out, dtype=np.int32).tolist(), k
    assert events.tolist() == [5, 3, 9, 1, 12, 7, 2, 30, 31, 8, 4]
    assert partition_events(events, 1, 1, shuffle).dtype == np.int32


BASE = {'number_of_samples': 1, 'gul_threshold': 0, 'gul_output': True, 'gul_summaries': [{'id': 1, 'ord_output': {'elt_sample': True}}]}


@pytest.mark.parametrize('settings, options, reason', [
    (BASE, {'_get_getmodel_cmd': lambda **kw: 'x'}, 'custom gulcalc'),
    (BASE, {'custom_gulcalc_cmd': 'supplier_model_gulcalc'}, 'custom gulcalc'),
    (BASE, {'gulmc': False}, 'gulpy'),
    (BASE, {'model_py_server': True}, 'data server'),
    (BASE, {'join_summary_info': True}, 'join-summary-info'),
    ({**BASE, 'join_summary_info': True}, {}, 'join-summary-info'),
    (BASE, {'custom_args': {'x': 1}}, 'custom_args'),
    ({**BASE, 'il_output': True, 'il_summaries': [{'id': 1}]}, {'num_gul_per_lb': 1, 'num_fm_per_lb': 1}, 'load balancer'),
    ({**BASE, 'gul_summaries': [{'id': 1, 'eltcalc': True}]}, {}, 'legacy'),
    ({**BASE, 'gul_summaries': [{'id': 1, 'lec_output': True, 'leccalc': {'full_uncertainty_aep': True}}]}, {}, 'legacy'),
])
def test_unsupported(tmp_path, monkeypatch, settings, options, reason):
    gulcalc = tmp_path / 'bin' / ('supplier_model_gulcalc.bat' if sys.platform == 'win32' else 'supplier_model_gulcalc')
    gulcalc.parent.mkdir()
    gulcalc.write_text('#!/bin/sh\n')
    gulcalc.chmod(0o755)
    monkeypatch.setenv('PATH', f"{gulcalc.parent}{os.pathsep}{os.environ['PATH']}")
    params = bash.bash_params(analysis_settings=copy.deepcopy(settings), max_process_id=2, fifo_tmp_dir=False,
                              model_run_dir=str(tmp_path), **options)
    with pytest.raises(Unsupported, match=reason) as info:
        plan_from_bash_params(params)
    assert reason in info.value.reason


def test_pla_spec_follows_the_plapy_command():
    params = {'number_of_samples': 1, 'gul_threshold': 0, 'gul_alloc_rule': 0, 'gul_output': True, 'gul_summaries': []}
    for settings, expected in [({}, None), ({'pla': False, 'pla_uniform_factor': 2}, None), ({'pla': True}, {}),
                               ({'pla': True, 'pla_secondary_factor': 0.3}, {'secondary_factor': 0.3}),
                               ({'pla': True, 'pla_uniform_factor': 2, 'pla_secondary_factor': 0.3}, {'uniform_factor': 2})]:
        spec = plan_from_bash_params({'analysis_settings': {**params, **settings}, **params}).spec
        assert spec.pla == expected, settings
        assert ChainSpec.from_json(spec.to_json()) == spec


def test_unsupported_full_correlation_and_shuffle():
    params = {'analysis_settings': {}, 'full_correlation': True}
    with pytest.raises(Unsupported, match='full correlation'):
        plan_from_bash_params(params)
    with pytest.raises(Unsupported, match='shuffle'):
        plan_from_bash_params({'analysis_settings': {}, 'event_shuffle': 9})


REFERENCE_PARAMS = sorted(glob.glob(os.path.expanduser('~/GitHub/runs/orch-bench-piwind-full/**/bash_params.json'), recursive=True))


@pytest.mark.skipif(not REFERENCE_PARAMS, reason='no saved full-kernel bash reference')
@pytest.mark.parametrize('params_file', REFERENCE_PARAMS)
def test_plan_matches_saved_reference_scripts(params_file):
    ref_dir = os.path.dirname(params_file)
    script = os.path.join(ref_dir, 'run_kernel.sh')
    if not os.path.exists(script):
        pytest.skip('no run_kernel.sh next to bash_params.json')
    with open(params_file) as f:
        params = json.load(f)
    plan = plan_from_bash_params(params, run_dir=ref_dir)
    fifo_dir = params['fifo_queue_dir']
    assert plan_commands(plan, fifo_dir) == script_commands(script, fifo_dir)


FULL_K = 2


def planned_run(ref_dir, run_dir, executor, n_workers=2, **options):
    """Run the plan of a reference's bash_params with an executor in a fresh run directory sharing its input and static."""
    run_dir.mkdir()
    for name in ('input', 'static'):
        (run_dir / name).symlink_to(os.path.realpath(ref_dir / name))
    with open(ref_dir / 'bash_params.json') as f:
        params = json.load(f)
    plan = plan_from_bash_params(params, run_dir=str(run_dir))
    chunks = plan.make_chunks(read_events(str(run_dir / 'input' / 'events.bin')))
    {'inline': inline, 'pool': pool}[executor].run_chunks(plan.spec, chunks, run_dir / 'taps', [], n_workers, **options)
    return plan


@pytest.mark.parametrize('executor, options', [('inline', {'chunk_order': [2, 1]}), ('pool', {})])
def test_planned_run_work_files_byte_identical_to_bash(full_kernel_reference, tmp_path, executor, options):
    ref_dir = full_kernel_reference(FULL_K)
    plan = planned_run(ref_dir, tmp_path / 'run', executor, **options)
    assert script_commands(ref_dir / 'run_kernel.sh') == plan_commands(plan)
    report = compare_trees_bytes(ref_dir, tmp_path / 'run', subdirs=('work',))
    assert report.ok, report.summary()
    assert report.compared == sum(len(plan.chunk_paths(k)) for k in range(1, FULL_K + 1))


def test_planned_run_low_memory_idx_files_byte_identical_to_bash(full_kernel_reference, tmp_path):
    ref_dir = full_kernel_reference(FULL_K, summarypy_low_memory=True)
    plan = planned_run(ref_dir, tmp_path / 'run', 'inline')
    assert any(s.get('idx_paths') for node in plan.spec.summaries for s in node['sets'])
    report = compare_trees_bytes(ref_dir, tmp_path / 'run', subdirs=('work',))
    assert report.ok, report.summary()
    assert report.compared == sum(len(plan.chunk_paths(k)) for k in range(1, FULL_K + 1))
