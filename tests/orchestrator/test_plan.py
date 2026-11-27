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
from oasislmf.execution.orchestrator.plan import (OutputJob, OutputRename, Plan, Unsupported, plan_from_bash_params,
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
    'gul_il_ri_intermediate': (
        {'number_of_samples': 10, 'gul_threshold': 0, 'gul_output': True, 'il_output': True, 'ri_output': True,
         'ri_inuring_priorities': [1],
         'gul_summaries': [{'id': 1, 'ord_output': {'plt_sample': True, 'elt_quantile': True, 'elt_sample': True, 'alt_period': True,
                                                    'ept_full_uncertainty_aep': True, 'psept_oep': True}}],
         'il_summaries': [{'id': 1, 'ord_output': {'elt_moment': True}}, {'id': 2, 'ord_output': {'alt_meanonly': True}}],
         'ri_summaries': [{'id': 1, 'ord_output': {'elt_moment': True, 'plt_quantile': True}}]},
        {'num_reinsurance_iterations': 2},
    ),
    'il_ri_options': (
        {'number_of_samples': 5, 'gul_threshold': 1e-3, 'il_output': True, 'ri_output': True,
         'il_summaries': [{'id': 1, 'ord_output': {'elt_sample': True, 'ept_full_uncertainty_oep': True}}],
         'ri_summaries': [{'id': 1, 'ord_output': ALL_ORD}, {'id': 2, 'ord_output': {'plt_sample': True}}]},
        {'num_reinsurance_iterations': 3, 'fmpy_low_memory': True, 'fmpy_sort_output': True, 'event_shuffle': EVE_NO_SHUFFLE,
         'peril_filter': ['WTC', 'WSS'], 'gulmc_effective_damageability': True, 'gulmc_vuln_cache_size': 50,
         'dynamic_footprint': True, 'section_merge': 'sum', 'gul_random_generator': 1, 'ri_alloc_rule': 2},
    ),
    'ri_only_one_layer': (
        {'number_of_samples': 2, 'gul_threshold': 0, 'ri_output': True, 'ri_summaries': [{'id': 1, 'ord_output': {'elt_sample': True}}]},
        {'num_reinsurance_iterations': 1, 'event_shuffle': EVE_FISHER_YATES},
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
    'join_summary_info_option': (
        {'number_of_samples': 4, 'gul_threshold': 0, 'gul_output': True, 'il_output': True, 'ri_output': True,
         'gul_summaries': [{'id': 1, 'ord_output': ALL_ORD}],
         'il_summaries': [{'id': 2, 'ord_output': {'elt_moment': True, 'alt_meanonly': True, 'psept_aep': True, 'parquet_format': True}}],
         'ri_summaries': [{'id': 1, 'ord_output': {'plt_sample': True, 'ept_mean_sample_oep': True}}]},
        {'num_reinsurance_iterations': 2, 'join_summary_info': True},
    ),
    'join_summary_info_setting': (
        {'number_of_samples': 4, 'gul_threshold': 0, 'gul_output': True, 'join_summary_info': True,
         'gul_summaries': [{'id': 1, 'ord_output': {'elt_sample': True, 'alt_period': True, 'alct_convergence': True}}]},
        {'num_reinsurance_iterations': 0},
    ),
    'legacy_outputs': (
        {'number_of_samples': 2, 'gul_threshold': 0, 'gul_output': True, 'il_output': True,
         'gul_summaries': [{'id': 1, 'eltcalc': True, 'summarycalc': True, 'ord_output': {'elt_sample': True}},
                           {'id': 2, 'lec_output': True, 'leccalc': {'full_uncertainty_aep': True}, 'aalcalc': True}],
         'il_summaries': [{'id': 1, 'pltcalc': True, 'aalcalcmeanonly': True}]},
        {'num_reinsurance_iterations': 0},
    ),
    'gulpy_gul_only': (
        {'number_of_samples': 10, 'gul_threshold': 0, 'gul_output': True, 'gul_summaries': [{'id': 1, 'ord_output': ALL_ORD}]},
        {'num_reinsurance_iterations': 0, 'gulmc': False},
    ),
    'gulpy_gul_il_ri': (
        {'number_of_samples': 4, 'gul_threshold': 1e-3, 'gul_output': True, 'il_output': True, 'ri_output': True,
         'gul_summaries': [{'id': 1, 'ord_output': {'elt_sample': True, 'alt_period': True}}],
         'il_summaries': [{'id': 1, 'ord_output': {'plt_sample': True}}],
         'ri_summaries': [{'id': 1, 'ord_output': {'ept_mean_sample_oep': True}}]},
        {'num_reinsurance_iterations': 2, 'gulmc': False, 'peril_filter': ['WTC', 'WSS'], 'gul_random_generator': 1,
         'gul_alloc_rule': 2, 'gulmc_effective_damageability': True, 'gulmc_vuln_cache_size': 50, 'dynamic_footprint': True,
         'model_df_engine': 'some.module.Reader', 'event_shuffle': EVE_STD_SHUFFLE},
    ),
    'gulpy_pla': (
        {'number_of_samples': 2, 'gul_threshold': 0, 'gul_output': True, 'il_output': True, 'pla': True, 'pla_secondary_factor': 0.5,
         'gul_summaries': [{'id': 1, 'ord_output': {'elt_sample': True}}], 'il_summaries': [{'id': 1, 'ord_output': {'plt_sample': True}}]},
        {'num_reinsurance_iterations': 0, 'gulmc': False},
    ),
}

DROP = {'(', ')', '2>>', '$LOG_DIR/stderror.err', '&'}
KINDS = ('evepy', 'summarypy', 'tee', 'eltpy', 'pltpy', 'katpy', 'aalpy', 'lecpy', 'join-summary-info')


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
        for argv in job.join_commands():
            found[normalise(argv, fifo_dir)] += 1
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
@pytest.mark.parametrize('case', ['gul_il_ri_low_memory', 'gul_il_ri_intermediate', 'il_ri_options'])
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
    settings, options = CASES['gul_il_ri_intermediate']
    run_dir, params = generate(tmp_path, settings, options, 2)
    as_json = json.loads(json.dumps(params, default=lambda v: None if callable(v) else str(v)))
    plan = plan_from_bash_params(as_json, run_dir=str(run_dir))
    assert plan == plan_from_bash_params(params, run_dir=str(run_dir))
    restored = Plan.from_json(plan.to_json())
    assert restored == plan
    assert plan_commands(restored) == script_commands(run_dir / 'run_kernel.sh')
    assert ChainSpec.from_json(plan.spec.to_json()) == plan.spec


def test_plan_tree_shape(tmp_path):
    settings, options = CASES['gul_il_ri_intermediate']
    run_dir, params = generate(tmp_path, settings, options, 2)
    spec = plan_from_bash_params(params, run_dir=str(run_dir)).spec
    assert spec.run_dir == str(run_dir)
    assert spec.gul == {'sample_size': 10, 'loss_threshold': 0.0, 'alloc_rule': 0, 'debug': 0, 'random_generator': 2, 'peril_filter': [],
                        'effective_damageability': False, 'max_cached_vuln_cdf_size_MB': 200,
                        'model_df_engine': 'oasis_data_manager.df_reader.reader.OasisPandasReader', 'dynamic_footprint': False,
                        'section_merge': 'max'}
    assert [(s['name'], s['static_path'], s['allocation_rule'], s['gross'], s['net'], s['feeds_next']) for s in spec.fm] == [
        ('il', 'input', 2, True, False, 'gross'), ('ri', ri_path(1), 3, False, True, 'net'), ('ri', ri_path(2), 3, False, True, None)]
    assert [(n['runtype'], n['inuring_priority'], n['source']['step'], n['source']['output'], n['stage']['run_type'],
             n['stage']['static_path'], n['stage']['summary_set_ids']) for n in spec.summaries] == [
        ('gul', '', None, None, 'gul', 'input', [1]),
        ('il', '', 0, 'gross', 'il', 'input', [1, 2]),
        ('ri', 'IP1_', 1, 'net', 'ri', ri_path(1), [1]),
        ('ri', 'IP2_', 2, 'net', 'ri', ri_path(2), [1]),
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
        'subfolder': 'gul_S1_summary_altmeanonly', 'meanonly': True, 'aal': 'output/gul_S1_altmeanonly.csv', 'ext': 'csv'}
    assert jobs[('lecpy', 'output/gul_S1_ept.csv')] == {
        'subfolder': 'gul_S1_summaryleccalc', 'use_return_period': True, 'ext': 'csv', 'ept': 'output/gul_S1_ept.csv',
        'psept': 'output/gul_S1_psept.csv', 'agg_full_uncertainty': True, 'occ_full_uncertainty': True, 'agg_sample_mean': True,
        'occ_sample_mean': True, 'agg_wheatsheaf_mean': True, 'occ_wheatsheaf_mean': True, 'agg_wheatsheaf': True,
        'occ_wheatsheaf': True}
    assert len(jobs) == 6 + 2 + 1
    assert OutputJob('katpy', {}).entry_point == 'oasislmf.pytools.kat.manager:main'


def test_spec_for_taps_switches_on_the_outputs_taps_read():
    spec = ChainSpec(run_dir='.', fm=[
        {'name': 'il', 'allocation_rule': 2, 'gross': False, 'net': False, 'feeds_next': 'gross'},
        {'name': 'ri', 'allocation_rule': 3, 'gross': False, 'net': True, 'feeds_next': 'net'},
        {'name': 'ri', 'allocation_rule': 3, 'gross': True, 'net': False, 'feeds_next': None}])
    assert [(s['gross'], s['net']) for s in spec_for_taps(spec, ['gul', 'il', 'ri']).fm] == [(True, False), (False, True), (True, True)]
    assert spec_for_taps(spec, ['gul']).to_json() == spec.to_json()
    assert spec.fm[2]['net'] is False
    without_flags = ChainSpec(run_dir='.', fm=[{'name': 'il', 'allocation_rule': 2, 'feeds_next': None}])
    assert spec_for_taps(without_flags, ['il', 'ri']).fm == without_flags.fm


@pytest.mark.parametrize('executor', ['inline', 'pool'])
def test_planned_run_with_ri_tap(tmp_path, executor):
    settings, options = CASES['ri_only_one_layer']
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
    ({**BASE, 'full_correlation': True}, {'gulmc': False}, 'full correlation'),
    (BASE, {'custom_args': {'x': 1}}, 'custom_args'),
    ({**BASE, 'rl_output': True, 'rl_summaries': [{'id': 1, 'ord_output': {'elt_sample': True}}]}, {'num_reinsurance_iterations': 1},
     'rlcombine'),
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


@pytest.mark.parametrize('ri_inuring_priorities, copy', [([], False), ([1], True), ([2], False)])
def test_ri_final_output_renames_or_copies_the_final_priority(tmp_path, ri_inuring_priorities, copy):
    settings = {**CASES['ri_only_one_layer'][0], 'ri_inuring_priorities': ri_inuring_priorities}
    run_dir, params = generate(tmp_path, settings, CASES['ri_only_one_layer'][1], 2)
    plan = plan_from_bash_params(params, run_dir=str(run_dir))
    assert plan.output_renames == [OutputRename(src='output/ri_IP1_', dst='output/ri_', copy=copy)]
    assert Plan.from_json(plan.to_json()) == plan
    output = tmp_path / 'run' / 'output'
    output.mkdir(parents=True)
    (output / 'ri_IP1_S1_selt.csv').write_text('a')
    (output / 'gul_S1_selt.csv').write_text('b')
    OutputRename(src='output/ri_IP1_', dst='output/ri_', copy=copy).apply(str(tmp_path / 'run'))
    expected = {'ri_S1_selt.csv', 'gul_S1_selt.csv'} | ({'ri_IP1_S1_selt.csv'} if copy else set())
    assert set(os.listdir(output)) == expected
    assert (output / 'ri_S1_selt.csv').read_text() == 'a'


def test_plans_without_ri_output_rename_nothing(tmp_path):
    settings, options = CASES['il_only']
    run_dir, params = generate(tmp_path, settings, options, 2)
    assert plan_from_bash_params(params, run_dir=str(run_dir)).output_renames == []


LB_SETTINGS = {**BASE, 'il_output': True, 'il_summaries': [{'id': 1, 'ord_output': {'elt_sample': True}}]}


@pytest.mark.parametrize('settings, options, message', [
    (BASE, {'model_py_server': True}, 'model_py_server'),
    (LB_SETTINGS, {'num_gul_per_lb': 2, 'num_fm_per_lb': 1}, 'load balancer'),
    ({**BASE, 'gul_summaries': [{'id': 1, 'eltcalc': True, 'ord_output': {'elt_sample': True}}]}, {}, 'Legacy'),
])
def test_ignored_settings_plan_as_without_them(tmp_path, caplog, settings, options, message):
    """Settings that change how bash spreads the work, not the losses, give the plan of the run without them."""
    without = {key: value for key, value in settings.items()}
    if message == 'Legacy':
        without['gul_summaries'] = [{'id': 1, 'ord_output': {'elt_sample': True}}]
    kwargs = dict(max_process_id=4, fifo_tmp_dir=False, model_run_dir=str(tmp_path))
    with caplog.at_level('INFO', logger='oasislmf.execution.orchestrator.plan'):
        plan = plan_from_bash_params(bash.bash_params(analysis_settings=copy.deepcopy(settings), **kwargs, **options))
    assert message in caplog.text
    assert plan == plan_from_bash_params(bash.bash_params(analysis_settings=copy.deepcopy(without), **kwargs))
    assert not plan.spec.gul.get('data_server')
    assert plan.n_chunks == 4


def test_gulpy_plan_as_without_model_py_server(tmp_path, caplog):
    """modelpy's data server is ignored as gulmc's is: the plan is that of the run without it, and never asks for it."""
    kwargs = dict(max_process_id=2, fifo_tmp_dir=False, model_run_dir=str(tmp_path), gulmc=False)
    with caplog.at_level('INFO', logger='oasislmf.execution.orchestrator.plan'):
        plan = plan_from_bash_params(bash.bash_params(analysis_settings=copy.deepcopy(BASE), model_py_server=True, **kwargs))
    assert 'model_py_server' in caplog.text
    assert plan == plan_from_bash_params(bash.bash_params(analysis_settings=copy.deepcopy(BASE), **kwargs))
    assert plan.spec.gul == {'modelpy': {}, 'gulpy': {'random_generator': 2, 'sample_size': 1, 'loss_threshold': 0.0, 'alloc_rule': 0}}


def test_gulpy_plan_tree_shape(tmp_path):
    """With gulmc off the gul stage is modelpy | gulpy, given only the options bash passes each command."""
    settings, options = CASES['gulpy_gul_il_ri']
    run_dir, params = generate(tmp_path, settings, options, 2)
    plan = plan_from_bash_params(params, run_dir=str(run_dir))
    spec = plan.spec
    assert spec.gul_engine == 'gulpy' and spec.gul_factory == 'gulpy'
    assert spec.gul == {'modelpy': {'peril_filter': ['WTC', 'WSS']},
                        'gulpy': {'random_generator': 1, 'sample_size': 4, 'loss_threshold': 1e-3, 'alloc_rule': 2}}
    assert [s['name'] for s in spec.fm] == ['il', 'ri', 'ri']
    assert ChainSpec.from_json(spec.to_json()) == spec
    assert Plan.from_json(plan.to_json()) == plan
    gulmc_spec = plan_from_bash_params(generate(tmp_path / 'gulmc', settings, {**options, 'gulmc': True}, 2)[1],
                                       run_dir=str(run_dir)).spec
    assert gulmc_spec.gul_engine == 'gulmc' and gulmc_spec.gul_factory == 'gul'
    assert gulmc_spec.fm == spec.fm and gulmc_spec.summaries == spec.summaries


def test_pla_spec_follows_the_plapy_command():
    params = {'number_of_samples': 1, 'gul_threshold': 0, 'gul_alloc_rule': 0, 'gul_output': True, 'gul_summaries': []}
    for settings, expected in [({}, None), ({'pla': False, 'pla_uniform_factor': 2}, None), ({'pla': True}, {}),
                               ({'pla': True, 'pla_secondary_factor': 0.3}, {'secondary_factor': 0.3}),
                               ({'pla': True, 'pla_uniform_factor': 2, 'pla_secondary_factor': 0.3}, {'uniform_factor': 2})]:
        spec = plan_from_bash_params({'analysis_settings': {**params, **settings}, **params}).spec
        assert spec.pla == expected, settings
        assert ChainSpec.from_json(spec.to_json()) == spec


def test_join_summary_info_jobs_round_trip(tmp_path):
    settings, options = CASES['join_summary_info_option']
    run_dir, params = generate(tmp_path, settings, options, 2)
    plan = plan_from_bash_params(params, run_dir=str(run_dir))
    assert plan.output_jobs and all(job.summary_info for job in plan.output_jobs)
    assert {job.summary_info for job in plan.output_jobs} == {
        'output/gul_S1_summary-info.csv', 'output/il_S2_summary-info.parquet', 'output/ri_S1_summary-info.csv'}
    assert Plan.from_json(plan.to_json()) == plan


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
