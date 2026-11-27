"""Tests of the output stage: katpy/aalpy/lecpy jobs run in-process by the inline and pool executors."""
import copy
import json
import logging
import os
import shutil
import time

import numpy as np
import pytest

from oasislmf.execution import bash
from oasislmf.execution.orchestrator import cli, inline, pool
from oasislmf.execution.orchestrator.chunks import make_chunks, read_events
from oasislmf.execution.orchestrator.outputs import (JobError, check_independent, isolated_logging, job_inputs, job_label,
                                                     job_outputs, jobs_for_chunks, run_job, run_jobs)
from oasislmf.execution.orchestrator.plan import OutputJob, plan_from_bash_params
from oasislmf.pytools.elt.data import SELT_headers
from oasislmf.pytools.utils import redirect_logging
from tests.orchestrator.tools.compare_outputs import compare_trees_bytes, list_files

from .test_chain import FakeFm, FakeGul, make_spec
from .test_pool import FAKE_FACTORIES, TAPS, descendants, run_pool

FULL_K = 2
SETTINGS = {'number_of_samples': 2, 'gul_threshold': 0, 'gul_output': True, 'il_output': True,
            'gul_summaries': [{'id': 1, 'ord_output': {'elt_sample': True, 'plt_moment': True, 'alt_period': True,
                                                       'alct_convergence': True, 'alt_meanonly': True,
                                                       'ept_full_uncertainty_aep': True, 'psept_oep': True}}],
            'il_summaries': [{'id': 1, 'ord_output': {'elt_quantile': True, 'parquet_format': True, 'alt_period': True}},
                             {'id': 2, 'ord_output': {'ept_mean_sample_oep': True}}]}


def plan_jobs(tmp_path, n_chunks=3):
    params = bash.bash_params(analysis_settings=copy.deepcopy(SETTINGS), max_process_id=n_chunks, fifo_tmp_dir=False)
    return plan_from_bash_params(params, run_dir=str(tmp_path)).output_jobs


def write_selt_csv(path, *event_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ''.join(f'{e},1,1,{e}.5,0\n' for e in event_ids)
    path.write_text(','.join(SELT_headers) + '\n' + rows)


def selt_job(out, *files_in):
    return OutputJob('katpy', {'out': out, 'file_type': 'csv', 'files_in': list(files_in), 'selt': True, 'unsorted': True})


def failing_job(out='output/bad.csv'):
    return OutputJob('aalpy', {'subfolder': 'no_such_summary_palt', 'aal': out, 'ext': 'csv'})


@pytest.fixture
def run_dir(tmp_path):
    """A run directory with two SELT csv inputs for katpy jobs."""
    path = tmp_path / 'run'
    (path / 'input').mkdir(parents=True)
    write_selt_csv(path / 'work' / 'a.csv', 3, 1)
    write_selt_csv(path / 'work' / 'b.csv', 2)
    return path


def expected_selt(*event_ids):
    return ','.join(SELT_headers) + '\n' + ''.join(f'{e},1,1,{e}.5,0\n' for e in event_ids)


def test_job_label_outputs_and_inputs(tmp_path):
    jobs = {job_label(job): job for job in plan_jobs(tmp_path)}
    aal = jobs['aalpy output/gul_S1_palt.csv, output/gul_S1_alct.csv']
    assert job_outputs(aal) == ['output/gul_S1_palt.csv', 'output/gul_S1_alct.csv']
    assert job_inputs(aal) == [os.path.join('work', 'gul_S1_summary_palt')]
    assert job_outputs(jobs['lecpy output/gul_S1_ept.csv, output/gul_S1_psept.csv'].to_dict()) == \
        ['output/gul_S1_ept.csv', 'output/gul_S1_psept.csv']
    kat = jobs['katpy output/il_S1_qelt.parquet']
    assert job_inputs(kat) == [f'work/kat/il_S1_elt_quantile_P{k}' for k in (1, 2, 3)]
    assert 'aalpy output/gul_S1_altmeanonly.csv' in jobs and 'lecpy output/il_S2_ept.csv' in jobs


def test_jobs_for_chunks_rewrites_katpy_inputs_only(tmp_path):
    jobs = plan_jobs(tmp_path, n_chunks=3)
    five = jobs_for_chunks([job.to_dict() for job in jobs], 5)
    assert [job.tool for job in five] == [job.tool for job in jobs]
    for before, after in zip(jobs, five):
        if before.tool == 'katpy':
            stem = before.kwargs['files_in'][0][:-1]
            assert after.kwargs['files_in'] == [f'{stem}{k}' for k in range(1, 6)]
            assert {**after.kwargs, 'files_in': None} == {**before.kwargs, 'files_in': None}
        else:
            assert after == before
    assert jobs_for_chunks(jobs, 3) == jobs
    with pytest.raises(ValueError, match='per-chunk'):
        jobs_for_chunks([selt_job('o.csv', 'work/a.csv')], 2)


def test_jobs_for_chunks_keeps_the_summary_info_to_join():
    job = OutputJob('katpy', {'out': 'output/gul_S1_selt.csv', 'files_in': ['work/kat/gul_S1_selt_ord_P1'], 'selt': True},
                    'output/gul_S1_summary-info.csv')
    [two] = jobs_for_chunks([job.to_dict()], 2)
    assert two.summary_info == job.summary_info
    assert two.join_commands() == [['join-summary-info', '-s', 'output/gul_S1_summary-info.csv',
                                    '-d', 'output/gul_S1_selt.csv', '-o', 'output/gul_S1_selt.csv']]
    assert OutputJob('katpy', {'out': 'x.csv'}).join_commands() == []


def test_plan_jobs_are_independent_and_conflicts_are_detected(tmp_path):
    jobs = plan_jobs(tmp_path)
    check_independent(jobs)
    assert len({out for job in jobs for out in job_outputs(job)}) == sum(len(job_outputs(job)) for job in jobs)
    with pytest.raises(ValueError, match='write the same file'):
        check_independent([selt_job('output/x.csv', 'work/a.csv'), selt_job('output/x.csv', 'work/b.csv')])
    with pytest.raises(ValueError, match="read each other's output"):
        check_independent([selt_job('output/x.csv', 'work/a.csv'), selt_job('output/y.csv', 'output/x.csv')])
    with pytest.raises(ValueError, match='share a work sub-directory'):
        check_independent([OutputJob('aalpy', {'subfolder': 's', 'aal': 'output/a.csv'}),
                           OutputJob('lecpy', {'subfolder': 's', 'ept': 'output/e.csv'})])


@redirect_logging(exec_name='fakepy')
def logging_tool(fail=False, **kwargs):
    logging.getLogger('oasislmf.pytools.fake').info('inside')
    if fail:
        raise RuntimeError('tool failed')


def test_isolated_logging_restores_what_redirect_logging_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    child = logging.getLogger('oasislmf.pytools.fake')
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    child.addHandler(handler)
    child.setLevel(logging.DEBUG)
    oasis = logging.getLogger('oasislmf')
    try:
        before = [(lg.level, lg.propagate, list(lg.handlers)) for lg in (child, oasis, logging.getLogger())]
        for fail in (False, True):
            with isolated_logging():
                try:
                    logging_tool(fail=fail)
                except RuntimeError:
                    pass
            assert [(lg.level, lg.propagate, list(lg.handlers)) for lg in (child, oasis, logging.getLogger())] == before
            assert handler.level == logging.DEBUG
    finally:
        child.removeHandler(handler)
        child.setLevel(logging.NOTSET)


def test_run_job_in_process_restores_cwd_and_names_the_failing_tool(run_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (run_dir / 'output').mkdir()
    result = run_job(selt_job('output/s.csv', 'work/a.csv', 'work/b.csv'), str(run_dir), index=4)
    assert result['job'] == 4 and result['tool'] == 'katpy' and result['outputs'] == ['output/s.csv'] and result['seconds'] >= 0
    assert os.getcwd() == str(tmp_path)
    assert (run_dir / 'output' / 's.csv').read_text() == expected_selt(3, 1, 2)
    with pytest.raises(JobError) as info:
        run_job(failing_job().to_dict(), str(run_dir), index=2)
    assert os.getcwd() == str(tmp_path)
    assert info.value.index == 2 and info.value.label == 'aalpy output/bad.csv'
    assert 'output job aalpy output/bad.csv failed: RuntimeError' in str(info.value)
    assert 'no_such_summary_palt' in str(info.value.__cause__)


def test_run_jobs_stops_at_the_first_failure(run_dir):
    jobs = [selt_job('out/1.csv', 'work/a.csv'), failing_job(), selt_job('out/3.csv', 'work/b.csv')]
    with pytest.raises(JobError) as info:
        run_jobs(jobs, str(run_dir))
    assert info.value.index == 1
    assert (run_dir / 'out' / '1.csv').exists() and not (run_dir / 'out' / '3.csv').exists()


def check_job_stats(stats, jobs, n_workers):
    assert [j['job'] for j in stats['jobs']] == list(range(len(jobs)))
    assert all(set(j) == {'job', 'tool', 'outputs', 'seconds', 'worker'} for j in stats['jobs'])
    assert [(j['tool'], j['outputs']) for j in stats['jobs']] == [(job.tool, job_outputs(job)) for job in jobs]
    assert all(0 <= j['worker'] < n_workers and j['seconds'] >= 0 for j in stats['jobs'])
    assert 0 <= stats['output_seconds'] <= stats['wall_seconds']
    assert sum(w['jobs'] for w in stats['workers']) == len(jobs)
    for w in stats['workers']:
        mine = [j for j in stats['jobs'] if j['worker'] == w['worker']]
        assert w['jobs'] == len(mine)
        assert w['job_seconds'] == pytest.approx(sum(j['seconds'] for j in mine))
    json.dumps(stats)


SELT_JOBS = [selt_job(f'output/s{i}.csv', 'work/a.csv', 'work/b.csv') for i in range(5)]


@pytest.mark.parametrize('executor, n_workers', [('inline', 1), ('pool', 1), ('pool', 3)])
def test_jobs_run_after_the_chunks_with_stats(tmp_path, run_dir, executor, n_workers):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 4)
    spec = make_spec(str(run_dir))
    if executor == 'inline':
        stats = inline.run_chunks(spec, chunks, tmp_path / 'out', TAPS, factories={'gul': FakeGul, 'fm': FakeFm},
                                  output_jobs=SELT_JOBS)
    else:
        stats = run_pool(spec, chunks, tmp_path / 'out', TAPS, n_workers, factories=FAKE_FACTORIES,
                         output_jobs=[job.to_dict() for job in SELT_JOBS])
        assert descendants() == []
    for i in range(5):
        assert (run_dir / 'output' / f's{i}.csv').read_text() == expected_selt(3, 1, 2)
    assert len(stats['chunks']) == 4
    check_job_stats(stats, SELT_JOBS, n_workers)


def test_pool_without_jobs_reports_empty_job_stats(tmp_path, run_dir):
    stats = run_pool(make_spec(str(run_dir)), [[1], [2]], tmp_path / 'out', TAPS, 2, factories=FAKE_FACTORIES)
    assert stats['jobs'] == [] and stats['output_seconds'] == 0.0
    assert all(w['jobs'] == 0 for w in stats['workers'])


class CountingGul(FakeGul):
    """FakeGul that appends its process id to ``gul_built.txt`` in the working directory (the run directory)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open('gul_built.txt', 'a') as f:
            f.write(f'{os.getpid()}\n')


@pytest.mark.parametrize('chunks, n_workers, expected_workers', [([[1]], 3, 3), ([[1]], 8, 5), ([[1], [2]], 1, 1), ([], 2, 2)])
def test_pool_job_only_workers_for_more_jobs_than_chunks(tmp_path, run_dir, chunks, n_workers, expected_workers):
    factories = {**FAKE_FACTORIES, 'gul': f'{__name__}:CountingGul'}
    stats = run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, n_workers, factories=factories,
                     output_jobs=[job.to_dict() for job in SELT_JOBS])
    n_chain = min(n_workers, len(chunks))
    assert stats['n_workers'] == expected_workers
    assert [w['jobs_only'] for w in stats['workers']] == [False] * n_chain + [True] * (expected_workers - n_chain)
    assert all(w['chunks'] == 0 for w in stats['workers'] if w['jobs_only'])
    built = (run_dir / 'gul_built.txt').read_text().split() if chunks else []
    assert len(built) == n_chain
    for i in range(5):
        assert (run_dir / 'output' / f's{i}.csv').read_text() == expected_selt(3, 1, 2)
    check_job_stats(stats, SELT_JOBS, expected_workers)
    assert descendants() == []


@pytest.mark.parametrize('n_workers', [1, 2])
def test_pool_failing_job_is_reported_and_stops_the_run(tmp_path, run_dir, n_workers):
    jobs = [selt_job('output/ok.csv', 'work/a.csv'), failing_job(), *SELT_JOBS[1:]]
    with pytest.raises(pool.WorkerError) as info:
        run_pool(make_spec(str(run_dir)), make_chunks(np.arange(1, 31, dtype=np.int32), 3), tmp_path / 'out', TAPS, n_workers,
                 factories=FAKE_FACTORIES, output_jobs=jobs)
    error = info.value
    assert (error.job, error.job_label, error.stage, error.chunk, error.event) == (1, 'aalpy output/bad.csv', 'aalpy', None, None)
    assert error.returncode not in (None, 0)
    text = str(error)
    assert 'stage aalpy, output job aalpy output/bad.csv' in text
    assert 'RuntimeError' in text and 'no_such_summary_palt' in text
    assert descendants() == []


def test_inline_failing_job_raises_after_the_chunks(tmp_path, run_dir):
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 2)
    with pytest.raises(JobError) as info:
        inline.run_chunks(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, factories={'gul': FakeGul, 'fm': FakeFm},
                          output_jobs=[failing_job(), SELT_JOBS[0]])
    assert info.value.index == 0
    assert sorted(os.listdir(tmp_path / 'out')) == sorted(f'{tap}_P{k}.bin' for tap in TAPS for k in (1, 2))
    assert not (run_dir / 'output' / 's0.csv').exists()


@pytest.mark.parametrize('executor', ['inline', 'pool'])
def test_executors_reject_dependent_jobs(tmp_path, run_dir, executor):
    jobs = [selt_job('output/x.csv', 'work/a.csv'), selt_job('output/y.csv', 'output/x.csv')]
    run_chunks = inline.run_chunks if executor == 'inline' else pool.run_chunks
    factories = {'gul': FakeGul, 'fm': FakeFm} if executor == 'inline' else FAKE_FACTORIES
    with pytest.raises(ValueError, match="read each other's output"):
        run_chunks(make_spec(str(run_dir)), [[1]], tmp_path / 'out', TAPS, 1, factories=factories, output_jobs=jobs)
    assert not (tmp_path / 'out' / 'gul_P1.bin').exists()


def orchestrated_run(ref_dir, run_dir, executor, n_workers=2, use_cli=False):
    """Run the plan of a reference's bash_params, chunks and output jobs, in a fresh run directory like the reference's.

    The run directory shares the reference's ``input`` and ``static`` and has its ``output/*summary-info*`` files, as
    the run directory bash ran in had.
    """
    run_dir.mkdir()
    for name in ('input', 'static'):
        (run_dir / name).symlink_to(os.path.realpath(ref_dir / name))
    (run_dir / 'output').mkdir()
    for path in (ref_dir / 'output').glob('*summary-info*'):
        shutil.copy2(path, run_dir / 'output' / path.name)
    if use_cli:
        stats_file = run_dir.parent / f'{run_dir.name}_stats.json'
        cli.main(['--plan', str(ref_dir / 'bash_params.json'), '--run-dir', str(run_dir), '--executor', executor,
                  '--workers', str(n_workers), '--stats-json', str(stats_file)])
        return plan_from_bash_params(json.loads((ref_dir / 'bash_params.json').read_text()), run_dir=str(run_dir)), \
            json.loads(stats_file.read_text())
    plan = plan_from_bash_params(json.loads((ref_dir / 'bash_params.json').read_text()), run_dir=str(run_dir))
    chunks = plan.make_chunks(read_events(str(run_dir / 'input' / 'events.bin')))
    run_chunks = inline.run_chunks if executor == 'inline' else pool.run_chunks
    stats = run_chunks(plan.spec, chunks, run_dir / 'taps', [], n_workers, output_jobs=plan.output_jobs)
    for rename in plan.output_renames:
        rename.apply(str(run_dir))
    return plan, stats


def final_outputs(plan):
    """The output files of a plan's jobs once its ``output_renames`` are applied."""
    paths = {p for job in plan.output_jobs for p in job_outputs(job)}
    for rename in plan.output_renames:
        for path in [p for p in paths if p.startswith(rename.src)]:
            paths.add(rename.dst + path[len(rename.src):])
            if not rename.copy:
                paths.discard(path)
    return sorted(f'output/{os.path.basename(p)}' for p in paths)


@pytest.mark.parametrize('executor, use_cli', [('inline', False), ('pool', True)])
def test_orchestrated_full_kernel_byte_identical_to_bash(full_kernel_reference, tmp_path, executor, use_cli):
    ref_dir = full_kernel_reference(FULL_K)
    plan, stats = orchestrated_run(ref_dir, tmp_path / 'run', executor, use_cli=use_cli)
    report = compare_trees_bytes(ref_dir, tmp_path / 'run')
    assert report.ok, report.summary()
    outputs = [rel for rel in list_files(ref_dir, subdirs=('output',)) if 'summary-info' not in rel]
    assert sorted(outputs) == final_outputs(plan)
    assert report.compared == len(list_files(ref_dir))
    check_job_stats(stats, plan.output_jobs, 1 if executor == 'inline' else 2)
    assert {j['tool'] for j in stats['jobs']} == {'katpy', 'aalpy', 'lecpy'}


def test_orchestrated_low_memory_run_byte_identical_to_its_bash_reference(full_kernel_reference, tmp_path):
    ref_dir = full_kernel_reference(FULL_K, summarypy_low_memory=True)
    plan, stats = orchestrated_run(ref_dir, tmp_path / 'run', 'pool')
    assert any(s.get('idx_paths') for node in plan.spec.summaries for s in node['sets'])
    report = compare_trees_bytes(ref_dir, tmp_path / 'run')
    assert report.ok, report.summary()
    assert any(rel.endswith('.idx') for rel in list_files(tmp_path / 'run', subdirs=('work',)))
    assert len(stats['jobs']) == len(plan.output_jobs)


def test_pool_output_jobs_without_chunks(tmp_path, run_dir):
    write_selt_csv(run_dir / 'work' / 'a.csv', 3, 1)
    stats = run_pool(make_spec(str(run_dir)), [], tmp_path / 'out', [], 3, factories=FAKE_FACTORIES,
                     output_jobs=[selt_job('output/out.csv', 'work/a.csv')])
    assert stats['chunks'] == []
    assert len(stats['jobs']) == 1 and stats['n_workers'] == 1
    assert (run_dir / 'output' / 'out.csv').read_text() == expected_selt(3, 1)


class SlowSecondChunkGul(FakeGul):
    """FakeGul that takes a while over the events of the second chunk, so that the other worker waits for it."""

    def process_event(self, event_id, out):
        if event_id > 5:
            time.sleep(0.3)
        super().process_event(event_id, out)


def test_pool_retired_waiting_worker_stops_before_the_last_chunk_and_jobs_still_run(tmp_path, run_dir):
    retired = []

    def policy(p):
        if not retired and any(w.waiting for w in p.workers):
            waiting = [w.number for w in p.workers if w.waiting]
            worker = p.retire()
            retired.append((worker.number, worker.stopped, p.chunks_done, waiting))

    factories = {**FAKE_FACTORIES, 'gul': f'{__name__}:SlowSecondChunkGul'}
    chunks = [np.arange(1, 6, dtype=np.int32), np.arange(6, 11, dtype=np.int32)]
    stats = run_pool(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, 2, factories=factories, policy=policy,
                     output_jobs=[job.to_dict() for job in SELT_JOBS])
    [(number, stopped, chunks_done, waiting)] = retired
    assert waiting == [number] and stopped and not chunks_done
    assert stats['workers'][number]['jobs'] == 0
    for i in range(5):
        assert (run_dir / 'output' / f's{i}.csv').read_text() == expected_selt(3, 1, 2)
    check_job_stats(stats, SELT_JOBS, 2)
    assert descendants() == []
