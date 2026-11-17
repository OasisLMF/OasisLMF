import copy
import json
import os
import types

import numpy as np
import pytest

from oasislmf.execution import bash
from oasislmf.execution.orchestrator import chain as chain_module
from oasislmf.execution.orchestrator import cli
from oasislmf.execution.orchestrator.chunks import make_chunks
from oasislmf.execution.orchestrator.plan import plan_from_bash_params
from oasislmf.execution.orchestrator.resources import MB, ResourceManager
from oasislmf.utils.defaults import EVE_NO_SHUFFLE

from .test_chain import TREE_FACTORIES

from .test_inline import PIWIND_ARGS


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / 'run'
    (path / 'input').mkdir(parents=True)
    np.arange(1, 31, dtype=np.int32).tofile(path / 'input' / 'events.bin')
    return path


@pytest.fixture
def recorded(monkeypatch):
    """Replace every executor by a fake whose ``run_chunks`` records its options."""
    calls = []

    def run_chunks(spec, chunks, out_dir, taps, n_workers, **options):
        calls.append({'n_workers': n_workers, 'options': options})
        return {'wall_seconds': 0.0, 'events': sum(len(c) for c in chunks)}

    monkeypatch.setattr(cli, 'get_executor', lambda name: types.SimpleNamespace(run_chunks=run_chunks))
    return calls


def run_main(run_dir, tmp_path, *argv):
    return cli.main(['--run-dir', str(run_dir), '--chunks', '3', '--workers', '4', '--out-dir', str(tmp_path / 'out'), *PIWIND_ARGS, *argv])


def test_no_executor_options_unless_set(tmp_path, run_dir, recorded):
    for executor in ('inline', 'pool'):
        run_main(run_dir, tmp_path, '--executor', executor)
    assert [c['options'] for c in recorded] == [{}, {}]
    assert [c['n_workers'] for c in recorded] == [4, 4]


def test_chunk_order_still_passed_to_inline(tmp_path, run_dir, recorded):
    run_main(run_dir, tmp_path, '--chunk-order', 'reverse')
    assert recorded[0]['options'] == {'chunk_order': [3, 2, 1]}


@pytest.mark.parametrize('argv', [
    ['--executor', 'pool', '--chunk-order', 'reverse'],
    ['--executor', 'inline', '--chunk-order', '1,2'],
    ['--executor', 'ray'],
    ['--executor', 'inline', '--resource-manager', 'on'],
    ['--executor', 'pool', '--initial-workers', '2'],
    ['--executor', 'pool', '--memory-margin-mb', '512'],
    ['--executor', 'pool', '--resource-manager', 'on', '--initial-workers', '0'],
])
def test_rejects_unsupported_arguments(tmp_path, run_dir, recorded, capsys, argv):
    with pytest.raises(SystemExit):
        run_main(run_dir, tmp_path, *argv)
    assert recorded == []
    assert 'error' in capsys.readouterr().err


def test_resource_manager_option(tmp_path, run_dir, recorded):
    run_main(run_dir, tmp_path, '--executor', 'pool', '--resource-manager', 'on', '--initial-workers', '1', '--memory-margin-mb', '512')
    manager = recorded[0]['options']['resource_manager']
    assert isinstance(manager, ResourceManager)
    assert (manager.max_workers, manager.initial_workers, manager.margin) == (4, 1, 512 * MB)
    run_main(run_dir, tmp_path, '--executor', 'pool', '--resource-manager', 'off')
    assert recorded[1]['options'] == {}


PLAN_SETTINGS = {'number_of_samples': 2, 'gul_threshold': 0, 'gul_output': True, 'il_output': True,
                 'gul_summaries': [{'id': 1, 'ord_output': {'elt_sample': True, 'alt_period': True}}],
                 'il_summaries': [{'id': 1, 'ord_output': {'plt_moment': True}}]}


def write_bash_params(run_dir, settings=PLAN_SETTINGS, n_chunks=4, **options):
    params = bash.bash_params(analysis_settings=copy.deepcopy(settings), max_process_id=n_chunks, fifo_tmp_dir=False, **options)
    path = run_dir / 'bash_params.json'
    path.write_text(json.dumps(params, default=lambda v: None if callable(v) else str(v)))
    return path


@pytest.fixture
def recorded_runs(monkeypatch):
    """Replace every executor by a fake whose ``run_chunks`` records all its arguments."""
    calls = []

    def run_chunks(spec, chunks, out_dir, taps, n_workers, **options):
        calls.append({'spec': spec, 'chunks': chunks, 'out_dir': out_dir, 'taps': taps, 'n_workers': n_workers, 'options': options})
        return {'wall_seconds': 0.0}

    monkeypatch.setattr(cli, 'get_executor', lambda name: types.SimpleNamespace(run_chunks=run_chunks))
    return calls


def test_plan_option_builds_spec_and_chunks_from_bash_params(tmp_path, run_dir, recorded_runs):
    params_file = write_bash_params(run_dir, event_shuffle=EVE_NO_SHUFFLE)
    stale = run_dir / 'work' / 'gul_S1_summary_palt' / 'P9.bin'
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b'old')
    (run_dir / 'work' / 'other').mkdir()
    cli.main(['--plan', str(params_file), '--run-dir', str(run_dir), '--executor', 'pool', '--workers', '3'])
    call, = recorded_runs
    plan = plan_from_bash_params(json.loads(params_file.read_text()), run_dir=str(run_dir))
    assert call['spec'] == plan.spec and call['spec'].run_dir == str(run_dir)
    assert [c.tolist() for c in call['chunks']] == [c.tolist() for c in plan.make_chunks(np.arange(1, 31), 4)]
    assert call['chunks'][0].tolist() == list(range(1, 9))
    assert call['taps'] == [] and call['n_workers'] == 3
    assert not stale.parent.exists() and (run_dir / 'work' / 'other').exists()


def test_plan_option_chunks_taps_and_keep_work(tmp_path, run_dir, recorded_runs):
    params_file = write_bash_params(run_dir)
    stale = run_dir / 'work' / 'kat' / 'x'
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b'old')
    cli.main(['--plan', str(params_file), '--run-dir', str(run_dir), '--chunks', '6', '--taps', 'gul,il', '--out-dir',
              str(tmp_path / 'out'), '--keep-work'])
    call, = recorded_runs
    assert len(call['chunks']) == 6 and call['taps'] == ['gul', 'il'] and call['out_dir'] == str(tmp_path / 'out')
    assert stale.exists()


def test_plan_option_unsupported_exits_3(run_dir, recorded_runs, capsys):
    params_file = write_bash_params(run_dir, gulmc=False)
    with pytest.raises(SystemExit) as info:
        cli.main(['--plan', str(params_file), '--run-dir', str(run_dir)])
    assert info.value.code == 3
    assert 'unsupported by the orchestrator, run with bash: the ground up loss is computed by modelpy | gulpy' in capsys.readouterr().err
    assert recorded_runs == []


def test_analysis_settings_option(tmp_path, run_dir, recorded_runs):
    settings = {**PLAN_SETTINGS, 'ri_output': True, 'ri_summaries': [{'id': 1, 'ord_output': {'elt_moment': True}}]}
    (run_dir / 'analysis_settings.json').write_text(json.dumps(settings))
    (run_dir / 'input' / 'ri_layers.json').write_text(json.dumps({'1': {}, '2': {}}))
    cli.main(['--analysis-settings', str(run_dir / 'analysis_settings.json'), '--run-dir', str(run_dir), '--chunks', '2'])
    call, = recorded_runs
    assert [s['static_path'] for s in call['spec'].fm] == ['input', os.path.join('input', 'RI_1'), os.path.join('input', 'RI_2')]
    assert [n['runtype'] for n in call['spec'].summaries] == ['gul', 'il', 'ri']
    assert len(call['chunks']) == 2


def test_plan_option_rejects_both_sources_and_tap_without_out_dir(run_dir, recorded_runs, capsys):
    params_file = write_bash_params(run_dir)
    with pytest.raises(SystemExit):
        cli.main(['--plan', str(params_file), '--analysis-settings', str(params_file)])
    with pytest.raises(SystemExit):
        cli.main(['--plan', str(params_file), '--run-dir', str(run_dir), '--taps', 'gul'])
    assert recorded_runs == []


def test_plan_option_inline_with_fake_stages(tmp_path, run_dir, monkeypatch):
    monkeypatch.setattr(chain_module, 'default_factories', lambda: dict(TREE_FACTORIES))
    params_file = write_bash_params(run_dir, n_chunks=3)
    cli.main(['--plan', str(params_file), '--run-dir', str(run_dir), '--chunks', '2', '--no-output-jobs'])
    assert not (run_dir / 'output').exists()
    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 2)
    for k in (1, 2):
        found = sorted(p.relative_to(run_dir).as_posix() for p in (run_dir / 'work').rglob(f'*P{k}*'))
        assert found == [f'work/gul_S1_summary_palt/P{k}.bin', f'work/kat/gul_S1_elt_sample_P{k}', f'work/kat/il_S1_plt_moment_P{k}']
        palt = (run_dir / 'work' / 'gul_S1_summary_palt' / f'P{k}.bin').read_bytes()
        gul = b''.join((np.array([e, 10 * e], dtype=np.int32) + 1).tobytes() for e in chunks[k - 1] if e not in (5, 10) and e % 100 != 8)
        assert palt == b'SUM\x01' + gul


def test_plan_option_passes_output_jobs_for_the_chunk_count(tmp_path, run_dir, recorded_runs):
    params_file = write_bash_params(run_dir, n_chunks=4)
    plan = plan_from_bash_params(json.loads(params_file.read_text()), run_dir=str(run_dir))
    cli.main(['--plan', str(params_file), '--run-dir', str(run_dir)])
    cli.main(['--plan', str(params_file), '--run-dir', str(run_dir), '--chunks', '2'])
    cli.main(['--plan', str(params_file), '--run-dir', str(run_dir), '--no-output-jobs'])
    full, two, none = (call['options'].get('output_jobs') for call in recorded_runs)
    assert full == plan.output_jobs
    assert [job.tool for job in two] == [job.tool for job in plan.output_jobs]
    katpy = [job for job in two if job.tool == 'katpy']
    assert katpy and all([f[-3:] for f in job.kwargs['files_in']] == ['_P1', '_P2'] for job in katpy)
    assert [job for job in two if job.tool != 'katpy'] == [job for job in plan.output_jobs if job.tool != 'katpy']
    assert none is None


def test_output_jobs_not_passed_without_a_plan(tmp_path, run_dir, recorded):
    run_main(run_dir, tmp_path, '--no-output-jobs')
    run_main(run_dir, tmp_path)
    assert [c['options'] for c in recorded] == [{}, {}]


@pytest.mark.parametrize('argv, error', [
    (['--chunks', '0'], '--chunks must be positive'),
    (['--workers', '0'], '--workers must be positive'),
    (['--ri-layers', '-1'], '--ri-layers must not be negative'),
    (['--taps', ','], 'no tap given'),
    (['--chunk-order', '1,x'], 'expected reverse, shuffle:<seed> or chunk numbers'),
])
def test_rejects_bad_values(tmp_path, run_dir, recorded, capsys, argv, error):
    with pytest.raises(SystemExit):
        run_main(run_dir, tmp_path, *argv)
    assert recorded == []
    assert error in capsys.readouterr().err
