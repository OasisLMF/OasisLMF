"""Phase 4: kernel steps run by the orchestrator through ``execution/runner.py``, against bash on the same run.

Unit tests of the pieces (the progress reporter, ``kernel.run_plan``'s side effects and errors) use fake stages;
the executors' chunk subsets and progress messages are tested in ``test_pool.py``. The end-to-end tests run the full
kernel of the test fixture with the runner, orchestrator on and off, as ``generate-losses`` does, and as the
platform's chunked steps do.
"""
import contextlib
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from oasislmf.execution import runner
from oasislmf.execution.bash import bash_params
from oasislmf.execution.orchestrator import kernel
from oasislmf.execution.orchestrator.chunks import make_chunks, read_events
from oasislmf.execution.orchestrator.plan import Plan, plan_from_bash_params
from oasislmf.execution.orchestrator.progress import ProgressReporter
from tests.orchestrator.tools.compare_outputs import compare_trees_bytes, list_files

from .test_chain import make_spec
from .test_pool import descendants

K = 2
SUMMARY_INFO = '*summary-info*'


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / 'run'
    (path / 'input').mkdir(parents=True)
    return path


def test_progress_reporter_pings_new_events_and_retries_failures():
    sent = []
    answers = iter([False, True, True, True])

    def ping(data):
        sent.append(dict(data))
        return next(answers)

    reporter = ProgressReporter(analysis_pk=7, interval=3600, ping=ping)
    reporter.update(5)
    assert not reporter._send()
    reporter.update(8)
    reporter.update(6)
    assert reporter._send()
    assert reporter._send()
    reporter.update(10)
    reporter.close()
    assert sent == [{'events_complete': 5, 'analysis_pk': 7}, {'events_complete': 8, 'analysis_pk': 7},
                    {'events_complete': 2, 'analysis_pk': 7}]
    assert reporter.sent == 10


def test_progress_reporter_bar_or_nothing_without_analysis_pk(capsys):
    calls = []
    with ProgressReporter(bar_total=4, ping=lambda data: calls.append(data)) as reporter:
        reporter.update(2)
        reporter.update(4)
    assert 'Gul events completed' in capsys.readouterr().err
    with ProgressReporter(ping=lambda data: calls.append(data)) as reporter:
        reporter.update(3)
    assert calls == [] and capsys.readouterr().err == ''


def test_remove_previous_outputs_keeps_summary_info_and_json(tmp_path):
    out = tmp_path / 'output'
    (out / 'sub').mkdir(parents=True)
    for name in ('gul_S1_summary-info.csv', 'analysis_settings.json', 'gul_S1_selt.csv', 'sub/x.parquet', 'sub/y.json'):
        (out / name).write_text('x')
    kernel.remove_previous_outputs(str(out))
    assert sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()) == [
        'analysis_settings.json', 'gul_S1_summary-info.csv', 'sub/y.json']


def test_empty_dir(tmp_path):
    (tmp_path / 'w' / 'kat').mkdir(parents=True)
    (tmp_path / 'w' / 'a.bin').write_text('x')
    kernel.empty_dir(str(tmp_path / 'w'))
    assert os.listdir(tmp_path / 'w') == []
    kernel.empty_dir(str(tmp_path / 'missing'), create=False)
    assert not (tmp_path / 'missing').exists()
    kernel.empty_dir(str(tmp_path / 'new'))
    assert (tmp_path / 'new').is_dir()


def fake_plan(run_dir, n_chunks=2, process_number=None):
    return Plan(spec=make_spec(str(run_dir)), n_chunks=n_chunks, process_number=process_number)


def test_run_plan_failure_is_a_called_process_error_with_the_logs(tmp_path, run_dir):
    np.arange(1, 5, dtype=np.int32).tofile(run_dir / 'input' / 'events.bin')
    (run_dir / 'log').mkdir()
    (run_dir / 'log' / 'old.log').write_text('stale')
    params = {'work_dir': 'work/', 'work_kat_dir': 'work/kat/', 'output_dir': 'output/'}
    with pytest.raises(subprocess.CalledProcessError) as info:
        kernel.run_plan(fake_plan(run_dir), params, 'analysis', 'log', str(run_dir / 'run_kernel.sh'))
    assert info.value.cmd == ['oasislmf-orchestrator', 'analysis', str(run_dir / 'run_kernel.sh')]
    assert b'FAILED' in info.value.output
    logs = sorted(os.listdir(run_dir / 'log'))
    assert 'old.log' not in logs and 'stderror.err' in logs and kernel.RUN_LOG in logs
    assert 'orchestrator analysis failed' in (run_dir / 'log' / 'stderror.err').read_text()
    assert (run_dir / 'work' / 'kat').is_dir()
    assert descendants() == []


def test_run_plan_rejects_unknown_mode_and_executor(run_dir):
    with pytest.raises(ValueError, match='mode'):
        kernel.run_plan(fake_plan(run_dir), {}, 'everything', 'log', 'x.sh')
    with pytest.raises(ValueError, match='executor'):
        kernel.run_plan(fake_plan(run_dir), {}, 'all', 'log', 'x.sh', executor='ray')


def test_plan_chunk_numbers():
    assert fake_plan('.', 3).chunk_numbers == [1, 2, 3]
    assert fake_plan('.', 3, process_number=2).chunk_numbers == [2]
    assert Plan.from_json(fake_plan('.', 3, process_number=2).to_json()).process_number == 2


# --- end to end, the full kernel of the test fixture through the runner -------------------------------------------


@contextlib.contextmanager
def cwd(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def new_run_dir(full_kernel_run, path):
    """A run directory as GenerateLossesDir leaves it: the fixture's input and static, its summary-info files, fifo/ and work/."""
    path.mkdir(parents=True)
    for name in ('input', 'static'):
        (path / name).symlink_to(os.path.realpath(full_kernel_run / name), target_is_directory=True)
    for subdir in ('fifo', 'output', 'work'):
        (path / subdir).mkdir()
    for src in (full_kernel_run / 'output').glob(SUMMARY_INFO):
        shutil.copy2(src, path / 'output' / src.name)
    return path


def settings_of(full_kernel_run):
    return json.loads((full_kernel_run / 'analysis_settings.json').read_text())


def num_ri(full_kernel_run, settings):
    if not (settings.get('ri_output') or settings.get('rl_output')):
        return 0
    return len(json.loads((full_kernel_run / 'input' / 'ri_layers.json').read_text()))


def run_whole(full_kernel_run, path, orchestrator, settings=None, n_processes=K, **kwargs):
    """``runner.run`` as ``GenerateLosses`` calls it, in a new run directory."""
    settings = settings or settings_of(full_kernel_run)
    run = new_run_dir(full_kernel_run, path)
    with cwd(run):
        runner.run(settings, number_of_processes=n_processes, filename=str(run / 'run_kernel.sh'),
                   num_reinsurance_iterations=num_ri(full_kernel_run, settings), disable_resource_report=True,
                   orchestrator=orchestrator, **kwargs)
    return run


def output_report(run_a, run_b):
    return compare_trees_bytes(run_a, run_b, subdirs=('output',))


@pytest.fixture(scope='module')
def whole_runs(full_kernel_run, tmp_path_factory):
    """The fixture's full kernel run by ``runner.run`` with K processes, bash and orchestrator."""
    base = tmp_path_factory.mktemp('whole_runs')
    return {'bash': run_whole(full_kernel_run, base / 'bash', False),
            'orchestrator': run_whole(full_kernel_run, base / 'orchestrator', True)}


def test_runner_orchestrator_output_byte_identical_to_bash(whole_runs):
    bash_run, orch_run = whole_runs['bash'], whole_runs['orchestrator']
    report = output_report(bash_run, orch_run)
    assert report.ok, report.summary()
    outputs = [rel for rel in list_files(orch_run, subdirs=('output',)) if 'summary-info' not in rel]
    assert len(outputs) > 20
    assert (orch_run / 'run_kernel.sh').is_file()
    assert (orch_run / 'log' / kernel.RUN_LOG).is_file()
    assert sorted(p.name for p in (orch_run / 'log').glob('orchestrator_worker_*.log')) == [
        f'orchestrator_worker_{w}.log' for w in range(K)]
    assert (orch_run / 'log' / kernel.STDERR_LOG).read_text() == ''
    assert os.listdir(orch_run / 'work') == []


def chunk_params(run, settings, n_ri, k, orchestrator, **extra):
    """bash_params of generate-losses-chunk k of K, as ``GenerateLossesPartial`` builds them."""
    params = bash_params(settings, max_process_id=K, num_reinsurance_iterations=n_ri, filename=str(run / f'{k}.run_analysis.sh'),
                         process_number=k, model_run_dir=str(run), fifo_tmp_dir=False)
    params.update(resource_monitor_interval=1.0, disable_resource_report=True, analysis_pk=extra.pop('analysis_pk', None))
    if orchestrator:
        params['orchestrator'] = True
    params.update(extra)
    return params


def output_params(run, settings, n_ri, orchestrator, remove_working_files):
    """bash_params of generate-losses-output, as ``GenerateLossesOutput`` builds them."""
    params = bash_params(settings, max_process_id=K, num_reinsurance_iterations=n_ri, filename=str(run / 'run_outputs.sh'),
                         model_run_dir=str(run), remove_working_files=remove_working_files, fifo_tmp_dir=False)
    params.update(resource_monitor_interval=1.0, disable_resource_report=True)
    if orchestrator:
        params['orchestrator'] = True
    return params


def run_chunked(full_kernel_run, path, chunk_engines, outputs_orchestrator, remove_working_files):
    """The platform's split run: each chunk k into ``k.work`` (bash or orchestrator), gathered into ``work/``, then outputs."""
    settings = settings_of(full_kernel_run)
    n_ri = num_ri(full_kernel_run, settings)
    run = new_run_dir(full_kernel_run, path)
    with cwd(run):
        for k, orchestrator in enumerate(chunk_engines, 1):
            runner.run_analysis(**chunk_params(run, settings, n_ri, k, orchestrator, analysis_pk=7 if orchestrator else None))
        work_dir = run / 'work'
        for k in range(1, K + 1):
            shutil.copytree(run / f'{k}.work', work_dir, dirs_exist_ok=True)
        runner.run_outputs(**output_params(run, settings, n_ri, outputs_orchestrator, remove_working_files))
    return run


def test_chunked_runs_mixed_bash_and_orchestrator_match_the_whole_run(full_kernel_run, whole_runs, tmp_path, monkeypatch):
    pinged = []
    monkeypatch.setattr('oasislmf.execution.orchestrator.progress.oasis_ping', lambda data: pinged.append(data) or True)
    run_a = run_chunked(full_kernel_run, tmp_path / 'a', [False, True], outputs_orchestrator=True, remove_working_files=False)
    run_b = run_chunked(full_kernel_run, tmp_path / 'b', [True, False], outputs_orchestrator=False, remove_working_files=True)

    for k in range(1, K + 1):
        report = compare_trees_bytes(run_a, run_b, subdirs=(f'{k}.work',))
        assert report.ok, f'chunk {k}, bash and orchestrator:\n' + report.summary()
        assert report.compared > 10
        orchestrated = run_a if k == 2 else run_b
        assert (orchestrated / 'log' / str(k) / kernel.RUN_LOG).is_file()
        assert list((orchestrated / f'{k}.work').rglob(f'P{k}.bin'))
        assert not list((orchestrated / f'{k}.work').rglob(f'P{3 - k}.bin'))
    for run in (run_a, run_b):
        report = output_report(whole_runs['bash'], run)
        assert report.ok, report.summary()
    assert (run_a / 'log' / 'out' / kernel.RUN_LOG).is_file()
    assert list((run_a / 'work').rglob('*.bin'))
    assert os.listdir(run_b / 'work') == []

    events = read_events(str(full_kernel_run / 'input' / 'events.bin'))
    chunks = make_chunks(events, K)
    assert {p['analysis_pk'] for p in pinged} == {7}
    assert sum(p['events_complete'] for p in pinged) == len(chunks[0]) + len(chunks[1])


def test_unsupported_run_falls_back_to_bash(full_kernel_run, tmp_path, caplog):
    settings = settings_of(full_kernel_run)
    settings = {'number_of_samples': settings['number_of_samples'], 'gul_threshold': 0, 'gul_output': True,
                'gul_summaries': [{'id': 1, 'ord_output': {'elt_sample': True}}], 'model_settings': settings.get('model_settings', {})}
    with caplog.at_level('WARNING', logger='oasislmf.execution.runner'):
        run = run_whole(full_kernel_run, tmp_path / 'run', True, settings=settings, n_processes=1, join_summary_info=True)
    assert 'join-summary-info' in caplog.text and 'running the kernel with the bash script instead' in caplog.text
    assert not (run / 'log' / kernel.RUN_LOG).exists()
    assert (run / 'output' / 'gul_S1_selt.csv').stat().st_size > 0


def test_plan_of_a_chunk_names_the_bash_chunk_files(full_kernel_run, tmp_path):
    settings = settings_of(full_kernel_run)
    run = new_run_dir(full_kernel_run, tmp_path / 'run')
    params = chunk_params(run, settings, num_ri(full_kernel_run, settings), 2, True)
    plan = plan_from_bash_params(params, run_dir=str(run))
    assert plan.chunk_numbers == [2]
    assert all(p.startswith(str(run / '2.work')) and ('_P2' in p or '/P2.' in p) for p in plan.chunk_paths(2))
