"""Phase 4: kernel steps run by the orchestrator through ``execution/runner.py``, against bash on the same run.

Unit tests of the pieces (the progress reporter, ``kernel.run_plan``'s side effects and errors) use fake stages;
the executors' chunk subsets and progress messages are tested in ``test_pool.py``. The end-to-end tests run the full
kernel of the test fixture with the runner, orchestrator on and off, as ``generate-losses`` does, and as the
platform's chunked steps do.
"""
import contextlib
import functools
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
from oasislmf.execution.orchestrator.plan import OutputJob, Plan, plan_from_bash_params
from oasislmf.execution.orchestrator.progress import ProgressReporter
from oasislmf.execution.orchestrator.resources import ResourceManager
from tests.orchestrator.tools.compare_outputs import compare_outputs_rows, compare_trees_bytes, list_files
from tests.orchestrator.tools.kernel_reference import link_dir

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


class FakeExecutor:
    """Stands in for the pool: writes what its workers would have written to stderr and the chunk files."""

    def __init__(self, stderr=(), files=(), error=None):
        self.stderr = stderr
        self.files = files
        self.error = error
        self.options = None

    def run_chunks(self, spec, chunks, out_dir, taps, n_workers, **options):
        self.options = options
        for n, text in enumerate(self.stderr):
            with open(os.path.join(options['log_dir'], f'orchestrator_worker_{n}.err'), 'w') as f:
                f.write(text)
        for path, content in self.files:
            path = os.path.join(spec.run_dir, path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(content)
        if self.error is not None:
            raise self.error
        return {'chunks': [], 'jobs': []}


def run_fake(run_dir, monkeypatch, executor, params, mode='analysis', process_number=None):
    np.arange(1, 5, dtype=np.int32).tofile(run_dir / 'input' / 'events.bin')
    monkeypatch.setitem(kernel.EXECUTORS, 'pool', executor)
    params = {'work_dir': 'work/', 'work_kat_dir': 'work/kat/', 'output_dir': 'output/', **params}
    return kernel.run_plan(fake_plan(run_dir, process_number=process_number), params, mode, 'log', 'run_kernel.sh')


def test_run_plan_stderr_guard_appends_worker_stderr(run_dir, monkeypatch):
    executor = FakeExecutor(stderr=['/a.py:1: UserWarning: w\n', '', 'x.py:2: FutureWarning: f\n'])
    run_fake(run_dir, monkeypatch, executor, {})
    assert executor.options['stderr_guard'] is True
    assert (run_dir / 'log' / 'stderror.err').read_text() == '/a.py:1: UserWarning: w\nx.py:2: FutureWarning: f\n'


def test_run_plan_stderr_guard_off(run_dir, monkeypatch):
    executor = FakeExecutor(stderr=['ERROR\n'], files=[('1.work/gul_S1_summary/P1.bin', b'')])
    run_fake(run_dir, monkeypatch, executor, {'stderr_guard': False, 'process_number': 1}, process_number=1)
    assert executor.options['stderr_guard'] is False
    assert (run_dir / 'log' / 'stderror.err').read_text() == ''


def test_run_plan_failure_keeps_worker_stderr_before_the_failure(run_dir, monkeypatch):
    executor = FakeExecutor(stderr=['Traceback\nERROR\n'], error=RuntimeError('pool worker 0 wrote to stderr: Traceback'))
    with pytest.raises(subprocess.CalledProcessError) as info:
        run_fake(run_dir, monkeypatch, executor, {'stderr_guard': True})
    assert info.value.returncode == 1
    text = (run_dir / 'log' / 'stderror.err').read_text()
    assert text.startswith('Traceback\nERROR\norchestrator analysis failed')
    assert 'wrote to stderr' in text


def test_run_plan_chunk_fails_on_an_empty_chunk_file(run_dir, monkeypatch):
    files = [('2.work/gul_S1_summary/P2.bin', b'header'), ('2.work/il_S1_summary/P2.bin', b''), ('2.work/kat/x_P2', b'')]
    with pytest.raises(subprocess.CalledProcessError) as info:
        run_fake(run_dir, monkeypatch, FakeExecutor(files=files), {'process_number': 2, 'work_dir': '2.work/'}, process_number=2)
    expected = "Chunk output error: File '" + os.path.join('2.work', 'il_S1_summary', 'P2.bin') + "' is empty"
    assert expected.encode() in info.value.output
    assert expected in (run_dir / 'log' / 'stderror.err').read_text()

    files[1] = ('2.work/il_S1_summary/P2.bin', b'header')
    run_fake(run_dir, monkeypatch, FakeExecutor(files=files), {'process_number': 2, 'work_dir': '2.work/'}, process_number=2)


def test_run_plan_rejects_unknown_mode_and_executor(run_dir):
    with pytest.raises(ValueError, match='mode'):
        kernel.run_plan(fake_plan(run_dir), {}, 'everything', 'log', 'x.sh')
    with pytest.raises(ValueError, match='executor'):
        kernel.run_plan(fake_plan(run_dir), {}, 'all', 'log', 'x.sh', executor='ray')


def test_run_plan_sizes_the_pool_with_the_resource_manager(run_dir, monkeypatch):
    executor = FakeExecutor()
    text = run_fake(run_dir, monkeypatch, executor, {}, mode='all')
    manager = executor.options['resource_manager']
    assert isinstance(manager, ResourceManager) and manager.max_workers == 2
    assert 'at most 2 workers (resource manager)' in text
    assert 'resource manager: at most 0 live workers of 2' in text
    assert text == (run_dir / 'log' / kernel.RUN_LOG).read_text()


def test_run_plan_fixed_pool_without_the_resource_manager(run_dir, monkeypatch):
    executor = FakeExecutor()
    np.arange(1, 5, dtype=np.int32).tofile(run_dir / 'input' / 'events.bin')
    monkeypatch.setitem(kernel.EXECUTORS, 'pool', executor)
    params = {'work_dir': 'work/', 'work_kat_dir': 'work/kat/', 'output_dir': 'output/'}
    text = kernel.run_plan(fake_plan(run_dir), params, 'all', 'log', 'run_kernel.sh', resource_manager=False)
    assert 'resource_manager' not in executor.options
    assert '2 workers, chunks [1, 2] of 2' in text and 'resource manager' not in text


def test_chunk_count():
    plan = fake_plan('.', 4)
    assert kernel.chunk_count(plan, 'all', 1000, 4, 1) == 4
    assert kernel.chunk_count(plan, 'all', 1000, 4, 4) == 16
    assert kernel.chunk_count(plan, 'all', 10, 4, 4) == 10
    assert kernel.chunk_count(plan, 'all', 2, 4, 4) == 4, 'never fewer chunks than the plan'
    assert kernel.chunk_count(plan, 'analysis', 1000, 4, 4) == 4, 'a chunked step keeps the work files the outputs step reads'
    assert kernel.chunk_count(fake_plan('.', 4, process_number=2), 'all', 1000, 4, 4) == 4


def test_run_plan_chunk_factor_splits_a_whole_run_finer(run_dir, monkeypatch):
    executor = FakeExecutor()
    np.arange(1, 31, dtype=np.int32).tofile(run_dir / 'input' / 'events.bin')
    monkeypatch.setitem(kernel.EXECUTORS, 'pool', executor)
    job = OutputJob('katpy', {'out': 'output/x.csv', 'files_in': ['work/kat/x_P1', 'work/kat/x_P2'], 'selt': True})
    plan = Plan(spec=make_spec(str(run_dir)), n_chunks=2, output_jobs=[job])
    params = {'work_dir': 'work/', 'work_kat_dir': 'work/kat/', 'output_dir': 'output/'}
    text = kernel.run_plan(plan, params, 'all', 'log', 'run_kernel.sh', chunk_factor=3)
    assert executor.options['chunk_numbers'] == [1, 2, 3, 4, 5, 6]
    assert executor.options['output_jobs'][0].kwargs['files_in'] == [f'work/kat/x_P{k}' for k in range(1, 7)]
    assert 'of 6' in text


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
        link_dir(path / name, full_kernel_run / name)
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
def orchestrated_whole_run(full_kernel_run, tmp_path_factory):
    """The fixture's full kernel run by ``runner.run`` with K processes and the orchestrator."""
    return run_whole(full_kernel_run, tmp_path_factory.mktemp('whole_runs') / 'orchestrator', True)


@pytest.fixture(scope='module')
def bash_whole_run(full_kernel_run, tmp_path_factory):
    """The same run by bash (``requires_bash``)."""
    return run_whole(full_kernel_run, tmp_path_factory.mktemp('whole_runs') / 'bash', False)


def test_runner_orchestrator_output_byte_identical_to_bash(bash_whole_run, orchestrated_whole_run):
    report = output_report(bash_whole_run, orchestrated_whole_run)
    assert report.ok, report.summary()


def test_runner_orchestrator_run_logs_and_outputs(orchestrated_whole_run):
    orch_run = orchestrated_whole_run
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


def test_chunked_orchestrated_run_matches_the_whole_run(full_kernel_run, orchestrated_whole_run, tmp_path, monkeypatch):
    pinged = []
    monkeypatch.setattr('oasislmf.execution.orchestrator.progress.oasis_ping', lambda data: pinged.append(data) or True)
    run = run_chunked(full_kernel_run, tmp_path / 'run', [True, True], outputs_orchestrator=True, remove_working_files=True)
    report = output_report(orchestrated_whole_run, run)
    assert report.ok, report.summary()
    assert report.compared > 20
    for k in range(1, K + 1):
        assert (run / 'log' / str(k) / kernel.RUN_LOG).is_file()
        assert (run / 'log' / str(k) / kernel.STDERR_LOG).read_text() == ''
    assert os.listdir(run / 'work') == []
    assert sum(p['events_complete'] for p in pinged) == len(read_events(str(full_kernel_run / 'input' / 'events.bin')))


def test_chunked_runs_mixed_bash_and_orchestrator_match_the_whole_run(full_kernel_run, bash_whole_run, tmp_path, monkeypatch):
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
        report = output_report(bash_whole_run, run)
        assert report.ok, report.summary()
    assert (run_a / 'log' / 'out' / kernel.RUN_LOG).is_file()
    assert list((run_a / 'work').rglob('*.bin'))
    assert os.listdir(run_b / 'work') == []

    events = read_events(str(full_kernel_run / 'input' / 'events.bin'))
    chunks = make_chunks(events, K)
    assert {p['analysis_pk'] for p in pinged} == {7}
    assert sum(p['events_complete'] for p in pinged) == len(chunks[0]) + len(chunks[1])


@pytest.mark.requires_bash
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


def run_whole_with(full_kernel_run, path, **run_plan_options):
    """``run_whole`` with the orchestrator, ``kernel.run_plan`` given ``run_plan_options``."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner, 'run_plan', functools.partial(kernel.run_plan, **run_plan_options))
        return run_whole(full_kernel_run, path, True)


def test_runner_resource_manager_on_and_off_byte_identical(full_kernel_run, orchestrated_whole_run, tmp_path):
    fixed = run_whole_with(full_kernel_run, tmp_path / 'fixed', resource_manager=False)
    report = compare_trees_bytes(orchestrated_whole_run, fixed, subdirs=('output',))
    assert report.ok, report.summary()
    assert report.compared > 20
    managed_log = (orchestrated_whole_run / 'log' / kernel.RUN_LOG).read_text()
    assert 'resource manager: 0.0 s: start 0 -> 2 workers' in managed_log
    assert 'resource manager' not in (fixed / 'log' / kernel.RUN_LOG).read_text()


def test_runner_resource_manager_finer_chunks_rows_equal(full_kernel_run, orchestrated_whole_run, tmp_path):
    run = run_whole_with(full_kernel_run, tmp_path / 'finer', chunk_factor=3, manager_options={'initial_workers': 1})
    text = (run / 'log' / kernel.RUN_LOG).read_text()
    assert f'of {3 * K}' in text
    assert 'start 0 -> 1 workers' in text
    report = compare_outputs_rows(orchestrated_whole_run, run)
    assert report.ok, report.summary()
    assert report.compared > 20
    assert os.listdir(run / 'work') == []
