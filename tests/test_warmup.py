"""Tests of the JIT warmup orchestration in oasislmf/warmup.py (the compile tasks themselves run in tests/pytools/test_jit_compilation.py)."""
import logging
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from oasislmf import warmup


def _succeeding_task():
    return None


def _failing_task():
    raise RuntimeError('compile failed')


def test_silence_suppresses_output_and_restores_it(capsys):
    stdout, stderr, disabled = sys.stdout, sys.stderr, logging.root.manager.disable

    with warmup._silence():
        print('hidden')
        print('hidden', file=sys.stderr)
        assert logging.root.manager.disable == logging.CRITICAL

    assert (sys.stdout, sys.stderr, logging.root.manager.disable) == (stdout, stderr, disabled)
    assert capsys.readouterr() == ('', '')


def test_silence_restores_output_when_the_task_raises():
    stdout, stderr, disabled = sys.stdout, sys.stderr, logging.root.manager.disable

    with pytest.raises(RuntimeError):
        with warmup._silence():
            raise RuntimeError('boom')

    assert (sys.stdout, sys.stderr, logging.root.manager.disable) == (stdout, stderr, disabled)


@pytest.mark.parametrize('name', sorted(warmup.ALL_SILENT_TASKS))
def test_every_task_can_be_sent_to_a_worker_process(name):
    task = warmup.ALL_SILENT_TASKS[name]

    assert pickle.loads(pickle.dumps(task)) is task


def test_warmup_is_skipped_when_the_assets_are_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(warmup, '_DATA_DIR', tmp_path / 'missing')

    assert warmup.warmup() == {}
    assert 'skipping' in capsys.readouterr().out


def test_warmup_reports_each_failed_task(monkeypatch):
    monkeypatch.setattr(warmup, 'ProcessPoolExecutor', ThreadPoolExecutor)
    monkeypatch.setattr(warmup, 'ALL_SILENT_TASKS', {'good': _succeeding_task, 'bad': _failing_task})

    errors = warmup.warmup(max_workers=2)

    assert list(errors) == ['bad']
    assert isinstance(errors['bad'], RuntimeError)


def test_main_skips_when_the_jit_is_disabled(monkeypatch, capsys):
    monkeypatch.setenv('NUMBA_DISABLE_JIT', '1')
    monkeypatch.setattr(warmup, 'warmup', lambda: pytest.fail('warmup ran with the JIT disabled'))

    warmup.main()

    assert 'skipping' in capsys.readouterr().out


def test_main_exits_with_an_error_when_a_task_fails(monkeypatch, capsys):
    monkeypatch.delenv('NUMBA_DISABLE_JIT', raising=False)
    monkeypatch.setattr(warmup, 'warmup', lambda: {'fmpy': RuntimeError('compile failed')})

    with pytest.raises(SystemExit) as exit_info:
        warmup.main()

    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert 'fmpy' in err and 'compile failed' in err


def test_main_reports_success(monkeypatch, capsys):
    monkeypatch.delenv('NUMBA_DISABLE_JIT', raising=False)
    monkeypatch.setattr(warmup, 'warmup', lambda: {})

    warmup.main()

    assert 'Done' in capsys.readouterr().out
