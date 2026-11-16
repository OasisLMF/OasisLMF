import types

import numpy as np
import pytest

from oasislmf.execution.orchestrator import cli

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
])
def test_rejects_unsupported_arguments(tmp_path, run_dir, recorded, capsys, argv):
    with pytest.raises(SystemExit):
        run_main(run_dir, tmp_path, *argv)
    assert recorded == []
    assert 'error' in capsys.readouterr().err
