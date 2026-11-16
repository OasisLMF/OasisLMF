import json
import os
import subprocess
import sys

import numpy as np
import pytest

from oasislmf.execution.orchestrator import chain as chain_module
from oasislmf.execution.orchestrator import cli, inline
from oasislmf.execution.orchestrator.chain import Chain
from oasislmf.execution.orchestrator.chunks import ChunkWriter, make_chunks
from tests.orchestrator.tools.compare_streams import compare_dirs_bytes

from .test_chain import (FACTORIES, TREE_FACTORIES, FakeFm, FakeGul, expected_streams, expected_tree_files, make_spec, make_tree_spec,
                         read_tree_files)

TAPS = ['gul', 'il', 'ri']
PIWIND_ARGS = ['--random-generator=2', '--model-df-engine=oasis_data_manager.df_reader.reader.OasisPandasReader',
               '--vuln-cache-size', '200', '-S10', '-L0', '-a1', '--ri-layers', '1']


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / 'run'
    (path / 'input').mkdir(parents=True)
    np.arange(1, 31, dtype=np.int32).tofile(path / 'input' / 'events.bin')
    return path


def check_outputs(out_dir, chunks, headers, taps=TAPS):
    assert sorted(os.listdir(out_dir)) == sorted(f'{tap}_P{k}.bin' for tap in taps for k in range(1, len(chunks) + 1))
    for k, chunk in enumerate(chunks, 1):
        expected = expected_streams([int(e) for e in chunk])
        for tap in taps:
            assert (out_dir / f'{tap}_P{k}.bin').read_bytes() == headers[tap] + expected[tap], (tap, k)


def fake_headers():
    chain = chain_module.Chain(make_spec(), factories=FACTORIES)
    return chain.headers


def test_inline_run_chunks_end_to_end(tmp_path, run_dir):
    gul_instances = []

    def gul_factory(**kwargs):
        gul_instances.append(FakeGul(**kwargs))
        return gul_instances[-1]

    events = np.arange(1, 31, dtype=np.int32)
    chunks = make_chunks(events, 4) + [np.array([], dtype=np.int32)]
    out_dir = tmp_path / 'out'
    cwd = os.getcwd()
    stats = inline.run_chunks(make_spec(str(run_dir)), chunks, out_dir, TAPS, n_workers=3, factories={'gul': gul_factory, 'fm': FakeFm})

    assert os.getcwd() == cwd
    assert len(gul_instances) == 1
    gul = gul_instances[0]
    assert gul.run_dir == str(run_dir)
    assert set(gul.cwd) == {str(run_dir)}
    assert gul.closed
    assert gul.events == [int(e) for chunk in chunks for e in chunk]
    check_outputs(out_dir, chunks, fake_headers())
    assert (out_dir / 'gul_P5.bin').read_bytes() == fake_headers()['gul']

    assert stats['executor'] == 'inline'
    assert stats['n_workers'] == 1
    assert stats['events'] == 30
    assert [c['chunk'] for c in stats['chunks']] == [1, 2, 3, 4, 5]
    assert [c['events'] for c in stats['chunks']] == [8, 8, 7, 7, 0]
    assert stats['wall_seconds'] >= stats['setup_seconds'] >= 0
    assert all(c['seconds'] >= 0 for c in stats['chunks'])
    json.dumps(stats)


def test_inline_relative_paths_and_cwd_restored_on_error(tmp_path, run_dir, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class FailingGul(FakeGul):
        def process_event(self, event_id, out):
            raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        inline.run_chunks(make_spec('run'), [np.array([1], dtype=np.int32)], 'out', ['gul'], factories={'gul': FailingGul, 'fm': FakeFm})
    assert os.getcwd() == str(tmp_path)

    inline.run_chunks(make_spec('run'), [np.array([1, 2], dtype=np.int32)], 'out', ['il'], factories=FACTORIES)
    assert (tmp_path / 'out' / 'il_P1.bin').read_bytes() == fake_headers()['il'] + expected_streams([1, 2])['il']


def test_cli_args_to_spec_piwind():
    args = cli.build_parser().parse_args(['--run-dir', '/run', '--chunks', '12', '--out-dir', 'o', *PIWIND_ARGS])
    spec = cli.spec_from_args(args)
    assert spec.run_dir == '/run'
    assert spec.gul == {
        'sample_size': 10, 'loss_threshold': 0.0, 'alloc_rule': 1, 'debug': 0, 'random_generator': 2,
        'ignore_file_type': [], 'peril_filter': [], 'data_server': False, 'ignore_correlation': False,
        'ignore_haz_correlation': False, 'effective_damageability': False, 'max_cached_vuln_cdf_size_MB': 200,
        'model_df_engine': 'oasis_data_manager.df_reader.reader.OasisPandasReader', 'dynamic_footprint': False,
    }
    fm_common = {'low_memory': False, 'sort_output': False, 'storage_method': 'sparse'}
    assert spec.fm == [
        {'allocation_rule': 2, 'static_path': 'input', 'name': 'il', 'feeds_next': 'gross', **fm_common},
        {'allocation_rule': 3, 'static_path': os.path.join('input', 'RI_1'), 'name': 'ri', 'feeds_next': None, **fm_common},
    ]
    assert cli.resolve_taps(args) == TAPS
    assert cli.resolve_taps(cli.build_parser().parse_args(['--plan', 'p.json'])) == []
    assert cli.resolve_taps(cli.build_parser().parse_args([])) == ['gul', 'il']
    assert cli.resolve_taps(cli.build_parser().parse_args(['--ri-layers', '2'])) == ['gul', 'il', 'ri']
    assert args.executor == 'inline'
    assert args.chunks == 12


def test_cli_args_to_spec_options():
    args = cli.build_parser().parse_args([
        '--run-dir', 'rel', '--out-dir', 'o', '--taps', 'gul, il', '--il-alloc-rule', '1', '--ri-layers', '3', '--ri-alloc-rule', '2',
        '--ignore-file-type', 'bin', 'parquet', '--peril-filter', 'WTC', 'WSS', '--effective-damageability', '--ignore-correlation',
        '--ignore-haz-correlation', '--data-server', '--dynamic-footprint', '-d', '1', '--sort-output', '--low-memory',
        '--storage-method', 'dense',
    ])
    spec = cli.spec_from_args(args)
    assert spec.run_dir == os.path.abspath('rel')
    assert args.taps == ['gul', 'il']
    assert spec.gul['ignore_file_type'] == ['bin', 'parquet']
    assert spec.gul['peril_filter'] == ['WTC', 'WSS']
    assert spec.gul['debug'] == 1
    for flag in ('effective_damageability', 'ignore_correlation', 'ignore_haz_correlation', 'data_server', 'dynamic_footprint'):
        assert spec.gul[flag] is True
    assert [(s['allocation_rule'], s['static_path'], s['feeds_next']) for s in spec.fm] == [
        (1, 'input', 'gross'), (2, os.path.join('input', 'RI_1'), 'net'), (2, os.path.join('input', 'RI_2'), 'net'),
        (2, os.path.join('input', 'RI_3'), None)]
    assert all(s['sort_output'] and s['low_memory'] and s['storage_method'] == 'dense' for s in spec.fm)
    assert cli.spec_from_args(args).to_json() == spec.to_json()


def test_cli_rejects_bad_taps_and_ri_without_layers(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(['--out-dir', 'o', '--taps', 'gul,xx'])
    with pytest.raises(SystemExit):
        cli.main(['--out-dir', 'o', '--taps', 'ri'])


def test_cli_get_executor():
    assert cli.get_executor('inline') is inline
    with pytest.raises(ValueError):
        cli.get_executor('threads')


def test_cli_main_inline_with_fake_stages(tmp_path, run_dir, monkeypatch):
    monkeypatch.setattr(chain_module, 'default_factories', lambda: dict(FACTORIES))
    out_dir = tmp_path / 'out'
    stats_json = tmp_path / 'stats.json'
    stats = cli.main(['--executor', 'inline', '--run-dir', str(run_dir), '--chunks', '3', '--workers', '2', '--out-dir', str(out_dir),
                      '--taps', 'gul,il,ri', '--stats-json', str(stats_json), *PIWIND_ARGS])
    check_outputs(out_dir, make_chunks(np.arange(1, 31, dtype=np.int32), 3), fake_headers())
    assert json.loads(stats_json.read_text())['events'] == stats['events'] == 30


def test_cli_main_default_taps_without_ri_layers(tmp_path, run_dir, monkeypatch):
    monkeypatch.setattr(chain_module, 'default_factories', lambda: dict(FACTORIES))
    out_dir = tmp_path / 'out'
    no_ri_args = [a for a in PIWIND_ARGS if a not in ('--ri-layers', '1')]
    cli.main(['--executor', 'inline', '--run-dir', str(run_dir), '--chunks', '2', '--out-dir', str(out_dir), *no_ri_args])
    check_outputs(out_dir, make_chunks(np.arange(1, 31, dtype=np.int32), 2), fake_headers(), taps=['gul', 'il'])


def test_cli_does_not_import_unselected_executors():
    code = ("import sys; from oasislmf.execution.orchestrator import cli; cli.get_executor('inline'); "
            "assert not {'oasislmf.execution.orchestrator.pool', 'oasislmf.execution.orchestrator.ray_executor', 'ray'} & set(sys.modules)")
    subprocess.run([sys.executable, '-c', code], check=True)


def test_inline_resets_chain_before_each_chunk(tmp_path, run_dir):
    log = []

    class LoggingGul(FakeGul):
        def process_event(self, event_id, out):
            log.append(int(event_id))
            super().process_event(event_id, out)

    class ResettingFm(FakeFm):
        def reset(self):
            log.append(('reset', self.static_path))

    chunks = [np.array([3, 1], dtype=np.int32), np.array([], dtype=np.int32), np.array([2], dtype=np.int32)]
    inline.run_chunks(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, factories={'gul': LoggingGul, 'fm': ResettingFm})
    reset = [('reset', 'input'), ('reset', 'input/RI_1')]
    assert log == [*reset, 3, 1, *reset, *reset, 2]
    check_outputs(tmp_path / 'out', chunks, fake_headers())


def test_inline_chunk_order_changes_processing_order_not_file_names(tmp_path, run_dir):
    log = []

    class LoggingGul(FakeGul):
        def process_event(self, event_id, out):
            log.append(int(event_id))
            super().process_event(event_id, out)

    chunks = make_chunks(np.arange(1, 31, dtype=np.int32), 3)
    stats = inline.run_chunks(make_spec(str(run_dir)), chunks, tmp_path / 'out', TAPS, factories={'gul': LoggingGul, 'fm': FakeFm},
                              chunk_order=[3, 1, 2])
    assert log == [int(e) for k in (3, 1, 2) for e in chunks[k - 1]]
    assert [c['chunk'] for c in stats['chunks']] == [3, 1, 2]
    check_outputs(tmp_path / 'out', chunks, fake_headers())

    for bad in ([1, 2], [1, 2, 2], [0, 1, 2]):
        with pytest.raises(ValueError, match='permutation'):
            inline.run_chunks(make_spec(str(run_dir)), chunks, tmp_path / 'bad', TAPS, factories=FACTORIES, chunk_order=bad)


def test_cli_chunk_order(tmp_path, run_dir, monkeypatch, capsys):
    assert cli.resolve_chunk_order(None, 4) is None
    assert cli.resolve_chunk_order(cli.parse_chunk_order('reverse'), 4) == [4, 3, 2, 1]
    assert cli.resolve_chunk_order(cli.parse_chunk_order('2,1,3'), 3) == [2, 1, 3]
    shuffled = cli.resolve_chunk_order(cli.parse_chunk_order('shuffle:7'), 6)
    assert sorted(shuffled) == [1, 2, 3, 4, 5, 6] and shuffled == cli.resolve_chunk_order('shuffle:7', 6)

    monkeypatch.setattr(chain_module, 'default_factories', lambda: dict(FACTORIES))
    stats = cli.main(['--run-dir', str(run_dir), '--chunks', '3', '--out-dir', str(tmp_path / 'out'), '--chunk-order', 'reverse',
                      *PIWIND_ARGS])
    assert [c['chunk'] for c in stats['chunks']] == [3, 2, 1]
    check_outputs(tmp_path / 'out', make_chunks(np.arange(1, 31, dtype=np.int32), 3), fake_headers())

    for argv in (['--chunk-order', '1,2'], ['--chunk-order', 'reverse', '--executor', 'pool'], ['--chunk-order', 'shuffle:x']):
        with pytest.raises(SystemExit):
            cli.main(['--run-dir', str(run_dir), '--chunks', '3', '--out-dir', str(tmp_path / 'bad'), *PIWIND_ARGS, *argv])
    capsys.readouterr()


def test_inline_one_chain_in_scrambled_chunk_order_matches_fresh_chains_and_bash(kernel_run, kernel_reference, tmp_path, monkeypatch):
    """One chain running the chunks out of order writes the same bytes as a fresh chain per chunk, and as bash."""
    n_chunks = 5
    args = cli.build_parser().parse_args([
        '--run-dir', str(kernel_run.run_dir), '--out-dir', 'unused', '-S', str(kernel_run.samples), *kernel_run.gul_args,
        '--il-alloc-rule', str(kernel_run.il_alloc_rule), '--ri-layers', '1', '--ri-alloc-rule', str(kernel_run.ri_alloc_rule)])
    spec = cli.spec_from_args(args)
    chunks = make_chunks(kernel_run.event_ids, n_chunks)

    one_chain = tmp_path / 'one_chain'
    inline.run_chunks(spec, chunks, one_chain, TAPS, chunk_order=[4, 2, 5, 1, 3])

    fresh = tmp_path / 'fresh'
    monkeypatch.chdir(kernel_run.run_dir)
    for k, chunk in enumerate(chunks, 1):
        with Chain(spec) as chain, ChunkWriter(fresh, k, chain.headers, TAPS) as writer:
            for event_id in chunk:
                chain.run_event(event_id, writer.sinks)

    for expected in (fresh, kernel_reference(n_chunks)):
        results = compare_dirs_bytes(expected, one_chain, taps=TAPS)
        assert len(results) == n_chunks * len(TAPS)
        assert all(r.ok for r in results), [r for r in results if not r.ok]


def test_inline_tree_writes_chunk_files_and_taps(tmp_path, run_dir):
    events = np.arange(1, 31, dtype=np.int32)
    chunks = make_chunks(events, 3) + [np.array([], dtype=np.int32)]
    out_dir = tmp_path / 'out'
    inline.run_chunks(make_tree_spec(str(run_dir), n_ri=1), chunks, out_dir, TAPS, factories=TREE_FACTORIES, chunk_order=[4, 2, 1, 3])
    check_outputs(out_dir, chunks, fake_headers())
    for k, chunk in enumerate(chunks, 1):
        assert read_tree_files(run_dir, k) == expected_tree_files([int(e) for e in chunk], n_ri=1, k=k), k
