import os
import subprocess
import sys

import numpy as np
import pytest

from oasislmf.execution.orchestrator.chunks import ChunkWriter, chunk_path, make_chunks
from oasislmf.pytools.eve.manager import event_id_dtype


@pytest.fixture
def events_file(tmp_path):
    rng = np.random.default_rng(42)
    events = rng.choice(np.arange(1, 100_000, dtype=event_id_dtype), size=103, replace=False)
    path = tmp_path / 'events.bin'
    events.tofile(path)
    return path, events


def evepy_events(events_file, k, n):
    out = subprocess.run([sys.executable, '-m', 'oasislmf.pytools.eve.cli', str(k), str(n), '-i', str(events_file)],
                         check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.int32)


@pytest.mark.parametrize('n_chunks', [1, 4, 12, 150])
def test_make_chunks_matches_evepy(events_file, n_chunks):
    path, events = events_file
    chunks = make_chunks(events, n_chunks)
    assert len(chunks) == n_chunks
    for k in sorted({k for k in (1, 2, n_chunks // 2, n_chunks) if 1 <= k <= n_chunks}):
        np.testing.assert_array_equal(chunks[k - 1], evepy_events(path, k, n_chunks))


def test_make_chunks_partitions_all_events(events_file):
    _, events = events_file
    chunks = make_chunks(events, 7)
    joined = np.concatenate(chunks)
    assert len(joined) == len(events)
    np.testing.assert_array_equal(np.sort(joined), np.sort(events))
    assert all(chunk.dtype == event_id_dtype for chunk in chunks)


def test_make_chunks_accepts_list_and_rejects_zero():
    assert [c.tolist() for c in make_chunks([5, 6, 7, 8, 9], 2)] == [[5, 7, 9], [6, 8]]
    with pytest.raises(ValueError):
        make_chunks([1, 2], 0)


def test_chunk_path():
    assert chunk_path('out', 'il', 3) == os.path.join('out', 'il_P3.bin')


def test_chunk_writer_writes_header_and_blocks(tmp_path):
    out_dir = tmp_path / 'out'
    headers = {'gul': b'GH', 'il': b'IH', 'ri': b'RH'}
    with ChunkWriter(out_dir, 2, headers, ['gul', 'il']) as writer:
        writer.sinks['gul'].write(np.frombuffer(b'g1', dtype=np.int8))
        writer.write('gul', b'g2')
        writer.write('il', memoryview(b'i1'))
    assert sorted(os.listdir(out_dir)) == ['gul_P2.bin', 'il_P2.bin']
    assert (out_dir / 'gul_P2.bin').read_bytes() == b'GHg1g2'
    assert (out_dir / 'il_P2.bin').read_bytes() == b'IHi1'


def test_chunk_writer_empty_chunk_gets_header_only(tmp_path):
    with ChunkWriter(tmp_path, 1, {'ri': b'header'}, ['ri']):
        pass
    assert (tmp_path / 'ri_P1.bin').read_bytes() == b'header'


def test_chunk_writer_unknown_tap_closes_opened_files(tmp_path):
    with pytest.raises(KeyError):
        ChunkWriter(tmp_path, 1, {'gul': b'G'}, ['gul', 'il'])
    assert (tmp_path / 'gul_P1.bin').read_bytes() == b'G'
