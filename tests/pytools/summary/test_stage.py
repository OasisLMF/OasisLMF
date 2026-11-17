"""SummaryStage fed in-memory blocks must write exactly what summarypy writes for the same input."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.common.event_stream import FM_STREAM_ID, GUL_STREAM_ID, ITEM_STREAM, SUMMARY_STREAM_ID, stream_info_to_bytes
from oasislmf.pytools.summary import manager
from oasislmf.pytools.summary.stage import SummaryStage

ASSETS = Path(__file__).parent.parent.parent / 'assets' / 'test_summarypy'
STATIC = ASSETS / 'multiple_summary_set'

RUN_TYPES = {
    'gul': (STATIC, (1, 2)),
    'il': (STATIC, (1, 2, 3)),
    'ri': (STATIC / 'RI_1', (1, 2)),
}


def split_events(body):
    """Cut a loss stream body at its event boundaries."""
    i32 = np.frombuffer(body, dtype=np.int32)
    bounds, pos, last = [0], 0, 0
    while pos < i32.shape[0]:
        if last and i32[pos] != last:
            bounds.append(pos * 4)
        last = i32[pos]
        pos += 2
        while i32[pos] != 0:
            pos += 2
        pos += 2
    bounds.append(len(body))
    return [body[start:end] for start, end in zip(bounds[:-1], bounds[1:])]


def read_input(run_type):
    raw = (ASSETS / f'{run_type}.bin').read_bytes()
    return raw[:8], raw[8:]


def run_cli(tmp_path, run_type, sets, low_memory=False, output_zeros=False, ext='bin'):
    """Run summarypy (manager.run) and return the bytes of each set's output (and .idx when low_memory)."""
    static, _ = RUN_TYPES[run_type]
    paths = {s: tmp_path / f'cli_S{s}.{ext}' for s in sets}
    manager.run(files_in=[str(ASSETS / f'{run_type}.bin')], static_path=str(static), run_type=run_type,
                low_memory=low_memory, output_zeros=output_zeros, output_format=ext,
                summary_sets_output=[a for s, p in paths.items() for a in (f'-{s}', str(p))])
    outs = {s: p.read_bytes() for s, p in paths.items()}
    idxs = {s: p.with_suffix('.idx').read_bytes() for s, p in paths.items()} if low_memory else None
    return outs, idxs


def run_stage(run_type, sets, blocks, low_memory=False, output_zeros=False, stage=None):
    """Feed ``blocks`` to a SummaryStage, returning each set's header + output and its .idx bytes."""
    header, _ = read_input(run_type)
    static, _ = RUN_TYPES[run_type]
    if stage is None:
        stage = SummaryStage(header, str(static), run_type, sets, low_memory=low_memory, output_zeros=output_zeros)
    outs = {s: io.BytesIO() for s in sets}
    idx_outs = {s: io.BytesIO() for s in sets}
    for s in sets:
        outs[s].write(stage.headers[s])
    for block in blocks:
        stage.process_block(block, outs, idx_outs if low_memory else None)
    return {s: o.getvalue() for s, o in outs.items()}, ({s: o.getvalue() for s, o in idx_outs.items()} if low_memory else None)


def whole_block(run_type):
    _, body = read_input(run_type)
    return [np.frombuffer(body, dtype=np.int8)]


def per_event_blocks(run_type):
    _, body = read_input(run_type)
    return [np.frombuffer(event, dtype=np.int8) for event in split_events(body)]


@pytest.mark.parametrize('run_type', ['gul', 'il', 'ri'])
@pytest.mark.parametrize('low_memory, output_zeros', [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('blocks', ['one', 'per_event'])
def test_stage_matches_cli(tmp_path, run_type, low_memory, output_zeros, blocks):
    sets = RUN_TYPES[run_type][1]
    expected, expected_idx = run_cli(tmp_path, run_type, sets, low_memory, output_zeros)
    block_list = whole_block(run_type) if blocks == 'one' else per_event_blocks(run_type)
    assert len(per_event_blocks(run_type)) > 1
    got, got_idx = run_stage(run_type, sets, block_list, low_memory, output_zeros)
    assert got == expected
    assert got_idx == expected_idx


@pytest.mark.parametrize('run_type', ['gul', 'il', 'ri'])
def test_requested_sets_subset_and_order(tmp_path, run_type):
    sets = tuple(reversed(RUN_TYPES[run_type][1]))[:2]
    expected, _ = run_cli(tmp_path, run_type, sets)
    got, _ = run_stage(run_type, sets, whole_block(run_type))
    assert got == expected


@pytest.mark.parametrize('run_type', ['gul', 'il', 'ri'])
def test_read_only_blocks(run_type):
    sets = RUN_TYPES[run_type][1]
    expected, expected_idx = run_stage(run_type, sets, per_event_blocks(run_type), low_memory=True)
    read_only = []
    for block in per_event_blocks(run_type):
        block = block.copy()
        block.setflags(write=False)
        read_only.append(block)
    got, got_idx = run_stage(run_type, sets, read_only, low_memory=True)
    assert got == expected
    assert got_idx == expected_idx


@pytest.mark.parametrize('run_type', ['gul', 'il', 'ri'])
def test_no_cross_event_state(run_type):
    """Each event's bytes are the same alone in a fresh stage as after any other events in the same stage."""
    sets = RUN_TYPES[run_type][1]
    header, _ = read_input(run_type)
    static, _ = RUN_TYPES[run_type]
    events = per_event_blocks(run_type)

    alone = []
    for block in events:
        stage = SummaryStage(header, str(static), run_type, sets)
        outs = {s: io.BytesIO() for s in sets}
        stage.process_block(block, outs)
        alone.append({s: o.getvalue() for s, o in outs.items()})
    assert any(any(event_out.values()) for event_out in alone)

    stage = SummaryStage(header, str(static), run_type, sets)
    for order in (range(len(events)), reversed(range(len(events)))):
        for i in order:
            outs = {s: io.BytesIO() for s in sets}
            stage.process_block(events[i], outs)
            assert {s: o.getvalue() for s, o in outs.items()} == alone[i], f'event block {i}'


@pytest.mark.parametrize('run_type', ['gul', 'il', 'ri'])
def test_reset_restarts_index_offsets(run_type):
    """After reset() a low-memory stage writes the .idx of a new file exactly as a fresh stage does."""
    sets = RUN_TYPES[run_type][1]
    events = per_event_blocks(run_type)
    first, second = events[:len(events) // 2], events[len(events) // 2:]
    fresh = run_stage(run_type, sets, second, low_memory=True)

    header, _ = read_input(run_type)
    stage = SummaryStage(header, str(RUN_TYPES[run_type][0]), run_type, sets, low_memory=True)
    run_stage(run_type, sets, first, low_memory=True, stage=stage)
    without_reset = run_stage(run_type, sets, second, low_memory=True, stage=stage)
    assert without_reset[0] == fresh[0]
    assert without_reset[1] != fresh[1]
    stage.reset()
    run_stage(run_type, sets, first, low_memory=True, stage=stage)
    stage.reset()
    assert run_stage(run_type, sets, second, low_memory=True, stage=stage) == fresh


@pytest.mark.parametrize('ext', ['csv', 'parquet'])
def test_row_writers(tmp_path, ext):
    """process_block with row writers writes the same SELT table as summarypy -E csv/parquet."""
    run_type, (static, sets) = 'il', RUN_TYPES['il']
    expected, _ = run_cli(tmp_path, run_type, sets, ext=ext)
    header, _ = read_input(run_type)
    stage = SummaryStage(header, str(static), run_type, sets)
    with manager.ExitStack() as stack:
        writers = {s: manager.SummaryRowWriter(stack, str(tmp_path / f'stage_S{s}.{ext}'), ext) for s in sets}
        for block in per_event_blocks(run_type):
            stage.process_block(block, None, row_writers=writers)
        for writer in writers.values():
            writer.flush()
    for s in sets:
        if ext == 'csv':
            assert (tmp_path / f'stage_S{s}.{ext}').read_bytes() == expected[s]
        else:
            pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / f'stage_S{s}.{ext}'), pd.read_parquet(tmp_path / f'cli_S{s}.{ext}'))


def test_headers():
    header, _ = read_input('il')
    stage = SummaryStage(header, str(STATIC), 'il', (2, 1))
    len_sample = np.frombuffer(header[4:], dtype=np.int32)[0]
    for s in (1, 2):
        assert stage.headers[s] == stream_info_to_bytes(SUMMARY_STREAM_ID, ITEM_STREAM) + len_sample.tobytes() + np.int32(s).tobytes()
    with pytest.raises(ValueError, match='each with its own header'):
        stage.header
    assert SummaryStage(header, str(STATIC), 'il', [3]).header == stage.headers[1][:8] + np.int32(3).tobytes()


@pytest.mark.parametrize('header, error', [
    (stream_info_to_bytes(SUMMARY_STREAM_ID, ITEM_STREAM) + np.int32(10).tobytes(), 'unsupported stream type'),
    (stream_info_to_bytes(GUL_STREAM_ID, 7) + np.int32(10).tobytes(), 'unsupported stream aggregation type'),
    (stream_info_to_bytes(FM_STREAM_ID, ITEM_STREAM), 'must be 8 bytes'),
])
def test_invalid_input_header(header, error):
    with pytest.raises(Exception, match=error):
        SummaryStage(header, str(STATIC), 'il', [1])


def test_unknown_summary_set():
    header, _ = read_input('gul')
    with pytest.raises(ValueError, match='not found in summaryxref'):
        SummaryStage(header, str(STATIC), 'gul', [3])


def test_low_memory_requires_idx_outs():
    header, body = read_input('gul')
    stage = SummaryStage(header, str(STATIC), 'gul', [1], low_memory=True)
    with pytest.raises(ValueError, match='idx_outs'):
        stage.process_block(np.frombuffer(body, dtype=np.int8), {1: io.BytesIO()})


def test_unsupported_run_type():
    header, _ = read_input('gul')
    with pytest.raises(Exception, match='not in supported list'):
        SummaryStage(header, str(STATIC), 'xx', [1])


def test_binary_output_requires_outs():
    header, body = read_input('gul')
    stage = SummaryStage(header, str(STATIC), 'gul', [1])
    with pytest.raises(ValueError, match='outs is required'):
        stage.process_block(np.frombuffer(body, dtype=np.int8), None)
