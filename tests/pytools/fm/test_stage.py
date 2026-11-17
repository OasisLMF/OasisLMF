"""FmStage fed in-memory blocks must write exactly what fmpy (run_synchronous) writes for the same input."""
import io
import json
import os
import tempfile

import numpy as np
import pytest

from oasislmf.pytools.common.event_stream import LOSS_STREAM_ID, PIPE_CAPACITY, SUMMARY_STREAM_ID, ITEM_STREAM, stream_info_to_bytes
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.fm import manager
from oasislmf.pytools.fm import stage as fm_stage
from oasislmf.pytools.fm.stage import FmStage

# two items in one aggregation, one level, two layers limited to 3000 and 1000
FM_PROGRAMME = """from_agg_id,level_id,to_agg_id
1,1,1
2,1,1
"""

FM_POLICYTC = """level_id,agg_id,layer_id,profile_id
1,1,1,2
1,1,2,3
"""

FM_PROFILE = """profile_id,calcrule_id,deductible1,deductible2,deductible3,attachment1,limit1,share1,share2,share3
1,100,0,0,0,0,0,0,0,0
2,14,0,0,0,0,3000,0,0,0
3,14,0,0,0,0,1000,0,0,0
"""

FM_XREF = """output,agg_id,layer_id
1,1,1
2,1,2
3,2,1
4,2,2
"""


class CountingSink(io.BytesIO):
    """BytesIO that counts write calls."""

    def __init__(self):
        super().__init__()
        self.n_writes = 0

    def write(self, data):
        self.n_writes += 1
        return super().write(data)


def write_gul(path, n_events, n_samples):
    rng = np.random.default_rng(n_events * 100 + n_samples)
    rows = ['event_id,item_id,sidx,loss']
    for event_id in range(1, n_events + 1):
        for item_id in (1, 2):
            rows.append(f'{event_id},{item_id},-3,5000.0')
            rows.append(f'{event_id},{item_id},-1,{rng.uniform(0, 4000):.2f}')
            for sidx in range(1, n_samples + 1):
                rows.append(f'{event_id},{item_id},{sidx},{rng.uniform(0, 4000):.2f}')
    (path / 'guls.csv').write_text('\n'.join(rows) + '\n')
    csvtobin(path / 'guls.csv', path / 'guls.bin', 'gul', stream_type=LOSS_STREAM_ID, max_sample_index=n_samples)
    return path / 'guls.bin'


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


@pytest.fixture
def fm_dir(tmp_path):
    for name, content in [('fm_programme', FM_PROGRAMME), ('fm_policytc', FM_POLICYTC),
                          ('fm_profile', FM_PROFILE), ('fm_xref', FM_XREF)]:
        (tmp_path / f'{name}.csv').write_text(content)
    for allocation_rule in (0, 1, 2):
        manager.run(create_financial_structure_files=True, allocation_rule=allocation_rule, static_path=str(tmp_path))
    return tmp_path


def run_fmpy(fm_dir, gul_path, allocation_rule, net, sort_output=False, low_memory=False):
    """Run fmpy (run_synchronous) writing gross to one file and, if net, net to another."""
    gross_path, net_path = fm_dir / 'gross.bin', fm_dir / 'net.bin'
    manager.run(create_financial_structure_files=False, allocation_rule=allocation_rule,
                static_path=str(fm_dir), files_in=[str(gul_path)], files_out=[str(gross_path)],
                net_loss=str(net_path) if net else None, storage_method='sparse', low_memory=low_memory,
                sort_output=sort_output)
    return gross_path.read_bytes(), net_path.read_bytes() if net else None


def run_stage(fm_dir, gul_path, allocation_rule, net, blocks='one', sort_output=False, low_memory=False):
    raw = gul_path.read_bytes()
    body = raw[8:]
    if blocks == 'one':
        block_list = [np.frombuffer(body, dtype=np.int8)]
    else:
        block_list = [np.frombuffer(event, dtype=np.int8) for event in split_events(body)]
    gross_out = CountingSink()
    net_out = CountingSink() if net else None
    with FmStage(raw[:8], allocation_rule, str(fm_dir), gross=True, net=net,
                 sort_output=sort_output, low_memory=low_memory) as stage:
        gross_out.write(stage.header)
        if net:
            net_out.write(stage.header)
        for block in block_list:
            stage.process_block(block, gross_out=gross_out, net_out=net_out)
    return gross_out, net_out


@pytest.mark.parametrize('blocks', ['one', 'per_event'])
@pytest.mark.parametrize('allocation_rule, net, sort_output, low_memory', [
    (0, False, False, False),
    (1, False, False, False),
    (2, False, False, False),
    (3, False, True, False),
    (1, True, False, False),
    (2, True, True, False),
    (3, True, False, True),
])
def test_stage_matches_fmpy(fm_dir, blocks, allocation_rule, net, sort_output, low_memory):
    gul_path = write_gul(fm_dir, n_events=4, n_samples=5)
    expected_gross, expected_net = run_fmpy(fm_dir, gul_path, allocation_rule, net, sort_output, low_memory)

    gross_out, net_out = run_stage(fm_dir, gul_path, allocation_rule, net, blocks, sort_output, low_memory)

    assert len(expected_gross) > 8
    assert gross_out.getvalue() == expected_gross
    if net:
        assert len(expected_net) > 8
        assert net_out.getvalue() == expected_net


def test_stage_writes_event_larger_than_writer_buffer_in_pieces(fm_dir):
    gul_path = write_gul(fm_dir, n_events=2, n_samples=6000)
    expected_gross, expected_net = run_fmpy(fm_dir, gul_path, 2, True)

    gross_out, net_out = run_stage(fm_dir, gul_path, 2, True, blocks='per_event')

    assert len(expected_gross) > 2 * PIPE_CAPACITY
    assert gross_out.n_writes > 1 + 2
    assert gross_out.getvalue() == expected_gross
    assert net_out.getvalue() == expected_net


def test_stage_net_only(fm_dir):
    gul_path = write_gul(fm_dir, n_events=3, n_samples=4)
    _, expected_net = run_fmpy(fm_dir, gul_path, 2, True)
    raw = gul_path.read_bytes()

    net_out = io.BytesIO()
    with FmStage(raw[:8], 2, str(fm_dir), gross=False, net=True) as stage:
        net_out.write(stage.header)
        stage.process_block(np.frombuffer(raw[8:], dtype=np.int8), net_out=net_out)
    assert net_out.getvalue() == expected_net


def test_stage_header_is_loss_stream_header(fm_dir):
    gul_path = write_gul(fm_dir, n_events=1, n_samples=7)
    raw = gul_path.read_bytes()
    with FmStage(raw[:8], 2, str(fm_dir)) as stage:
        assert stage.header == stream_info_to_bytes(LOSS_STREAM_ID, ITEM_STREAM) + np.int32(7).tobytes()


def test_stage_rejects_unsupported_input(fm_dir):
    good = stream_info_to_bytes(LOSS_STREAM_ID, ITEM_STREAM) + np.int32(3).tobytes()
    with pytest.raises(Exception, match='unsupported stream_type'):
        FmStage(stream_info_to_bytes(SUMMARY_STREAM_ID, ITEM_STREAM) + np.int32(3).tobytes(), 2, str(fm_dir))
    with pytest.raises(Exception, match='unsupported stream aggregation type'):
        FmStage(stream_info_to_bytes(LOSS_STREAM_ID, 7) + np.int32(3).tobytes(), 2, str(fm_dir))
    with pytest.raises(NotImplementedError):
        FmStage(good, 0, str(fm_dir), net=True)
    with pytest.raises(ValueError):
        FmStage(good, 2, str(fm_dir), storage_method='dense')


def test_stage_refuses_output_not_requested(fm_dir):
    gul_path = write_gul(fm_dir, n_events=1, n_samples=2)
    raw = gul_path.read_bytes()
    block = np.frombuffer(raw[8:], dtype=np.int8)
    with FmStage(raw[:8], 2, str(fm_dir), gross=True, net=False) as stage:
        with pytest.raises(ValueError):
            stage.process_block(block, net_out=io.BytesIO())
    with FmStage(raw[:8], 2, str(fm_dir), gross=False, net=True) as stage:
        with pytest.raises(ValueError):
            stage.process_block(block, gross_out=io.BytesIO())


def test_reset_clears_chance_of_loss_carried_from_earlier_events(fm_dir):
    """A -4 (chance of loss) record is kept by fm for later events; reset() must make the stage fresh again."""
    rows = ['event_id,item_id,sidx,loss',
            '1,1,-4,0.5', '1,1,-3,5000.0', '1,1,-1,100.0', '1,1,1,200.0',
            '1,2,-4,0.25', '1,2,-3,5000.0', '1,2,-1,300.0', '1,2,1,400.0']
    (fm_dir / 'first.csv').write_text('\n'.join(rows) + '\n')
    (fm_dir / 'second.csv').write_text('\n'.join(['event_id,item_id,sidx,loss', '2,1,-3,5000.0', '2,1,-1,100.0', '2,1,1,200.0',
                                                  '2,2,-3,5000.0', '2,2,-1,300.0', '2,2,1,400.0']) + '\n')
    blocks = {}
    for name in ('first', 'second'):
        csvtobin(fm_dir / f'{name}.csv', fm_dir / f'{name}.bin', 'gul', stream_type=LOSS_STREAM_ID, max_sample_index=1)
        raw = (fm_dir / f'{name}.bin').read_bytes()
        header, blocks[name] = raw[:8], np.frombuffer(raw[8:], dtype=np.int8)

    def second_event_output(first, do_reset):
        out = io.BytesIO()
        with FmStage(header, 2, str(fm_dir)) as stage:
            if first:
                stage.process_block(blocks['first'], gross_out=io.BytesIO())
            if do_reset:
                stage.reset()
            stage.process_block(blocks['second'], gross_out=out)
        return out.getvalue()

    fresh = second_event_output(first=False, do_reset=False)
    assert second_event_output(first=True, do_reset=False) != fresh
    assert second_event_output(first=True, do_reset=True) == fresh


def test_stage_needs_gross_or_net_and_cleans_up_a_failed_setup(fm_dir, tmp_path, monkeypatch):
    header = stream_info_to_bytes(LOSS_STREAM_ID, ITEM_STREAM) + np.int32(3).tobytes()
    with pytest.raises(ValueError, match='at least one of gross or net'):
        FmStage(header, 2, str(fm_dir), gross=False, net=False)
    made = []
    real_tempdir = tempfile.TemporaryDirectory

    def recording_tempdir(*args, **kwargs):
        tempdir = real_tempdir(*args, **kwargs)
        made.append(tempdir.name)
        return tempdir

    monkeypatch.setattr(tempfile, 'TemporaryDirectory', recording_tempdir)
    with pytest.raises(OSError):
        FmStage(header, 2, str(tmp_path / 'missing'))
    assert len(made) == 1 and not os.path.exists(made[0])


def test_stage_debug_profile_is_written_on_close(fm_dir, monkeypatch):
    profile = fm_dir / 'profile.tsv'
    monkeypatch.setattr(fm_stage, 'DEBUG_PROFILE', True)
    monkeypatch.setenv('FM_PROFILE_OUT', str(profile))
    raw = write_gul(fm_dir, n_events=3, n_samples=2).read_bytes()
    with FmStage(raw[:8], 2, str(fm_dir)) as stage:
        stage.process_block(np.frombuffer(raw[8:], dtype=np.int8), gross_out=io.BytesIO())
    assert profile.read_text().splitlines()[0].startswith('PHASE\tread=') and 'events=3' in profile.read_text()


def test_stage_failure_writes_event_error_and_no_profile(fm_dir, monkeypatch):
    monkeypatch.chdir(fm_dir)
    profile = fm_dir / 'profile.tsv'
    monkeypatch.setattr(fm_stage, 'DEBUG_PROFILE', True)
    monkeypatch.setenv('FM_PROFILE_OUT', str(profile))
    real_compute = fm_stage.compute_event_sparse
    calls = []

    def failing_compute(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ZeroDivisionError('boom')
        return real_compute(*args)

    monkeypatch.setattr(fm_stage, 'compute_event_sparse', failing_compute)
    raw = write_gul(fm_dir, n_events=3, n_samples=2).read_bytes()
    with FmStage(raw[:8], 2, str(fm_dir)) as stage:
        with pytest.raises(ZeroDivisionError):
            stage.process_block(np.frombuffer(raw[8:], dtype=np.int8), gross_out=io.BytesIO())
    error = json.loads((fm_dir / 'event_error.json').read_text())
    assert str(error['event_id']) == '2' and error['event_index'] == 1
    assert not profile.exists()
