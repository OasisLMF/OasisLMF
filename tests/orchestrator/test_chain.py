import io
import os
import subprocess
import sys

import numpy as np
import pytest

from oasislmf.execution.orchestrator.chain import Chain, ChainSpec, build_chain_spec, summary_set_header

EMPTY_GUL_EVENTS = (5, 10)


class FakeGul:
    """Writes the int32 pair (event_id, 10 * event_id) in two writes; nothing for EMPTY_GUL_EVENTS."""

    def __init__(self, run_dir, sample_size=1, **kwargs):
        self.run_dir = run_dir
        self.kwargs = kwargs
        self.header = np.array([1, sample_size], dtype=np.int32).tobytes()
        self.events = []
        self.cwd = []
        self.closed = False

    def process_event(self, event_id, out):
        self.events.append(int(event_id))
        self.cwd.append(os.getcwd())
        if event_id in EMPTY_GUL_EVENTS:
            return
        out.write(np.array([event_id], dtype=np.int32))
        out.write(np.array([10 * event_id], dtype=np.int32).tobytes())

    def close(self):
        self.closed = True


class FakeFm:
    """gross = values * 2 + rule, net = values * 3 + rule; writes nothing for an input whose first value is 7 (mod 100)."""

    def __init__(self, input_header, allocation_rule, static_path='input', gross=True, net=False, **kwargs):
        self.input_header = input_header
        self.allocation_rule = allocation_rule
        self.static_path = static_path
        self.gross = gross
        self.net = net
        self.kwargs = kwargs
        self.header = np.array([2, allocation_rule], dtype=np.int32).tobytes() + static_path.encode()
        self.calls = []
        self.closed = False

    def process_block(self, block, gross_out=None, net_out=None):
        assert block.dtype == np.int8
        assert gross_out is None or self.gross
        assert net_out is None or self.net
        self.calls.append((block.tobytes(), gross_out is not None, net_out is not None))
        values = np.frombuffer(block.tobytes(), dtype=np.int32)
        if values[0] % 100 == 7:
            return
        if gross_out is not None:
            gross_out.write(values * 2 + self.allocation_rule)
        if net_out is not None:
            net_out.write(values * 3 + self.allocation_rule)

    def close(self):
        self.closed = True


FACTORIES = {'gul': FakeGul, 'fm': FakeFm}


def expected_streams(event_ids, il_rule=2, ri_rules=(3,)):
    """Reference implementation of the fake pipeline gul | fm(il) | fm(ri_1) -n | ... for each tap."""
    out = {'gul': b'', 'il': b'', 'ri': b''}
    for e in event_ids:
        if e in EMPTY_GUL_EVENTS:
            continue
        values = np.array([e, 10 * e], dtype=np.int32)
        out['gul'] += values.tobytes()
        if values[0] % 100 == 7:
            continue
        values = values * 2 + il_rule
        out['il'] += values.tobytes()
        ri_bytes = b''
        for rule in ri_rules:
            if values[0] % 100 == 7:
                ri_bytes = b''
                break
            values = values * 3 + rule
            ri_bytes = values.tobytes()
        out['ri'] += ri_bytes
    return out


def make_spec(run_dir='.', n_ri=1):
    il = {'allocation_rule': 2, 'static_path': 'input'}
    ri = [{'allocation_rule': 3, 'static_path': f'input/RI_{i}'} for i in range(1, n_ri + 1)]
    return build_chain_spec(run_dir, {'sample_size': 10, 'loss_threshold': 0.0}, il=il, ri=ri)


def run(chain, event_ids, taps):
    sinks = {tap: io.BytesIO() for tap in taps}
    for e in event_ids:
        chain.run_event(e, sinks)
    return {tap: sink.getvalue() for tap, sink in sinks.items()}


def test_build_chain_spec_wiring():
    spec = make_spec('/run', n_ri=2)
    assert spec.run_dir == '/run'
    assert [(s['name'], s['feeds_next'], s['static_path']) for s in spec.fm] == [
        ('il', 'gross', 'input'), ('ri', 'net', 'input/RI_1'), ('ri', None, 'input/RI_2')]
    assert build_chain_spec('/run', {}, il={'allocation_rule': 2}).fm == [{'allocation_rule': 2, 'name': 'il', 'feeds_next': None}]


def test_chain_spec_json_round_trip():
    spec = make_spec('/some/run', n_ri=2)
    spec.gul['ignore_file_type'] = ['bin']
    spec.gul['peril_filter'] = []
    restored = ChainSpec.from_json(spec.to_json())
    assert restored == spec
    assert ChainSpec.from_dict(spec.to_dict()) == spec


def test_chain_builds_stages_with_upstream_headers():
    chain = Chain(make_spec('/run', n_ri=2), factories=FACTORIES)
    assert chain.gul.run_dir == '/run'
    assert chain.gul.kwargs == {'loss_threshold': 0.0}
    il, ri1, ri2 = chain.fm_steps
    assert il.input_header == chain.gul.header
    assert ri1.input_header == il.header
    assert ri2.input_header == ri1.header
    assert [s.static_path for s in chain.fm_steps] == ['input', 'input/RI_1', 'input/RI_2']
    assert all('name' not in s.kwargs and 'feeds_next' not in s.kwargs for s in chain.fm_steps)
    assert [(s.gross, s.net) for s in chain.fm_steps] == [(True, False), (False, True), (False, True)]
    assert chain.headers == {'gul': chain.gul.header, 'il': il.header, 'ri': ri2.header}


def test_chain_explicit_gross_net_and_run_dir_override():
    spec = make_spec('/run')
    spec.gul['run_dir'] = '/elsewhere'
    spec.fm[0]['net'] = True
    chain = Chain(spec, factories=FACTORIES)
    assert chain.gul.run_dir == '/elsewhere'
    assert (chain.fm_steps[0].gross, chain.fm_steps[0].net) == (True, True)


@pytest.mark.parametrize('taps', [('gul',), ('il',), ('ri',), ('gul', 'il'), ('gul', 'ri'), ('il', 'ri'), ('gul', 'il', 'ri')])
@pytest.mark.parametrize('n_ri', [1, 2])
def test_chain_tap_bytes(taps, n_ri):
    event_ids = [1, 2, 5, 3, 7, 107, 10, 4]
    chain = Chain(make_spec(n_ri=n_ri), factories=FACTORIES)
    got = run(chain, event_ids, taps)
    expected = expected_streams(event_ids, ri_rules=(3,) * n_ri)
    assert got == {tap: expected[tap] for tap in taps}


def test_chain_requests_only_needed_outputs():
    chain = Chain(make_spec(n_ri=2), factories=FACTORIES)
    il, ri1, ri2 = chain.fm_steps

    run(chain, [1], ['gul'])
    assert il.calls == [] and ri1.calls == [] and ri2.calls == []

    run(chain, [1], ['il'])
    assert il.calls == [(np.array([1, 10], dtype=np.int32).tobytes(), True, False)]
    assert ri1.calls == [] and ri2.calls == []

    run(chain, [2], ['ri'])
    gul_bytes = np.array([2, 20], dtype=np.int32).tobytes()
    il_bytes = (np.array([2, 20], dtype=np.int32) * 2 + 2).tobytes()
    ri1_bytes = (np.array([6, 42], dtype=np.int32) * 3 + 3).tobytes()
    assert il.calls[-1] == (gul_bytes, True, False)
    assert ri1.calls == [(il_bytes, False, True)]
    assert ri2.calls == [(ri1_bytes, False, True)]


def test_chain_net_feed_requests_net_only():
    spec = build_chain_spec('.', {}, il={'allocation_rule': 2}, ri=[{'allocation_rule': 3}])
    spec.fm[0]['feeds_next'] = 'net'
    spec.fm[0]['net'] = True
    chain = Chain(spec, factories=FACTORIES)
    run(chain, [1], ['ri'])
    assert chain.fm_steps[0].calls == [(np.array([1, 10], dtype=np.int32).tobytes(), False, True)]
    run(chain, [1], ['il', 'ri'])
    assert chain.fm_steps[0].calls[-1][1:] == (True, True)


def test_chain_skips_fm_for_empty_gul_event():
    chain = Chain(make_spec(), factories=FACTORIES)
    got = run(chain, [5, 10], ['gul', 'il', 'ri'])
    assert chain.gul.events == [5, 10]
    assert chain.fm_steps[0].calls == []
    assert chain.fm_steps[1].calls == []
    assert got == {'gul': b'', 'il': b'', 'ri': b''}


def test_chain_skips_next_step_on_empty_fm_output_without_stale_bytes():
    chain = Chain(make_spec(), factories=FACTORIES)
    got = run(chain, [1, 7], ['gul', 'il', 'ri'])
    assert len(chain.fm_steps[0].calls) == 2
    assert len(chain.fm_steps[1].calls) == 1
    assert got == expected_streams([1, 7])


def test_chain_without_il_or_ri_rejects_taps():
    chain = Chain(build_chain_spec('.', {}, il={'allocation_rule': 2}), factories=FACTORIES)
    assert set(chain.headers) == {'gul', 'il'}
    with pytest.raises(ValueError, match="'ri'"):
        chain.run_event(1, {'ri': io.BytesIO()})
    with pytest.raises(ValueError, match='unknown tap'):
        chain.run_event(1, {'xx': io.BytesIO()})
    chain = Chain(build_chain_spec('.', {}), factories=FACTORIES)
    with pytest.raises(ValueError, match="'il'"):
        chain.run_event(1, {'il': io.BytesIO()})


def test_chain_rejects_missing_feeds_next():
    spec = make_spec()
    spec.fm[0]['feeds_next'] = None
    with pytest.raises(ValueError, match='feeds_next'):
        Chain(spec, factories=FACTORIES)


def test_chain_close_and_context_manager():
    with Chain(make_spec(), factories=FACTORIES) as chain:
        pass
    assert chain.gul.closed
    assert all(s.closed for s in chain.fm_steps)


def test_chain_factories_as_import_strings():
    factories = {'gul': f'{__name__}:FakeGul', 'fm': f'{__name__}:FakeFm'}
    chain = Chain(make_spec(), factories=factories)
    assert isinstance(chain.gul, FakeGul)
    assert all(isinstance(s, FakeFm) for s in chain.fm_steps)


def test_chain_module_does_not_import_real_stages():
    code = ("import sys, oasislmf.execution.orchestrator.chain; "
            "assert not {'oasislmf.execution.orchestrator.stages', 'oasislmf.pytools.gulmc.stage', 'oasislmf.pytools.fm.stage'} "
            "& set(sys.modules)")
    subprocess.run([sys.executable, '-c', code], check=True)


def test_chain_reset_calls_stages_that_have_reset():
    log = []

    class ResettingGul(FakeGul):
        def reset(self):
            log.append('gul')

    class ResettingFm(FakeFm):
        def reset(self):
            log.append(self.static_path)

    chain = Chain(make_spec(n_ri=2), factories={'gul': ResettingGul, 'fm': ResettingFm})
    chain.reset()
    assert log == ['gul', 'input', 'input/RI_1', 'input/RI_2']

    chain = Chain(make_spec(), factories={'gul': FakeGul, 'fm': ResettingFm})
    log.clear()
    chain.reset()
    assert log == ['input', 'input/RI_1']

    Chain(make_spec(), factories=FACTORIES).reset()


class FakeSummary:
    """headers[n] = b'SUM' + n; set n gets (values + n) as int32, nothing for an input whose first value is 8 (mod 100)."""

    def __init__(self, input_header, static_path, run_type, summary_set_ids, low_memory=False, output_zeros=False, **kwargs):
        self.input_header = input_header
        self.static_path = static_path
        self.run_type = run_type
        self.summary_set_ids = list(summary_set_ids)
        self.low_memory = low_memory
        self.headers = {n: b'SUM' + bytes([n]) for n in self.summary_set_ids}
        self.calls = []
        self.closed = False

    def process_block(self, block, outs, idx_outs=None):
        assert set(outs) == set(self.summary_set_ids)
        assert (idx_outs is not None) == self.low_memory
        values = np.frombuffer(block.tobytes(), dtype=np.int32)
        self.calls.append(values.tolist())
        if values[0] % 100 == 8:
            return
        for n, out in outs.items():
            out.write(values + n)
            if self.low_memory:
                idx_outs[n].write(np.int64(values[0] * 1000 + n).tobytes())

    def close(self):
        self.closed = True


class FakeTable:
    """Writes b'H' + kind + set id at start_chunk, then each block's int32 values * 100 + table flag count, per path."""

    kind = b'T'

    def __init__(self, run_dir, input_header, output_format='bin', noheader=False, **flags):
        self.run_dir = run_dir
        self.input_header = input_header
        self.flags = sorted(name for name, on in flags.items() if on)
        self.files = None
        self.log = []
        self.closed = False

    def reset(self):
        self.log.append('reset')

    def start_chunk(self, paths):
        assert sorted(paths) == self.flags
        self.log.append(('start', dict(paths)))
        self.files = {name: open(path, 'wb') for name, path in paths.items()}
        for f in self.files.values():
            f.write(b'H' + self.kind + self.input_header[3:4])

    def process_block(self, block):
        values = np.frombuffer(block.tobytes(), dtype=np.int32)
        for f in self.files.values():
            f.write(values * 100 + len(self.flags))

    def finish_chunk(self):
        self.log.append('finish')
        for f in self.files.values():
            f.close()
        self.files = None

    def close(self):
        self.closed = True


class FakeElt(FakeTable):
    kind = b'E'


class FakePlt(FakeTable):
    kind = b'P'


TREE_FACTORIES = {**FACTORIES, 'summary': FakeSummary, 'elt': FakeElt, 'plt': FakePlt}


def summary_node(runtype, step, output, set_ids=(1,), elt=True, plt=False, files=('palt',)):
    """A summary node spec writing work/<runtype>_S<n>_* files."""
    sets = []
    for n in set_ids:
        outputs = []
        if plt:
            outputs.append({'kind': 'plt', 'kwargs': {'splt': True}, 'paths': {'splt': f'work/kat/{runtype}_S{n}_plt_sample_P{{k}}'}})
        if elt:
            outputs.append({'kind': 'elt', 'kwargs': {'qelt': True, 'melt': True},
                            'paths': {'qelt': f'work/kat/{runtype}_S{n}_elt_quantile_P{{k}}', 'melt': f'work/kat/{runtype}_S{n}_elt_moment_P{{k}}'}})
        outputs += [{'kind': 'file', 'path': f'work/{runtype}_S{n}_summary_{name}/P{{k}}.bin'} for name in files]
        sets.append({'id': n, 'outputs': outputs})
    return {'runtype': runtype, 'inuring_priority': '', 'source': {'step': step, 'output': output},
            'stage': {'run_type': runtype, 'static_path': 'input', 'summary_set_ids': list(set_ids)}, 'sets': sets}


def make_tree_spec(run_dir='.', n_ri=2, rl=True):
    spec = make_spec(run_dir, n_ri=n_ri)
    spec.summaries = [summary_node('gul', None, None, plt=True), summary_node('il', 0, 'gross', set_ids=(1, 2), files=('palt', 'leccalc')),
                      summary_node('ri', n_ri, 'net', elt=False, plt=True)]
    if rl:
        spec.summaries.append(summary_node('rl', 1, 'gross', files=()))
    return spec


def fake_losses(event_ids, n_ri=2):
    """Per event, the gul, il, ri_<i> net and ri_<i> gross int32 values of the fake chain, None where nothing is written."""
    out = []
    for e in event_ids:
        streams = {}
        if e not in EMPTY_GUL_EVENTS:
            values = np.array([e, 10 * e], dtype=np.int32)
            streams['gul'] = values
            if values[0] % 100 != 7:
                values = values * 2 + 2
                streams['il'] = values
                for i in range(1, n_ri + 1):
                    if values[0] % 100 == 7:
                        break
                    streams[f'rl{i}'] = values * 2 + 3
                    values = values * 3 + 3
                    streams[f'ri{i}'] = values
        out.append(streams)
    return out


def expected_tree_files(event_ids, n_ri=2, k=1):
    """Bytes of every file chunk k of make_tree_spec writes for event_ids."""
    files = {}
    losses = fake_losses(event_ids, n_ri)
    for runtype, key, set_ids, elt, plt, names in [('gul', 'gul', (1,), True, True, ('palt',)), ('il', 'il', (1, 2), True, False, ('palt', 'leccalc')),
                                                   ('ri', f'ri{n_ri}', (1,), False, True, ('palt',)), ('rl', 'rl1', (1,), True, False, ())]:
        for n in set_ids:
            blocks = b''.join((s[key] + n).tobytes() for s in losses if key in s and s[key][0] % 100 != 8)
            values = np.frombuffer(blocks, dtype=np.int32)
            if plt:
                files[f'work/kat/{runtype}_S{n}_plt_sample_P{k}'] = b'HP' + bytes([n]) + (values * 100 + 1).tobytes()
            if elt:
                for table in ('quantile', 'moment'):
                    files[f'work/kat/{runtype}_S{n}_elt_{table}_P{k}'] = b'HE' + bytes([n]) + (values * 100 + 2).tobytes()
            for name in names:
                files[f'work/{runtype}_S{n}_summary_{name}/P{k}.bin'] = b'SUM' + bytes([n]) + blocks
    return files


def read_tree_files(root, k=1):
    root = str(root)
    found = {}
    for dirpath, _dirs, names in os.walk(os.path.join(root, 'work')):
        for name in names:
            path = os.path.relpath(os.path.join(dirpath, name), root)
            if path.endswith(f'_P{k}') or path.endswith(f'/P{k}.bin'):
                with open(os.path.join(root, path), 'rb') as f:
                    found[path] = f.read()
    return found


def test_chain_tree_writes_every_summary_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    event_ids = [1, 2, 5, 3, 7, 107, 8, 10, 4, 108]
    with Chain(make_tree_spec(), factories=TREE_FACTORIES) as chain:
        chain.start_chunk(1)
        taps = run(chain, event_ids, ['gul', 'il', 'ri'])
        chain.finish_chunk()
    assert read_tree_files(tmp_path) == expected_tree_files(event_ids)
    assert taps == expected_streams(event_ids, ri_rules=(3, 3))


def test_chain_tree_builds_stages_with_headers_and_outputs(tmp_path):
    chain = Chain(make_tree_spec(str(tmp_path)), factories=TREE_FACTORIES)
    il, ri1, ri2 = chain.fm_steps
    assert [(s.gross, s.net) for s in chain.fm_steps] == [(True, False), (True, True), (False, True)]
    gul_node, il_node, ri_node, rl_node = chain.summary_nodes
    assert gul_node.stage.input_header == chain.gul.header
    assert il_node.stage.input_header == il.header and il_node.stage.summary_set_ids == [1, 2]
    assert ri_node.stage.input_header == ri2.header and rl_node.stage.input_header == ri1.header
    table, paths = il_node.tables[2][0]
    assert isinstance(table, FakeElt) and table.run_dir == str(tmp_path) and table.input_header == b'SUM\x02'
    assert paths == {'qelt': 'work/kat/il_S2_elt_quantile_P{k}', 'melt': 'work/kat/il_S2_elt_moment_P{k}'}
    assert il_node.files[2] == ['work/il_S2_summary_palt/P{k}.bin', 'work/il_S2_summary_leccalc/P{k}.bin']
    assert chain.plan([])[0] == [('gross',), ('gross', 'net'), ('net',)]
    assert [name for name, _ in chain.stages()] == [
        'gul', 'il (fm step 1)', 'ri (fm step 2)', 'ri (fm step 3)', 'summary gul', 'summary gul S1 FakePlt', 'summary gul S1 FakeElt',
        'summary il', 'summary il S1 FakeElt', 'summary il S2 FakeElt', 'summary ri', 'summary ri S1 FakePlt', 'summary rl',
        'summary rl S1 FakeElt']
    assert sorted(chain.chunk_paths(3))[:2] == ['work/gul_S1_summary_palt/P3.bin', 'work/il_S1_summary_leccalc/P3.bin']


def test_chain_tree_summary_only_runs_needed_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = make_spec(n_ri=2)
    spec.summaries = [summary_node('il', 0, 'gross')]
    chain = Chain(spec, factories=TREE_FACTORIES)
    chain.start_chunk(1)
    run(chain, [1, 2], [])
    chain.finish_chunk()
    assert len(chain.fm_steps[0].calls) == 2 and chain.fm_steps[1].calls == [] and chain.fm_steps[2].calls == []
    assert read_tree_files(tmp_path)['work/il_S1_summary_palt/P1.bin'] == b'SUM\x01' + b''.join(
        (s['il'] + 1).tobytes() for s in fake_losses([1, 2]))


def test_chain_start_finish_chunk_order_reset_and_empty_chunk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = []

    class ResettingFm(FakeFm):
        def reset(self):
            log.append(('reset', self.static_path))

    chain = Chain(make_tree_spec(), factories={**TREE_FACTORIES, 'fm': ResettingFm})
    gul_table = chain.summary_nodes[0].tables[1][1][0]
    chain.start_chunk(4)
    assert log == [('reset', 'input'), ('reset', 'input/RI_1'), ('reset', 'input/RI_2')]
    assert gul_table.log == ['reset', ('start', {'qelt': 'work/kat/gul_S1_elt_quantile_P4', 'melt': 'work/kat/gul_S1_elt_moment_P4'})]
    with pytest.raises(ValueError, match='still open'):
        chain.start_chunk(5)
    chain.finish_chunk()
    assert gul_table.log[-1] == 'finish'
    with pytest.raises(ValueError, match='no chunk is open'):
        chain.finish_chunk()
    files = read_tree_files(tmp_path, 4)
    assert files == expected_tree_files([], k=4) and files['work/il_S2_summary_leccalc/P4.bin'] == b'SUM\x02'
    chain.start_chunk(5)
    chain.close()
    assert chain.chunk is None
    assert all(table.closed for node in chain.summary_nodes for n in node.set_ids for table, _ in node.tables[n])
    assert all(node.stage.closed for node in chain.summary_nodes)


def test_chain_tree_one_chain_many_chunks_any_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chunks = {1: [1, 4, 7, 10], 2: [2, 5, 8, 108], 3: [3, 6, 9]}
    with Chain(make_tree_spec(), factories=TREE_FACTORIES) as chain:
        for k in (3, 1, 2):
            chain.start_chunk(k)
            run(chain, chunks[k], [])
            chain.finish_chunk()
    for k, events in chunks.items():
        assert read_tree_files(tmp_path, k) == expected_tree_files(events, k=k)


def test_chain_tree_rejects_bad_source_and_kind():
    spec = make_spec(n_ri=1)
    spec.summaries = [summary_node('ri', 5, 'net')]
    with pytest.raises(ValueError, match='not an output'):
        Chain(spec, factories=TREE_FACTORIES)
    spec.summaries = [summary_node('il', 0, 'gross')]
    spec.summaries[0]['sets'][0]['outputs'].append({'kind': 'xx'})
    with pytest.raises(ValueError, match='unknown output kind'):
        Chain(spec, factories=TREE_FACTORIES)


def test_chain_tree_spec_json_round_trip():
    spec = make_tree_spec('/run')
    assert ChainSpec.from_json(spec.to_json()) == spec
    assert ChainSpec.from_dict(spec.to_dict()) == spec


def test_summary_set_header():
    class Single:
        header = b'12345678'

    assert summary_set_header(Single(), 3) == b'12345678' + np.int32(3).tobytes()
    assert summary_set_header(FakeSummary(b'', 'input', 'gul', [2]), 2) == b'SUM\x02'


def test_chain_tree_low_memory_idx_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = make_spec(n_ri=1)
    node = summary_node('il', 0, 'gross', set_ids=(1, 2), elt=False)
    node['stage']['low_memory'] = True
    node['sets'][0]['idx_paths'] = ['work/il_S1_summary_palt/P{k}.idx', 'work/il_S1_summaryleccalc/P{k}.idx']
    spec.summaries = [node]
    with Chain(spec, factories=TREE_FACTORIES) as chain:
        assert sorted(chain.chunk_paths(2)) == ['work/il_S1_summary_palt/P2.bin', 'work/il_S1_summary_palt/P2.idx',
                                                'work/il_S1_summaryleccalc/P2.idx', 'work/il_S2_summary_palt/P2.bin']
        chain.start_chunk(2)
        run(chain, [1, 5, 2, 8, 3], [])
        chain.finish_chunk()
    idx = b''.join(np.int64(s['il'][0] * 1000 + 1).tobytes() for s in fake_losses([1, 5, 2, 8, 3]) if 'il' in s and s['il'][0] % 100 != 8)
    assert (tmp_path / 'work/il_S1_summary_palt/P2.idx').read_bytes() == idx
    assert (tmp_path / 'work/il_S1_summaryleccalc/P2.idx').read_bytes() == idx
    assert (tmp_path / 'work/il_S2_summary_palt/P2.bin').read_bytes()[:4] == b'SUM\x02'
