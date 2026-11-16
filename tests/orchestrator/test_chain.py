import io
import os
import subprocess
import sys

import numpy as np
import pytest

from oasislmf.execution.orchestrator.chain import Chain, ChainSpec, build_chain_spec

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
