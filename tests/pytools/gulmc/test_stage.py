"""GulmcStage gives exactly the bytes the gulmc command writes, whatever order the events come in."""
import io
import json
from pathlib import Path

import numpy as np
import pytest

from oasislmf.pytools.gulmc import manager as gulmc_manager
from oasislmf.pytools.gulmc.manager import run as run_gulmc
from oasislmf.pytools.gulmc.stage import GulmcStage

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets")
TEST_MODEL_DIRS = sorted(x for x in TESTS_ASSETS_DIR.glob("test_model_*") if x.is_dir())

CONFIGS = {
    'S0': dict(sample_size=0, alloc_rule=1, random_generator=1),
    'S10_rng1_a2': dict(sample_size=10, alloc_rule=2, random_generator=1),
    'S10_rng2_eff': dict(sample_size=10, alloc_rule=3, random_generator=2, effective_damageability=True),
    'S10_rng0_nocorr': dict(sample_size=10, alloc_rule=1, random_generator=0, ignore_correlation=True),
    'S10_debug1': dict(sample_size=10, alloc_rule=0, random_generator=1, debug=1),
}


def make_stage(model_dir, sample_size, alloc_rule, random_generator, debug=0, **kwargs):
    """Build a GulmcStage on ``model_dir`` with the options the gulmc tests use."""
    return GulmcStage(model_dir, sample_size, 0., alloc_rule, debug, random_generator, **kwargs)


def stage_bytes_per_event(model_dir, event_ids, config):
    """Run ``event_ids`` through one stage, in the order given; return the header and {event_id: bytes}."""
    per_event = {}
    with make_stage(model_dir, **config) as stage:
        for event_id in event_ids:
            out = io.BytesIO()
            stage.process_event(event_id, out)
            per_event[int(event_id)] = out.getvalue()
        return stage.header, per_event


def run_cli(model_dir, out_path, config):
    """Run ``gulmc``'s run() on the model's events.bin and return the bytes it writes."""
    config = dict(config)
    run_gulmc(
        run_dir=model_dir,
        ignore_file_type=set(),
        file_in=model_dir / 'input' / 'events.bin',
        file_out=out_path,
        sample_size=config.pop('sample_size'),
        loss_threshold=0.,
        alloc_rule=config.pop('alloc_rule'),
        debug=config.pop('debug', 0),
        random_generator=config.pop('random_generator'),
        **config,
    )
    return out_path.read_bytes()


def read_event_ids(model_dir):
    """Event ids of the model's events.bin, in file order."""
    return np.fromfile(model_dir / 'input' / 'events.bin', dtype='i4')


@pytest.mark.parametrize("config_name", list(CONFIGS))
@pytest.mark.parametrize("model_dir", TEST_MODEL_DIRS[:1], ids=lambda x: x.name)
def test_stage_matches_run_and_is_order_independent(model_dir, config_name, tmp_path, monkeypatch):
    """Header + per-event stage output is run()'s output; per-event bytes do not depend on event order."""
    monkeypatch.chdir(tmp_path)
    config = CONFIGS[config_name]
    event_ids = read_event_ids(model_dir)
    assert len(event_ids) > 1

    expected = run_cli(model_dir, tmp_path / 'gul.bin', config)

    header, in_order = stage_bytes_per_event(model_dir, event_ids, config)
    assert header + b''.join(in_order[int(e)] for e in event_ids) == expected
    assert any(in_order.values())

    _, reversed_order = stage_bytes_per_event(model_dir, event_ids[::-1], config)
    assert reversed_order == in_order

    for event_id in event_ids:
        _, alone = stage_bytes_per_event(model_dir, [event_id], config)
        assert alone[int(event_id)] == in_order[int(event_id)]


@pytest.mark.parametrize("model_dir", TEST_MODEL_DIRS, ids=lambda x: x.name)
def test_stage_matches_run_on_every_test_model(model_dir, tmp_path, monkeypatch):
    """On every test model, the stage reproduces run()'s stream, events also given in reverse."""
    monkeypatch.chdir(tmp_path)
    config = CONFIGS['S10_rng1_a2']
    event_ids = read_event_ids(model_dir)

    expected = run_cli(model_dir, tmp_path / 'gul.bin', config)

    header, in_order = stage_bytes_per_event(model_dir, event_ids, config)
    assert header + b''.join(in_order[int(e)] for e in event_ids) == expected

    _, reversed_order = stage_bytes_per_event(model_dir, event_ids[::-1], config)
    assert reversed_order == in_order


def test_stage_writes_whole_buffers_in_several_calls_for_a_large_event(tmp_path, monkeypatch):
    """An event larger than the output buffer reaches the sink in several writes that add up to run()'s bytes."""
    monkeypatch.chdir(tmp_path)
    model_dir = TEST_MODEL_DIRS[0]
    config = dict(sample_size=2000, alloc_rule=1, random_generator=1)
    event_ids = read_event_ids(model_dir)

    class RecordingSink:
        def __init__(self):
            self.chunks = []

        def write(self, data):
            self.chunks.append(bytes(data))
            return len(self.chunks[-1])

    sink = RecordingSink()
    with make_stage(model_dir, **config) as stage:
        header = stage.header
        for event_id in event_ids:
            stage.process_event(event_id, sink)

    assert header + b''.join(sink.chunks) == run_cli(model_dir, tmp_path / 'gul.bin', config)
    assert len(sink.chunks) > len(event_ids)
    assert all(sink.chunks)


@pytest.mark.parametrize("alloc_rule,debug", [(4, 0), (1, 1)])
def test_stage_rejects_invalid_options(alloc_rule, debug):
    """The stage validates alloc_rule and debug as gulmc does."""
    with pytest.raises(ValueError):
        make_stage(TEST_MODEL_DIRS[0], 10, alloc_rule, 1, debug=debug)


def test_stage_writes_event_error_file_when_the_kernel_fails(tmp_path, monkeypatch):
    """A failing event leaves event_error.json naming it in the working directory, and the error propagates."""
    monkeypatch.chdir(tmp_path)

    def failing_kernel(*args):
        raise ZeroDivisionError("boom")

    event_id = int(read_event_ids(TEST_MODEL_DIRS[0])[0])
    with make_stage(TEST_MODEL_DIRS[0], 10, 1, 1) as stage:
        monkeypatch.setattr(gulmc_manager, 'compute_event_losses', failing_kernel)
        with pytest.raises(ZeroDivisionError):
            stage.process_event(event_id, io.BytesIO())

    assert json.loads((tmp_path / 'event_error.json').read_text()) == {"event_id": str(event_id)}


def test_stage_raises_when_the_kernel_makes_no_progress(tmp_path, monkeypatch):
    """A kernel asking to be resumed at the coverage it started from raises instead of looping forever."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gulmc_manager, 'compute_event_losses', lambda *args: False)

    with make_stage(TEST_MODEL_DIRS[0], 10, 1, 1) as stage:
        with pytest.raises(RuntimeError, match="made no progress"):
            stage.process_event(int(read_event_ids(TEST_MODEL_DIRS[0])[0]), io.BytesIO())
