"""A packed item far larger than the default buffer is still emitted correctly.

compute_event_losses returns only BETWEEN coverages, so a coverage's whole output accumulates
in the buffer before anything is written out, and the buffer is grown to the largest coverage.
These tests pin that a packed item well past the default buffer size comes out complete: every
building's block present, every sidx once, and the item header written exactly once.

They used to test a flush landing in the MIDDLE of an item, when a fused coverage was emitted
per building block and could return between them. That per-block return is gone -- it was the
one flush point that existed only for the fused path, and the buffer bound is now uniform.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.common.data import correlations_dtype, items_dtype
from oasislmf.pytools.common.event_stream import PIPE_CAPACITY
from oasislmf.pytools.gul.common import (NUM_IDX, gulSampleslevelHeader_size,
                                         gulSampleslevelRec_size)
from oasislmf.pytools.gulmc.manager import run as run_gulmc

SRC_MODEL = Path(__file__).parents[2].joinpath("assets", "test_model_1")
SAMPLE_SIZE = 64


def _bytes_per_block(sample_size):
    return gulSampleslevelHeader_size + (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size


def _buildings_past_the_default_buffer(sample_size):
    """Enough buildings that the item outgrows the buffer gulpy and gulmc start with.

    Derived from PIPE_CAPACITY rather than hardcoded, so raising the default cannot quietly make
    these tests vacuous -- the buffer must actually have to grow for them to mean anything.
    """
    return (PIPE_CAPACITY * 2) // _bytes_per_block(sample_size) + 64


def _run(n_buildings, alloc_rule, sample_size=SAMPLE_SIZE, packed_sign=-1):
    """Run one item per coverage, each packing n_buildings, and parse the stream back.

    Returns {(event_id, item_id): [sidx, ...]} in the order written.
    """
    with tempfile.TemporaryDirectory() as t:
        run_dir = Path(t) / 'assets'
        shutil.copytree(SRC_MODEL, run_dir)
        shutil.rmtree(run_dir / 'input' / 'gulmc_structure', ignore_errors=True)

        # one item per coverage, so every coverage takes the fused per-building path
        items = pd.read_csv(run_dir / 'input' / 'items.csv').groupby(
            'coverage_id', as_index=False).first()
        items['item_id'] = np.arange(1, len(items) + 1)
        items = items[list(items_dtype.names)]
        items.to_csv(run_dir / 'input' / 'items.csv', index=False)
        packed = np.zeros(len(items), dtype=items_dtype)
        for name in items_dtype.names:
            packed[name] = items[name].to_numpy()
        packed.tofile(run_dir / 'input' / 'items.bin')

        corr = np.zeros(len(items), dtype=correlations_dtype)
        corr['item_id'] = items['item_id'].to_numpy()
        corr['packed_buildings'] = packed_sign * n_buildings
        corr['peril_correlation_group'] = 1
        corr['damage_correlation_value'] = 0.5
        corr['hazard_group_id'] = 1
        corr.tofile(run_dir / 'input' / 'correlations.bin')
        pd.DataFrame({k: corr[k] for k in corr.dtype.names}).to_csv(
            run_dir / 'input' / 'correlations.csv', index=False)
        # one event: the building counts needed to outgrow the buffer are large, and every
        # event repeats the same code path
        np.fromfile(run_dir / 'input' / 'events.bin', dtype='i4')[:1].tofile(
            run_dir / 'input' / 'events.bin')

        out = run_dir / 'out.bin'
        run_gulmc(run_dir=run_dir, ignore_file_type=set(), file_in=run_dir / 'input' / 'events.bin',
                  file_out=out, sample_size=sample_size, loss_threshold=0., alloc_rule=alloc_rule,
                  debug=0, random_generator=2, ignore_correlation=False,
                  effective_damageability=False)
        raw = np.fromfile(out, dtype=np.int32)

    records = {}
    order = []
    i = 2                                    # stream type + max sample index
    while i + 1 < len(raw):
        key = (int(raw[i]), int(raw[i + 1]))
        i += 2
        sidxs = []
        while i + 1 < len(raw):
            sidx = int(raw[i])
            i += 2
            if sidx == 0:
                break
            sidxs.append(sidx)
        order.append(key)
        records.setdefault(key, []).extend(sidxs)
    return records, order


@pytest.mark.parametrize("alloc_rule", [0, 1, 2])
def test_an_item_larger_than_the_default_buffer_is_written_once(alloc_rule):
    """The buffer grows to fit it, and the header is written exactly once."""
    n_buildings = _buildings_past_the_default_buffer(SAMPLE_SIZE)
    records, order = _run(n_buildings, alloc_rule)
    assert len(order) == len(set(order)), "an item was opened more than once"
    assert records, "no items emitted"


@pytest.mark.parametrize("alloc_rule", [0, 1, 2])
def test_every_building_block_is_present(alloc_rule):
    """Each building contributes its specials once and its samples once, none lost or repeated."""
    n_buildings = _buildings_past_the_default_buffer(SAMPLE_SIZE)
    records, _ = _run(n_buildings, alloc_rule)

    for key, sidxs in records.items():
        specials = [s for s in sidxs if s < 0]
        samples = [s for s in sidxs if s > 0]
        assert len(specials) == n_buildings * NUM_IDX, f"{key}: wrong special count"
        assert len(samples) == n_buildings * SAMPLE_SIZE, f"{key}: wrong sample count"
        # a packed sample sidx is (b - 1) * S + s, so the set is exactly 1..N*S
        assert sorted(samples) == list(range(1, n_buildings * SAMPLE_SIZE + 1)), \
            f"{key}: sample sidx are not the contiguous packed range"
        assert len(set(specials)) == len(specials), f"{key}: a special sidx repeated"


def test_a_run_that_fits_the_default_buffer_is_unaffected():
    """The small case takes the same path and must still be complete."""
    n_buildings = 4
    assert n_buildings * _bytes_per_block(SAMPLE_SIZE) < PIPE_CAPACITY * 2
    records, order = _run(n_buildings, alloc_rule=1)
    assert len(order) == len(set(order))
    for key, sidxs in records.items():
        samples = [s for s in sidxs if s > 0]
        assert sorted(samples) == list(range(1, n_buildings * SAMPLE_SIZE + 1))
