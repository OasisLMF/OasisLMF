"""A packed item whose output exceeds the buffer is flushed part-way and stays well formed.

The buffer used to be grown to hold a whole dependency subtree, so a location with enough
buildings forced a multi-GB allocation. It is now sized to a single building block and flushed
between blocks, which means a flush can land in the MIDDLE of an item -- after its header and
some of its buildings, before the rest and the delimiter.

That is only sound because an item's records have to be contiguous in the STREAM, not in the
buffer. These tests pin the two ways it could go wrong: a repeated item header (written again
on re-entry) and a lost or duplicated building block.
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


def _run(n_buildings, alloc_rule, sample_size=SAMPLE_SIZE):
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
        corr['packed_buildings'] = -n_buildings      # negative: kept separate, one block each
        corr['peril_correlation_group'] = 1
        corr['damage_correlation_value'] = 0.5
        corr['hazard_group_id'] = 1
        corr.tofile(run_dir / 'input' / 'correlations.bin')
        pd.DataFrame({k: corr[k] for k in corr.dtype.names}).to_csv(
            run_dir / 'input' / 'correlations.csv', index=False)

        out = run_dir / 'out.bin'
        run_gulmc(run_dir=run_dir, ignore_file_type=set(), file_in=run_dir / 'input' / 'events.bin',
                  file_out=out, sample_size=sample_size, loss_threshold=0., alloc_rule=alloc_rule,
                  debug=0, random_generator=1, ignore_correlation=False,
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
def test_item_spanning_several_buffers_is_written_once(alloc_rule):
    """The header must not be re-emitted when a flush lands inside an item."""
    n_buildings = 256
    # guard against the test quietly becoming vacuous if the buffer constant changes
    assert n_buildings * _bytes_per_block(SAMPLE_SIZE) > PIPE_CAPACITY * 2, \
        "one item must not fit the buffer, or nothing is being flushed mid-item"

    records, order = _run(n_buildings, alloc_rule)
    assert len(order) == len(set(order)), "an item was opened more than once"
    assert records, "no items emitted"


@pytest.mark.parametrize("alloc_rule", [0, 1, 2])
def test_every_building_block_survives_the_flush(alloc_rule):
    """Each building contributes its specials once and its samples once, none lost or repeated."""
    n_buildings = 256
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


def test_a_run_that_fits_the_buffer_is_unaffected():
    """The small case takes the same path and must still be complete."""
    n_buildings = 4
    assert n_buildings * _bytes_per_block(SAMPLE_SIZE) < PIPE_CAPACITY * 2
    records, order = _run(n_buildings, alloc_rule=1)
    assert len(order) == len(set(order))
    for key, sidxs in records.items():
        samples = [s for s in sidxs if s > 0]
        assert sorted(samples) == list(range(1, n_buildings * SAMPLE_SIZE + 1))
