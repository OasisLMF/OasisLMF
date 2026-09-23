"""gulpy: a buffer that fills at an item boundary must not repeat the item header.

gulpy emits each building as it is computed whenever the alloc rule never has to look across a
coverage's items. The header is written once per item, so the buffer has to be reserved for the
header and the item's first block TOGETHER: a header written before a reservation that then
fails goes out with the flushed bytes and is written again on re-entry, and the reader decodes
the second copy as a sidx/loss pair.

The existing gulpy runs never reach this: they use alloc rules that take the whole-coverage
write_losses path instead.
"""
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from oasislmf.pytools.common.event_stream import PIPE_CAPACITY
from oasislmf.pytools.gul.common import (NUM_IDX, gulSampleslevelHeader_size,
                                         gulSampleslevelRec_size)
from oasislmf.pytools.gul.manager import run as gulpy_run

TEST_MODEL_DIR = Path(__file__).parents[2].joinpath("assets", "test_model_1")


def _bytes_per_block(sample_size):
    return gulSampleslevelHeader_size + (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size


def _sample_size_forcing_an_item_boundary_flush():
    """S at which two items no longer fit the buffer, so a flush lands BETWEEN items."""
    return (PIPE_CAPACITY - gulSampleslevelHeader_size) // gulSampleslevelRec_size - NUM_IDX


def _parse(path):
    """Return the stream as [((event_id, item_id), [sidx, ...]), ...] in the order written."""
    raw = np.fromfile(path, dtype=np.int32)
    order, i = [], 2                      # stream type, then max sample index
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
        order.append((key, sidxs))
    return order


@pytest.mark.skipif(os.environ.get("NUMBA_DISABLE_JIT", "0") != "0",
                    reason="the sample size needed to fill the buffer is too slow unjitted")
def test_flush_between_items_does_not_repeat_the_header():
    """alloc_rule 0 takes the fused per-building path for every coverage, whatever its item count."""
    sample_size = _sample_size_forcing_an_item_boundary_flush()
    # not vacuous: one item fits the buffer, two do not, so a flush has to land on a boundary
    assert _bytes_per_block(sample_size) <= PIPE_CAPACITY * 2
    assert 2 * _bytes_per_block(sample_size) > PIPE_CAPACITY * 2

    with TemporaryDirectory() as tmp:
        stream = Path(tmp).joinpath("getmodel_stream.bin")
        out_bin = Path(tmp).joinpath("out.bin")
        with open(stream, "wb") as fout:
            subprocess.run("evepy 1 1 | modelpy", cwd=TEST_MODEL_DIR, shell=True,
                           check=True, stdout=fout)
        gulpy_run(run_dir=TEST_MODEL_DIR, ignore_file_type=set(), sample_size=sample_size,
                  loss_threshold=0., alloc_rule=0, debug=False, random_generator=1,
                  file_in=str(stream), file_out=str(out_bin))
        order = _parse(out_bin)

    keys = [key for key, _ in order]
    assert len(keys) > 1, "need several items for a boundary to be crossed"
    assert len(keys) == len(set(keys)), "an item header was emitted twice"
    for key, sidxs in order:
        assert len(sidxs) == NUM_IDX + sample_size, f"{key}: wrong record count"
        assert len(set(sidxs)) == len(sidxs), f"{key}: a sidx repeated"
        assert sorted(s for s in sidxs if s > 0) == list(range(1, sample_size + 1)), \
            f"{key}: sample sidx are not 1..S"
