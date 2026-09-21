"""The golden-ratio mapping from (building, sample) to an entry of a stratified pool.

A pool holds one stratified value per entry, so the accuracy of the scheme rests on the
buildings covering the entries evenly at each sample. These pin the two properties that gives:
even coverage within a sample, and no building pinned to one quantile across samples.
"""
import numpy as np
import pytest

from oasislmf.pytools.gul.random import GOLDEN_RATIO_CONJUGATE, pool_index


@pytest.mark.parametrize("pool_size", [1, 2, 8, 256, 1024])
def test_every_entry_is_used_once_per_full_cycle(pool_size):
    """At a fixed sample, pool_size consecutive buildings must hit every entry exactly once.
    Anything less is a stratum over- or under-represented, which is the whole accuracy claim."""
    for sample_idx in (1, 2, 17, 1000):
        seen = [pool_index(b, sample_idx, pool_size) for b in range(1, pool_size + 1)]
        assert sorted(seen) == list(range(pool_size)), \
            f"pool_size={pool_size}, sample={sample_idx}: coverage is not a permutation"


@pytest.mark.parametrize("pool_size", [8, 256, 1024])
def test_a_building_is_not_pinned_to_one_entry(pool_size):
    """The reason for the rotation. Without it a building would read the same entry in every
    sample, fixing its quantile for the whole event."""
    n_samples = 200
    reachable = min(n_samples, pool_size)          # cannot exceed one entry per sample
    for building in (1, 7, 500):
        entries = {pool_index(building, s, pool_size) for s in range(1, n_samples + 1)}
        assert len(entries) >= 0.9 * reachable, \
            f"building {building} reaches {len(entries)} entries, expected about {reachable}"


@pytest.mark.parametrize("pool_size", [1, 3, 8, 257, 1024])
def test_the_index_is_always_in_range(pool_size):
    """Read straight into the pool with no bounds check, so the range is load-bearing. The
    rotation is computed in floating point, where a fraction of 0.999... could otherwise round
    up to pool_size."""
    for building in (1, 2, 630_510):
        for sample_idx in (1, 2, 999_999):
            i = pool_index(building, sample_idx, pool_size)
            assert 0 <= i < pool_size, f"b={building} s={sample_idx} M={pool_size} -> {i}"


def test_the_constant_is_the_golden_ratio_conjugate():
    """Its continued fraction is all 1s, which is what makes {n*PHI} the most evenly spread of
    any additive recurrence. A nearby but rational-ish value would clump."""
    assert GOLDEN_RATIO_CONJUGATE == pytest.approx((np.sqrt(5) - 1) / 2, rel=1e-15)


def test_only_groups_past_the_gate_are_pooled():
    """Below the gate a group keeps its own per-building draws. The gate sits past the band where
    the buildings divide unevenly among the entries, which is worst just above POOL_SIZE."""
    from oasislmf.pytools.gul.random import POOL_GATE_RATIO, POOL_SIZE, group_is_pooled

    gate = POOL_GATE_RATIO * POOL_SIZE
    for n in (1, 2, POOL_SIZE, POOL_SIZE + 1, gate - 1):
        assert not group_is_pooled(n), f"{n} buildings should draw individually"
    for n in (gate, gate + 1, 630_510):
        assert group_is_pooled(n), f"{n} buildings should read from the pool"
