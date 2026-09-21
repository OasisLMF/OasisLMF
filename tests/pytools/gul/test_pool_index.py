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


def test_only_items_past_the_gate_may_pool():
    """Below the gate an item keeps its own per-building draws. The gate sits past the band where
    the buildings divide unevenly among the entries, which is worst just above POOL_SIZE.

    It is the ITEM's count that must clear it, not its group's largest: a group pools only when
    every item in it does, or a small item sharing a group with a huge one would read a
    POOL_SIZE pool at exactly the ratio the gate exists to avoid."""
    from oasislmf.pytools.gul.random import POOL_GATE_RATIO, POOL_SIZE, item_is_past_pool_gate

    gate = POOL_GATE_RATIO * POOL_SIZE
    for n in (1, 2, POOL_SIZE, POOL_SIZE + 1, gate - 1):
        assert not item_is_past_pool_gate(n), f"{n} buildings must keep its own draws"
    for n in (gate, gate + 1, 630_510):
        assert item_is_past_pool_gate(n), f"{n} buildings may read from a pool"


POOLED_N = 8192          # POOL_GATE_RATIO * POOL_SIZE, the smallest pooled group


@pytest.mark.parametrize("generator", [0, 1, 2], ids=["MersenneTwister", "LatinHypercube", "LH-Philox"])
def test_a_pooled_group_gets_one_value_per_stratum(generator):
    """Whichever generator fills it, a pool is stratified.

    Its entries stand in for the whole building population, and plain uniforms would leave gaps
    and clumps in that population -- an iid pool converges as 1/sqrt(M), so it would need about
    as many entries as there are buildings to match drawing per building, i.e. save nothing.
    A Latin Hypercube of M puts exactly one value in each 1/M-wide stratum, which is the property
    asserted here.
    """
    from oasislmf.pytools.gul.random import (POOL_SIZE, build_packed_rndm_offsets,
                                             get_sample_generator)
    sample_size = 100
    seeds = np.array([12345, 67890], dtype='i8')
    n_buildings = np.array([4, POOLED_N], dtype='i8')     # one ordinary group, one pooled
    pooled = np.array([0, 1], dtype=np.int8)     # group 0 ordinary, group 1 past the gate
    offsets = build_packed_rndm_offsets(n_buildings, pooled, sample_size)

    assert offsets[2] - offsets[1] == POOL_SIZE, "a pooled group must reserve exactly POOL_SIZE"
    assert offsets[1] - offsets[0] == 4 * sample_size, "an ordinary group keeps its full block"

    pool = get_sample_generator(generator)(seeds, sample_size, n_buildings, pooled, offsets)[offsets[1]:offsets[2]]
    assert pool.min() >= 0.0 and pool.max() < 1.0
    strata = np.floor(pool * POOL_SIZE).astype(int)
    assert sorted(strata) == list(range(POOL_SIZE)), \
        "not a Latin Hypercube: some stratum is empty and another holds two"


@pytest.mark.parametrize("generator", [0, 1, 2], ids=["MersenneTwister", "LatinHypercube", "LH-Philox"])
def test_the_pool_is_not_merely_sorted(generator):
    """The entries must be permuted, not left in ascending stratum order.

    pool_index walks the entries in order, so ascending entries would hand building b the
    quantile (b + rotation)/M in BOTH the hazard and the damage pool -- locking the two
    dimensions into near-perfect rank correlation across the buildings. The permutation inside
    the Latin Hypercube is what keeps them independent.
    """
    from oasislmf.pytools.gul.random import (POOL_SIZE, build_packed_rndm_offsets,
                                             get_sample_generator)
    seeds = np.array([4242], dtype='i8')
    n_buildings = np.array([POOLED_N], dtype='i8')
    offsets = build_packed_rndm_offsets(n_buildings, np.array([1], dtype=np.int8), 100)
    pool = get_sample_generator(generator)(seeds, 100, n_buildings, np.array([1], dtype=np.int8), offsets)

    rank_corr = np.corrcoef(np.arange(POOL_SIZE), pool)[0, 1]
    assert abs(rank_corr) < 0.2, \
        f"pool entries track their index (corr {rank_corr:+.2f}); the two dimensions would couple"


def test_a_group_pools_only_when_every_item_clears_the_gate():
    """One small item keeps its whole group off the pool.

    The imbalance error goes as POOL_SIZE/n_buildings, so it is the item's own count that has to
    clear the gate. Deciding on the group's LARGEST item would let a 1,500-building item sharing
    a group with a huge one read a 1,024-entry pool at a ratio of 1.5 -- the worst point of the
    band the gate exists to avoid.
    """
    from oasislmf.pytools.gul.random import (POOL_GATE_RATIO, POOL_SIZE,
                                             build_packed_rndm_offsets, item_is_past_pool_gate)
    sample_size = 100

    def group_pools(counts):
        return all(item_is_past_pool_gate(c) for c in counts)

    assert group_pools([20_000, 9_000]), "every item past the gate: pool"
    assert not group_pools([20_000, 1_500]), "one item at the bad ratio: must not pool"
    assert not group_pools([4, 7]), "all small: nothing to pool"

    # and the offsets follow the flag, not the largest count
    nb = np.array([20_000], dtype='i8')
    assert build_packed_rndm_offsets(nb, np.array([0], dtype=np.int8), sample_size)[1] == 20_000 * sample_size
    assert build_packed_rndm_offsets(nb, np.array([1], dtype=np.int8), sample_size)[1] == POOL_SIZE
