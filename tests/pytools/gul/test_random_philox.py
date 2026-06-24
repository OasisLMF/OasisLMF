"""Tests for the Philox-based Latin Hypercube generator (random_generator 2).

Covers:
  * Philox4x32 round-function correctness: an independent reference reproduces the
    official Random123 known-answer vectors (at 10 rounds), and the production 7-round
    core matches that reference (at 7 rounds).
  * get_random_generator id -> function mapping.
  * The determinism guarantees gulmc requires (repeatable, order-independent per
    seed, batch-independent) and the skip_seeds contract.
  * Valid Latin Hypercube structure (exactly one sample per stratum, range [0,1)).
"""
import numpy as np
import pytest

from oasislmf.pytools.gul.random import (
    _philox4x32_7,
    generate_hash,
    get_random_generator,
    random_LatinHypercube_Philox7,
)

GEN = random_LatinHypercube_Philox7  # random_generator 2

# Official Random123 philox4x32-10 known-answer vectors: (ctr[0..3], key[0..1]) -> out[0..3]
PHILOX_KAT = [
    ((0x00000000, 0x00000000, 0x00000000, 0x00000000), (0x00000000, 0x00000000),
     (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
    ((0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), (0xffffffff, 0xffffffff),
     (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
    ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
     (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)),
]


def _ref_philox(ctr, key, rounds):
    """Independent plain-Python Philox4x32 block (R rounds), used to validate production.

    Mirrors the Random123 philox4x32 round + bumpkey logic. Validated against the
    published 10-round known-answer vectors below, then used to check the production
    7-round njit core uses the identical (validated) round function.
    """
    M0, M1 = 0xD2511F53, 0xCD9E8D57
    W0, W1 = 0x9E3779B9, 0xBB67AE85
    M = 0xFFFFFFFF
    c = list(ctr)
    k = list(key)
    for _ in range(rounds):
        p0 = M0 * c[0]
        p1 = M1 * c[2]
        c = [(p1 >> 32) ^ c[1] ^ k[0], p1 & M, (p0 >> 32) ^ c[3] ^ k[1], p0 & M]
        k = [(k[0] + W0) & M, (k[1] + W1) & M]
    return tuple(c)


def _seeds(event_id=4242, groups=(7, 13, 99, 100000, 3)):
    return np.array([generate_hash(g, event_id) for g in groups], dtype=np.int64)


@pytest.mark.parametrize("ctr,key,expected", PHILOX_KAT)
def test_reference_matches_known_answer(ctr, key, expected):
    """The independent reference reproduces the Random123 philox4x32-10 vectors bit-exactly."""
    assert _ref_philox(ctr, key, 10) == expected


@pytest.mark.parametrize("ctr,key,expected", PHILOX_KAT)
def test_production_core_matches_reference(ctr, key, expected):
    """The production 7-round njit core matches the validated reference at 7 rounds."""
    out = _philox4x32_7(np.uint32(ctr[0]), np.uint32(ctr[1]), np.uint32(ctr[2]), np.uint32(ctr[3]),
                        np.uint32(key[0]), np.uint32(key[1]))
    assert tuple(int(np.uint32(x)) for x in out) == _ref_philox(ctr, key, 7)


def test_get_random_generator_id():
    """id 2 maps to the Philox-7 LH generator."""
    assert get_random_generator(2) is random_LatinHypercube_Philox7


def test_get_random_generator_unknown():
    # 3 was the removed Philox-10 id; both 3 and 4 must now be rejected.
    for bad in (3, 4):
        with pytest.raises(ValueError):
            get_random_generator(bad)


def test_repeatable():
    seeds = _seeds()
    np.testing.assert_array_equal(GEN(seeds, 40, 0), GEN(seeds, 40, 0))


def test_order_independent_per_seed():
    """A row's values depend only on its seed, not its position (per-group determinism)."""
    seeds = _seeds()
    base = GEN(seeds, 40, 0)
    perm = np.array([3, 0, 4, 1, 2])
    reordered = GEN(seeds[perm], 40, 0)
    for new_i, old_i in enumerate(perm):
        np.testing.assert_array_equal(reordered[new_i], base[old_i])


def test_batch_independent():
    """A single-seed draw equals that seed's row within a larger batch."""
    seeds = _seeds()
    base = GEN(seeds, 40, 0)
    np.testing.assert_array_equal(GEN(seeds[2:3], 40, 0)[0], base[2])


def test_skip_seeds():
    """Skipped leading rows are left as zeros; the rest match the unskipped draw."""
    seeds = _seeds()
    full = GEN(seeds, 40, 0)
    skipped = GEN(seeds, 40, 2)
    assert np.all(skipped[:2] == 0.0)
    np.testing.assert_array_equal(skipped[2:], full[2:])


@pytest.mark.parametrize("n", [1, 4, 7, 50, 128])
def test_valid_latin_hypercube(n):
    """Each row is a valid LHS in (0,1]: exactly one sample falls in each of the n strata."""
    seeds = _seeds()
    out = GEN(seeds, n, 0)
    assert out.shape == (len(seeds), n)
    # Range is (0,1] -- the jitter w*INV32 lies in [0,1), so (perms[k]-jitter)/n
    # can reach exactly 1.0 (w=0, perms[k]=n) but is strictly above 0.
    assert (out > 0.0).all() and (out <= 1.0).all()
    for row in out:
        strata = np.minimum((row * n).astype(np.int64), n - 1)
        assert (np.bincount(strata, minlength=n) == 1).all()
