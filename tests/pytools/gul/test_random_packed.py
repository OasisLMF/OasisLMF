"""Tests for building-packed random draws, across all three generators.

Building-packing mode multiplexes N buildings of a location into the sample dimension of
one item. Each location seed must therefore yield ``N * S`` random numbers instead of ``S``,
with building ``b`` (1-based), sample ``s`` (1-based) at
``rndms[offsets[seed] + (b - 1) * S + (s - 1)]``.

Two properties are required of every generator:

* **Building 1 reproduces the unpacked draw**, so the single-building case is unchanged.
* For the two Latin Hypercube generators, **each building gets a Latin Hypercube of its own**
  — not a slice of one larger sample, which would leave each building holding a random subset
  of the strata (see ``test_slicing_one_large_lh_is_not_stratified``, the control that pins the
  reason this design exists).

The per-building stream coordinate differs: the Mersenne Twister generators continue the
group's single seeded stream, Philox uses a counter word.
"""
from unittest import main, TestCase

import numpy as np

from oasislmf.pytools.gul.random import (
    build_packed_rndm_offsets,
    get_correlation_generator,
    get_sample_generator,
    random_MersenneTwister,
    random_MersenneTwister_packed,
)


class TestRandomMersenneTwisterPacked(TestCase):

    def setUp(self):
        self.seeds = np.array([12345, 67890, 11111], dtype=np.int64)
        self.S = 8

    def test_offsets_prefix_sum(self):
        n_buildings = np.array([1, 3, 2], dtype=np.int64)
        offsets = build_packed_rndm_offsets(n_buildings, self.S)
        self.assertEqual(offsets.tolist(), [0, 8, 32, 48])
        self.assertEqual(offsets[-1], n_buildings.sum() * self.S)

    def test_single_building_matches_legacy(self):
        """N=1 for every seed -> packed draw is byte-identical to the legacy 2-d draw."""
        n_buildings = np.ones(len(self.seeds), dtype=np.int64)
        offsets = build_packed_rndm_offsets(n_buildings, self.S)
        packed = random_MersenneTwister_packed(self.seeds, self.S, n_buildings, offsets)
        legacy = random_MersenneTwister(self.seeds, self.S)
        for seed_i in range(len(self.seeds)):
            np.testing.assert_array_equal(packed[offsets[seed_i]: offsets[seed_i] + self.S], legacy[seed_i])

    def test_building_one_slice_matches_legacy_when_multi(self):
        """Building 1's slice equals the legacy single-building draw even when N>1."""
        n_buildings = np.array([1, 3, 2], dtype=np.int64)
        offsets = build_packed_rndm_offsets(n_buildings, self.S)
        packed = random_MersenneTwister_packed(self.seeds, self.S, n_buildings, offsets)
        legacy = random_MersenneTwister(self.seeds, self.S)
        for seed_i in range(len(self.seeds)):
            building_1 = packed[offsets[seed_i]: offsets[seed_i] + self.S]
            np.testing.assert_array_equal(building_1, legacy[seed_i])

    def test_buildings_are_contiguous_sequence_of_one_seed(self):
        """The N*S block for a seed is exactly np.random.random(N*S) from that seed."""
        n_buildings = np.array([1, 3, 2], dtype=np.int64)
        offsets = build_packed_rndm_offsets(n_buildings, self.S)
        packed = random_MersenneTwister_packed(self.seeds, self.S, n_buildings, offsets)
        for seed_i in range(len(self.seeds)):
            count = int(n_buildings[seed_i]) * self.S
            np.random.seed(self.seeds[seed_i])
            expected = np.random.random(count)
            np.testing.assert_array_equal(packed[offsets[seed_i]: offsets[seed_i] + count], expected)

    def test_buildings_differ_from_each_other(self):
        """Buildings 2..N consume the continuation -> distinct from building 1."""
        n_buildings = np.array([3], dtype=np.int64)
        offsets = build_packed_rndm_offsets(n_buildings, self.S)
        packed = random_MersenneTwister_packed(np.array([42], dtype=np.int64), self.S, n_buildings, offsets)
        b1 = packed[0: self.S]
        b2 = packed[self.S: 2 * self.S]
        b3 = packed[2 * self.S: 3 * self.S]
        self.assertFalse(np.array_equal(b1, b2))
        self.assertFalse(np.array_equal(b2, b3))

    def test_skip_seeds_left_as_zeros(self):
        n_buildings = np.array([2, 2], dtype=np.int64)
        offsets = build_packed_rndm_offsets(n_buildings, self.S)
        packed = random_MersenneTwister_packed(self.seeds[:2], self.S, n_buildings, offsets, skip_seeds=1)
        self.assertTrue(np.all(packed[offsets[0]: offsets[1]] == 0.0))
        self.assertFalse(np.all(packed[offsets[1]: offsets[2]] == 0.0))


def _strata(block, n):
    """The stratum index each sample falls in, sorted. A Latin Hypercube hits each exactly once."""
    return np.sort(np.floor(np.asarray(block) * n).astype(int))


class TestPackedGeneratorsAgree(TestCase):
    """Contract shared by every packed generator, exercised through the public dispatcher."""

    S = 20
    SEEDS = np.array([11, 22, 33], dtype=np.int64)
    N_BUILDINGS = np.array([1, 3, 4], dtype=np.int64)

    def setUp(self):
        self.offsets = build_packed_rndm_offsets(self.N_BUILDINGS, self.S)

    def _packed(self, gen):
        return get_sample_generator(gen)(self.SEEDS, self.S, self.N_BUILDINGS, self.offsets)

    def _block(self, packed, seed_i, b):
        start = self.offsets[seed_i] + b * self.S
        return packed[start: start + self.S]

    def test_every_generator_has_a_packed_variant(self):
        for gen in (0, 1, 2):
            with self.subTest(generator=gen):
                self.assertIsNotNone(get_sample_generator(gen))

    def test_unknown_generator_is_rejected(self):
        with self.assertRaises(ValueError):
            get_sample_generator(3)

    def test_building_one_reproduces_the_unpacked_draw(self):
        """The single-building case must be unchanged by packing, for every generator."""
        for gen in (0, 1, 2):
            with self.subTest(generator=gen):
                unpacked = get_correlation_generator(gen)(self.SEEDS, self.S)
                packed = self._packed(gen)
                for seed_i in range(len(self.SEEDS)):
                    np.testing.assert_array_equal(self._block(packed, seed_i, 0), unpacked[seed_i])

    def test_buildings_are_distinct(self):
        for gen in (0, 1, 2):
            with self.subTest(generator=gen):
                packed = self._packed(gen)
                for seed_i, nb in enumerate(self.N_BUILDINGS):
                    blocks = [self._block(packed, seed_i, b) for b in range(nb)]
                    for x in range(len(blocks)):
                        for y in range(x + 1, len(blocks)):
                            self.assertFalse(np.array_equal(blocks[x], blocks[y]))

    def test_values_are_in_the_unit_interval(self):
        for gen in (0, 1, 2):
            with self.subTest(generator=gen):
                packed = self._packed(gen)
                self.assertTrue((packed > 0).all())
                self.assertTrue((packed <= 1).all())

    def test_skip_seeds_leaves_the_head_zeroed(self):
        for gen in (0, 1, 2):
            with self.subTest(generator=gen):
                packed = get_sample_generator(gen)(
                    self.SEEDS, self.S, self.N_BUILDINGS, self.offsets, skip_seeds=1)
                self.assertTrue((packed[self.offsets[0]: self.offsets[1]] == 0).all())
                self.assertTrue((packed[self.offsets[1]:] != 0).any())

    def test_every_building_gets_its_own_latin_hypercube(self):
        """The point of the design: stratification is per building, not shared across them."""
        for gen in (1, 2):
            with self.subTest(generator=gen):
                packed = self._packed(gen)
                for seed_i, nb in enumerate(self.N_BUILDINGS):
                    for b in range(nb):
                        np.testing.assert_array_equal(
                            _strata(self._block(packed, seed_i, b), self.S), np.arange(self.S))

    def test_slicing_one_large_lh_is_not_stratified(self):
        """Control for the above: the rejected alternative really does lose stratification.

        Drawing one Latin Hypercube of ``n_buildings * S`` and cutting it into per-building
        blocks gives each building a random subset of the strata — ordinary Monte Carlo. If this
        ever starts passing, the packed generators have stopped being needed.
        """
        for gen in (1, 2):
            with self.subTest(generator=gen):
                big = get_correlation_generator(gen)(np.array([22], dtype=np.int64), 3 * self.S)[0]
                stratified = all(
                    np.array_equal(_strata(big[b * self.S:(b + 1) * self.S], self.S), np.arange(self.S))
                    for b in range(3)
                )
                self.assertFalse(stratified)


class TestPackedLatinHypercubeStreams(TestCase):
    """How each Latin Hypercube generator separates the buildings."""

    S = 12
    SEED = np.array([4242], dtype=np.int64)

    def test_mt_buildings_do_not_come_from_derived_seeds(self):
        """Generator 1 continues the group's one seeded stream rather than seeding per building.

        This is a deliberate choice, not an implementation detail: ``numpy.random.seed`` takes
        less than 2**32 and the group seeds already occupy only 31 bits, so folding a building
        index into the seed would multiply seed collisions by the building count. Continuing the
        stream introduces no new seeds at all.

        (Do not try to pin the stream by replaying it in plain numpy — numba's
        ``np.random.shuffle`` does not match numpy's, so even the *unpacked* generator cannot be
        reproduced that way. Building 1 against the unpacked draw is the check that matters, and
        ``TestPackedGeneratorsAgree`` makes it.)
        """
        n_buildings = np.array([3], dtype=np.int64)
        offsets = build_packed_rndm_offsets(n_buildings, self.S)
        packed = get_sample_generator(1)(self.SEED, self.S, n_buildings, offsets)

        for b in range(1, 3):
            block = packed[b * self.S:(b + 1) * self.S]
            for derived in (self.SEED[0] + b, self.SEED[0] * (b + 1), self.SEED[0] ^ b):
                candidate = get_correlation_generator(1)(np.array([derived], dtype=np.int64), self.S)[0]
                self.assertFalse(np.array_equal(block, candidate),
                                 f"building {b + 1} looks like it was seeded with {derived}")

    def test_philox_buildings_are_a_counter_coordinate(self):
        """Generator 2 keeps random access: a building's block does not depend on the others.

        Asking for four buildings and asking for two must give the same first two blocks.
        """
        wide = build_packed_rndm_offsets(np.array([4], dtype=np.int64), self.S)
        narrow = build_packed_rndm_offsets(np.array([2], dtype=np.int64), self.S)
        four = get_sample_generator(2)(self.SEED, self.S, np.array([4], dtype=np.int64), wide)
        two = get_sample_generator(2)(self.SEED, self.S, np.array([2], dtype=np.int64), narrow)
        np.testing.assert_array_equal(four[:2 * self.S], two)

    def test_mt_stream_is_sequential_not_random_access(self):
        """The counterpart: MT blocks also come out the same, because the stream is replayed."""
        wide = build_packed_rndm_offsets(np.array([4], dtype=np.int64), self.S)
        narrow = build_packed_rndm_offsets(np.array([2], dtype=np.int64), self.S)
        four = get_sample_generator(1)(self.SEED, self.S, np.array([4], dtype=np.int64), wide)
        two = get_sample_generator(1)(self.SEED, self.S, np.array([2], dtype=np.int64), narrow)
        np.testing.assert_array_equal(four[:2 * self.S], two)


if __name__ == "__main__":
    main()
