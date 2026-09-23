"""Tests for building-packed random draws, which exist on random_generator 2 alone.

Packing multiplexes a location's N buildings into the sample dimension of one item, so each
location needs N independent blocks of S draws instead of one. Only Philox can supply them
cheaply: it is counter-based, so the building is a third counter coordinate and
``_lh_philox_block`` produces a block where that block is consumed. The Mersenne Twister
generators can only reach building b by replaying their seed's stream from the start, which
means materialising all N*S values before the event runs, so packing is refused for them --
see ``check_packing_supported``, pinned at the bottom of this file.

Two properties carry the design:

* **Building 0 reproduces the unpacked draw**, so an unpacked run is unchanged by packing.
* **Each building gets a Latin Hypercube of its own** -- not a slice of one larger sample, which
  would leave each building holding a random subset of the strata. See
  ``test_slicing_one_large_lh_is_not_stratified``, the control that pins why this design exists.
"""
from unittest import main, TestCase

import numpy as np

from oasislmf.pytools.common.event_stream import check_packing_supported
from oasislmf.pytools.gul.random import (
    PHILOX_SHIFT32,
    PHILOX_U32_MASK,
    _lh_philox_block,
    get_random_generator,
    random_LatinHypercube_Philox7,
)
from oasislmf.utils.exceptions import OasisException


def _strata(block, n):
    """The stratum index each sample falls in, sorted. A Latin Hypercube hits each exactly once."""
    return np.sort(np.floor(np.asarray(block) * n).astype(int))


def _block(seed, building, n):
    """One building's block, the way gulpy and gulmc ask for it."""
    s = np.uint64(seed)
    perms = np.empty(n, dtype='float64')
    out = np.empty(n, dtype='float64')
    _lh_philox_block(np.uint32(s & PHILOX_U32_MASK), np.uint32(s >> PHILOX_SHIFT32),
                     building, n, perms, out)
    return out


class TestPhiloxPackedBlocks(TestCase):
    """The per-building block contract that building packing rests on."""

    S = 20
    SEEDS = np.array([11, 22, 33], dtype=np.int64)

    def test_building_zero_reproduces_the_unpacked_draw(self):
        """The single-building case must be unchanged by packing."""
        unpacked = random_LatinHypercube_Philox7(self.SEEDS, self.S)
        for seed_i, seed in enumerate(self.SEEDS):
            np.testing.assert_array_equal(_block(seed, 0, self.S), unpacked[seed_i])

    def test_buildings_are_distinct(self):
        for seed in self.SEEDS:
            blocks = [_block(seed, b, self.S) for b in range(4)]
            for x in range(len(blocks)):
                for y in range(x + 1, len(blocks)):
                    self.assertFalse(np.array_equal(blocks[x], blocks[y]))

    def test_values_are_in_the_unit_interval(self):
        for seed in self.SEEDS:
            for b in range(4):
                block = _block(seed, b, self.S)
                self.assertTrue((block > 0).all())
                self.assertTrue((block <= 1).all())

    def test_every_building_gets_its_own_latin_hypercube(self):
        """The point of the design: stratification is per building, not shared across them."""
        for seed in self.SEEDS:
            for b in range(4):
                np.testing.assert_array_equal(_strata(_block(seed, b, self.S), self.S),
                                              np.arange(self.S))

    def test_slicing_one_large_lh_is_not_stratified(self):
        """Control for the above: the rejected alternative really does lose stratification.

        Drawing one Latin Hypercube of ``n_buildings * S`` and cutting it into per-building
        blocks gives each building a random subset of the strata -- ordinary Monte Carlo. If this
        ever starts passing, per-building blocks have stopped being needed.
        """
        big = random_LatinHypercube_Philox7(np.array([22], dtype=np.int64), 3 * self.S)[0]
        stratified = all(
            np.array_equal(_strata(big[b * self.S:(b + 1) * self.S], self.S), np.arange(self.S))
            for b in range(3)
        )
        self.assertFalse(stratified)

    def test_a_block_does_not_depend_on_the_others(self):
        """Random access, which is what lets the draw be lazy: block b is a pure function of (key, b).

        Asking for building 3 without having asked for 0..2 must give the same values. This is the
        property the Mersenne Twister generators lack, and the reason they cannot pack cheaply.
        """
        seed = self.SEEDS[0]
        in_order = [_block(seed, b, self.S) for b in range(4)]
        np.testing.assert_array_equal(_block(seed, 3, self.S), in_order[3])
        np.testing.assert_array_equal(_block(seed, 1, self.S), in_order[1])

    def test_repeatable(self):
        seed = self.SEEDS[1]
        np.testing.assert_array_equal(_block(seed, 2, self.S), _block(seed, 2, self.S))


class TestPackingIsGeneratorTwoOnly(TestCase):
    """check_packing_supported: packing is accepted on generator 2 and refused elsewhere."""

    def test_packed_items_are_accepted_on_generator_2(self):
        check_packing_supported(2, np.array([1, -8, 40], dtype=np.int32))

    def test_unpacked_items_are_accepted_on_every_generator(self):
        for gen in (0, 1, 2):
            with self.subTest(generator=gen):
                check_packing_supported(gen, np.array([1, 1, 1], dtype=np.int32))

    def test_an_empty_item_array_is_accepted(self):
        for gen in (0, 1, 2):
            with self.subTest(generator=gen):
                check_packing_supported(gen, np.zeros(0, dtype=np.int32))

    def test_packed_items_are_refused_on_the_mersenne_twister_generators(self):
        for gen in (0, 1):
            for packed in (np.array([1, 40], dtype=np.int32), np.array([1, -40], dtype=np.int32)):
                with self.subTest(generator=gen, packed_buildings=list(packed)):
                    with self.assertRaises(OasisException) as raised:
                        check_packing_supported(gen, packed)
                    # the message has to name the way out, not just the refusal
                    self.assertIn("--random-generator=2", str(raised.exception))
                    self.assertIn("40", str(raised.exception))

    def test_the_sign_does_not_change_the_verdict(self):
        """A summed item packs just as much as a kept-separate one; only the magnitude counts."""
        for packed in (np.array([-2], dtype=np.int32), np.array([2], dtype=np.int32)):
            with self.subTest(packed_buildings=list(packed)):
                with self.assertRaises(OasisException):
                    check_packing_supported(0, packed)


class TestGeneratorDispatch(TestCase):
    """One dispatch serves both the per-rng-group and per-correlation-group draws."""

    def test_every_generator_id_resolves(self):
        for gen in (0, 1, 2):
            with self.subTest(generator=gen):
                self.assertIsNotNone(get_random_generator(gen))

    def test_unknown_generator_is_rejected(self):
        with self.assertRaises(ValueError):
            get_random_generator(3)


if __name__ == "__main__":
    main()
