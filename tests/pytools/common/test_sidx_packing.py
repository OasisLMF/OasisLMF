"""Tests for the building-packed sidx encode/decode helpers.

Buildings of one OED location are multiplexed into the sample dimension of a single
stream item. These tests pin down the packing scheme used by the producer (gulmc/gulpy)
and the consumers (fmpy/summarypy/plapy):

    positive sample : sidx = (building - 1) * S + local_sidx        local_sidx in [1, S]
    special sample  : sidx = local_sidx - (building - 1) * NUM_SPECIAL_SIDX
                                                                    local_sidx in [-5, -1]
"""
from unittest import main, TestCase

from oasislmf.pytools.common.event_stream import (
    NUM_SPECIAL_SIDX,
    MEAN_IDX,
    STD_DEV_IDX,
    TIV_IDX,
    CHANCE_OF_LOSS_IDX,
    MAX_LOSS_IDX,
    encode_sidx,
    decode_building,
    decode_local_sidx,
)

SPECIAL_SIDX = [MEAN_IDX, STD_DEV_IDX, TIV_IDX, CHANCE_OF_LOSS_IDX, MAX_LOSS_IDX]


class TestSidxPacking(TestCase):

    def test_roundtrip_positive_samples(self):
        """encode then decode recovers (building, local_sidx) for random samples."""
        for sample_size in (1, 2, 10, 1000):
            for building in range(1, 6):
                for local_sidx in range(1, sample_size + 1):
                    sidx = encode_sidx(building, local_sidx, sample_size)
                    self.assertGreater(sidx, 0)
                    self.assertEqual(decode_building(sidx, sample_size), building)
                    self.assertEqual(decode_local_sidx(sidx, sample_size), local_sidx)

    def test_roundtrip_special_samples(self):
        """encode then decode recovers (building, local_sidx) for special indices."""
        for sample_size in (1, 10, 1000):
            for building in range(1, 6):
                for local_sidx in SPECIAL_SIDX:
                    sidx = encode_sidx(building, local_sidx, sample_size)
                    self.assertLess(sidx, 0)
                    self.assertEqual(decode_building(sidx, sample_size), building)
                    self.assertEqual(decode_local_sidx(sidx, sample_size), local_sidx)

    def test_single_building_identity(self):
        """For building 1, the packed sidx equals the local sidx (back-compatible)."""
        for sample_size in (1, 10, 256):
            for local_sidx in list(range(1, sample_size + 1)) + SPECIAL_SIDX:
                self.assertEqual(encode_sidx(1, local_sidx, sample_size), local_sidx)
                self.assertEqual(decode_building(local_sidx, sample_size), 1)
                self.assertEqual(decode_local_sidx(local_sidx, sample_size), local_sidx)

    def test_known_packed_values(self):
        """Pin the exact wire encoding so the scheme cannot drift silently."""
        S = 10
        # building 2 random samples occupy sidx 11..20
        self.assertEqual(encode_sidx(2, 1, S), 11)
        self.assertEqual(encode_sidx(2, S, S), 20)
        self.assertEqual(encode_sidx(3, 1, S), 21)
        # special blocks: building 1 -> -1..-5, building 2 -> -6..-10
        self.assertEqual(encode_sidx(1, MEAN_IDX, S), -1)
        self.assertEqual(encode_sidx(1, MAX_LOSS_IDX, S), -5)
        self.assertEqual(encode_sidx(2, MEAN_IDX, S), -6)
        self.assertEqual(encode_sidx(2, MAX_LOSS_IDX, S), -10)

    def test_positive_blocks_are_contiguous_and_disjoint(self):
        """Random-sample blocks tile [1, N*S] with no gaps or overlaps."""
        S, N = 7, 4
        seen = []
        for building in range(1, N + 1):
            for local_sidx in range(1, S + 1):
                seen.append(encode_sidx(building, local_sidx, S))
        self.assertEqual(sorted(seen), list(range(1, N * S + 1)))

    def test_special_blocks_are_contiguous_and_disjoint(self):
        """Special-index blocks tile [-N*K, -1] with no gaps or overlaps."""
        S, N = 7, 4
        seen = []
        for building in range(1, N + 1):
            for local_sidx in SPECIAL_SIDX:
                seen.append(encode_sidx(building, local_sidx, S))
        self.assertEqual(sorted(seen), list(range(-N * NUM_SPECIAL_SIDX, 0)))

    def test_delimiter_decodes_to_zero(self):
        """sidx == 0 is the item delimiter and carries no building/local sidx."""
        for sample_size in (1, 10, 1000):
            self.assertEqual(decode_building(0, sample_size), 0)
            self.assertEqual(decode_local_sidx(0, sample_size), 0)


if __name__ == "__main__":
    main()
