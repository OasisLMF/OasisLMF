"""Tests for gulmc building-packed stream writing (write_losses_packed).

Verifies the on-wire layout: one stream item (header + single delimiter) per item_id, with
each building's 5 special records (identical values, building-shifted sidx) followed by its
random samples, all multiplexed via encode_sidx. Decoding with decode_building /
decode_local_sidx must recover the original (building, local_sidx, loss) values.

An item whose buildings nothing downstream can tell apart is instead written summed, as an
ordinary unpacked item -- see TestWriteSummedAtSource. That is what keeps a packed stream
unambiguous for the financial module, whose reader can only discriminate on the sidx range.
"""
from unittest import main, TestCase

import numpy as np

from oasislmf.pytools.common.data import oasis_float, oasis_int
from oasislmf.pytools.common.event_stream import decode_building, decode_local_sidx, encode_sidx
from oasislmf.pytools.gul.common import SPECIAL_SIDX, NUM_IDX, CHANCE_OF_LOSS_IDX
from oasislmf.pytools.gulmc.manager import write_losses_packed

record_dtype = np.dtype([('sidx', '<i4'), ('loss', oasis_float)])


def _decode_stream(byte_mv, cursor):
    """Parse the written buffer into a list of (event_id, item_id, [(sidx, loss), ...])."""
    raw = byte_mv[:cursor].tobytes()
    items = []
    pos = 0
    while pos < len(raw):
        event_id, item_id = np.frombuffer(raw[pos:pos + 8], dtype='<i4')
        pos += 8
        records = []
        while True:
            sidx = np.frombuffer(raw[pos:pos + 4], dtype='<i4')[0]
            loss = np.frombuffer(raw[pos + 4:pos + 8], dtype=oasis_float)[0]
            pos += 8
            if sidx == 0:  # delimiter ends the item
                break
            records.append((int(sidx), float(loss)))
        items.append((int(event_id), int(item_id), records))
    return items


class TestWriteLossesPacked(TestCase):

    def setUp(self):
        self.S = 4
        self.event_id = 7
        # two items: item 10 has 3 buildings, item 20 has 1 building
        self.item_ids = np.array([10, 20], dtype=oasis_int)
        self.n_buildings = np.array([3, 1], dtype=np.int64)
        max_items = 2
        max_buildings = 3

        # specials buffer (S + NUM_IDX + 1, max_items); negative-indexed special rows
        self.losses = np.zeros((self.S + NUM_IDX + 1, max_items), dtype=oasis_float)
        for item_j in range(max_items):
            for special_idx in SPECIAL_SIDX:
                self.losses[special_idx, item_j] = 100 * (item_j + 1) + (-int(special_idx))

        # building samples (S, max_items, max_buildings) — distinct per (sample, item, building)
        self.building_losses = np.zeros((self.S, max_items, max_buildings), dtype=oasis_float)
        for s in range(self.S):
            for item_j in range(max_items):
                for b in range(max_buildings):
                    self.building_losses[s, item_j, b] = 1000 * (item_j + 1) + 10 * (b + 1) + (s + 1)

        self.byte_mv = np.zeros(1 << 16, dtype='b')

    def _write(self, loss_threshold=0.0, keep_separate=True):
        # n_buildings is signed: negative means "keep the buildings separate"
        signed = -np.abs(self.n_buildings) if keep_separate else np.abs(self.n_buildings)
        cursor = write_losses_packed(self.event_id, self.S, loss_threshold, self.losses,
                                     self.building_losses, self.item_ids, signed,
                                     0, 0.0, self.byte_mv, 0)
        return _decode_stream(self.byte_mv, cursor)

    def test_one_stream_item_per_item_id(self):
        items = self._write()
        self.assertEqual(len(items), 2)
        self.assertEqual([it[1] for it in items], [10, 20])
        self.assertTrue(all(it[0] == self.event_id for it in items))

    def test_record_counts(self):
        """Each item emits n_buildings * (NUM_IDX specials + S samples) records."""
        items = self._write()
        self.assertEqual(len(items[0][2]), 3 * (NUM_IDX + self.S))
        self.assertEqual(len(items[1][2]), 1 * (NUM_IDX + self.S))

    def test_specials_replicated_per_building(self):
        """Special values are building-independent; sidx is building-shifted and decodes back."""
        items = self._write()
        _, item_id, records = items[0]  # item 10, item_j == 0
        for sidx, loss in records:
            if sidx < 0:
                b = decode_building(sidx, self.S)
                local = decode_local_sidx(sidx, self.S)
                self.assertIn(b, (1, 2, 3))
                self.assertEqual(loss, self.losses[local, 0])

    def test_samples_decode_to_correct_building_and_value(self):
        items = self._write()
        for item_idx, (_, item_id, records) in enumerate(items):
            for sidx, loss in records:
                if sidx > 0:
                    b = decode_building(sidx, self.S)
                    s = decode_local_sidx(sidx, self.S)
                    self.assertEqual(loss, self.building_losses[s - 1, item_idx, b - 1])

    def test_threshold_filters_only_positive_samples(self):
        """A high threshold drops random samples but keeps all special records."""
        # threshold above every sample value for item 10 building 1 sample 1 (==1011) but below others
        items = self._write(loss_threshold=1e9)
        for _, item_id, records in items:
            self.assertTrue(all(sidx < 0 for sidx, _ in records))  # only specials survive
        # specials still all present: n_buildings * NUM_IDX
        self.assertEqual(len(items[0][2]), 3 * NUM_IDX)
        self.assertEqual(len(items[1][2]), 1 * NUM_IDX)

    def test_single_building_matches_legacy_sidx(self):
        """For a 1-building item, packed sidx are the plain legacy sidx (identity)."""
        items = self._write()
        _, item_id, records = items[1]  # item 20, single building
        sidxs = sorted(sidx for sidx, _ in records)
        expected = sorted(list(SPECIAL_SIDX) + list(range(1, self.S + 1)))
        self.assertEqual(sidxs, expected)


class TestWriteSummedAtSource(TestCase):
    """keep_buildings_separate == 0 collapses the buildings into an ordinary item.

    The financial module's reader discriminates only on the sidx range, so it cannot tell a
    packed item it should collapse from one it should not. Items whose buildings the site levels
    would sum anyway are therefore summed here, and never reach it packed.
    """

    def setUp(self):
        self.S = 4
        self.event_id = 7
        self.item_ids = np.array([10], dtype=oasis_int)
        self.n_buildings = np.array([3], dtype='i4')
        self.losses = np.zeros((self.S + NUM_IDX + 1, 1), dtype=oasis_float)
        for special_idx in SPECIAL_SIDX:
            self.losses[special_idx, 0] = 100 + (-int(special_idx))
        self.building_losses = np.zeros((self.S, 1, 3), dtype=oasis_float)
        for s in range(self.S):
            for b in range(3):
                self.building_losses[s, 0, b] = 10 * (b + 1) + (s + 1)
        self.byte_mv = np.zeros(1 << 16, dtype='b')

    def _records(self):
        cursor = write_losses_packed(self.event_id, self.S, 0.0, self.losses, self.building_losses,
                                     self.item_ids, np.abs(self.n_buildings),
                                     0, 0.0, self.byte_mv, 0)
        items = _decode_stream(self.byte_mv, cursor)
        self.assertEqual(len(items), 1)
        return dict(items[0][2])

    def test_no_packed_sidx_is_emitted(self):
        """Every sidx is an ordinary one, so a consumer cannot mistake it for a packed item."""
        recs = self._records()
        for sidx in recs:
            self.assertGreaterEqual(sidx, -len(SPECIAL_SIDX))
            self.assertLessEqual(sidx, self.S)

    def test_samples_are_the_sum_over_buildings(self):
        recs = self._records()
        for sample_idx in range(1, self.S + 1):
            expected = sum(self.building_losses[sample_idx - 1, 0, b] for b in range(3))
            self.assertAlmostEqual(recs[sample_idx], expected, places=4)

    def test_additive_specials_scale_with_the_building_count(self):
        """max, tiv, mean (and the std the financial module ignores) aggregate across buildings."""
        recs = self._records()
        for special_idx in SPECIAL_SIDX:
            if int(special_idx) == CHANCE_OF_LOSS_IDX:
                continue
            self.assertAlmostEqual(recs[int(special_idx)],
                                   self.losses[special_idx, 0] * 3, places=4)

    def test_chance_of_loss_is_taken_once(self):
        """It is a property of the risk, not a quantity to add up."""
        recs = self._records()
        self.assertAlmostEqual(recs[CHANCE_OF_LOSS_IDX], self.losses[CHANCE_OF_LOSS_IDX, 0], places=4)

    def test_single_building_is_unchanged_by_summing(self):
        self.n_buildings = np.array([1], dtype='i4')
        recs = self._records()
        for sample_idx in range(1, self.S + 1):
            self.assertAlmostEqual(recs[sample_idx], self.building_losses[sample_idx - 1, 0, 0], places=4)
        for special_idx in SPECIAL_SIDX:
            self.assertAlmostEqual(recs[int(special_idx)], self.losses[special_idx, 0], places=4)


if __name__ == "__main__":
    main()


class TestPerCoverageTivCap(TestCase):
    """alloc_rule caps a coverage's item losses at its TIV, and the cap is per building.

    Generation divides a location's TIV by its building count, so the coverage TIV reaching the
    writer is the per-building share -- the same share row disaggregation gives each building's
    own coverage. Capping across the location instead would let one building's losses eat another
    building's insured value.
    """

    S = 2
    TIV = 100.0

    def setUp(self):
        self.event_id = 7
        self.item_ids = np.array([10, 20], dtype=oasis_int)      # two perils on one coverage
        self.n_buildings = np.array([2, 2], dtype='i4')
        self.losses = np.zeros((self.S + NUM_IDX + 1, 2), dtype=oasis_float)
        self.building_losses = np.zeros((self.S, 2, 2), dtype=oasis_float)
        # building 0 is over the per-building TIV (60 + 90 = 150), building 1 is under (30 + 30)
        for s in range(self.S):
            self.building_losses[s, 0, 0], self.building_losses[s, 1, 0] = 60.0, 90.0
            self.building_losses[s, 0, 1], self.building_losses[s, 1, 1] = 30.0, 30.0
        self.byte_mv = np.zeros(1 << 16, dtype='b')

    def _records(self, alloc_rule, keep_separate=1):
        signed = -np.abs(self.n_buildings) if keep_separate else np.abs(self.n_buildings)
        cursor = write_losses_packed(
            self.event_id, self.S, 0.0, self.losses, self.building_losses.copy(),
            self.item_ids, signed, alloc_rule, self.TIV, self.byte_mv, 0)
        return {item_id: dict(records) for _, item_id, records in _decode_stream(self.byte_mv, cursor)}

    def test_each_building_is_capped_against_its_own_tiv(self):
        """Building 0 is scaled to the TIV; building 1 was already under it and is untouched."""
        recs = self._records(alloc_rule=1)
        for sample_idx in range(1, self.S + 1):
            b0 = sum(recs[i][encode_sidx(1, sample_idx, self.S)] for i in (10, 20))
            b1 = sum(recs[i][encode_sidx(2, sample_idx, self.S)] for i in (10, 20))
            self.assertAlmostEqual(b0, self.TIV, places=3)   # 150 -> 100
            self.assertAlmostEqual(b1, 60.0, places=3)       # under the cap, unchanged

    def test_the_cap_is_shared_in_proportion(self):
        recs = self._records(alloc_rule=1)
        sidx = encode_sidx(1, 1, self.S)
        self.assertAlmostEqual(recs[10][sidx], 60.0 * self.TIV / 150.0, places=3)
        self.assertAlmostEqual(recs[20][sidx], 90.0 * self.TIV / 150.0, places=3)

    def test_alloc_rule_zero_does_not_cap(self):
        recs = self._records(alloc_rule=0)
        sidx = encode_sidx(1, 1, self.S)
        self.assertAlmostEqual(recs[10][sidx], 60.0, places=3)
        self.assertAlmostEqual(recs[20][sidx], 90.0, places=3)

    def test_capping_across_the_location_would_be_wrong(self):
        """Control: the per-building cap is not the same as capping the location total.

        Across the location the two buildings sum to 210 against a location TIV of
        n_buildings * TIV = 200, which would scale building 1 down too. Per building it is
        untouched.
        """
        recs = self._records(alloc_rule=1)
        sidx = encode_sidx(2, 1, self.S)
        b1_total = recs[10][sidx] + recs[20][sidx]
        self.assertAlmostEqual(b1_total, 60.0, places=3)
        self.assertNotAlmostEqual(b1_total, 60.0 * (2 * self.TIV) / 210.0, places=3)

    def test_a_summed_item_is_capped_at_the_location_tiv(self):
        """Blocks are capped before they are summed, so the summed item lands at n * TIV."""
        recs = self._records(alloc_rule=1, keep_separate=0)
        for sample_idx in range(1, self.S + 1):
            total = sum(recs[i][sample_idx] for i in (10, 20))
            self.assertAlmostEqual(total, self.TIV + 60.0, places=3)
            self.assertLessEqual(total, 2 * self.TIV + 1e-6)


class TestItemsWithDifferentBuildingCounts(TestCase):
    """Items on one coverage need not agree on their building count.

    The alloc-rule passes work across items at a fixed building index, so a slot belonging to no
    item at that index has to read 0. The buffer is reused between coverages and the compute loops
    only fill the slots each item owns, so the writer clears the rest itself rather than trusting
    its caller to have done it.

    Not reachable through generation today -- NumberOfBuildings is a location attribute and a
    coverage belongs to one location -- but the writer is shared by two compute loops and takes
    the counts as an argument, so nothing in its own contract says they must match.
    """

    S = 2
    # low enough that the cap binds only if a stale slot is counted: item 20 alone is under it at
    # every building index, but a leftover 999 alongside would push the pair well over
    TIV = 100.0

    def setUp(self):
        self.event_id = 7
        self.item_ids = np.array([10, 20], dtype=oasis_int)
        self.n_buildings = np.array([1, 3], dtype='i4')      # deliberately different
        self.losses = np.zeros((self.S + NUM_IDX + 1, 2), dtype=oasis_float)
        # a buffer left dirty by a previous coverage, in the slots item 10 does not own
        self.building_losses = np.full((self.S, 2, 3), 999.0, dtype=oasis_float)
        for s in range(self.S):
            self.building_losses[s, 0, 0] = 10.0
            for b in range(3):
                self.building_losses[s, 1, b] = 20.0 + b

    def _records(self, alloc_rule):
        cursor = write_losses_packed(
            self.event_id, self.S, 0.0, self.losses, self.building_losses.copy(),
            self.item_ids, -np.abs(self.n_buildings),
            alloc_rule, self.TIV, self.byte_mv, 0)
        return {item_id: dict(records) for _, item_id, records in _decode_stream(self.byte_mv, cursor)}

    byte_mv = np.zeros(1 << 16, dtype='b')

    def test_a_stale_slot_cannot_reach_the_output(self):
        """Item 10 owns one building, so its blocks 2 and 3 must not appear."""
        for alloc_rule in (0, 1, 2, 3):
            with self.subTest(alloc_rule=alloc_rule):
                recs = self._records(alloc_rule)
                for building in (2, 3):
                    self.assertNotIn(encode_sidx(building, 1, self.S), recs[10])

    def test_a_stale_slot_does_not_bind_the_cap_for_another_item(self):
        """The failure this guards: item 20 is under the TIV at every building index on its own.

        Counting item 10's leftover 999 at buildings 2 and 3 would put the pair over it and scale
        item 20 down, on losses that were never anywhere near its insured value.
        """
        recs = self._records(alloc_rule=1)
        for building in (1, 2, 3):
            self.assertAlmostEqual(recs[20][encode_sidx(building, 1, self.S)],
                                   20.0 + building - 1, places=3)
        self.assertAlmostEqual(recs[10][encode_sidx(1, 1, self.S)], 10.0, places=3)
