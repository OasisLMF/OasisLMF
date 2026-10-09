"""Post loss amplification over a building-packed GUL stream.

plapy scales every loss in an item by one factor, looked up from
``(event_id, amplification_id)`` -- per item and event, never per sample index. So a packed item's
buildings are all scaled identically, which is what amplification should do to them, and nothing
here needs to know the building dimension exists.

That is a claim about the code rather than about the stream format, and it is worth pinning:
plapy sits directly on the GUL stream, so it is one of only three things that see a packed item
before the financial module collapses it.
"""
from unittest import TestCase

import numpy as np
from numba import types
from numba.typed import Dict

from oasislmf.pytools.common.data import loss_pair_dtype, oasis_int
from oasislmf.pytools.common.event_stream import (CHANCE_OF_LOSS_IDX, MAX_LOSS_IDX, MEAN_IDX,
                                                  PIPE_CAPACITY, encode_sidx)
from oasislmf.pytools.pla.streams import read_buffer

EVENT_ID = 1
AMPLIFICATION_ID = 7
FACTOR = 1.25
CHANCE_OF_LOSS = 0.8
S = 4
N_BUILDINGS = 3


def _packed_item(item_id):
    """One packed item: every building's specials and samples, values distinct per record."""
    records = []
    for building in range(1, N_BUILDINGS + 1):
        for special_idx in (MAX_LOSS_IDX, MEAN_IDX):
            records.append((encode_sidx(building, special_idx, S), 100.0 * building - special_idx))
        for sample_idx in range(1, S + 1):
            records.append((encode_sidx(building, sample_idx, S), 10.0 * building + sample_idx))
    return sorted(records)


def _stream(item_id, records):
    header = np.array([EVENT_ID, item_id], dtype=oasis_int).tobytes()
    pairs = np.zeros(len(records) + 1, dtype=loss_pair_dtype)
    for i, (sidx, loss) in enumerate(records):
        pairs[i]['sidx'], pairs[i]['loss'] = sidx, loss
    return np.frombuffer(header + pairs.tobytes(), dtype='b').copy()


def _amplify(item_id, records, factor=FACTOR):
    """Run plapy's reader over one item and return {sidx: loss} as written out."""
    byte_mv = _stream(item_id, records)
    out_byte_mv = np.zeros(PIPE_CAPACITY, dtype='b')

    items_amps = np.zeros(item_id + 1, dtype=oasis_int)
    items_amps[item_id] = AMPLIFICATION_ID
    plafactors = Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=types.float64)
    plafactors[(EVENT_ID, AMPLIFICATION_ID)] = factor

    out_cursor = np.empty(1, dtype='i4')                # the reader reports its length here
    read_buffer(byte_mv, 0, byte_mv.shape[0], 0, 0,
                items_amps, plafactors, 1.0, out_byte_mv, out_cursor)

    raw = out_byte_mv[:out_cursor[0]].tobytes()[8:]     # skip the (event_id, item_id) header
    pairs = np.frombuffer(raw[:len(records) * 8], dtype=loss_pair_dtype)
    return {int(p['sidx']): float(p['loss']) for p in pairs}


class TestPackedAmplification(TestCase):

    def test_every_building_is_amplified(self):
        records = _packed_item(11)
        got = _amplify(11, records)
        for sidx, loss in records:
            self.assertAlmostEqual(got[sidx], loss * FACTOR, places=3,
                                   msg=f"sidx {sidx} was not amplified")

    def test_one_factor_across_the_whole_item(self):
        """The factor is per item and event, so the buildings must not drift apart."""
        records = _packed_item(11)
        got = _amplify(11, records)
        factors = {round(got[sidx] / loss, 6) for sidx, loss in records if loss}
        self.assertEqual(factors, {FACTOR})

    def test_packed_indices_are_preserved(self):
        """plapy rewrites losses in place; the building dimension must survive it untouched."""
        records = _packed_item(11)
        got = _amplify(11, records)
        self.assertEqual(sorted(got.keys()), sorted(sidx for sidx, _ in records))
        self.assertTrue(any(sidx > S for sidx in got), "expected packed sample indices")
        self.assertTrue(any(sidx < -5 for sidx in got), "expected packed special indices")


def _packed_item_with_chance_of_loss(item_id):
    """Like _packed_item, plus each building's chance-of-loss.

    Kept separate from _packed_item because the tests above assert that every record it contains
    is amplified, which is exactly what must NOT happen to chance-of-loss.
    """
    records = _packed_item(item_id)
    for building in range(1, N_BUILDINGS + 1):
        records.append((encode_sidx(building, CHANCE_OF_LOSS_IDX, S), CHANCE_OF_LOSS))
    return sorted(records)


class TestChanceOfLossIsNotAmplified(TestCase):
    """Chance-of-loss is a probability, so an amplification factor must not touch it.

    Every other special scales with the loss -- mean and max obviously, and tiv deliberately, so
    that an amplified loss is not clipped by the coverage cap. Chance-of-loss is P(loss > 0);
    multiplying it is meaningless and pushes it above 1 for any factor above 1/p.

    plapy sits directly on the ground-up item stream, before the financial module, and the
    financial module passes -4 through untouched, so a corrupted value survives the whole
    pipeline.
    """

    def test_it_survives_a_factor_unchanged(self):
        records = _packed_item_with_chance_of_loss(item_id=1)
        out = _amplify(1, records)
        for building in range(1, N_BUILDINGS + 1):
            with self.subTest(building=building):
                self.assertAlmostEqual(out[encode_sidx(building, CHANCE_OF_LOSS_IDX, S)],
                                       CHANCE_OF_LOSS, places=5)

    def test_every_other_special_still_scales(self):
        """The guard must be surgical: only chance-of-loss is spared."""
        records = _packed_item_with_chance_of_loss(item_id=1)
        out = _amplify(1, records)
        for building in range(1, N_BUILDINGS + 1):
            for special_idx in (MAX_LOSS_IDX, MEAN_IDX):
                with self.subTest(building=building, special=special_idx):
                    sidx = encode_sidx(building, special_idx, S)
                    self.assertAlmostEqual(out[sidx], (100.0 * building - special_idx) * FACTOR, places=4)

    def test_it_is_spared_for_every_building_not_just_the_first(self):
        """The reason the check decodes rather than comparing to -4.

        A packed item carries one chance-of-loss PER BUILDING, at -4, -9, -14 ... Testing
        ``sidx == CHANCE_OF_LOSS_IDX`` would spare building 1 and amplify all the others, which
        is the shape of bug this guards.
        """
        records = _packed_item_with_chance_of_loss(item_id=1)
        out = _amplify(1, records)
        packed_sidx = [encode_sidx(b, CHANCE_OF_LOSS_IDX, S) for b in range(2, N_BUILDINGS + 1)]
        self.assertNotIn(CHANCE_OF_LOSS_IDX, packed_sidx)   # they really are different indices
        for sidx in packed_sidx:
            with self.subTest(sidx=sidx):
                self.assertAlmostEqual(out[sidx], CHANCE_OF_LOSS, places=5)
