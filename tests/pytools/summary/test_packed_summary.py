"""Tests for how the summary reader handles building-packed GUL streams.

A summary is per item and a packed item covers all its buildings, so the reader decodes each
packed index onto the sample it stands for and lets the accumulation add the buildings up.

The zero-sample case matters on its own. ``number_of_samples`` defaults to 0, giving a mean-only
run, and a packed item still carries per-building specials -- building 2's mean, TIV and max sit
at -6, -8 and -10 whatever the sample size. Skipping the decode there drops buildings 2..N from
every summary, silently and with no error.
"""
from unittest import TestCase

import numpy as np

from oasislmf.pytools.common.data import loss_pair_dtype, oasis_float, oasis_int
from oasislmf.pytools.common.event_stream import MAX_LOSS_IDX, MEAN_IDX, TIV_IDX, encode_sidx
from oasislmf.pytools.summary.manager import SPECIAL_SIDX_COUNT, read_buffer

EVENT_ID, ITEM_ID = 1, 1


def _stream(records):
    """Serialise one item: (event_id, item_id) header, (sidx, loss) pairs, then the delimiter."""
    buf = np.array([EVENT_ID, ITEM_ID], dtype=oasis_int).tobytes()
    pairs = np.zeros(len(records) + 1, dtype=loss_pair_dtype)
    for i, (sidx, loss) in enumerate(records):
        pairs[i]['sidx'], pairs[i]['loss'] = sidx, loss
    return np.frombuffer(buf + pairs.tobytes(), dtype='b').copy()


def _read(records, len_sample):
    """Run the reader over one item and return its summary row."""
    loss_summary = np.zeros((1, len_sample + SPECIAL_SIDX_COUNT), dtype=oasis_float)
    byte_mv = _stream(records)
    read_buffer(
        byte_mv, 0, byte_mv.shape[0], 0, 0,
        summary_sets_id=np.array([1], dtype=oasis_int),
        summary_set_index_to_loss_ptr=np.array([0, 1], dtype=oasis_int),
        item_id_to_summary_id=np.array([[0], [1]], dtype=oasis_int),  # item_id -> one id per summary set
        loss_index=np.zeros(1, dtype=oasis_int),
        loss_summary=loss_summary,
        present_summary_id=np.zeros(1, dtype=oasis_int),
        summary_set_index_to_present_loss_ptr_end=np.array([0, 1], dtype=oasis_int),
        item_id_to_risks_i=np.zeros(2, dtype=oasis_int),
        is_risk_affected=np.zeros(2, dtype=np.uint8),
        has_affected_risk=None,
    )
    return loss_summary[0]


class TestPackedSummary(TestCase):

    def test_buildings_are_summed_into_one_summary(self):
        S = 4
        records = []
        for building, scale in ((1, 1.0), (2, 10.0)):
            records.append((encode_sidx(building, MEAN_IDX, S), 7.0 * scale))
            for sample_idx in range(1, S + 1):
                records.append((encode_sidx(building, sample_idx, S), sample_idx * scale))
        got = _read(sorted(records), S)
        for sample_idx in range(1, S + 1):
            self.assertAlmostEqual(got[sample_idx], sample_idx * 11.0, places=4)
        self.assertAlmostEqual(got[MEAN_IDX], 7.0 * 11.0, places=4)

    def test_a_mean_only_run_keeps_every_building(self):
        """number_of_samples defaults to 0, and the specials are still per building."""
        records = []
        for building, scale in ((1, 1.0), (2, 10.0)):
            for special_idx, value in ((MAX_LOSS_IDX, 5.0), (TIV_IDX, 100.0), (MEAN_IDX, 3.0)):
                records.append((encode_sidx(building, special_idx, 0), value * scale))
        got = _read(sorted(records), 0)
        self.assertAlmostEqual(got[MAX_LOSS_IDX], 5.0 * 11.0, places=4)
        self.assertAlmostEqual(got[TIV_IDX], 100.0 * 11.0, places=4)
        self.assertAlmostEqual(got[MEAN_IDX], 3.0 * 11.0, places=4)

    def test_an_ordinary_stream_is_untouched(self):
        S = 4
        records = [(MEAN_IDX, 7.0)] + [(s, float(s)) for s in range(1, S + 1)]
        got = _read(records, S)
        for sample_idx in range(1, S + 1):
            self.assertAlmostEqual(got[sample_idx], float(sample_idx), places=4)
        self.assertAlmostEqual(got[MEAN_IDX], 7.0, places=4)
