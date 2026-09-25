"""Unit test for how fmpy reads building-packed GUL streams.

A building-packed GUL stream multiplexes N buildings of one item into the sample dimension via
``encode_sidx`` (building 1 == identity). ``read_buffer`` **preserves** that dimension: the
records are stored at their packed sidx so the site FM levels can apply their terms per building,
and the collapse happens afterwards at ``compute_info['site_collapse_level']``.

Only items whose buildings must survive arrive packed. The site levels aggregate on
('loc_id', 'risk_id'), and risk_id is 1 for every building of a non-aggregate location, so those
buildings would be summed before any term is applied -- nothing downstream can tell them apart
and the ground-up tool sums them at source. An aggregate location gets one site node per building
instead, each carrying term/NumberOfRisks, and those are the ones that reach this reader packed.

A normal single-building stream is unaffected: for building 1 the packed encoding is the identity.
"""
from unittest import TestCase

import numpy as np

from oasislmf.pytools.common.data import oasis_int, loss_pair_dtype
from oasislmf.pytools.common.event_stream import decode_local_sidx, encode_sidx
from oasislmf.pytools.fm.common import compute_idx_dtype
from oasislmf.pytools.fm.financial_structure import nodes_array_dtype
from oasislmf.pytools.fm.stream_sparse import read_buffer

S = 4  # logical sample size


def _build_item_buffer(event_id, item_id, records):
    """Serialise one item (header + (sidx, loss) pairs + delimiter) to a byte view."""
    header = np.array([event_id, item_id], dtype=oasis_int).tobytes()
    pairs = np.zeros(len(records) + 1, dtype=loss_pair_dtype)
    for i, (sidx, loss) in enumerate(records):
        pairs[i]['sidx'] = sidx
        pairs[i]['loss'] = loss
    # trailing (0, 0) delimiter terminates the item
    buf = header + pairs.tobytes()
    return np.frombuffer(buf, dtype='b')


def _fresh_state():
    """Minimal sparse-array state for a single item (item_id == 1, node_id == 1)."""
    nodes_array = np.zeros(2, dtype=nodes_array_dtype)
    nodes_array[1]['node_id'] = 1
    nodes_array[1]['loss'] = 0
    nodes_array[1]['layer_len'] = 1
    state = dict(
        nodes_array=nodes_array,
        sidx_indexes=np.zeros(2, dtype=oasis_int),
        sidx_indptr=np.zeros(4, dtype=np.int64),
        sidx_val=np.zeros(32, dtype=oasis_int),
        loss_val=np.zeros(32, dtype=np.float64),
        loss_indptr=np.zeros(4, dtype=np.int64),
        pass_through=np.zeros(4, dtype=np.float64),
        computes=np.zeros(4, dtype=oasis_int),
        compute_idx=np.zeros(1, dtype=compute_idx_dtype)[0],
    )
    return state


def _read(state, byte_mv, collapse_on_read=False):
    read_buffer(
        byte_mv, 0, byte_mv.shape[0], 0, 0,
        state['nodes_array'], state['sidx_indexes'], state['sidx_indptr'],
        state['sidx_val'], state['loss_indptr'], state['loss_val'], state['pass_through'],
        state['computes'], state['compute_idx'], S, True, collapse_on_read,
    )


def _stored(state):
    """Return (sidx, loss) the reader stored for the single item, plus its pass-through."""
    end = state['sidx_indptr'][1]
    sidx = state['sidx_val'][:end].tolist()
    loss = state['loss_val'][:end].tolist()
    return dict(zip(sidx, loss)), float(state['pass_through'][1])


class TestFmDepacking(TestCase):

    def setUp(self):
        # two buildings; specials MAX(-5)/TIV(-3)/MEAN(-1) and samples are per-building values
        # that sum to the location total; chance-of-loss (-4) is building-independent.
        self.packed_records = [
            (encode_sidx(1, -5, S), 50.), (encode_sidx(1, -4, S), 0.5), (encode_sidx(1, -3, S), 30.),
            (encode_sidx(1, -1, S), 20.),
            (encode_sidx(1, 1, S), 11.), (encode_sidx(1, 2, S), 12.), (encode_sidx(1, 3, S), 13.), (encode_sidx(1, 4, S), 14.),
            (encode_sidx(2, -5, S), 500.), (encode_sidx(2, -4, S), 0.5), (encode_sidx(2, -3, S), 300.),
            (encode_sidx(2, -1, S), 200.),
            (encode_sidx(2, 1, S), 101.), (encode_sidx(2, 2, S), 102.), (encode_sidx(2, 3, S), 103.), (encode_sidx(2, 4, S), 104.),
        ]
        # an ordinary single-building stream
        self.normal_records = [
            (-5, 550.), (-4, 0.5), (-3, 330.), (-1, 220.),
            (1, 112.), (2, 114.), (3, 116.), (4, 118.),
        ]
        self.normal_expected = {-5: 550., -3: 330., -1: 220., 1: 112., 2: 114., 3: 116., 4: 118.}

    def test_each_building_keeps_its_own_records(self):
        """Nothing is merged: every packed record is stored where it arrived."""
        state = _fresh_state()
        _read(state, _build_item_buffer(1, 1, self.packed_records))
        stored, chance = _stored(state)
        expected = {sidx: loss for sidx, loss in self.packed_records
                    if decode_local_sidx(sidx, S) != -4}
        self.assertEqual(stored, expected)
        self.assertEqual(chance, 0.5)  # a property of the risk, taken once

    def test_the_two_buildings_stay_distinguishable(self):
        """The whole point: building 2's records must not land on building 1's sidx."""
        state = _fresh_state()
        _read(state, _build_item_buffer(1, 1, self.packed_records))
        stored, _ = _stored(state)
        for building in (1, 2):
            for local_sidx in (-5, -3, -1, 1, 2, 3, 4):
                self.assertIn(encode_sidx(building, local_sidx, S), stored)

    def test_sidx_are_stored_in_ascending_order(self):
        """Aggregation relies on it, and the packed specials run below the unpacked ones."""
        state = _fresh_state()
        _read(state, _build_item_buffer(1, 1, self.packed_records))
        end = state['sidx_indptr'][1]
        stored_sidx = state['sidx_val'][:end].tolist()
        self.assertEqual(stored_sidx, sorted(stored_sidx))

    def test_normal_stream_unchanged(self):
        """For building 1 the packed encoding is the identity, so a normal stream is verbatim."""
        state = _fresh_state()
        _read(state, _build_item_buffer(1, 1, self.normal_records))
        stored, chance = _stored(state)
        self.assertEqual(stored, self.normal_expected)
        self.assertEqual(chance, 0.5)

    def test_a_repeated_sidx_is_still_stream_corruption(self):
        state = _fresh_state()
        with self.assertRaises(ValueError):
            _read(state, _build_item_buffer(1, 1, [(1, 1.), (1, 2.)]))


if __name__ == "__main__":
    from unittest import main
    main()
