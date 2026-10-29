"""What fmpy guarantees about a building-packed loss stream, beyond the losses themselves.

A packed item carries its N buildings in the sample dimension, one block of sidx per building.
That the financial module gives the same losses packed as with one item per building is shown
end to end in ``tests/fm/test_building_packing_matrix.py``. What is here is what a loss
comparison cannot show: the input fmpy refuses rather than mis-reads, a pipe read splitting an
item, and the sample indices of the output.
"""
import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from oasislmf.preparation.il_inputs import write_fm_structure_info
from oasislmf.pytools.common.data import (fm_policytc_dtype, fm_profile_dtype,
                                          fm_programme_dtype, fm_xref_dtype, oasis_float, oasis_int)
from oasislmf.pytools.common.event_stream import (ITEM_STREAM, LOSS_STREAM_ID, PIPE_CAPACITY,
                                                  encode_sidx, stream_info_to_bytes)
from oasislmf.pytools.fm.manager import run as run_fm
from oasislmf.utils.exceptions import OasisStreamException

S = 4
DEDUCTIBLE = 100.0
EVENT_ID = 1


def _write_structure(d, programme, policytc, xref, site_collapse_level, max_buildings):
    """Write fm inputs whose level-1 nodes take ``DEDUCTIBLE`` (profile 1) or nothing (profile 0)."""
    profile = np.zeros(2, dtype=fm_profile_dtype)
    profile[0]['profile_id'], profile[0]['calcrule_id'] = 0, 12
    profile[1]['profile_id'], profile[1]['calcrule_id'] = 1, 12
    profile[1]['deductible1'] = DEDUCTIBLE
    for name, arr in (('fm_programme', np.array(programme, dtype=fm_programme_dtype)),
                      ('fm_policytc', np.array(policytc, dtype=fm_policytc_dtype)),
                      ('fm_profile', profile),
                      ('fm_xref', np.array(xref, dtype=fm_xref_dtype))):
        arr.tofile(os.path.join(d, f'{name}.bin'))
    if max_buildings > 1:
        n_items = len({from_agg for from_agg, level, _ in programme if level == 1})
        write_fm_structure_info(d, site_collapse_level, max_buildings,
                                total_packed_buildings=max_buildings * n_items)


def _write_gul_stream(path, items, sample_size=S):
    with open(path, 'wb') as f:
        f.write(stream_info_to_bytes(LOSS_STREAM_ID, ITEM_STREAM))
        f.write(np.int32(sample_size).tobytes())
        for item_id, records in items:
            f.write(np.array([EVENT_ID, item_id], dtype=oasis_int).tobytes())
            for sidx, loss in list(records) + [(0, 0.)]:
                f.write(np.array([sidx], dtype=oasis_int).tobytes())
                f.write(np.array([loss], dtype=oasis_float).tobytes())


def _read_fm_stream(path):
    """{output_id: {sidx: loss}}"""
    raw = open(path, 'rb').read()[8:]  # skip the stream header
    out, pos = {}, 0
    while pos < len(raw):
        _, output_id = np.frombuffer(raw[pos:pos + 8], dtype=oasis_int)
        pos += 8
        records = out.setdefault(int(output_id), {})
        while True:
            sidx = int(np.frombuffer(raw[pos:pos + 4], dtype=oasis_int)[0])
            loss = float(np.frombuffer(raw[pos + 4:pos + 8], dtype=oasis_float)[0])
            pos += 8
            if sidx == 0:
                break
            records[sidx] = loss
    return out


def _run(write_structure, items, allocation_rule=2, sample_size=S):
    with TemporaryDirectory() as d:
        write_structure(d)
        gul, out = os.path.join(d, 'gul.bin'), os.path.join(d, 'fm.bin')
        _write_gul_stream(gul, items, sample_size)
        run_fm(True, allocation_rule=allocation_rule, static_path=d)
        run_fm(False, allocation_rule=allocation_rule, static_path=d, files_in=[gul], files_out=[out],
               net_loss=None, storage_method='sparse', low_memory=False, sort_output=False)
        return _read_fm_stream(out)


def _packed_item(item_id, losses_by_building, sample_size=S):
    """One item whose buildings ride in the sample dimension, in the ascending order required."""
    return (item_id, sorted((encode_sidx(b, s, sample_size), loss)
                            for b, losses in losses_by_building.items()
                            for s, loss in enumerate(losses, start=1)))


def _total_per_sample(outputs):
    totals = {}
    for records in outputs.values():
        for sidx, loss in records.items():
            totals[sidx] = totals.get(sidx, 0.0) + loss
    return {sidx: round(loss, 3) for sidx, loss in totals.items() if loss}


TWO_BUILDINGS = {1: [10.0, 200.0, 30.0, 400.0], 2: [300.0, 20.0, 500.0, 5.0]}


def _site_terms_structure(d):
    """One item, one site node with the deductible, collapsed at level 1, one node above."""
    _write_structure(d, [(1, 1, 1), (1, 2, 1)], [(1, 1, 1, 1), (2, 1, 1, 0)], [(1, 1, 1)],
                     site_collapse_level=1, max_buildings=2)


def _no_site_terms_structure(d):
    """Two items, each straight onto a level-1 node with no terms: the buildings are summed as read."""
    _write_structure(d, [(1, 1, 1), (2, 1, 2), (1, 2, 1), (2, 2, 1)],
                     [(1, 1, 1, 0), (1, 2, 1, 0), (2, 1, 1, 0)], [(1, 1, 1)],
                     site_collapse_level=0, max_buildings=2)


class TestWhatFmpyRefuses(TestCase):

    def test_a_packed_stream_needs_the_structure_to_declare_packing(self):
        """fm_structure_info lives beside the binaries and has to be staged with them. Without it
        the arrays are sized for one building, and numba does not bounds-check, so carrying on
        would corrupt memory rather than fail."""
        def declares_no_packing(d):
            _write_structure(d, [(1, 1, 1), (1, 2, 1)], [(1, 1, 1, 1), (2, 1, 1, 0)], [(1, 1, 1)],
                             site_collapse_level=0, max_buildings=1)

        with self.assertRaises(Exception) as caught:
            _run(declares_no_packing, [_packed_item(1, TWO_BUILDINGS)])
        self.assertIn('fm_structure_info', str(caught.exception))

    def test_an_item_must_ascend_in_sidx(self):
        """The stream's ordering rule, which is also its only duplicate check: a descent is
        refused where a repeat is equal. Whether the buildings are kept apart for site terms or
        summed as they are read, which loses the order they arrived in."""
        item_id, ascending = _packed_item(1, TWO_BUILDINGS)
        last_building_first = ascending[S:] + ascending[:S]
        for structure in (_site_terms_structure, _no_site_terms_structure):
            for records in (last_building_first, [(1, 1.), (1, 2.)]):
                with self.subTest(structure=structure.__name__, records=records[:3]):
                    with self.assertRaisesRegex(OasisStreamException, "out of order or repeated"):
                        _run(structure, [(item_id, records)], allocation_rule=0)


class TestTheOutput(TestCase):

    def test_an_item_split_across_two_reads_is_summed_whole(self):
        """A pipe delivers PIPE_CAPACITY bytes at a time, so a large item arrives in pieces, and the
        reader summing its buildings has to carry the partial sums across the read."""
        sample_size = PIPE_CAPACITY // 8   # two buildings of this many samples cannot fit one read
        losses = {b: [float(b * 1000 + s) for s in range(1, sample_size + 1)] for b in (1, 2)}
        out = _run(_no_site_terms_structure,
                   [_packed_item(item_id, losses, sample_size) for item_id in (1, 2)],
                   allocation_rule=0, sample_size=sample_size)
        totals = _total_per_sample(out)
        for s in range(1, sample_size + 1):
            self.assertAlmostEqual(totals[s], 2 * (losses[1][s - 1] + losses[2][s - 1]), places=0)
        self.assertEqual(max(totals), sample_size, 'a packed sidx reached the output')

    def test_rule_1_reports_the_same_sample_indices_as_separate_items(self):
        """Rule 1 reads the input loss kept beside each leaf. Out of step with the leaf's sidx, it
        invented special indices, which a comparison of losses at the shared indices cannot see."""
        multi_losses = {
            (1, 'a'): [10.0, 200.0, 30.0, 400.0], (1, 'b'): [5.0, 100.0, 15.0, 200.0],
            (2, 'a'): [300.0, 20.0, 500.0, 5.0], (2, 'b'): [150.0, 10.0, 250.0, 2.0],
        }

        def packed(d):
            """Two packed items (perils a and b) -> one site node -> one node above."""
            _write_structure(d, [(1, 1, 1), (2, 1, 1), (1, 2, 1)], [(1, 1, 1, 1), (2, 1, 1, 0)],
                             [(1, 1, 1), (2, 2, 1)], site_collapse_level=1, max_buildings=2)

        def separate(d):
            """Four items -> a site node per building -> one node above."""
            _write_structure(d, [(1, 1, 1), (2, 1, 1), (3, 1, 2), (4, 1, 2), (1, 2, 1), (2, 2, 1)],
                             [(1, 1, 1, 1), (1, 2, 1, 1), (2, 1, 1, 0)],
                             [(i, i, 1) for i in range(1, 5)], site_collapse_level=0, max_buildings=1)

        packed_out = _run(packed, [_packed_item(item_id, {b: multi_losses[(b, peril)] for b in (1, 2)})
                                   for item_id, peril in ((1, 'a'), (2, 'b'))], allocation_rule=1)
        separate_out = _run(separate, [(item_id, list(enumerate(multi_losses[key], start=1)))
                                       for item_id, key in ((1, (1, 'a')), (2, (1, 'b')),
                                                            (3, (2, 'a')), (4, (2, 'b')))],
                            allocation_rule=1)
        self.assertEqual(sorted(_total_per_sample(packed_out)), sorted(_total_per_sample(separate_out)))
