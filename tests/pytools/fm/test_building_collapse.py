"""End-to-end test that the financial module applies site terms per building, then collapses.

A building-packed item carries one block per building through the site levels -- the ones whose
aggregation key includes ``risk_id`` -- so their terms apply to each building separately, and the
blocks merge only once the item is aggregated into a node above ``site_collapse_level``.

The test runs fmpy twice over the same per-building losses:

* **packed** -- one item whose two buildings ride in the sample dimension, through a structure
  with one site node and ``site_collapse_level = 1``;
* **disaggregated** -- two items, one per building, through a structure with a site node each and
  no collapse. This is the behaviour packing has to reproduce, and it is what row disaggregation
  produces today.

Both must give the same answer. The deductible is chosen to **bind on one building but not the
other**, which is the only regime where applying it per building differs from applying it to the
location total -- so ``test_it_is_not_the_same_as_deducting_from_the_total`` is what gives the
comparison teeth. With a non-binding deductible the two agree for any input, since
``sum(L_b - d) == sum(L_b) - N*d``, which is why end-to-end parity on the analytical mean never
caught the site levels being applied to the wrong thing.
"""
import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from oasislmf.preparation.il_inputs import write_fm_structure_info
from oasislmf.pytools.common.data import (fm_policytc_dtype, fm_profile_dtype,
                                          fm_programme_dtype, fm_xref_dtype, oasis_float, oasis_int)
from oasislmf.pytools.common.event_stream import LOSS_STREAM_ID, ITEM_STREAM, encode_sidx, stream_info_to_bytes
from oasislmf.pytools.fm.manager import run as run_fm

S = 4
DEDUCTIBLE = 100.0
EVENT_ID = 1

# building 1 is mostly below the deductible, building 2 mostly above, so it binds unevenly
BUILDING_LOSSES = {
    1: [10.0, 200.0, 30.0, 400.0],
    2: [300.0, 20.0, 500.0, 5.0],
}
# what applying the deductible per building then summing must give
EXPECTED = [max(0.0, BUILDING_LOSSES[1][s] - DEDUCTIBLE) + max(0.0, BUILDING_LOSSES[2][s] - DEDUCTIBLE)
            for s in range(S)]


def _profiles():
    """Two profiles: 0 is a pass-through, 1 is the per-building deductible."""
    profile = np.zeros(2, dtype=fm_profile_dtype)
    profile[0]['profile_id'], profile[0]['calcrule_id'] = 0, 12   # deductible-only, deductible 0
    profile[1]['profile_id'], profile[1]['calcrule_id'] = 1, 12
    profile[1]['deductible1'] = DEDUCTIBLE
    return profile


def _write(d, programme, policytc, xref, site_collapse_level, max_buildings):
    for name, arr in (('fm_programme', programme), ('fm_policytc', policytc),
                      ('fm_profile', _profiles()), ('fm_xref', xref)):
        arr.tofile(os.path.join(d, f'{name}.bin'))
    if site_collapse_level:
        write_fm_structure_info(d, site_collapse_level, max_buildings)


def write_packed_structure(d):
    """One item -> one site node (per-building deductible) -> one node above the collapse point."""
    _write(
        d,
        np.array([(1, 1, 1), (1, 2, 1)], dtype=fm_programme_dtype),
        np.array([(1, 1, 1, 1), (2, 1, 1, 0)], dtype=fm_policytc_dtype),
        np.array([(1, 1, 1)], dtype=fm_xref_dtype),
        site_collapse_level=1, max_buildings=2,
    )


def write_disaggregated_structure(d):
    """Two items -> a site node each (same deductible) -> one node summing them. No collapse."""
    _write(
        d,
        np.array([(1, 1, 1), (2, 1, 2), (1, 2, 1), (2, 2, 1)], dtype=fm_programme_dtype),
        np.array([(1, 1, 1, 1), (1, 2, 1, 1), (2, 1, 1, 0)], dtype=fm_policytc_dtype),
        np.array([(1, 1, 1)], dtype=fm_xref_dtype),
        site_collapse_level=0, max_buildings=1,
    )


def write_gul_stream(path, items, sample_size=S):
    """Write a GUL stream: header, then (event, item) blocks of (sidx, loss) pairs."""
    with open(path, 'wb') as f:
        f.write(stream_info_to_bytes(LOSS_STREAM_ID, ITEM_STREAM))
        f.write(np.int32(sample_size).tobytes())
        for item_id, records in items:
            f.write(np.array([EVENT_ID, item_id], dtype=oasis_int).tobytes())
            for sidx, loss in records:
                f.write(np.array([sidx], dtype=oasis_int).tobytes())
                f.write(np.array([loss], dtype=oasis_float).tobytes())
            f.write(np.array([0], dtype=oasis_int).tobytes())
            f.write(np.array([0.0], dtype=oasis_float).tobytes())


def read_fm_stream(path):
    """Parse an fm stream into {output_id: {sidx: loss}}."""
    raw = open(path, 'rb').read()[8:]  # skip the stream header
    out, pos = {}, 0
    while pos < len(raw):
        _, output_id = np.frombuffer(raw[pos:pos + 8], dtype=oasis_int)
        pos += 8
        records = {}
        while True:
            sidx = int(np.frombuffer(raw[pos:pos + 4], dtype=oasis_int)[0])
            loss = float(np.frombuffer(raw[pos + 4:pos + 8], dtype=oasis_float)[0])
            pos += 8
            if sidx == 0:
                break
            records[sidx] = loss
        out[int(output_id)] = records
    return out


def _run(structure_writer, stream_items, allocation_rule=0, xref=None):
    with TemporaryDirectory() as d:
        structure_writer(d)
        if xref is not None:
            xref.tofile(os.path.join(d, 'fm_xref.bin'))
        gul = os.path.join(d, 'gul.bin')
        out = os.path.join(d, 'fm.bin')
        write_gul_stream(gul, stream_items)
        run_fm(True, allocation_rule=allocation_rule, static_path=d)
        run_fm(False, allocation_rule=allocation_rule, static_path=d, files_in=[gul], files_out=[out],
               net_loss=None, storage_method='sparse', low_memory=False, sort_output=False)
        return read_fm_stream(out)


def _total_per_sample(outputs):
    """Sum an fm result across its outputs, so one packed item compares with N disaggregated ones."""
    totals = {}
    for records in outputs.values():
        for sidx, loss in records.items():
            totals[sidx] = totals.get(sidx, 0.0) + loss
    return {sidx: round(loss, 3) for sidx, loss in totals.items() if loss}


def packed_stream_items():
    records = []
    for building, losses in BUILDING_LOSSES.items():
        for s, loss in enumerate(losses, start=1):
            records.append((encode_sidx(building, s, S), loss))
    return [(1, sorted(records))]


def disaggregated_stream_items():
    return [(item_id, [(s, loss) for s, loss in enumerate(losses, start=1)])
            for item_id, losses in BUILDING_LOSSES.items()]


class TestSiteTermsApplyPerBuilding(TestCase):

    def test_packed_matches_disaggregated(self):
        """The whole point of the collapse: packing must reproduce row disaggregation."""
        packed = _run(write_packed_structure, packed_stream_items())
        disaggregated = _run(write_disaggregated_structure, disaggregated_stream_items())
        self.assertEqual(packed, disaggregated)

    def test_the_deductible_is_applied_per_building(self):
        packed = _run(write_packed_structure, packed_stream_items())
        losses = packed[1]
        for s in range(1, S + 1):
            self.assertAlmostEqual(losses.get(s, 0.0), EXPECTED[s - 1], places=3)

    def test_it_is_not_the_same_as_deducting_from_the_total(self):
        """Gives the comparison teeth: the two answers really do differ on this input."""
        from_total = [max(0.0, BUILDING_LOSSES[1][s] + BUILDING_LOSSES[2][s] - DEDUCTIBLE) for s in range(S)]
        self.assertNotEqual(EXPECTED, from_total)

    def test_no_packed_sidx_survives_the_collapse(self):
        """Everything above the collapse point sees ordinary sample indices."""
        packed = _run(write_packed_structure, packed_stream_items())
        for records in packed.values():
            for sidx in records:
                self.assertLessEqual(sidx, S)
                self.assertGreaterEqual(sidx, -5)

    def test_a_structure_that_declares_no_packing_refuses_a_packed_stream(self):
        """Negative control, and the guard for a real deployment failure.

        The financial module sizes its arrays from max_buildings, so a packed stream arriving at a
        structure that declares none would write past them -- and numba does not bounds-check, so
        that corrupts memory instead of failing. It is reachable: fm_structure_info.bin lives
        beside the binaries and has to be staged with them.

        Refusing the stream is also what keeps the tests above honest. They would pass just as
        well if the collapse did nothing, were it not that a structure without packing declared
        cannot process this stream at all.
        """
        def declares_no_packing(d):
            _write(d,
                   np.array([(1, 1, 1), (1, 2, 1)], dtype=fm_programme_dtype),
                   np.array([(1, 1, 1, 1), (2, 1, 1, 0)], dtype=fm_policytc_dtype),
                   np.array([(1, 1, 1)], dtype=fm_xref_dtype),
                   site_collapse_level=0, max_buildings=1)

        with self.assertRaises(Exception) as caught:
            _run(declares_no_packing, packed_stream_items())
        self.assertIn("fm_structure_info", str(caught.exception))


class TestBackAllocationOntoThePackedItem(TestCase):
    """With an allocation rule above 0 the output is read from the leaves, below the collapse.

    Nothing aggregates a leaf, so its building blocks would never merge and the output would come
    out at packed sample indices. The leaves are therefore given collapsed storage once the site
    terms have been applied, which is also what lets back-allocation stay unchanged: it looks its
    factor up by sample index, and a collapsed leaf's indices line up with the node's.

    A packed item covers all its buildings, so it produces one output where disaggregation
    produces one per building -- the comparison is against their sum.
    """

    XREF_PACKED = np.array([(1, 1, 1)], dtype=fm_xref_dtype)

    @staticmethod
    def _xref(allocation_rule, leaf_count):
        """Where the output sits: at the aggregate node under rule 0, at the leaves above it.

        Only the top level has an agg_id 2 to point at once the leaves are the output, so asking
        for one per leaf under rule 0 addresses a node that does not exist.
        """
        if allocation_rule == 0:
            return np.array([(1, 1, 1)], dtype=fm_xref_dtype)
        return np.array([(i, i, 1) for i in range(1, leaf_count + 1)], dtype=fm_xref_dtype)

    def test_packed_matches_disaggregated_for_every_allocation_rule(self):
        for allocation_rule in (0, 1, 2, 3):
            with self.subTest(allocation_rule=allocation_rule):
                packed = _run(write_packed_structure, packed_stream_items(),
                              allocation_rule, self._xref(allocation_rule, 1))
                disaggregated = _run(write_disaggregated_structure, disaggregated_stream_items(),
                                     allocation_rule, self._xref(allocation_rule, 2))
                self.assertEqual(_total_per_sample(packed), _total_per_sample(disaggregated))

    def test_the_allocated_output_is_still_the_per_building_answer(self):
        packed = _run(write_packed_structure, packed_stream_items(), 2, self.XREF_PACKED)
        totals = _total_per_sample(packed)
        for s in range(1, S + 1):
            self.assertAlmostEqual(totals.get(s, 0.0), EXPECTED[s - 1], places=3)

    def test_the_allocated_output_carries_no_packed_sidx(self):
        """The failure this fixes: without collapsed leaves the output came out packed."""
        packed = _run(write_packed_structure, packed_stream_items(), 2, self.XREF_PACKED)
        for records in packed.values():
            for sidx in records:
                self.assertLessEqual(sidx, S)
                self.assertGreaterEqual(sidx, -5)


class TestNoSiteLevelStillCollapses(TestCase):
    """A packed item with no site-level terms must still have its buildings merged.

    Generation records site_collapse_level as the last level keyed on risk_id, so an input set
    with no location terms writes no such level and the value is legitimately 0. The items are
    still packed, though -- what is packed is decided from IsAggregate, not from whether there
    are terms -- so 0 has to mean "merge as soon as the items are aggregated", level 0 being the
    items, and not "never merge".

    Treating 0 as "nothing to do" let packed indices escape into levels sized for the sample
    count, which a real PiWind portfolio with no location deductible turned into
    `cannot assign slice of shape (17,) from input of shape (16,)`.
    """

    @staticmethod
    def _structure(d):
        """Two packed items, no site terms, into one pass-through node. site_collapse_level = 0.

        Two rather than one because a single item and a single programme row collapse into a node
        with no children at all, which generation never produces -- it always writes a policy
        layer, and every location's items feed into it.
        """
        _write(d,
               np.array([(1, 1, 1), (2, 1, 1)], dtype=fm_programme_dtype),
               np.array([(1, 1, 1, 0)], dtype=fm_policytc_dtype),
               np.array([(1, 1, 1)], dtype=fm_xref_dtype),
               site_collapse_level=0, max_buildings=2)
        # _write only emits the file when a collapse level is set, so write it explicitly
        write_fm_structure_info(d, 0, 2)

    @staticmethod
    def _stream():
        """Both items packed with two buildings each."""
        items = []
        for item_id in (1, 2):
            records = []
            for building, losses in BUILDING_LOSSES.items():
                for s, loss in enumerate(losses, start=1):
                    records.append((encode_sidx(building, s, S), loss))
            items.append((item_id, sorted(records)))
        return items

    def test_buildings_merge_even_with_no_collapse_level(self):
        out = _run(self._structure, self._stream())
        totals = _total_per_sample(out)
        for sample_idx in range(1, S + 1):
            # two items, each the sum of its two buildings
            expected = 2 * (BUILDING_LOSSES[1][sample_idx - 1] + BUILDING_LOSSES[2][sample_idx - 1])
            self.assertAlmostEqual(totals.get(sample_idx, 0.0), expected, places=3)

    def test_no_packed_sidx_escapes(self):
        out = _run(self._structure, self._stream())
        for records in out.values():
            for sidx in records:
                self.assertLessEqual(sidx, S)
                self.assertGreaterEqual(sidx, -5)


# Per (building, peril) losses for the multi-item shape below. Building 1 sits mostly under the
# deductible once its perils are summed, building 2 over it.
MULTI_LOSSES = {
    (1, 'a'): [10.0, 200.0, 30.0, 400.0], (1, 'b'): [5.0, 100.0, 15.0, 200.0],
    (2, 'a'): [300.0, 20.0, 500.0, 5.0], (2, 'b'): [150.0, 10.0, 250.0, 2.0],
}


class TestSiteNodeWithSeveralItems(TestCase):
    """A site node aggregating more than one item -- the ordinary shape, and the one that broke.

    The site FM levels merge every coverage type and peril of a location, so a site node almost
    always has several item children. The single-child fixtures above take the forced-aggregation
    path instead, and so missed two defects:

    * the packed leaves were being collapsed by the site node itself, before its own storage was
      collapsed, leaving back-allocation to look factors up at packed indices and read building
      1's for every building;
    * ``net_loss`` was left in the packed layout while the leaf's sidx array became collapsed, so
      allocation rule 1 read the two out of step and emitted fabricated specials.

    Both showed only at allocation rules above 0, and only with more than one item under the site
    node.
    """

    @staticmethod
    def _packed_structure(d):
        """Two packed items -> one site node (per-building deductible) -> one node above."""
        _write(d,
               np.array([(1, 1, 1), (2, 1, 1), (1, 2, 1)], dtype=fm_programme_dtype),
               np.array([(1, 1, 1, 1), (2, 1, 1, 0)], dtype=fm_policytc_dtype),
               np.array([(1, 1, 1)], dtype=fm_xref_dtype),
               site_collapse_level=1, max_buildings=2)

    @staticmethod
    def _disaggregated_structure(d):
        """Four items -> a site node per building -> one node above."""
        _write(d,
               np.array([(1, 1, 1), (2, 1, 1), (3, 1, 2), (4, 1, 2),
                         (1, 2, 1), (2, 2, 1)], dtype=fm_programme_dtype),
               np.array([(1, 1, 1, 1), (1, 2, 1, 1), (2, 1, 1, 0)], dtype=fm_policytc_dtype),
               np.array([(1, 1, 1)], dtype=fm_xref_dtype),
               site_collapse_level=0, max_buildings=1)

    @staticmethod
    def _packed_stream():
        out = []
        for item_id, peril in ((1, 'a'), (2, 'b')):
            recs = [(encode_sidx(b, s, S), MULTI_LOSSES[(b, peril)][s - 1])
                    for b in (1, 2) for s in range(1, S + 1)]
            out.append((item_id, sorted(recs)))
        return out

    @staticmethod
    def _disaggregated_stream():
        return [(item_id, [(s, MULTI_LOSSES[key][s - 1]) for s in range(1, S + 1)])
                for item_id, key in ((1, (1, 'a')), (2, (1, 'b')),
                                     (3, (2, 'a')), (4, (2, 'b')))]

    @staticmethod
    def _xref(allocation_rule, leaf_count):
        if allocation_rule == 0:
            return np.array([(1, 1, 1)], dtype=fm_xref_dtype)
        return np.array([(i, i, 1) for i in range(1, leaf_count + 1)], dtype=fm_xref_dtype)

    def test_packed_matches_disaggregated_for_every_allocation_rule(self):
        for allocation_rule in (0, 1, 2, 3):
            with self.subTest(allocation_rule=allocation_rule):
                packed = _run(self._packed_structure, self._packed_stream(),
                              allocation_rule, self._xref(allocation_rule, 2))
                disaggregated = _run(self._disaggregated_structure, self._disaggregated_stream(),
                                     allocation_rule, self._xref(allocation_rule, 4))
                self.assertEqual(_total_per_sample(packed), _total_per_sample(disaggregated))

    def test_the_deductible_applies_to_a_building_across_its_perils(self):
        """Perils merge inside the site node first, so the deductible sees the building total."""
        packed = _total_per_sample(_run(self._packed_structure, self._packed_stream(),
                                        2, self._xref(2, 2)))
        for sample_idx in range(1, S + 1):
            expected = sum(
                max(0.0, MULTI_LOSSES[(b, 'a')][sample_idx - 1] + MULTI_LOSSES[(b, 'b')][sample_idx - 1]
                    - DEDUCTIBLE)
                for b in (1, 2)
            )
            self.assertAlmostEqual(packed.get(sample_idx, 0.0), expected, places=3)

    def test_no_fabricated_specials_under_allocation_rule_one(self):
        """net_loss and the sidx array must stay in step, or rule 1 invents special indices."""
        packed = _run(self._packed_structure, self._packed_stream(), 1, self._xref(1, 2))
        disaggregated = _run(self._disaggregated_structure, self._disaggregated_stream(),
                             1, self._xref(1, 4))
        # the sample indices themselves: fabricated specials show up as extra keys. The losses
        # at those indices are compared in full by the test above.
        self.assertEqual(sorted(_total_per_sample(packed).keys()),
                         sorted(_total_per_sample(disaggregated).keys()))


class TestNoSiteLevelSinglePeril(TestCase):
    """The no-site-terms case again, but with the items mapped 1:1 into the first level.

    ``TestNoSiteLevelStillCollapses`` above feeds two items into one level-1 node, which makes the
    structure multi-peril, and multi-peril puts the item nodes at level 0. A single-peril
    structure maps each item straight onto a level-1 node instead, and generation then sets
    ``start_level = 1`` and skips the item level entirely -- so the item nodes carry
    ``level_id == 1``, not 0.

    That is what made ``site_collapse_level == 0`` mean two different things. With the items at
    level 0 it reads as "merge as soon as they are aggregated", which is right. With the items at
    level 1 nothing is at or below level 0 at all, so nothing was ever marked packable and nothing
    collapsed: the packed sample indices ran all the way to the output, spreading one location's
    loss across ``N * S`` samples instead of ``S``. It came out as a wrong answer, not an error.

    Clamping the collapse level up to ``start_level`` would fix the indices and break the losses,
    because level 1 may carry a real site term that must be applied to the location once, not once
    per building. So the reader sums the buildings away as it reads them instead, and the
    financial module sees an ordinary stream.
    """

    @staticmethod
    def _structure(d):
        """Two items, 1:1 onto their own level-1 nodes with no terms, aggregated at level 2."""
        _write(d,
               np.array([(1, 1, 1), (2, 1, 2), (1, 2, 1), (2, 2, 1)], dtype=fm_programme_dtype),
               np.array([(1, 1, 1, 0), (1, 2, 1, 0), (2, 1, 1, 0)], dtype=fm_policytc_dtype),
               np.array([(1, 1, 1)], dtype=fm_xref_dtype),
               site_collapse_level=0, max_buildings=2)
        # _write only emits the file when a collapse level is set, so write it explicitly
        write_fm_structure_info(d, 0, 2)

    @staticmethod
    def _stream():
        items = []
        for item_id in (1, 2):
            records = []
            for building, losses in BUILDING_LOSSES.items():
                for s, loss in enumerate(losses, start=1):
                    records.append((encode_sidx(building, s, S), loss))
            items.append((item_id, sorted(records)))
        return items

    def test_the_buildings_are_summed(self):
        out = _run(self._structure, self._stream())
        totals = _total_per_sample(out)
        for sample_idx in range(1, S + 1):
            expected = 2 * (BUILDING_LOSSES[1][sample_idx - 1] + BUILDING_LOSSES[2][sample_idx - 1])
            self.assertAlmostEqual(totals.get(sample_idx, 0.0), expected, places=3)

    def test_no_packed_sidx_reaches_the_output(self):
        """The failure this guards: sidx above S surviving to the output."""
        out = _run(self._structure, self._stream())
        for records in out.values():
            for sidx in records:
                self.assertLessEqual(sidx, S)
                self.assertGreaterEqual(sidx, -5)

    def test_it_holds_under_back_allocation(self):
        xref = np.array([(1, 1, 1), (2, 2, 1)], dtype=fm_xref_dtype)
        for allocation_rule in (1, 2, 3):
            with self.subTest(allocation_rule=allocation_rule):
                out = _run(self._structure, self._stream(), allocation_rule=allocation_rule, xref=xref)
                for records in out.values():
                    for sidx in records:
                        self.assertLessEqual(sidx, S)


class TestArenaHoldsTheNetLossSlice(TestCase):
    """The loss arena must budget the packed ``net_loss`` slice, not just the layers.

    The sparse arrays are one arena filled by a bump allocator, and packing inflates it by an
    extra full-size slice per packable node -- room for the node's ``max_buildings`` blocks plus
    the collapsed copy that :func:`collapse_packed_leaves` appends rather than shrinking in place.

    ``net_loss`` is a further slice per node in that same arena, holding the leaf's pre-profile
    loss in the leaf's layout, so it is packed and collapsed with the rest. It is in use whenever
    ``keep_input_loss`` is set -- allocation rule 1, *or* any net-loss output mode at any
    allocation rule, which is the ordinary reinsurance path. Budgeting only ``max_layer`` slices
    left it out, and the run wrote past the end of ``loss_val``.

    Numba does not bounds-check, so the overflow was a live heap write: this shape aborted the
    process with ``free(): invalid pointer`` rather than raising. Under ``NUMBA_DISABLE_JIT=1`` it
    surfaces as ``IndexError``/``ValueError`` from the same line. The shape matters -- almost
    every node here is packable, so there is no slack from ordinary nodes to absorb the shortfall,
    which is why smaller fixtures with a couple of unpacked nodes passed while this one corrupted
    memory.
    """

    # The smallest shape that still catches the missing net_loss slice -- verified by reverting
    # the fix, where 2 items / 3 buildings already overflows. Sample size has to be 8 rather than
    # the module's 4: at S=4 the ordinary nodes' unused allowance absorbs the shortfall and
    # nothing overflows.
    SAMPLE_SIZE = 8
    BUILDINGS = 3
    ITEMS = 4
    ITEMS_PER_SITE = 2

    def _structure(self, d):
        sites = self.ITEMS // self.ITEMS_PER_SITE
        programme = [(i, 1, (i - 1) // self.ITEMS_PER_SITE + 1) for i in range(1, self.ITEMS + 1)]
        programme += [(a, 2, 1) for a in range(1, sites + 1)]
        # profile 0 is a pass-through: this fixture is about arena capacity, not term arithmetic,
        # so the expected total stays the plain sum of the input
        policytc = [(1, a, 1, 0) for a in range(1, sites + 1)] + [(2, 1, 1, 0)]
        _write(d,
               np.array(programme, dtype=fm_programme_dtype),
               np.array(policytc, dtype=fm_policytc_dtype),
               np.array([(1, 1, 1)], dtype=fm_xref_dtype),
               site_collapse_level=1, max_buildings=self.BUILDINGS)

    def _stream(self):
        items = []
        for item_id in range(1, self.ITEMS + 1):
            records = [(encode_sidx(building, s, self.SAMPLE_SIZE), 10.0 * building + s + item_id)
                       for building in range(1, self.BUILDINGS + 1)
                       for s in range(1, self.SAMPLE_SIZE + 1)]
            items.append((item_id, sorted(records)))
        return items

    def _run_with_net_loss(self, allocation_rule):
        with TemporaryDirectory() as d:
            self._structure(d)
            np.array([(i, i, 1) for i in range(1, self.ITEMS + 1)],
                     dtype=fm_xref_dtype).tofile(os.path.join(d, 'fm_xref.bin'))
            gul = os.path.join(d, 'gul.bin')
            out = os.path.join(d, 'fm.bin')
            write_gul_stream(gul, self._stream(), sample_size=self.SAMPLE_SIZE)
            run_fm(True, allocation_rule=allocation_rule, static_path=d)
            run_fm(False, allocation_rule=allocation_rule, static_path=d, files_in=[gul],
                   files_out=[out], net_loss=os.path.join(d, 'net.bin'),
                   storage_method='sparse', low_memory=False, sort_output=False)
            return read_fm_stream(out)

    def _expected_total(self):
        """Every building's every sample, with a zero deductible, reaches the output."""
        return sum(10.0 * building + s + item_id
                   for item_id in range(1, self.ITEMS + 1)
                   for building in range(1, self.BUILDINGS + 1)
                   for s in range(1, self.SAMPLE_SIZE + 1))

    def test_a_net_loss_run_survives_every_allocation_rule(self):
        for allocation_rule in (1, 2, 3):
            with self.subTest(allocation_rule=allocation_rule):
                out = self._run_with_net_loss(allocation_rule)
                total = sum(loss for records in out.values()
                            for sidx, loss in records.items() if sidx > 0)
                self.assertAlmostEqual(total, self._expected_total(), places=2)

    def test_the_output_carries_no_packed_sidx(self):
        out = self._run_with_net_loss(2)
        for records in out.values():
            for sidx in records:
                self.assertLessEqual(sidx, self.SAMPLE_SIZE)
