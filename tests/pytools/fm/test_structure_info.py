"""Tests for the building-packing structure info the financial module reads at load time.

Building-packed items keep their buildings apart until the site levels -- the ones whose
aggregation key includes ``risk_id`` -- have applied their terms per building, and collapse
straight after. Which fm level that is cannot be derived from the fm input files: their levels
are compacted (only levels carrying terms get one), so the numbering varies per portfolio.
IL generation records it in ``fm_structure_info.bin`` and the financial module carries it on
``compute_info``.

An input set without the file reads as 0, meaning there is nothing to collapse. That is every
input set not generated with building-packing, so the default has to stay backward compatible.
"""
import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from oasislmf.preparation.il_inputs import write_fm_structure_info
from oasislmf.pytools.common.data import (FM_STRUCTURE_INFO_FILE, coverages_bin_dtype,
                                          fm_policytc_dtype, fm_profile_dtype,
                                          fm_programme_dtype, fm_xref_dtype, items_dtype)
from oasislmf.utils.exceptions import OasisException
from oasislmf.pytools.fm.financial_structure import (
    compute_info_dtype,
    create_financial_structure,
    load_financial_structure,
    load_fm_structure_info,
)


def _write_minimal_fm_inputs(d):
    """Write the smallest coherent fm input set: two items, one level, one output.

    Built here rather than borrowed from another asset directory. The extraction runs in numba
    without bounds checking, so feeding it an input set whose files disagree corrupts the heap
    instead of raising -- which is what an incompatible fixture did while this test was written.
    """
    programme = np.array([(1, 1, 1), (2, 1, 1)], dtype=fm_programme_dtype)
    policytc = np.array([(1, 1, 1, 1)], dtype=fm_policytc_dtype)
    profile = np.zeros(2, dtype=fm_profile_dtype)
    profile[0]['profile_id'], profile[0]['calcrule_id'] = 0, 12   # 12 == pass-through
    profile[1]['profile_id'], profile[1]['calcrule_id'] = 1, 12
    xref = np.array([(1, 1, 1)], dtype=fm_xref_dtype)
    for name, arr in (('fm_programme', programme), ('fm_policytc', policytc),
                      ('fm_profile', profile), ('fm_xref', xref)):
        arr.tofile(os.path.join(d, f'{name}.bin'))


class TestLoadFmStructureInfo(TestCase):

    def test_absent_file_reads_as_nothing_to_collapse(self):
        """Every input set generated before building-packing has no such file."""
        with TemporaryDirectory() as d:
            self.assertEqual(load_fm_structure_info(d), (0, 1, 0, 0))

    def test_values_are_read(self):
        """Round trip through the writer generation actually uses."""
        with TemporaryDirectory() as d:
            write_fm_structure_info(d, 3, 7, 4200, 9100)
            self.assertEqual(load_fm_structure_info(d), (3, 7, 4200, 9100))

    def test_an_empty_file_falls_back_to_the_default(self):
        """The fixed-width equivalent of a malformed file: no record to read."""
        with TemporaryDirectory() as d:
            open(os.path.join(d, FM_STRUCTURE_INFO_FILE), "wb").close()
            self.assertEqual(load_fm_structure_info(d), (0, 1, 0, 0))

    def test_max_buildings_is_never_below_one(self):
        """It multiplies array sizes, so a bad value must not shrink them."""
        with TemporaryDirectory() as d:
            write_fm_structure_info(d, 1, 0, 0)
            self.assertEqual(load_fm_structure_info(d), (1, 1, 0, 0))

    def test_a_file_of_the_wrong_layout_is_rejected(self):
        """Reading it as "no packing" would drop the collapse silently and give wrong losses.

        np.fromfile yields zero records for a record of the wrong width, which is indistinguishable
        from an empty file unless the size is checked. Older layouts are not supported -- the point
        here is that they fail loudly.
        """
        short = np.dtype([("site_collapse_level", "<i4"), ("max_buildings", "<i4")])
        with TemporaryDirectory() as d:
            np.array([(2, 9)], dtype=short).tofile(os.path.join(d, FM_STRUCTURE_INFO_FILE))
            with self.assertRaises(OasisException):
                load_fm_structure_info(d)

    def test_a_total_below_the_maximum_is_rejected(self):
        """The total is a SUM over the items the maximum is taken from, so it cannot be smaller.

        It sizes the arena, and under-reserving there is a write past the end of a numba array --
        corruption rather than an exception.
        """
        with TemporaryDirectory() as d:
            write_fm_structure_info(d, 1, 9, 4)
            with self.assertRaises(OasisException):
                load_fm_structure_info(d)


class TestComputeInfoCarriesIt(TestCase):

    def test_compute_info_has_the_field(self):
        self.assertIn('site_collapse_level', compute_info_dtype.dtype.names)

    def test_round_trip_through_the_structure_cache(self):
        """The value survives extract -> save -> load, and defaults to 0 without the file."""
        for written, expected_level, expected_nb in ((None, 0, 1), ((2, 5), 2, 5)):
            with self.subTest(written=written):
                with TemporaryDirectory() as d:
                    _write_minimal_fm_inputs(d)
                    if written is not None:
                        write_fm_structure_info(d, written[0], written[1], written[1] * 2)

                    create_financial_structure(0, d)
                    compute_info = load_financial_structure(0, d)[0][0]
                    self.assertEqual(compute_info['site_collapse_level'], expected_level)
                    self.assertEqual(compute_info['max_buildings'], expected_nb)
                    # the rest of the structure is unaffected by the new fields
                    self.assertGreater(compute_info['max_level'], 0)
                    self.assertGreater(compute_info['node_len'], 0)

    def test_packable_node_len_counts_only_nodes_up_to_the_collapse_level(self):
        """Nodes above the collapse level see the collapsed loss and keep their ordinary size."""
        with TemporaryDirectory() as d:
            _write_minimal_fm_inputs(d)
            create_financial_structure(0, d)
            self.assertEqual(load_financial_structure(0, d)[0][0]['packable_node_len'], 0)

        with TemporaryDirectory() as d:
            _write_minimal_fm_inputs(d)
            write_fm_structure_info(d, 1, 3, 6)
            create_financial_structure(0, d)
            compute_info = load_financial_structure(0, d)[0][0]
            self.assertGreater(compute_info['packable_node_len'], 0)
            self.assertLessEqual(compute_info['packable_node_len'], compute_info['node_len'])


# --- the extras arena's packed budget -----------------------------------------------------------

def _write_extras_structure(d, extras_items, n_buildings, with_coverages=True):
    """Four packed items, each its own site node at level 1, summed by one node at level 2.

    ``extras_items`` are the item_ids whose site node carries a min/max deductible -- a
    ``need_extras`` calcrule. The rest take a plain pass-through, so the fixture can show the
    budget tracking which nodes were actually marked rather than how many could have been.

    ``with_coverages`` writes items.bin and coverages.bin, which is where the per-node building
    count comes from; omitting them is the fall-back case.
    """
    n_items = len(n_buildings)
    programme = np.array([(i, 1, i) for i in range(1, n_items + 1)]
                         + [(i, 2, 1) for i in range(1, n_items + 1)],
                         dtype=fm_programme_dtype)
    policytc = np.array([(1, i, 1, 1 if i in extras_items else 0) for i in range(1, n_items + 1)]
                        + [(2, 1, 1, 0)], dtype=fm_policytc_dtype)
    profile = np.zeros(2, dtype=fm_profile_dtype)
    profile[0]['profile_id'], profile[0]['calcrule_id'] = 0, 12    # pass-through
    profile[1]['profile_id'], profile[1]['calcrule_id'] = 1, 13    # deductible with a minimum
    profile[1]['deductible1'], profile[1]['deductible2'] = 100., 50.
    xref = np.array([(1, 1, 1)], dtype=fm_xref_dtype)

    for name, arr in (('fm_programme', programme), ('fm_policytc', policytc),
                      ('fm_profile', profile), ('fm_xref', xref)):
        arr.tofile(os.path.join(d, f'{name}.bin'))

    if with_coverages:
        items = np.zeros(n_items, dtype=items_dtype)
        items['item_id'] = np.arange(1, n_items + 1)
        items['coverage_id'] = np.arange(1, n_items + 1)
        items.tofile(os.path.join(d, 'items.bin'))
        coverages = np.zeros(n_items, dtype=coverages_bin_dtype)
        coverages['tiv'] = 1000.
        coverages['n_building'] = n_buildings          # signed: negative keeps them separate
        coverages.tofile(os.path.join(d, 'coverages.bin'))

    max_buildings = int(max(abs(b) for b in n_buildings))
    write_fm_structure_info(d, 1, max_buildings,
                            total_packed_buildings=max_buildings * n_items)


def _extras_budget(extras_items, n_buildings, with_coverages=True):
    with TemporaryDirectory() as d:
        _write_extras_structure(d, extras_items, n_buildings, with_coverages)
        create_financial_structure(0, d)
        compute_info = load_financial_structure(0, d)[0][0]
    return compute_info


class TestExtrasArenaPackedBudget(TestCase):
    """The extras arena is charged per marked node, not per packable node.

    Extras are 3 floats a slot against the loss arena's 1, so budgeting every packable node for
    them makes the extras array the largest in the module on a book where a handful of locations
    carry a min/max deductible. The counts come from coverages.bin, which is why the fall-back
    below still has to work.
    """

    def test_only_the_marked_nodes_are_charged(self):
        # four packed items, two of them with the min-deductible calcrule
        info = _extras_budget(extras_items={1, 2}, n_buildings=[-4, -4, -1, -1])
        # 2 nodes x 1 layer, each at its own building count
        self.assertEqual(int(info['extra_len']), 2)
        self.assertEqual(int(info['packable_extra_slots']), 4 + 4)
        # the loss arena still covers every packable node, so the two now differ
        self.assertGreater(int(info['packable_layer_slots']), int(info['packable_extra_slots']))

    def test_a_single_building_node_is_still_charged_one_slice(self):
        """The collapse appends the collapsed copy rather than shrinking in place, so even an
        unpacked packable node owes a slice. A count of 1 must not read as 'nothing to reserve'."""
        info = _extras_budget(extras_items={3}, n_buildings=[-4, -4, -1, -1])
        self.assertEqual(int(info['packable_extra_slots']), 1)

    def test_no_extras_rule_reserves_nothing(self):
        info = _extras_budget(extras_items=set(), n_buildings=[-4, -4, -1, -1])
        self.assertEqual(int(info['extra_len']), 0)
        self.assertEqual(int(info['packable_extra_slots']), 0)

    def test_every_node_marked_matches_the_portfolio_bound(self):
        """With every packable node marked the exact sum is the bound, which pins the two against
        each other -- a drift in either shows up here rather than as a corrupt arena."""
        info = _extras_budget(extras_items={1, 2, 3, 4}, n_buildings=[-4, -4, -4, -4])
        self.assertEqual(int(info['packable_extra_slots']), int(info['packable_layer_slots']))

    def test_without_coverages_it_falls_back_to_the_bound(self):
        """No items/coverages means no per-node counts, and the budget has to over-reserve rather
        than guess: under-reserving is a write past the end of a numba array."""
        info = _extras_budget(extras_items={1}, n_buildings=[-4, -4, -1, -1], with_coverages=False)
        self.assertEqual(int(info['packable_extra_slots']), int(info['packable_layer_slots']))
        self.assertGreater(int(info['packable_extra_slots']), 0)


# --- which nodes the extras closure marks --------------------------------------------------------
