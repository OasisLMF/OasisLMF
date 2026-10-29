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
from oasislmf.pytools.common.data import (FM_STRUCTURE_INFO_FILE, COVERAGE_BUILDINGS_FILE,
                                          coverage_buildings_dtype, oasis_float,
                                          fm_policytc_dtype, fm_profile_dtype,
                                          fm_programme_dtype, fm_xref_dtype, items_dtype)
from oasislmf.utils.exceptions import OasisException
from oasislmf.pytools.fm.financial_structure import (
    create_financial_structure,
    load_fm_structure_info,
)


class TestLoadFmStructureInfo(TestCase):

    def test_absent_or_empty_file_reads_as_nothing_to_collapse(self):
        """Every input set generated before building-packing has no such file."""
        with TemporaryDirectory() as d:
            self.assertEqual(load_fm_structure_info(d), (0, 1, 0, 0))
            open(os.path.join(d, FM_STRUCTURE_INFO_FILE), "wb").close()
            self.assertEqual(load_fm_structure_info(d), (0, 1, 0, 0))

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


# --- the two files that size the packed arena -------------------------------------------------

def _write_packed_structure(d, n_buildings):
    """One packed item per coverage, each its own site node at level 1, summed at level 2.

    Item 1's site node carries a min/max deductible, so the extras arena is in play as well as the
    loss arena. ``n_buildings`` is the signed count per coverage.
    """
    n_items = len(n_buildings)
    programme = np.array([(i, 1, i) for i in range(1, n_items + 1)]
                         + [(i, 2, 1) for i in range(1, n_items + 1)],
                         dtype=fm_programme_dtype)
    policytc = np.array([(1, i, 1, 1 if i == 1 else 0) for i in range(1, n_items + 1)]
                        + [(2, 1, 1, 0)], dtype=fm_policytc_dtype)
    profile = np.zeros(2, dtype=fm_profile_dtype)
    profile[0]['profile_id'], profile[0]['calcrule_id'] = 0, 12    # pass-through
    profile[1]['profile_id'], profile[1]['calcrule_id'] = 1, 13    # deductible with a minimum
    profile[1]['deductible1'], profile[1]['deductible2'] = 100., 50.
    xref = np.array([(1, 1, 1)], dtype=fm_xref_dtype)

    for name, arr in (('fm_programme', programme), ('fm_policytc', policytc),
                      ('fm_profile', profile), ('fm_xref', xref)):
        arr.tofile(os.path.join(d, f'{name}.bin'))

    items = np.zeros(n_items, dtype=items_dtype)
    items['item_id'] = np.arange(1, n_items + 1)
    items['coverage_id'] = np.arange(1, n_items + 1)
    items.tofile(os.path.join(d, 'items.bin'))
    # coverages.bin stays the published tiv-only format; the count has its own file
    np.full(n_items, 1000., dtype=oasis_float).tofile(os.path.join(d, 'coverages.bin'))
    buildings = np.zeros(n_items, dtype=coverage_buildings_dtype)
    buildings['coverage_id'] = np.arange(1, n_items + 1)
    buildings['n_building'] = n_buildings          # signed: negative keeps them separate
    buildings.tofile(os.path.join(d, f'{COVERAGE_BUILDINGS_FILE}.bin'))

    max_buildings = int(max(abs(b) for b in n_buildings))
    write_fm_structure_info(d, 1, max_buildings,
                            total_packed_buildings=max_buildings * n_items)


class TestTheTwoBuildingFilesMustAgree(TestCase):
    """fm_structure_info and coverage_buildings size the arena between them, and are written together.

    A folder where they disagree -- copied by hand, or partly regenerated -- under-reserves the
    arena, which numba does not bounds-check, so it has to be refused at load rather than run.
    """

    def _build(self, n_buildings, max_buildings=None, drop_coverage_buildings=False):
        with TemporaryDirectory() as d:
            _write_packed_structure(d, n_buildings)
            if max_buildings is not None:
                write_fm_structure_info(d, 1, max_buildings, total_packed_buildings=sum(abs(b) for b in n_buildings))
            if drop_coverage_buildings:
                os.remove(os.path.join(d, f'{COVERAGE_BUILDINGS_FILE}.bin'))
            create_financial_structure(0, d)

    def test_packed_structure_info_without_the_counts_is_rejected(self):
        """Read as all ones, the counts would size the extras arena for one building a node."""
        with self.assertRaises(OasisException) as caught:
            self._build([-4, -4, -1, -1], drop_coverage_buildings=True)
        self.assertIn('coverage_buildings', str(caught.exception))

    def test_a_separated_count_above_max_buildings_is_rejected(self):
        """max_buildings sizes the dense temporaries, so a coverage above it overruns them."""
        with self.assertRaises(OasisException) as caught:
            self._build([-4, -2, -1, -1], max_buildings=2)
        self.assertIn('4', str(caught.exception))

    def test_matching_files_are_accepted(self):
        self._build([-4, -2, -1, -1])
        # A positive count was summed at source and never reaches fm packed, so max_buildings,
        # which only bounds the separated ones, may well be below it.
        self._build([6, -2, -1, -1], max_buildings=2)


# --- which nodes the extras closure marks --------------------------------------------------------

def _write_shared_item_structure(d, extras_on_agg):
    r"""Three items, two level-2 nodes, and item 2 belonging to BOTH of them.

        items/level1:   1     2     3
                         \   / \   /
        level2:           (2,1)  (2,2)      <- item 2 has two parents at the same level
                             \   /
        level3:               (3,1)

    ``extras_on_agg`` puts the min/max deductible on one of the two level-2 nodes. The closure
    that marks extras walks node -> its items -> every parent of those items AT THE NODE'S OWN
    LEVEL -> all their descendants, so either placement has to mark the same set: the two nodes
    share item 2 and its loss is split between them.
    """
    programme = np.array([
        (1, 1, 1), (2, 1, 2), (3, 1, 3),       # items -> their own level-1 nodes
        (1, 2, 1), (2, 2, 1),                  # (2,1) <- items 1, 2
        (2, 2, 2), (3, 2, 2),                  # (2,2) <- items 2, 3
        (1, 3, 1), (2, 3, 1),                  # both -> (3,1)
    ], dtype=fm_programme_dtype)
    policytc = np.array(
        [(1, a, 1, 0) for a in (1, 2, 3)]
        + [(2, a, 1, 1 if a == extras_on_agg else 0) for a in (1, 2)]
        + [(3, 1, 1, 0)], dtype=fm_policytc_dtype)
    profile = np.zeros(2, dtype=fm_profile_dtype)
    profile[0]['profile_id'], profile[0]['calcrule_id'] = 0, 12     # pass-through
    profile[1]['profile_id'], profile[1]['calcrule_id'] = 1, 13     # deductible with a minimum
    profile[1]['deductible1'], profile[1]['deductible2'] = 100., 50.
    xref = np.array([(a, a, 1) for a in (1, 2, 3)], dtype=fm_xref_dtype)
    for name, arr in (('fm_programme', programme), ('fm_policytc', policytc),
                      ('fm_profile', profile), ('fm_xref', xref)):
        arr.tofile(os.path.join(d, f'{name}.bin'))


def _write_tree_structure(d):
    """The same programme with item 2 under ONE parent -- the shape generation actually emits."""
    programme = np.array([
        (1, 1, 1), (2, 1, 2), (3, 1, 3),
        (1, 2, 1), (2, 2, 1),                  # (2,1) <- items 1, 2
        (3, 2, 2),                             # (2,2) <- item 3 only
        (1, 3, 1), (2, 3, 1),
    ], dtype=fm_programme_dtype)
    policytc = np.array(
        [(1, a, 1, 0) for a in (1, 2, 3)]
        + [(2, 1, 1, 1), (2, 2, 1, 0)]
        + [(3, 1, 1, 0)], dtype=fm_policytc_dtype)
    profile = np.zeros(2, dtype=fm_profile_dtype)
    profile[0]['profile_id'], profile[0]['calcrule_id'] = 0, 12
    profile[1]['profile_id'], profile[1]['calcrule_id'] = 1, 13
    profile[1]['deductible1'], profile[1]['deductible2'] = 100., 50.
    xref = np.array([(a, a, 1) for a in (1, 2, 3)], dtype=fm_xref_dtype)
    for name, arr in (('fm_programme', programme), ('fm_policytc', policytc),
                      ('fm_profile', profile), ('fm_xref', xref)):
        arr.tofile(os.path.join(d, f'{name}.bin'))


class TestOneParentPerLevel(TestCase):
    """A node feeding two nodes at ONE level is rejected, loudly, when the structure is built.

    The structure is a forest, not a tree, and several parents on one node are fine as long as
    they sit at different levels -- root_start produces exactly that, and the insurance_policy_coverage
    units in tests/fm build such sets end to end, so a check that also rejected them would fail
    there.

    The extraction walks nodes in index order and, for a node carrying a min/max deductible,
    marks every node that shares an item with it -- with one parent per level, itself and its
    descendants, all already written. A second parent at the same level would put a node the
    loop has not reached into that set, and whether it kept its extras would then depend on the
    agg_id ordering.
    """

    def test_a_shared_parent_level_is_rejected(self):
        """The shape is what is wrong, not where the deductible sits -- and it was the placement
        that decided whether the old extraction went wrong, so both have to be refused."""
        for extras_on_agg in (1, 2):
            with self.subTest(extras_on_agg=extras_on_agg):
                with TemporaryDirectory() as d:
                    _write_shared_item_structure(d, extras_on_agg)
                    with self.assertRaises(OasisException) as raised:
                        create_financial_structure(2, d)
                message = str(raised.exception)
                self.assertIn("more than one node at the same level", message)
                self.assertIn("(2, 2)", message, "the message must name the offending (level, child)")

    def test_a_plain_tree_is_accepted(self):
        """The same programme with item 2 under one parent only must build."""
        with TemporaryDirectory() as d:
            _write_tree_structure(d)
            create_financial_structure(2, d)          # must not raise
