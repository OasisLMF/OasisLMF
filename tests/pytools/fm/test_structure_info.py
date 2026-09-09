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
from oasislmf.pytools.common.data import (FM_STRUCTURE_INFO_FILE, fm_policytc_dtype,
                                          fm_profile_dtype, fm_programme_dtype, fm_xref_dtype)
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
            self.assertEqual(load_fm_structure_info(d), (0, 1))

    def test_values_are_read(self):
        """Round trip through the writer generation actually uses."""
        with TemporaryDirectory() as d:
            write_fm_structure_info(d, 3, 7)
            self.assertEqual(load_fm_structure_info(d), (3, 7))

    def test_an_empty_file_falls_back_to_the_default(self):
        """The fixed-width equivalent of a malformed file: no record to read."""
        with TemporaryDirectory() as d:
            open(os.path.join(d, FM_STRUCTURE_INFO_FILE), "wb").close()
            self.assertEqual(load_fm_structure_info(d), (0, 1))

    def test_max_buildings_is_never_below_one(self):
        """It multiplies array sizes, so a bad value must not shrink them."""
        with TemporaryDirectory() as d:
            write_fm_structure_info(d, 1, 0)
            self.assertEqual(load_fm_structure_info(d), (1, 1))


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
                        write_fm_structure_info(d, written[0], written[1])

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
            write_fm_structure_info(d, 1, 3)
            create_financial_structure(0, d)
            compute_info = load_financial_structure(0, d)[0][0]
            self.assertGreater(compute_info['packable_node_len'], 0)
            self.assertLessEqual(compute_info['packable_node_len'], compute_info['node_len'])
