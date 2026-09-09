"""Building packing in gulpy (the gul manager).

gulpy mirrors gulmc: an item carrying more than one building draws a block of samples per
building and writes them into the sample dimension of one stream item. What is packed is
decided in build_structures from the correlations table, so packing is derived from the input
set rather than configured.

These tests pin the plumbing that carries it -- the per-item arrays out of build_structures and
the per-seed counts the packed draw needs -- because the compute loop itself is only reachable
through a getmodel stream, which the end-to-end runs cover.
"""
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pytest

from oasislmf.pytools.common.data import correlations_dtype, oasis_int
from oasislmf.pytools.common.event_stream import check_packed_sidx_fits
from oasislmf.utils.exceptions import OasisException
from oasislmf.pytools.gul.structure import build_structures

MODEL = Path(__file__).parents[2] / "assets" / "test_model_1"


def _with_correlations(dst, number_of_buildings, keep_separate):
    """Copy the test model and set the packing fields on every correlations row."""
    shutil.copytree(MODEL, dst, dirs_exist_ok=True)
    path = os.path.join(dst, 'input', 'correlations.bin')
    corr = np.fromfile(path, dtype=correlations_dtype)
    # one signed field on the wire: magnitude is the count, negative means "keep separate"
    corr['number_of_buildings'] = -number_of_buildings if keep_separate else number_of_buildings
    corr.tofile(path)
    # a stale cached structure would be loaded in preference to the files
    cache = os.path.join(dst, 'input', 'gulpy_structure')
    if os.path.isdir(cache):
        shutil.rmtree(cache)
    return corr


class TestPackedSidxMustFitTheStream(TestCase):
    """Packing must not push a sidx past what an int32 stream field can hold.

    ``encode_sidx`` computes ``(b - 1) * S + s`` in int64, but a sidx is written as int32. Inside
    njit that store wraps silently instead of raising, and the wrapped value is often NEGATIVE --
    which every reader classifies as a packed special rather than a sample. The result is corrupt
    output rather than a failure, so the bound is checked once up front.
    """

    def test_ordinary_configurations_are_allowed(self):
        for max_buildings, sample_size in ((1, 10 ** 9), (1000, 100_000), (5, 1000), (0, 10)):
            with self.subTest(max_buildings=max_buildings, sample_size=sample_size):
                check_packed_sidx_fits(max_buildings, sample_size, oasis_int)

    def test_an_overflowing_configuration_is_rejected(self):
        with self.assertRaises(OasisException) as caught:
            check_packed_sidx_fits(300_000, 10_000, oasis_int)
        message = str(caught.exception)
        self.assertIn("300000", message)
        self.assertIn("10000", message)

    def test_the_boundary(self):
        """Exactly at the limit is fine; one sample more is not."""
        limit = np.iinfo(oasis_int).max
        check_packed_sidx_fits(2, limit // 2, oasis_int)
        with self.assertRaises(OasisException):
            check_packed_sidx_fits(2, limit // 2 + 1, oasis_int)

    def test_what_would_happen_without_it(self):
        """The value the guard prevents being written -- negative, so read as a special."""
        wrapped = np.array([300_000 * 10_000], dtype=np.int64).astype(oasis_int)[0]
        self.assertLess(int(wrapped), 0)


@pytest.mark.skipif(not MODEL.exists(), reason="test_model_1 assets not available")
class TestGulpyPackingStructures(TestCase):

    def test_one_building_reads_as_no_packing(self):
        """The default, and every input set generated without building packing."""
        with TemporaryDirectory() as d:
            _with_correlations(d, 1, 0)
            s = build_structures(d, set(), [])
            self.assertEqual(s['building_packing'], 0)
            # signed: +1 == one building, summed at source
            self.assertEqual(s['n_buildings_by_item_id'].tolist(), [1])

    def test_several_buildings_turn_packing_on(self):
        with TemporaryDirectory() as d:
            corr = _with_correlations(d, 3, 1)
            s = build_structures(d, set(), [])
            self.assertEqual(s['building_packing'], 1)
            # indexed by item_id, so entry 0 is unused and the rest carry the count
            for item_id in corr['item_id']:
                # negative == keep the buildings separate
                self.assertEqual(s['n_buildings_by_item_id'][item_id], -3)

    def test_the_separability_flag_is_carried_independently(self):
        """Packed but summed at source: several buildings, none needing to stay apart."""
        with TemporaryDirectory() as d:
            corr = _with_correlations(d, 4, 0)
            s = build_structures(d, set(), [])
            self.assertEqual(s['building_packing'], 1)
            for item_id in corr['item_id']:
                # positive == summed at source
                self.assertEqual(s['n_buildings_by_item_id'][item_id], 4)

    def test_the_structure_cache_round_trips_the_packing_fields(self):
        """run() prefers the cached structure, so the fields have to survive it."""
        from oasislmf.pytools.gul.structure import create_gulpy_structure, load_gulpy_structure
        with TemporaryDirectory() as d:
            corr = _with_correlations(d, 5, 1)
            create_gulpy_structure(d, set(), [])
            s = load_gulpy_structure(d)
            self.assertEqual(s['building_packing'], 1)
            self.assertEqual(s['n_buildings_by_item_id'][corr['item_id'][0]], -5)

    def test_an_unusable_cache_reads_as_absent(self):
        """A cache that cannot be read must rebuild, not fail the run.

        run() prefers a cached structure whenever one is reported present, so reporting an
        unusable one present makes the load raise instead of falling back to building it. The
        cache is written and read by the same version -- it is built once per run and then
        memory-mapped by the parallel gulpy processes -- so the case to survive is a partially
        written one, from a build that was interrupted.

        The metadata is read positionally, so a short array is the one shape that would fail at
        load rather than fall back.
        """
        from oasislmf.pytools.gul.structure import (METADATA_FIELDS, create_gulpy_structure,
                                                    gulpy_structure_exists)
        with TemporaryDirectory() as d:
            _with_correlations(d, 2, 1)
            create_gulpy_structure(d, set(), [])
            cache = os.path.join(d, 'input', 'gulpy_structure')
            self.assertTrue(gulpy_structure_exists(d))

            # too few scalars to index positionally
            metadata = os.path.join(cache, 'metadata.npy')
            np.save(metadata, np.zeros(len(METADATA_FIELDS) - 1, dtype=np.int64))
            self.assertFalse(gulpy_structure_exists(d))

            # truncated mid-write
            with open(metadata, 'r+b') as f:
                f.truncate(8)
            self.assertFalse(gulpy_structure_exists(d))

            # absent entirely
            os.remove(metadata)
            self.assertFalse(gulpy_structure_exists(d))
