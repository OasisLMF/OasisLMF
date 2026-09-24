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

from oasislmf.pytools.common.data import correlations_dtype, items_dtype, oasis_int
from oasislmf.pytools.common.event_stream import (SIDX_DTYPE, check_packed_item_fits, max_emitted_blocks,
                                                  max_packed_buildings)
from oasislmf.utils.exceptions import OasisException
from oasislmf.pytools.gul.structure import build_structures

MODEL = Path(__file__).parents[2] / "assets" / "test_model_1"


def _with_correlations(dst, number_of_buildings, keep_separate):
    """Copy the test model and set the packing fields on every correlations row."""
    shutil.copytree(MODEL, dst, dirs_exist_ok=True)
    path = os.path.join(dst, 'input', 'correlations.bin')
    corr = np.fromfile(path, dtype=correlations_dtype)
    # one signed field on the wire: magnitude is the count, negative means "keep separate"
    corr['packed_buildings'] = -number_of_buildings if keep_separate else number_of_buildings
    corr.tofile(path)
    # a stale cached structure would be loaded in preference to the files
    cache = os.path.join(dst, 'input', 'gulpy_structure')
    if os.path.isdir(cache):
        shutil.rmtree(cache)
    return corr


class TestPackedItemMustFitTheStream(TestCase):
    """Packing must not push a sidx past what an int32 stream field can hold.

    ``encode_sidx`` computes ``(b - 1) * S + s`` in int64, but a sidx is written as int32. Inside
    njit that store wraps silently instead of raising, and the wrapped value is often NEGATIVE --
    which every reader classifies as a packed special rather than a sample. The result is corrupt
    output rather than a failure, so the bound is checked once up front.

    This is a property of the stream FORMAT. The bytes those records occupy are ordinary memory
    and impose no such bound -- see TestOutputBufferIsNotBoundedByAnInt32.
    """

    def test_ordinary_configurations_are_allowed(self):
        for separate, sample_size in ((1, 10 ** 9), (1000, 100_000), (5, 1000), (0, 10)):
            with self.subTest(separate=separate, sample_size=sample_size):
                check_packed_item_fits(separate, sample_size)

    def test_an_overflowing_configuration_is_rejected(self):
        with self.assertRaises(OasisException) as caught:
            check_packed_item_fits(300_000, 10_000)
        message = str(caught.exception)
        self.assertIn("300,000", message)
        self.assertIn("10000", message)
        self.assertIn("IsAggregate", message)

    def test_the_boundary(self):
        """Exactly at the limit is fine; one building more is not."""
        for sample_size in (10, 1000):
            allowed = max_packed_buildings(sample_size)
            with self.subTest(sample_size=sample_size, allowed=allowed):
                check_packed_item_fits(allowed, sample_size)
                with self.assertRaises(OasisException):
                    check_packed_item_fits(allowed + 1, sample_size)

    def test_a_summed_item_is_not_subject_to_the_ceiling(self):
        """A positive count writes one block at sidx 1..S, so it never encodes a packed sidx
        however many buildings it carries. max_emitted_blocks keeps it out of the number checked."""
        packed = np.array([630_510, 1, 1], dtype='i4')      # huge, but summed at source
        check_packed_item_fits(max_emitted_blocks(packed), 1000)   # must not raise

    def test_the_ceiling_is_the_wire_sidx_not_oasis_int(self):
        """OASIS_INT is configurable; the sidx on the wire is not.

        mv_write_sidx_loss writes through the "sidx" record definition, which is int32 whatever
        OASIS_INT says. Keying the ceiling to oasis_int would raise it on a width nothing writes:
        at OASIS_INT=i8, 300,000 buildings at S=10,000 passed the check and wrote sidx
        3,000,000,000 as -1,294,967,296, which every reader takes for a packed special.
        """
        self.assertEqual(SIDX_DTYPE, np.dtype('i4'))
        self.assertEqual(max_packed_buildings(10_000), np.iinfo(np.int32).max // 10_000)
        with self.assertRaises(OasisException):
            check_packed_item_fits(300_000, 10_000)

    def test_what_would_happen_without_it(self):
        """The value the guard prevents being written -- negative, so read as a special."""
        wrapped = np.array([300_000 * 10_000], dtype=np.int64).astype(oasis_int)[0]
        self.assertLess(int(wrapped), 0)


class TestOutputBufferIsNotBoundedByAnInt32(TestCase):
    """The byte counters must be int64.

    They count bytes of an ordinary numpy buffer, which no stream rule bounds. Holding them in an
    int32 put a 2 GB ceiling on the output buffer for no reason beyond the choice of field, and a
    kept-separate item with enough buildings reaches it: at 1000 samples, 630,510 buildings needs
    about 5 GB of estimate. On numpy 2 the assignment raises OverflowError rather than wrapping,
    so the run dies at setup with an error about an int32 that says nothing about the cause.
    """

    def test_the_byte_counters_are_int64(self):
        from oasislmf.pytools.gulmc.common import gulmc_compute_info_type
        fields = gulmc_compute_info_type.dtype
        for name in ('cursor', 'max_bytes_per_item'):
            with self.subTest(field=name):
                self.assertEqual(fields[name], np.dtype(np.int64))

    def test_a_multi_gigabyte_estimate_is_storable(self):
        from oasislmf.pytools.gulmc.common import gulmc_compute_info_type
        info = np.zeros(1, dtype=gulmc_compute_info_type.dtype)
        estimate = (8 + (1000 + 6) * 8) * 630_510          # ~5 GB
        self.assertGreater(estimate, np.iinfo(np.int32).max, "not the case under test")
        info[0]['max_bytes_per_item'] = estimate           # must not raise
        self.assertEqual(int(info[0]['max_bytes_per_item']), estimate)


@pytest.mark.skipif(not MODEL.exists(), reason="test_model_1 assets not available")
class TestGulpyPackingStructures(TestCase):

    def test_one_building_reads_as_no_packing(self):
        """The default, and every input set generated without building packing."""
        with TemporaryDirectory() as d:
            corr = _with_correlations(d, 1, 0)
            s = build_structures(d, set(), [])
            self.assertEqual(s['building_packing'], 0)
            # The array always spans every item, so the compute can index it unconditionally:
            # an unpacked run is the all-ones case, +1 being one building summed at source.
            self.assertEqual(len(s['n_buildings_by_item_id']), int(corr['item_id'].max()) + 1)
            self.assertTrue((s['n_buildings_by_item_id'] == 1).all())

    def test_the_array_spans_the_items_even_when_correlations_is_shorter(self):
        """The compute indexes by item_id with no bounds check, so the array must cover them all.

        It used to be a length-1 sentinel when nothing was packed, which is why the reader
        carried an ``item_id < shape[0]`` guard. Collapsing the packed and unpacked paths
        removed that guard, so the sizing is now load-bearing. Sizing from correlations alone
        is not enough -- items is the table that is indexed -- so truncate correlations and
        check the array still reaches the last item.
        """
        with TemporaryDirectory() as d:
            _with_correlations(d, 1, 0)
            path = os.path.join(d, 'input', 'correlations.bin')
            corr = np.fromfile(path, dtype=correlations_dtype)
            corr[:len(corr) // 2].tofile(path)
            items = np.fromfile(os.path.join(d, 'input', 'items.bin'), dtype=items_dtype)
            max_item_id = int(items['item_id'].max())
            self.assertGreater(max_item_id, len(corr) // 2, 'items must outrun correlations here')

            s = build_structures(d, set(), [])
            self.assertGreater(len(s['n_buildings_by_item_id']), max_item_id)
            for item_id in items['item_id']:
                self.assertEqual(s['n_buildings_by_item_id'][item_id], 1)

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


class TestCorrelationIsLookedUpByItemId(TestCase):
    """corr_data_by_item_id is indexed by item_id, so it must be SCATTERED by item_id.

    It used to be filled positionally (``[1:] = data[...]``) and then read as
    ``corr_data_by_item_id[item['item_id']]``, which only lines up when item_id happens to be a
    dense 1..N. With sparse ids that reads the wrong row, and an id past the end reads out of
    bounds -- unchecked, inside njit. A guard in read_correlations had been masking this by
    rejecting any non-dense table outright, including valid ones.
    """

    def _structures(self, item_ids, rhos, groups):
        with TemporaryDirectory() as tmp:
            dst = Path(tmp) / "model"
            shutil.copytree(MODEL, dst, dirs_exist_ok=True)
            shutil.rmtree(dst / "input" / "gulpy_structure", ignore_errors=True)

            items = np.fromfile(dst / "input" / "items.bin", dtype=items_dtype)[:len(item_ids)]
            items["item_id"] = item_ids
            items.tofile(dst / "input" / "items.bin")

            corr = np.zeros(len(item_ids), dtype=correlations_dtype)
            corr["item_id"] = item_ids
            corr["peril_correlation_group"] = groups
            corr["damage_correlation_value"] = rhos
            corr["packed_buildings"] = 1
            corr.tofile(dst / "input" / "correlations.bin")
            return build_structures(str(dst), set(), [])

    def test_sparse_item_ids_reach_their_own_row(self):
        item_ids, rhos, groups = [5, 6, 9], [0.1, 0.5, 0.9], [1, 2, 3]
        s = self._structures(item_ids, rhos, groups)
        table = s["corr_data_by_item_id"]
        self.assertGreater(len(table), max(item_ids), "table must span every item_id it is indexed by")
        for item_id, rho, group in zip(item_ids, rhos, groups):
            self.assertAlmostEqual(float(table[item_id]["damage_correlation_value"]), rho, places=6,
                                   msg=f"item {item_id} read the wrong correlation")
            self.assertEqual(int(table[item_id]["peril_correlation_group"]), group)

    def test_an_item_id_absent_from_correlations_reads_as_uncorrelated(self):
        """The gaps are the default row, not another item's."""
        s = self._structures([2, 4, 6], [0.3, 0.6, 0.9], [1, 1, 1])
        table = s["corr_data_by_item_id"]
        for absent in (1, 3, 5):
            self.assertEqual(float(table[absent]["damage_correlation_value"]), 0.0,
                             f"item_id {absent} is not in correlations and must read as uncorrelated")

    def test_dense_item_ids_are_unchanged(self):
        """The ordinary case must be untouched by the change."""
        item_ids, rhos = [1, 2, 3], [0.2, 0.4, 0.6]
        table = self._structures(item_ids, rhos, [1, 1, 1])["corr_data_by_item_id"]
        for item_id, rho in zip(item_ids, rhos):
            self.assertAlmostEqual(float(table[item_id]["damage_correlation_value"]), rho, places=6)
