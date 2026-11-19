"""gulpy as a stage, and ``modelpy | gulpy`` as one ground up loss stage.

``GulpyStage`` holds everything the ``gulpy`` command builds before its event loop and runs the loop body
for one event of the getmodel (CDF) stream on demand. ``oasislmf.pytools.gul.manager.run`` (the ``gulpy``
command) is built on it: it reads the CDF stream with ``read_getmodel_stream`` into the stage's arrays and
hands each event to ``process_cdf_event``.

``ModelpyGulpyStage`` chains a ``ModelpyStage`` and a ``GulpyStage`` in one process: for each event id it
computes the event's CDF records into an in-memory block, reads the block as gulpy reads its stream, and
writes the ground up losses. It has the interface of ``GulmcStage``, so a chain can use either.
"""
import logging

import numpy as np

from oasislmf.pytools.common.data import correlations_dtype, items_dtype
from oasislmf.pytools.common.event_stream import (PIPE_CAPACITY, ITEM_PACKED_STREAM, ITEM_STREAM, LOSS_STREAM_ID,
                                                  check_packed_item_fits, check_packing_supported,
                                                  max_emitted_blocks, stream_info_to_bytes)
from oasislmf.pytools.common.stage import EventBlock
from oasislmf.pytools.getmodel.common import oasis_float
from oasislmf.pytools.getmodel.stage import ModelpyStage
from oasislmf.pytools.gul import manager as gul_manager
from oasislmf.pytools.gul import structure as gul_structure
from oasislmf.pytools.gul.common import NUM_IDX, gulSampleslevelHeader_size, gulSampleslevelRec_size
from oasislmf.pytools.gul.core import HERMITE_TERMS
from oasislmf.pytools.gul.io import read_getmodel_block
from oasislmf.pytools.gul.random import cdf_min, generate_correlated_hash_vector, get_random_generator, inv_factor, norm_factor, x_min

# the manager's logger, so the gulpy log reads exactly as it did before the stage was split out of run
logger = logging.getLogger('oasislmf.pytools.gul.manager')


class GulpyStage:
    """The gulpy ground up loss calculation, set up once and then run one event of the CDF stream at a time.

    The constructor does everything the ``gulpy`` command does before its first event: it loads or builds the
    structures, checks the building packing, sets up the random generator and correlation, and allocates the
    reusable buffers. ``process_cdf_event`` computes one event, as read from the getmodel stream by
    ``read_getmodel_stream`` (given the stage's arrays) or from an in-memory block by ``read_block``, and writes exactly the bytes ``gulpy`` writes
    for it. ``header`` is the stream header ``gulpy`` writes before its first event.

    The defaults are those of the ``gulpy`` command line.

    The stage is not reentrant: its buffers are reused from one event to the next, and the bytes handed to a
    sink are a view of an internal buffer, valid only until the sink's ``write`` returns.

    Args:
        run_dir (str): the run directory.
        sample_size (int, optional): number of random samples to draw. Defaults to 0.
        loss_threshold (float, optional): threshold above which losses are printed to the output stream.
            Defaults to 1e-6.
        alloc_rule (int, optional): back-allocation rule. Defaults to 0.
        debug (bool, optional): if True, for each random sample, write the random value instead of the loss.
            Defaults to False.
        random_generator (int, optional): random generator function id. Defaults to 2.
        ignore_file_type (Iterable[str], optional): file extensions to ignore when loading. Defaults to ().
        peril_filter (Iterable[str], optional): perils to include (all if empty or None). Defaults to None.
        ignore_correlation (bool, optional): if True, do not compute correlated random samples. Defaults to False.

    Attributes:
        header (bytes): the stream header (stream type and sample size) gulpy writes before its first event.

    Raises:
        ValueError: if alloc_rule is not 0, 1, 2, or 3.
    """

    def __init__(self, run_dir, sample_size=0, loss_threshold=1e-6, alloc_rule=0, debug=False, random_generator=2,
                 ignore_file_type=(), peril_filter=None, ignore_correlation=False):
        if gul_structure.gulpy_structure_exists(run_dir):
            logger.info("loading pre-computed gulpy structures (shared memory)")
            structures = gul_structure.load_gulpy_structure(run_dir)
        else:
            logger.info("building gulpy structures from input files")
            structures = gul_structure.build_structures(run_dir, set(ignore_file_type), peril_filter)

        self.damage_bins = structures['damage_bins']
        # writable copy: cur_items/start_items are mutated per event
        self.coverages = structures['coverages'].copy()
        self.items = structures['items']
        self.item_map_hm = structures['item_map_hm']
        self.item_map_hm_keys = structures['item_map_hm_keys']
        self.item_map_ja_offsets = structures['item_map_ja_offsets']

        # coverages are numbered from 1, therefore skip element 0.
        self.compute = np.zeros(self.coverages.shape[0] + 1, self.items.dtype['coverage_id'])

        self.generate_rndm = get_random_generator(random_generator)

        # Building packing is the N > 1 case of one mechanism, not a second path: an unpacked run is every
        # item carrying one building, and the packed generator's first block per seed is the legacy draw
        # byte-for-byte. Signed: magnitude is the building count, a negative sign means "keep the buildings
        # separate".
        n_buildings_by_item_id = structures['n_buildings_by_item_id']
        if debug:
            # debug writes the random values drawn, one per building; summed at source they would add up to
            # anywhere in [0, N] instead of [0, 1], so every item keeps its buildings separate
            n_buildings_by_item_id = -np.abs(n_buildings_by_item_id)
        damage_correlation_by_item_id = structures['damage_correlation_by_item_id']
        max_buildings = int(np.abs(n_buildings_by_item_id).max())
        check_packing_supported(random_generator, n_buildings_by_item_id)
        # only kept-separate items meet either stream ceiling: a summed one writes a single block at
        # sidx 1..S however many buildings it carries
        emitted_blocks = max_emitted_blocks(n_buildings_by_item_id)
        check_packed_item_fits(emitted_blocks, sample_size)

        # The aggregation type says whether a reader may meet a packed sidx, and only a kept-separate item
        # emits one: a run that packs nothing, or packs only items summed at source, declares ITEM_STREAM.
        self.header = (stream_info_to_bytes(LOSS_STREAM_ID, ITEM_PACKED_STREAM if emitted_blocks > 1 else ITEM_STREAM)
                       + np.int32(sample_size).tobytes())

        if alloc_rule not in [0, 1, 2, 3]:
            raise ValueError(f"Expect alloc_rule to be 0, 1, 2, or 3, got {alloc_rule}")

        self.seeds = np.zeros(len(np.unique(self.items['group_id'])), dtype=items_dtype['group_id'])

        do_correlation = bool(structures['do_correlation'])
        if ignore_correlation:
            do_correlation = False
            logger.info("Correlated random number generation: switched OFF because --ignore-correlation is True.")
            # The structures were built with the file's correlation, and --ignore-correlation is a run-time
            # flag the build never saw. A summed packed item's std_dev is combined with this value, so it is
            # zeroed rather than left to report the spread of a correlation nothing drew.
            damage_correlation_by_item_id = np.zeros_like(damage_correlation_by_item_id)

        if do_correlation:
            logger.info("Correlated random number generation: switched ON.")
            self.corr_data_by_item_id = structures['corr_data_by_item_id']
            self.unique_peril_correlation_groups = structures['unique_peril_correlation_groups']
            self.norm_inv_cdf = structures['norm_inv_cdf']
            self.norm_cdf = structures['norm_cdf']
            self.corr_seeds = np.zeros(np.max(self.unique_peril_correlation_groups) + 1, dtype='int64')
            self.arr_min, self.arr_min_cdf = x_min, cdf_min
            self.arr_inv_factor, self.arr_norm_factor = inv_factor, norm_factor
            self.z_unif = np.zeros(sample_size, dtype='float64')
        else:
            if not ignore_correlation:
                logger.info("Correlated random number generation: switched OFF because 0 peril correlation groups were detected or "
                            "the correlation value is zero for all peril correlation groups.")
            # dummy data structures with proper dtypes to allow correct numba compilation
            self.corr_seeds = np.zeros(1, dtype='int64')
            self.corr_data_by_item_id = np.ndarray(1, dtype=correlations_dtype)
            self.arr_min, self.arr_min_cdf = 0., 0.
            self.arr_inv_factor, self.arr_norm_factor = 0., 0.
            self.norm_inv_cdf, self.norm_cdf = np.zeros(1, dtype='float64'), np.zeros(1, dtype='float64')
            self.z_unif = np.zeros(1, dtype='float64')
        self.dummy_eps_ij = np.zeros((1, 1), dtype='float64')

        max_items_per_coverage = np.max(self.coverages[1:]['max_items'])
        self.losses_buffer = np.zeros((sample_size + NUM_IDX + 1, max_items_per_coverage), dtype=oasis_float)
        # Per-building samples: the specials stay on losses_buffer, being building-independent. write_losses
        # reduces across items at a fixed building and then sums the buildings, so a coverage's every building
        # has to be resident before any of it can be written.
        self.building_losses = np.zeros((max(sample_size, 1), max_items_per_coverage, max_buildings), dtype=oasis_float)
        # per item of the current coverage, the correlation between two of its buildings' losses
        self.loss_correlation_by_item = np.zeros(max_items_per_coverage, dtype=oasis_float)
        # Generator 2 is counter-based, so a building's block is produced where it is consumed; these hold one
        # building's worth, and empty_draws stands in for the per-group array that is then never built.
        self.lazy_draws = np.int8(1 if random_generator == 2 else 0)
        self.draw_scratch = np.zeros(max(sample_size, 1), dtype='float64')
        self.perm_scratch = np.zeros(max(sample_size, 1), dtype='float64')
        # 2d like the real array, so that numba types both the same
        self.empty_draws = np.empty((1, 1), dtype='float64')
        self.hermite_coeffs = np.zeros(HERMITE_TERMS, dtype='float64')
        self.byte_mv = np.empty(PIPE_CAPACITY * 2, dtype='b')

        # One block: the item header, a building's NUM_IDX specials and S samples, and the delimiter. A
        # kept-separate item writes one block per building; a summed one writes a single block.
        max_bytes_per_block = gulSampleslevelHeader_size + (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size
        self.max_bytes_per_item = max_bytes_per_block * emitted_blocks

        self.sample_size = sample_size
        self.loss_threshold = loss_threshold
        self.alloc_rule = alloc_rule
        self.debug = debug
        self.do_correlation = do_correlation
        self.n_buildings_by_item_id = n_buildings_by_item_id
        self.damage_correlation_by_item_id = damage_correlation_by_item_id
        self._items_data = None

    def read_block(self, block):
        """Read one event of the getmodel stream held in memory, as ``read_getmodel_stream`` reads it from a stream.

        Args:
            block (numpy.ndarray): 1-D int8 array holding the CDF records ``modelpy`` writes for one event.

        Returns:
            tuple | None: the event's data for ``process_cdf_event``, or None if the block is empty.
        """
        event_data = read_getmodel_block(block, self.items, self.item_map_hm, self.item_map_hm_keys, self.item_map_ja_offsets,
                                         self.coverages, self.compute, self.seeds, self._items_data)
        if event_data is not None:
            self._items_data = event_data[2]
        return event_data

    def process_cdf_event(self, event_data, out):
        """Compute the ground up losses of one event of the getmodel stream and write its bytes to ``out``.

        ``out.write`` is called once each time the compute kernel stops, with an ``int8`` numpy view of the
        stage's buffer that must be consumed before ``write`` returns.

        Args:
            event_data (tuple): the event, from ``read_getmodel_stream`` or ``read_block``.
            out: sink with a ``write(data)`` method.

        Raises:
            RuntimeError: if the compute kernel asks to be resumed without having made progress.
        """
        event_id, compute_i, items_data, damagecdfrecs, recs, rec_idx_ptr, rng_index = event_data
        sample_size = self.sample_size

        # One row of sample_size per rng group. Generator 2 draws each building's block where it is used
        # instead, so nothing is built here for it.
        if self.lazy_draws:
            rndms_base = self.empty_draws
        else:
            rndms_base = self.generate_rndm(self.seeds[:rng_index], sample_size)

        if self.do_correlation:
            generate_correlated_hash_vector(self.unique_peril_correlation_groups, event_id, self.corr_seeds)
            eps_ij = self.generate_rndm(self.corr_seeds, sample_size, skip_seeds=1)
        else:
            eps_ij = self.dummy_eps_ij

        last_processed_coverage_ids_idx = 0
        cursor = 0

        # The buffer has to fit the longest coverage: compute_event_losses returns only between coverages.
        required_bytes = int(self.coverages['cur_items'].max()) * self.max_bytes_per_item
        self.byte_mv = gul_manager.adjust_byte_mv_size(self.byte_mv, required_bytes)
        while last_processed_coverage_ids_idx < compute_i:
            resume_point_before = last_processed_coverage_ids_idx
            cursor, last_processed_coverage_ids_idx = gul_manager.compute_event_losses(
                event_id, self.coverages, self.compute[:compute_i], items_data,
                last_processed_coverage_ids_idx, sample_size, recs, rec_idx_ptr,
                self.damage_bins, self.loss_threshold, self.losses_buffer, self.alloc_rule, self.do_correlation, eps_ij,
                self.corr_data_by_item_id, self.arr_min, self.arr_inv_factor, self.norm_inv_cdf, self.arr_min_cdf,
                self.arr_norm_factor, self.norm_cdf, self.z_unif, self.debug,
                self.building_losses, rndms_base,
                self.seeds, self.lazy_draws, self.draw_scratch, self.perm_scratch,
                self.loss_correlation_by_item, self.hermite_coeffs,
                self.n_buildings_by_item_id, self.damage_correlation_by_item_id,
                self.max_bytes_per_item, self.byte_mv, cursor
            )

            # A call that stops short must have advanced the resume point: it only stops because the buffer
            # is full, and the buffer is empty on entry, so stopping at the same place would loop forever.
            if (last_processed_coverage_ids_idx < compute_i
                    and last_processed_coverage_ids_idx <= resume_point_before):
                raise RuntimeError(
                    f"gulpy made no progress on event {event_id}: it asked to resume at "
                    f"coverage {last_processed_coverage_ids_idx}, no further on than the "
                    f"{resume_point_before} it started from, having written {cursor} bytes "
                    f"into a {self.byte_mv.shape[0]} byte buffer, which must hold the largest "
                    f"coverage whole.")

            if cursor > 0:
                out.write(self.byte_mv[:cursor])
            cursor = 0

        logger.info(f"event {event_id} DONE")

    def close(self):
        """Nothing to release: the stage holds no file or connection."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class ModelpyGulpyStage:
    """``modelpy | gulpy`` as one stage: the ground up losses of one event at a time, from its event id.

    Each event's CDF records are computed by a ``ModelpyStage`` into an in-memory block, read back as ``gulpy``
    reads them from its input stream, and turned into losses by a ``GulpyStage``. ``header`` followed by the
    bytes of each event in turn is what ``evepy | modelpy | gulpy`` writes for those events. An event modelpy
    writes no record for is not in gulpy's input stream, so it writes nothing.

    The footprint is read by the stage itself: modelpy's data server is never used.

    Args:
        run_dir (str): the run directory.
        modelpy (dict, optional): ``ModelpyStage`` keyword arguments, without ``run_dir`` and ``data_server``: the
            options of the ``modelpy`` command. Defaults to modelpy's defaults.
        gulpy (dict, optional): ``GulpyStage`` keyword arguments, without ``run_dir``: the options of the ``gulpy``
            command. Defaults to gulpy's defaults.

    Attributes:
        header (bytes): the stream header gulpy writes before its first event.

    Raises:
        ValueError: if ``modelpy`` asks for the data server.
    """

    def __init__(self, run_dir, modelpy=None, gulpy=None):
        modelpy = dict(modelpy or {})
        if modelpy.pop('data_server', False):
            raise ValueError('ModelpyGulpyStage reads the footprint itself, it cannot use the data server')
        self.modelpy = ModelpyStage(run_dir, **modelpy)
        try:
            self.gulpy = GulpyStage(run_dir, **(gulpy or {}))
        except BaseException:
            self.modelpy.close()
            raise
        self.header = self.gulpy.header
        self._cdf_block = EventBlock()

    def process_event(self, event_id, out):
        """Compute the ground up losses of one event and write its bytes to ``out``.

        Args:
            event_id (int): the event to compute.
            out: sink with a ``write(data)`` method.

        Returns:
            bool: False if modelpy wrote no record for the event (so nothing is written), True otherwise.
        """
        block = self._cdf_block
        block.clear()
        self.modelpy.process_event(event_id, block)
        event_data = self.gulpy.read_block(block.view())
        if event_data is None:
            return False
        self.gulpy.process_cdf_event(event_data, out)
        return True

    def close(self):
        """Release the footprint."""
        self.gulpy.close()
        self.modelpy.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
