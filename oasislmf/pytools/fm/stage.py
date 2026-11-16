"""Financial module stage: set up once, then compute and write events one block at a time.

``FmStage`` holds everything ``fmpy`` builds before its event loop (financial structure,
computation arrays, reader and writer) and processes events handed to it in memory. The
``fmpy`` command is built on it (see ``manager.run_synchronous_sparse``), so both share one
implementation of the per-event work.
"""
import json
import logging
import os
import tempfile
import time

import numpy as np

from oasislmf.pytools.common.event_stream import (GUL_STREAM_ID, FM_STREAM_ID, LOSS_STREAM_ID, LOSS_STREAM_AGG_TYPES,
                                                  bytes_to_stream_types)
from oasislmf.utils.exceptions import OasisStreamException
from .compute_sparse import compute_event as compute_event_sparse
from .compute_sparse import DEBUG_PROFILE, PROFILE_LEVELS, PROFILE_METRICS
from .compute_sparse import init_variable as init_variable_sparse
from .compute_sparse import reset_variable as reset_variable_sparse
from .compute_sparse import load_net_value
from .financial_structure import load_financial_structure
from .stream_sparse import FMReader, EventWriterSparse, EventWriterOrderedOutputSparse

logger = logging.getLogger(__name__)

STREAM_HEADER_SIZE = 8


def resolve_allocation_rule(allocation_rule, net):
    """Return the allocation rule fm computes with, refusing the combinations it does not support.

    Args:
        allocation_rule (int): allocation rule requested.
        net (bool): whether net loss will be output.

    Returns:
        int: the allocation rule to use (rule 3 is computed as rule 2).

    Raises:
        NotImplementedError: net loss with allocation rule 0.
    """
    if allocation_rule == 3:
        return 2
    elif allocation_rule == 0 and net:
        raise NotImplementedError("net loss option is not implemented for alloc rule 0")
    return allocation_rule


def parse_input_header(input_header):
    """Validate the header of an fm input stream as fmpy does and return its sample count.

    Args:
        input_header (bytes): the 8-byte header of the upstream loss stream.

    Returns:
        np.int32: the sample count (max sidx) of the stream.

    Raises:
        Exception: the stream type or the aggregation type is not one fm reads.
    """
    input_header = bytes(input_header)
    if len(input_header) != STREAM_HEADER_SIZE:
        raise ValueError(f'stream header must be {STREAM_HEADER_SIZE} bytes, got {len(input_header)}')
    stream_source_type, stream_agg_type = bytes_to_stream_types(input_header[:4])
    max_sidx_val = np.frombuffer(input_header[4:], dtype=np.int32)[0]

    if stream_source_type not in [GUL_STREAM_ID, FM_STREAM_ID, LOSS_STREAM_ID]:
        raise Exception(f'unsupported stream_type {stream_source_type} (most probable cause is that the up stream data are incorrect)')
    # ITEM_PACKED_STREAM is read by the same path: an unpacked item is the one-building case
    # of the packed encoding. The check is here so an aggregation type we have NOT defined --
    # a future layout, or a stream from a newer writer -- is refused rather than read as if
    # it were items, which would be a wrong loss and not a failure.
    if stream_agg_type not in LOSS_STREAM_AGG_TYPES:
        raise Exception(f'unsupported stream aggregation type {stream_agg_type}, expected one of '
                        f'{LOSS_STREAM_AGG_TYPES} (item, or item with packed buildings)')
    return max_sidx_val


def _dump_fm_profile(stats, phase=None):
    """Append the per-level profile to $FM_PROFILE_OUT, one row per level.

    fmpy writes the IL stream on stdout, so profiling output has to go to a file.
    """
    if not DEBUG_PROFILE:
        return
    path = os.environ.get("FM_PROFILE_OUT")
    if not path:
        return
    with open(path, "a") as fh:
        if phase is not None:
            fh.write("PHASE\tread=%.4f\tcompute=%.4f\twrite=%.4f\tevents=%d\n" % (
                phase['read'], phase['compute'], phase['write'], phase['events']))
        for lvl in range(stats.shape[0]):
            if stats[lvl, 1] == 0 and stats[lvl, 0] == 0:
                continue
            fh.write("%d\t%.4f\t%d\t%d\t%d\t%d\t%d\n" % (
                lvl, stats[lvl, 0], stats[lvl, 1], stats[lvl, 2],
                stats[lvl, 3], stats[lvl, 4], stats[lvl, 5]))


class FmStage:
    """One financial module step (IL or an RI inuring step), set up once and fed events in memory.

    Paths resolve as in ``fmpy``, so ``static_path`` is relative to the working directory.

    Args:
        input_header (bytes): the 8-byte header of the upstream loss stream. It is validated as
            fmpy validates its input and gives the sample count.
        allocation_rule (int): back-allocation rule (0, 1, 2 or 3; 3 is computed as 2).
        static_path (str): folder holding the financial structure files.
        gross (bool): whether gross loss can be requested.
        net (bool): whether net loss can be requested.
        low_memory (bool): store the loss arrays in memory maps in a temporary directory.
        sort_output (bool): write each event's output nodes sorted by output id.
        storage_method (str): only 'sparse' is supported.
        **kwargs: ignored, so that the fmpy command-line options can be passed through.

    Attributes:
        header (bytes): the stream header fmpy writes before its first event (the same for gross
            and net).
        reader (FMReader): the reader loading events into the computation arrays.
    """

    def __init__(self, input_header, allocation_rule, static_path, gross=True, net=False,
                 low_memory=False, sort_output=False, storage_method='sparse', **kwargs):
        if not (gross or net):
            raise ValueError("at least one of gross or net must be requested")
        allocation_rule = resolve_allocation_rule(allocation_rule, net)
        max_sidx_val = parse_input_header(input_header)
        if storage_method != "sparse":
            raise ValueError(f"storage_method {storage_method} is not supported for this version")

        self.gross = gross
        self.net = net
        self._closed = False
        self._failed = False
        self._tempdir = tempfile.TemporaryDirectory()
        try:
            self._setup(max_sidx_val, allocation_rule, static_path, low_memory, sort_output)
        except BaseException:
            self._tempdir.cleanup()
            raise

    def _setup(self, max_sidx_val, allocation_rule, static_path, low_memory, sort_output):
        compute_info, nodes_array, node_parents_array, node_profiles_array, output_array, fm_profile = load_financial_structure(
            allocation_rule, static_path)

        compute_info = compute_info[0]
        stepped = True if compute_info['stepped'] else None  # https://github.com/numba/numba/issues/4108

        # net_loss storage is reserved by init, so resolve the flag before it rather than after:
        # rule 1 always keeps the input loss, and every net-loss output mode does too.
        keep_input_loss = compute_info['allocation_rule'] == 1 or self.net

        (max_sidx_val, max_sidx_count, len_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, pass_through,
         extras_indptr, extras_val, children, computes, item_parent_i, compute_idx) = init_variable_sparse(
            compute_info, max_sidx_val, self._tempdir.name, low_memory, keep_input_loss)

        if allocation_rule == 0:
            pass_through_out = np.zeros_like(pass_through)
        else:
            pass_through_out = pass_through

        event_writer_cls = EventWriterOrderedOutputSparse if sort_output else EventWriterSparse
        self.writer = event_writer_cls(
            nodes_array, output_array, sidx_indexes,
            sidx_indptr, sidx_val, loss_indptr, loss_val,
            pass_through_out, max_sidx_val, computes
        )
        self.header = self.writer.header

        self.reader = FMReader(nodes_array, sidx_indexes, sidx_indptr, sidx_val,
                               loss_indptr, loss_val, pass_through, len_array, computes, compute_idx, max_sidx_val,
                               compute_info['max_buildings'] > 1,
                               # packed input, but no level applies terms per building. 0 is the "no risk-keyed level"
                               # marker, so it means nothing to collapse whatever start_level is -- hence max(1, ...).
                               compute_info['max_buildings'] > 1
                               and compute_info['site_collapse_level'] < max(1, compute_info['start_level']))

        self._compute_args = (
            compute_info,
            keep_input_loss,
            nodes_array,
            node_parents_array,
            node_profiles_array,
            len_array, max_sidx_val, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, extras_indptr, extras_val,
            children,
            computes,
            compute_idx,
            item_parent_i,
            fm_profile,
            stepped)
        self.nodes_array = nodes_array
        self.computes = computes
        self.compute_idx = compute_idx
        self.children = children
        self.sidx_indexes = sidx_indexes
        self.sidx_indptr = sidx_indptr
        self.loss_indptr = loss_indptr
        self.loss_val = loss_val
        self.pass_through = pass_through
        self.pass_through_out = pass_through_out

        self.fm_profile_stats = np.zeros((PROFILE_LEVELS, PROFILE_METRICS), dtype=np.float64)
        self.phase = {'read': 0.0, 'compute': 0.0, 'write': 0.0, 'events': 0}
        self.event_i = -1
        self.event_id = None

    def process_block(self, block, gross_out=None, net_out=None):
        """Compute each event of an in-memory block, in order, and write its gross and/or net loss.

        Args:
            block (np.ndarray): 1-D int8 array holding one or more whole events of the input loss
                stream, without the stream header. It may be read-only.
            gross_out: sink (object with ``write(data)``) for the gross loss, or None.
            net_out: sink for the net loss, or None.
        """
        self.process_events(self.reader.read_block(block), gross_out, net_out)

    def process_events(self, events, gross_out=None, net_out=None):
        """Compute and write each event yielded by ``events``.

        ``events`` must be an iterator over event ids produced by ``self.reader`` (``read_block`` or
        ``read_streams``), which loads each event into the computation arrays before yielding it.
        On failure the event is recorded in ``event_error.json`` in the working directory.

        Args:
            events: iterator of event ids loaded by ``self.reader``.
            gross_out: sink for the gross loss, or None.
            net_out: sink for the net loss, or None.
        """
        if gross_out is not None and not self.gross:
            raise ValueError("gross loss was not requested when the stage was created")
        if net_out is not None and not self.net:
            raise ValueError("net loss was not requested when the stage was created")

        compute_args = self._compute_args
        fm_profile_stats = self.fm_profile_stats
        phase = self.phase
        writer = self.writer
        computes, compute_idx = self.computes, self.compute_idx
        try:
            while True:
                if DEBUG_PROFILE:
                    _t = time.perf_counter()
                try:
                    event_id = next(events)
                except StopIteration:
                    break
                self.event_i += 1
                self.event_id = event_id
                if DEBUG_PROFILE:
                    _r = time.perf_counter()
                    phase['read'] += _r - _t
                    phase['events'] += 1
                compute_event_sparse(*compute_args, fm_profile_stats)
                if DEBUG_PROFILE:
                    _c = time.perf_counter()
                    phase['compute'] += _c - _r
                if gross_out is not None:
                    writer.write(event_id, compute_idx, gross_out)
                if net_out is not None:
                    load_net_value(computes, compute_idx, self.nodes_array, self.sidx_indptr, self.sidx_indexes,
                                   self.loss_indptr, self.loss_val)
                    writer.write(event_id, compute_idx, net_out)
                reset_variable_sparse(self.children, compute_idx, computes)
                if DEBUG_PROFILE:
                    phase['write'] += time.perf_counter() - _c
        except OasisStreamException:
            self._failed = True
            logger.error("Stream read error in fm")
            raise
        except Exception:
            self._failed = True
            node = self.nodes_array[computes[compute_idx['compute_i']]]
            data = {
                "event_index": self.event_i,
                "event_id": self.event_id,
                "agg_id": node['agg_id'],
                "node_level_id": node['level_id']
            }

            with open("event_error.json", "w") as f:
                json.dump(data, f, default=str)

            logger.error(f"event index={self.event_i} id={self.event_id}, at node level_id={node['level_id']} agg_id={node['agg_id']} failed in fm")
            raise

    def reset(self):
        """Clear the state fm carries from one event to the next, so the stage then behaves as a fresh one.

        The chance of loss (sidx -4) read for an event is kept in ``pass_through``, which nothing
        clears between events, and is written out for later events whose own input has no -4
        record. fmpy keeps this behaviour within a process (its output is unchanged), so call this
        at a chunk boundary to make a chunk's output independent of the events processed before it.
        The error report's event index restarts at 0; the debug profile keeps accumulating.
        """
        self.pass_through.fill(0)
        if self.pass_through_out is not self.pass_through:
            self.pass_through_out.fill(0)
        self.event_i = -1
        self.event_id = None

    def close(self):
        """Write the debug profile (if enabled and no event failed) and release the temporary directory."""
        if self._closed:
            return
        self._closed = True
        if not self._failed:
            _dump_fm_profile(self.fm_profile_stats, self.phase)
        self._tempdir.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._failed = True
        self.close()
