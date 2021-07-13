import numpy as np

from .financial_structure import create_financial_structure, load_financial_structure
from .stream import read_stream_header, read_streams, read_streams_sparse, EventWriter, EventWriterOrderedOutput, EventWriterSparse, EXTRA_VALUES
from .compute import compute_event, init_variable, reset_variabe, init_variable_sparse, compute_sparse, reset_variabe_sparse
from .common import allowed_allocation_rule


import tempfile
import sys
import logging
logger = logging.getLogger(__name__)


def run(create_financial_structure_files, low_memory, **kwargs):
    if create_financial_structure_files:
        create_financial_structure(kwargs['allocation_rule'], kwargs['static_path'])
    elif low_memory:
        return run_synchronous_sparse(**kwargs)
    else:
        return run_synchronous(low_memory=low_memory, **kwargs)


def run_synchronous(allocation_rule, static_path, files_in, files_out, low_memory, net_loss, sort_output, **kwargs):
    if allocation_rule == 3:
        allocation_rule = 2
    elif allocation_rule == 0 and net_loss:
        raise NotImplementedError("net loss option is not implemented for alloc rule 0")

    compute_info, nodes_array, node_parents_array, node_profiles_array, output_array, fm_profile = load_financial_structure(
        allocation_rule, static_path)

    compute_info = compute_info[0]
    stepped = True if compute_info['stepped'] else None  # https://github.com/numba/numba/issues/4108
    if files_out is not None:
        files_out = files_out[0]

    if files_in is None:
        streams_in = [sys.stdin.buffer]
    else:
        streams_in = [open(file_in, 'rb') for file_in in files_in]

    if sort_output:
        event_writer_cls = EventWriterOrderedOutput
    else:
        event_writer_cls = EventWriter

    try:
        for stream_in in streams_in:
            stream_type, len_sample = read_stream_header(stream_in)
        len_array = len_sample + EXTRA_VALUES

        with tempfile.TemporaryDirectory() as tempdir:
            losses, loss_indexes, extras, extra_indexes, children, computes = init_variable(compute_info, len_array, tempdir, low_memory)

            with event_writer_cls(files_out, nodes_array, output_array, losses, loss_indexes, computes, len_sample) as event_writer:
                for event_id, compute_i in read_streams(streams_in, nodes_array, losses, loss_indexes, computes):
                    compute_i, loss_i, extra_i = compute_event(compute_info,
                                                               net_loss,
                                                               nodes_array,
                                                               node_parents_array,
                                                               node_profiles_array,
                                                               losses,
                                                               loss_indexes,
                                                               extras,
                                                               extra_indexes,
                                                               children,
                                                               computes,
                                                               compute_i,
                                                               fm_profile,
                                                               stepped)
                    compute_i = event_writer.write(event_id, compute_i)
                    reset_variabe(children, compute_i, computes, loss_i, losses)
    finally:
        if files_in is not None:
            for stream_in in streams_in:
                stream_in.close()


def run_synchronous_sparse(allocation_rule, static_path, files_in, files_out, net_loss, sort_output,
                           compute_size, loss_threshold, **kwargs):
    if allocation_rule == 3:
        allocation_rule = 2
    elif allocation_rule == 0 and net_loss:
        raise NotImplementedError("net loss option is not implemented for alloc rule 0")

    compute_info, nodes_array, node_parents_array, node_profiles_array, output_array, fm_profile = load_financial_structure(
        allocation_rule, static_path)

    compute_info = compute_info[0]
    stepped = True if compute_info['stepped'] else None  # https://github.com/numba/numba/issues/4108
    if files_out is not None:
        files_out = files_out[0]

    if files_in is None:
        streams_in = [sys.stdin.buffer]
    else:
        streams_in = [open(file_in, 'rb') for file_in in files_in]

    event_writer_cls = EventWriterSparse

    try:
        for stream_in in streams_in:
            stream_type, len_sample = read_stream_header(stream_in)
        len_array = len_sample + EXTRA_VALUES

        with tempfile.TemporaryDirectory() as tempdir:
            (loss_indexes, extra_indexes, children, computes, losses, extras,
             event_items, in_indptr_array, in_sidx_array, in_loss_array,
             output_count, output_id_array, out_sidx, out_loss) = init_variable_sparse(compute_info, len_array, compute_size, tempdir)

            with event_writer_cls(files_out, nodes_array, output_array, output_count, output_id_array, out_sidx, out_loss, len_sample) as event_writer:
                for event_id, event_items_count in read_streams_sparse(streams_in, in_indptr_array, in_sidx_array, in_loss_array, event_items):
                    output_id_count = compute_sparse(net_loss, compute_size, loss_threshold, stepped,
                                   compute_info, nodes_array, node_parents_array, node_profiles_array, output_array,
                                   fm_profile,
                                   len_array, event_items_count,
                                   loss_indexes, extra_indexes, children, computes, losses, extras,
                                   event_items, in_indptr_array, in_sidx_array, in_loss_array,
                                   output_count, output_id_array, out_sidx, out_loss)
                    event_writer.write(event_id, output_id_count)
                    reset_variabe_sparse(event_items_count, event_items, in_indptr_array, output_count)
    finally:
        if files_in is not None:
            for stream_in in streams_in:
                stream_in.close()

