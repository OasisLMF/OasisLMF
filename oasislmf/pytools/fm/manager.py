import numpy as np

from .financial_structure import create_financial_structure, load_financial_structure
from .stream import read_stream_header, read_streams, EventWriter, EventWriterOrderedOutput, EXTRA_VALUES
from .stream_sparse import EventWriterSparse, read_streams_sparse, EventWriterOrderedOutputSparse
from .compute import compute_event, init_variable, reset_variable
from .compute_sparse import compute_event as compute_event_sparse
from .compute_sparse import init_variable as init_variable_sparse
from .compute_sparse import reset_variable as reset_variable_sparse

from .common import allowed_allocation_rule


import tempfile
import sys
import logging
logger = logging.getLogger(__name__)


def run(create_financial_structure_files, **kwargs):
    if create_financial_structure_files:
        create_financial_structure(kwargs['allocation_rule'], kwargs['static_path'])
    else:
        return run_synchronous(**kwargs)


def run_synchronous(allocation_rule, files_in, files_out, net_loss,  storage_method, **kwargs):
    if allocation_rule == 3:
        allocation_rule = 2
    elif allocation_rule == 0 and net_loss:
        raise NotImplementedError("net loss option is not implemented for alloc rule 0")

    if files_out is not None:
        files_out = files_out[0]

    if files_in is None:
        streams_in = [sys.stdin.buffer]
    else:
        streams_in = [open(file_in, 'rb') for file_in in files_in]

    try:
        for stream_in in streams_in:
            stream_type, max_sidx_val = read_stream_header(stream_in)

        if storage_method == "sparse":
            run_synchronous_sparse(max_sidx_val, allocation_rule, streams_in=streams_in, files_out = files_out, net_loss=net_loss, **kwargs)
        else:
            raise ValueError(f"storage_method {storage_method} is not supported for this version")
            run_synchronous_dense(max_sidx_val, allocation_rule, streams_in=streams_in, files_out = files_out, net_loss=net_loss, **kwargs)

    finally:
        if files_in is not None:
            for stream_in in streams_in:
                stream_in.close()


def run_synchronous_dense(max_sidx_val, allocation_rule, static_path, streams_in, files_out, low_memory, net_loss, sort_output, **kwargs):
    compute_info, nodes_array, node_parents_array, node_profiles_array, output_array, fm_profile = load_financial_structure(
        allocation_rule, static_path)

    compute_info = compute_info[0]
    stepped = True if compute_info['stepped'] else None  # https://github.com/numba/numba/issues/4108

    if sort_output:
        event_writer_cls = EventWriterOrderedOutput
    else:
        event_writer_cls = EventWriter

    len_array = max_sidx_val + EXTRA_VALUES

    with tempfile.TemporaryDirectory() as tempdir:
        losses, loss_indexes, extras, extra_indexes, children, computes = init_variable(compute_info, len_array, tempdir, low_memory)

        with event_writer_cls(files_out, nodes_array, output_array, losses, loss_indexes, computes, max_sidx_val) as event_writer:
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
                reset_variable(children, compute_i, computes, loss_i, losses)


def run_synchronous_sparse(max_sidx_val, allocation_rule, static_path, streams_in, files_out, low_memory, net_loss, sort_output,  **kwargs):
    compute_info, nodes_array, node_parents_array, node_profiles_array, output_array, fm_profile = load_financial_structure(
        allocation_rule, static_path)

    compute_info = compute_info[0]
    stepped = True if compute_info['stepped'] else None  # https://github.com/numba/numba/issues/4108

    if sort_output:
        event_writer_cls = EventWriterOrderedOutputSparse
    else:
        event_writer_cls = EventWriterSparse

    with tempfile.TemporaryDirectory() as tempdir:
        (max_sidx_val, max_sidx_count, len_array, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, pass_through,
         extras_indptr, extras_val, children, computes, item_parent_i) = init_variable_sparse(compute_info, max_sidx_val, tempdir, low_memory)

        if allocation_rule == 0:
            pass_through_out = np.zeros_like(pass_through)
        else:
            pass_through_out = pass_through
        
        with event_writer_cls(files_out, nodes_array, output_array, sidx_indexes, sidx_indptr, sidx_val,
                              loss_indptr, loss_val, pass_through_out, max_sidx_val, computes) as event_writer:
            for event_id, compute_i in read_streams_sparse(streams_in, nodes_array, sidx_indexes, sidx_indptr, sidx_val,
                                                           loss_indptr, loss_val, pass_through, len_array, computes):
                compute_i = compute_event_sparse(compute_info,
                                           net_loss,
                                           nodes_array,
                                           node_parents_array,
                                           node_profiles_array,
                                           len_array, max_sidx_val, sidx_indexes, sidx_indptr, sidx_val, loss_indptr, loss_val, extras_indptr, extras_val,
                                           children,
                                           computes,
                                           compute_i,
                                           item_parent_i,
                                           fm_profile,
                                           stepped)
                if allocation_rule == 0:
                    compute_i = event_writer.write(event_id, compute_i)
                else:
                    event_writer.write(event_id, 0)
                reset_variable_sparse(children, compute_i, computes)
