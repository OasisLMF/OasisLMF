import tempfile
import logging
import numpy as np
from contextlib import ExitStack

from .financial_structure import create_financial_structure, load_financial_structure
from .stream_sparse import FMReader, EventWriterSparse, EventWriterOrderedOutputSparse
from .compute_sparse import compute_event as compute_event_sparse
from .compute_sparse import init_variable as init_variable_sparse
from .compute_sparse import reset_variable as reset_variable_sparse
from .compute_sparse import load_net_value
from oasislmf.pytools.utils import redirect_logging
from oasislmf.pytools.common.event_stream import init_streams_in, GUL_STREAM_ID, FM_STREAM_ID, LOSS_STREAM_ID


logger = logging.getLogger(__name__)


def run(create_financial_structure_files, **kwargs):
    if create_financial_structure_files:
        create_financial_structure(kwargs['allocation_rule'], kwargs['static_path'])
    else:
        return run_synchronous(**kwargs)


@redirect_logging(exec_name='fmpy')
def run_synchronous(allocation_rule, files_in, files_out, net_loss, storage_method, **kwargs):
    if allocation_rule == 3:
        allocation_rule = 2
    elif allocation_rule == 0 and net_loss is not None:
        raise NotImplementedError("net loss option is not implemented for alloc rule 0")

    if files_out is not None:
        files_out = files_out[0]

    with ExitStack() as stack:
        streams_in, (stream_source_type, stream_agg_type, max_sidx_val) = init_streams_in(files_in, stack)

        if stream_source_type not in [GUL_STREAM_ID, FM_STREAM_ID, LOSS_STREAM_ID]:
            raise Exception(f'unsupported stream_type {stream_source_type} (most probable cause is that the up stream data are incorrect)')

        if storage_method == "sparse":
            run_synchronous_sparse(max_sidx_val, allocation_rule, streams_in=streams_in, files_out=files_out, net_loss=net_loss, stack=stack,
                                   **kwargs)
        else:
            raise ValueError(f"storage_method {storage_method} is not supported for this version")


def run_synchronous_sparse(max_sidx_val, allocation_rule, static_path, streams_in, files_out, low_memory, net_loss, sort_output, stack, **kwargs):
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
         extras_indptr, extras_val, children, computes, item_parent_i, compute_idx) = init_variable_sparse(compute_info, max_sidx_val, tempdir, low_memory)

        if allocation_rule == 0:
            pass_through_out = np.zeros_like(pass_through)
        else:
            pass_through_out = pass_through

        keep_input_loss = False
        if compute_info['allocation_rule'] == 1:
            keep_input_loss = True

        if net_loss is None:  # stream out need to provide gross loss
            gross_writer = stack.enter_context(
                event_writer_cls(
                    files_out, nodes_array, output_array, sidx_indexes,
                    sidx_indptr, sidx_val, loss_indptr, loss_val,
                    pass_through_out, max_sidx_val, computes
                )
            )
            net_writer = False
        elif net_loss == '':  # stream out need to provide net_loss instead of gross loss
            gross_writer = False
            net_writer = stack.enter_context(
                event_writer_cls(
                    files_out, nodes_array, output_array, sidx_indexes,
                    sidx_indptr, sidx_val, loss_indptr, loss_val,
                    pass_through_out, max_sidx_val, computes
                )
            )
            keep_input_loss = True
        elif net_loss == '-':  # double stream out, net_loss written to stdout
            gross_writer = False
            if files_out is not None:  # gross loss written out only if files_out path given
                gross_writer = stack.enter_context(
                    event_writer_cls(
                        files_out, nodes_array, output_array, sidx_indexes,
                        sidx_indptr, sidx_val, loss_indptr, loss_val,
                        pass_through_out, max_sidx_val, computes
                    )
                )
            net_writer = stack.enter_context(
                event_writer_cls(
                    None, nodes_array, output_array, sidx_indexes, sidx_indptr,
                    sidx_val, loss_indptr, loss_val, pass_through_out,
                    max_sidx_val, computes
                )
            )
            keep_input_loss = True
        else:  # double stream out, net_loss is the extra path to write the net to
            gross_writer = stack.enter_context(
                event_writer_cls(
                    files_out, nodes_array, output_array, sidx_indexes,
                    sidx_indptr, sidx_val, loss_indptr, loss_val,
                    pass_through_out, max_sidx_val, computes
                )
            )
            net_writer = stack.enter_context(
                event_writer_cls(
                    net_loss, nodes_array, output_array, sidx_indexes,
                    sidx_indptr, sidx_val, loss_indptr, loss_val,
                    pass_through_out, max_sidx_val, computes
                )
            )
            keep_input_loss = True

        fm_reader = FMReader(nodes_array, sidx_indexes, sidx_indptr, sidx_val,
                             loss_indptr, loss_val, pass_through, len_array, computes, compute_idx)

        for i, event_id in enumerate(fm_reader.read_streams(streams_in)):
            try:
                compute_event_sparse(
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
                if gross_writer:
                    gross_writer.write(event_id, compute_idx)
                if net_writer:
                    load_net_value(computes, compute_idx, nodes_array, sidx_indptr, sidx_indexes, loss_indptr, loss_val)
                    net_writer.write(event_id, compute_idx)
                reset_variable_sparse(children, compute_idx, computes)
            except Exception:
                node = nodes_array[computes[compute_idx['compute_i']]]
                logger.error(f"event index={i} id={event_id}, "
                             f"at node level_id={node['level_id']} agg_id={node['agg_id']}")
                raise
