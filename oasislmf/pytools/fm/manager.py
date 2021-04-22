from .financial_structure import create_financial_structure, load_financial_structure
from .stream import read_stream_header, read_streams, EventWriter, EventWriterOrderedOutput, EXTRA_VALUES
from .compute import compute_event, init_variable, reset_variabe
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
