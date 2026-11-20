import sys
import logging
from contextlib import ExitStack

import numpy as np

from .financial_structure import create_financial_structure
from .stage import FmStage, resolve_allocation_rule, _dump_fm_profile  # noqa: F401
from oasislmf.pytools.common.stage import StreamSink
from oasislmf.pytools.utils import redirect_logging
from oasislmf.pytools.common.event_stream import init_streams_in, stream_info_to_bytes


logger = logging.getLogger(__name__)


def run(create_financial_structure_files, **kwargs):
    if create_financial_structure_files:
        create_financial_structure(kwargs['allocation_rule'], kwargs['static_path'])
    else:
        return run_synchronous(**kwargs)


@redirect_logging(exec_name='fmpy')
def run_synchronous(allocation_rule, files_in, files_out, net_loss, storage_method, **kwargs):
    resolve_allocation_rule(allocation_rule, net_loss is not None)

    if files_out is not None:
        files_out = files_out[0]

    with ExitStack() as stack:
        streams_in, (stream_source_type, stream_agg_type, max_sidx_val) = init_streams_in(files_in, stack)
        input_header = stream_info_to_bytes(stream_source_type, stream_agg_type) + np.int32(max_sidx_val).tobytes()
        run_synchronous_sparse(input_header, allocation_rule, streams_in=streams_in, files_out=files_out, net_loss=net_loss, stack=stack,
                               storage_method=storage_method, **kwargs)


def _open_stream_out(path, stack):
    """Return stdout if path is None, otherwise path opened for binary writing on the stack."""
    if path is None:
        return sys.stdout.buffer
    return stack.enter_context(open(path, 'wb'))


def run_synchronous_sparse(input_header, allocation_rule, static_path, streams_in, files_out, low_memory, net_loss, sort_output, stack,
                           storage_method='sparse', **kwargs):
    """Run fm over the input streams, writing gross and/or net loss as selected by net_loss and files_out.

    Output modes:

    - ``net_loss`` None: gross to ``files_out`` (stdout if None).
    - ``net_loss`` '': net to ``files_out`` (stdout if None).
    - ``net_loss`` '-': net to stdout, and gross to ``files_out`` if given.
    - ``net_loss`` a path: gross to ``files_out`` (stdout if None) and net to that path.

    Args:
        input_header (bytes): header of the input streams.
        allocation_rule (int): back-allocation rule.
        static_path (str): folder holding the financial structure files.
        streams_in (list): open input streams, positioned after their header.
        files_out (str | None): gross (or net, if net_loss is '') output path; None for stdout.
        low_memory (bool): store loss arrays in memory maps.
        net_loss (str | None): net loss output mode, see above.
        sort_output (bool): sort each event's output by output id.
        stack (ExitStack): stack the output files are opened on.
        storage_method (str): only 'sparse' is supported.
        **kwargs: remaining fmpy options, unused.
    """
    if net_loss is None:  # stream out need to provide gross loss
        gross_path, net_path = files_out, False
    elif net_loss == '':  # stream out need to provide net_loss instead of gross loss
        gross_path, net_path = False, files_out
    elif net_loss == '-':  # double stream out, net_loss written to stdout
        # gross loss written out only if files_out path given
        gross_path, net_path = (False if files_out is None else files_out), None
    else:  # double stream out, net_loss is the extra path to write the net to
        gross_path, net_path = files_out, net_loss

    stage = stack.enter_context(FmStage(input_header, allocation_rule, static_path,
                                        gross=gross_path is not False, net=net_path is not False,
                                        low_memory=low_memory, sort_output=sort_output, storage_method=storage_method))

    gross_out = net_out = None
    if gross_path is not False:
        gross_stream = _open_stream_out(gross_path, stack)
        gross_stream.write(stage.header)
        gross_out = StreamSink(gross_stream)
    if net_path is not False:
        net_stream = _open_stream_out(net_path, stack)
        net_stream.write(stage.header)
        net_out = StreamSink(net_stream)

    stage.process_events(stage.reader.read_streams(streams_in), gross_out, net_out)
