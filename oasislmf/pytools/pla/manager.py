from contextlib import ExitStack
import os
import sys
import logging

from .streams import read_and_write_streams
from .structure import (
    get_items_amplifications,
    get_post_loss_amplification_factors
)
from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.utils import redirect_logging
from oasislmf.pytools.common.event_stream import get_streams_in

logger = logging.getLogger(__name__)


@redirect_logging(exec_name='plapy')
def run(
    run_dir, file_in, file_out, input_path, static_path, secondary_factor,
    uniform_factor
):
    """
    Execute the main Post Loss Amplification workflow.

    Args:
        run_dir (str): the directory of where the process is running
        file_in (str): file name of input stream
        file_out (str): file name of output streak
        input_path (str): path to amplifications.bin
        static_path (str): path to lossfactors.bin
        secondary_factor (float): secondary factor to apply to post loss
          amplification
        uniform_factor (float): uniform factor to apply across all losses

    Returns:
        0 (int): if no errors occurred
    """
    input_path = os.path.join(run_dir, input_path)
    model_storage = get_storage_from_config_path(
        os.path.join(run_dir, 'model_storage.json'),
        os.path.join(run_dir, static_path),
    )

    items_amps = get_items_amplifications(input_path)
    plafactors = get_post_loss_amplification_factors(model_storage, secondary_factor, uniform_factor)

    # Set default factor should post loss amplification factor be missing
    default_factor = 1.0 if uniform_factor == 0.0 else uniform_factor

    with ExitStack() as stack:
        streams_in = get_streams_in(file_in, stack)

        if file_out is None:
            stream_out = sys.stdout.buffer
        else:
            stream_out = stack.enter_context(open(file_out, 'wb'))

        read_and_write_streams(
            streams_in, stream_out, items_amps, plafactors, default_factor
        )

    return 0
