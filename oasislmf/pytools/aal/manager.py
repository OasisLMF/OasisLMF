# aal/manager.py

import logging
import numpy as np
import numba as nb

from oasislmf.pytools.common.data import (oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import (EventReader, mv_read)
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


class PLTReader(EventReader):
    def __init__(self, len_sample, compute_splt, compute_mplt, compute_qplt):
        self.logger = logger

        read_buffer_state_dtype = np.dtype([
            ('len_sample', oasis_int),
            ('reading_losses', np.bool_),
            ('summary_id', oasis_int),
        ])

        self.state = np.zeros(1, dtype=read_buffer_state_dtype)[0]
        self.state["reading_losses"] = False  # Set to true after reading header in read_buffer
        self.state["len_sample"] = len_sample

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id):
        # Pass state variables to read_buffer
        cursor, event_id, item_id, ret = read_buffer(
            byte_mv, cursor, valid_buff, event_id, item_id,
            self.state,
        )
        return cursor, event_id, item_id, ret


@nb.njit(cache=True, error_model="numpy")
def read_buffer(
        byte_mv, cursor, valid_buff, event_id, item_id,
        state,
):
    # Initialise idxs
    last_event_id = event_id

    # Read input loop
    while cursor < valid_buff:
        if not state["reading_losses"]:
            if valid_buff - cursor >= 3 * oasis_int_size + oasis_float_size:
                # Read summary header
                _, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                event_id_new, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                if last_event_id != 0 and event_id_new != last_event_id:
                    # New event, return to process the previous event
                    return cursor - (2 * oasis_int_size), last_event_id, item_id, 1
                event_id = event_id_new
                state["summary_id"], cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
                state["exposure_value"], cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
                state["reading_losses"] = True
            else:
                break  # Not enough for whole summary header

        if state["reading_losses"]:
            if valid_buff - cursor < oasis_int_size + oasis_float_size:
                break  # Not enough for whole record

            # Read sidx
            sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            if sidx == 0:  # sidx == 0, end of record
                # TODO: Do something
                continue

            # Read loss
            loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
            # TODO: Do something else
        else:
            pass  # Should never reach here anyways

    # Update the indices
    return cursor, event_id, item_id, 0


def run(run_dir, subfolder, aal=None, alct=None, meanonly=False):
    """Runs AAL calculations
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
        subfolder (str): Workspace subfolder inside <run_dir>/work/<subfolder>
        aal (str, optional): Path to AAL output file. Defaults to None
        alct (str, optional): Path to ALCT output file. Defaults to None
        meanonly (bool): Output AAL with mean only
    """
    pass


@redirect_logging(exec_name='pltpy')
def main(run_dir='.', subfolder=None, aal=None, alct=None, meanonly=False,):
    run(
        run_dir,
        subfolder,
        aal=aal,
        meanonly=meanonly,
        alct=alct,
    )
