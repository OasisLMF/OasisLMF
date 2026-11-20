"""modelpy as a stage: set up once, then compute the damage CDFs of one event at a time.

``ModelpyStage`` holds everything the ``modelpy`` command builds before its event loop and runs the
loop body for one event on demand. ``oasislmf.pytools.getmodel.manager.run`` (the ``modelpy``
command) is built on it: it reads event ids from its input stream and hands each to ``process_event``.
"""
import atexit
import logging
import os
from contextlib import ExitStack

import numpy as np

from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.common.event_stream import PIPE_CAPACITY
from oasislmf.pytools.data_layer.footprint_layer import FootprintLayerClient
from oasislmf.pytools.getmodel import manager as getmodel_manager
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.getmodel import structure as getmodel_structure

# the manager's logger, so the modelpy log reads exactly as it did before the stage was split out of run
logger = logging.getLogger('oasislmf.pytools.getmodel.manager')

DEFAULT_DF_ENGINE = "oasis_data_manager.df_reader.reader.OasisPandasReader"
CDF_STREAM_HEADER = np.uint32(1).tobytes()


class ModelpyStage:
    """The modelpy damage CDF calculation, set up once and then run one event at a time.

    The constructor does everything the ``modelpy`` command does before its first event: it registers
    with the data server when asked, loads or builds the model structures, opens the footprint and
    allocates the output buffer. ``process_event`` computes one event and writes exactly the bytes
    ``modelpy`` writes for it. ``header`` is the stream header ``modelpy`` writes before its first event,
    so ``header`` followed by the bytes of each event in turn is the ``modelpy`` output stream.

    The defaults are those of the ``modelpy`` command line.

    The stage is not reentrant: its buffer is reused from one event to the next, and the bytes handed
    to a sink are a view of it, valid only until the sink's ``write`` returns.

    Args:
        run_dir (str): the run directory, holding ``input/`` and ``static/`` (or ``model_storage.json``).
        ignore_file_type (Iterable[str], optional): file extensions to ignore when loading. Defaults to ().
        data_server (bool, optional): if True, get the footprint from the data server. Defaults to False.
        peril_filter (Iterable[str], optional): perils to include (all if empty or None). Defaults to None.
        df_engine (str, optional): the engine to use when loading dataframes. Defaults to OasisPandasReader.

    Attributes:
        header (bytes): the stream header (the CDF stream type) modelpy writes before its first event.
    """

    header = CDF_STREAM_HEADER

    def __init__(self, run_dir, ignore_file_type=(), data_server=False, peril_filter=None, df_engine=DEFAULT_DF_ENGINE):
        model_storage = get_storage_from_config_path(
            os.path.join(run_dir, 'model_storage.json'),
            os.path.join(run_dir, 'static'),
        )
        ignore_file_type = set(ignore_file_type)
        self.data_server = data_server

        if data_server:
            logger.debug("data server active")
            FootprintLayerClient.register()
            logger.debug("registered with data server")
            atexit.register(FootprintLayerClient.unregister)
        else:
            logger.debug("data server not active")

        with ExitStack() as stack:
            if getmodel_structure.getmodel_structure_exists(run_dir):
                logger.info("loading pre-computed getmodel structures (shared memory)")
                structures = getmodel_structure.load_getmodel_structure(run_dir)
            else:
                logger.info("building getmodel structures from input files")
                structures = getmodel_structure.build_structures(run_dir, ignore_file_type, peril_filter, df_engine)

            self.vuln_array = structures['vuln_array']
            self.vulns_id = structures['vulns_id']
            self.areaperil_id_ind = structures['areaperil_id_ind']
            self.areaperil_to_vulns_idx_array = structures['areaperil_to_vulns_idx_array']
            self.areaperil_to_vulns = structures['areaperil_to_vulns']
            unique_areaperil_ids = structures['unique_areaperil_ids']
            self.mean_damage_bins = structures['mean_damage_bins']
            self.num_damage_bins = structures['num_damage_bins']
            num_intensity_bins = structures['num_intensity_bins']

            logger.debug('init footprint')
            self.footprint_obj = stack.enter_context(
                Footprint.load(model_storage, ignore_file_type, df_engine=df_engine,
                               areaperil_ids=list(unique_areaperil_ids)))

            if data_server:
                num_intensity_bins: int = FootprintLayerClient.get_number_of_intensity_bins()
                logger.info(f"got {num_intensity_bins} intensity bins from server")
            self.num_intensity_bins = num_intensity_bins

            # event_id, areaperil_id, vulnerability_id, num_result, [oasis_float] * num_result
            self.max_result_relative_size = (1 + getmodel_manager.areaperil_int_relative_size + 1 + 1
                                             + self.num_damage_bins * getmodel_manager.results_relative_size)
            self.byte_mv = np.empty(PIPE_CAPACITY, dtype=np.int8)
            self.int32_mv = self.byte_mv.view(np.int32)
            self._stack = stack.pop_all()

    def process_event(self, event_id, out):
        """Compute the damage CDFs of one event and write its bytes to ``out``.

        ``out.write`` is called each time the buffer is full and once at the end of the event, with a
        ``int8`` numpy view of the stage's buffer that must be consumed before ``write`` returns. An event
        with no footprint, or none of whose area perils is in the portfolio, writes nothing.

        Args:
            event_id (int): the event to compute.
            out: sink with a ``write(data)`` method.

        Returns:
            bool: False for an event modelpy counts as empty (no footprint, or no CDF written at the end of
            the event), True otherwise.
        """
        event_id = np.int32(event_id)
        if self.data_server:
            event_footprint = FootprintLayerClient.get_event(event_id)
        else:
            event_footprint = self.footprint_obj.get_event(event_id)

        if event_footprint is None:
            return False

        for cursor_bytes in getmodel_manager.doCdf(event_id,
                                                   self.num_intensity_bins, event_footprint,
                                                   self.areaperil_id_ind, self.areaperil_to_vulns_idx_array,
                                                   self.areaperil_to_vulns,
                                                   self.vuln_array, self.vulns_id, self.num_damage_bins, self.mean_damage_bins,
                                                   self.int32_mv, self.max_result_relative_size):
            if cursor_bytes:
                out.write(self.byte_mv[:cursor_bytes])
            else:
                return False
        return True

    def close(self):
        """Release the footprint."""
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
