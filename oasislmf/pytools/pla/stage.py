"""The post loss amplification of ``plapy`` as an in-process stage, applied to blocks of whole events.

``plapy`` multiplies each loss of the ground up stream by the factor of its (event, amplification id) pair, or by
a uniform factor, and passes the stream on otherwise unchanged. ``PlaStage`` loads the same factors once and
applies the same Numba kernels to a block in place, so a chain runs ``gulmc | plapy`` as one step.
"""
import os

import numpy as np
from oasis_data_manager.filestore.config import get_storage_from_config_path

from oasislmf.pytools.common.input_files import read_amplifications

from .streams import read_buffer, read_buffer_uniform
from .structure import get_post_loss_amplification_factors


class PlaStage:
    """Post loss amplification of a ground up loss stream, block by block."""

    def __init__(self, run_dir, input_header, secondary_factor=1.0, uniform_factor=0.0, input_path='input',
                 static_path='static'):
        """Load the amplification factors as ``plapy`` does.

        Args:
            run_dir (str): run directory the paths are relative to.
            input_header (bytes): header of the stream the stage amplifies (stream type and sample size).
            secondary_factor (float): ``plapy -f``, the relative secondary factor in [0, 1].
            uniform_factor (float): ``plapy -F``, a factor applied to every loss; when positive the per-event factors
                are not read and the secondary factor is ignored.
            input_path (str): directory of ``amplifications.bin``.
            static_path (str): directory of ``lossfactors.bin``.
        """
        self.header = bytes(input_header)
        """bytes: the stream header ``plapy`` writes, the same as the one it reads."""
        if uniform_factor > 0:
            self.items_amps = None
            self.plafactors = None
            self._kernel = read_buffer_uniform
        else:
            model_storage = get_storage_from_config_path(os.path.join(run_dir, 'model_storage.json'),
                                                         os.path.join(run_dir, static_path))
            self.items_amps = read_amplifications(os.path.join(run_dir, input_path))
            self.plafactors = get_post_loss_amplification_factors(model_storage, secondary_factor)
            self._kernel = read_buffer
        self.default_factor = 1.0 if uniform_factor == 0.0 else uniform_factor
        self._cursor = np.empty(1, dtype='i4')

    def amplify(self, block):
        """Amplify, in place, the losses of a block of whole events.

        Args:
            block (np.ndarray): writable 1-D int8 array holding whole events of the stream, without its header.
        """
        if block.shape[0]:
            self._kernel(block, 0, block.shape[0], 0, 0, self.items_amps, self.plafactors, self.default_factor,
                         block, self._cursor)
