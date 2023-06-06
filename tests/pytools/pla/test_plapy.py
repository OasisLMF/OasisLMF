import filecmp
from mock import patch
import numpy as np
import os
from pathlib import Path
import pytest
from tempfile import NamedTemporaryFile
from unittest import TestCase

from oasislmf.pytools.pla.common import (
    DATA_SIZE,
    event_item_dtype,
    sidx_loss_dtype,
    event_count_dtype,
    amp_factor_dtype,
    ITEMS_AMPLIFICATIONS_FILE_NAME,
    LOSS_FACTORS_FILE_NAME
)
from oasislmf.pytools.pla.manager import run
from oasislmf.pytools.pla.structure import get_items_amplifications


# Reduce BUFFER_SIZE to ensure that loop in
# oasislmf.pytools.pla.streams:read_and_write_buffers() iterates multiple times
@patch('oasislmf.pytools.pla.common.BUFFER_SIZE', 24)
@patch('oasislmf.pytools.pla.common.N_PAIRS', 24 // DATA_SIZE)
class TestPostLossAmplification(TestCase):

    def write_items_amplifications_file(
        self, n_items, itemsamps_file, formula
    ):
        """
        Class method to write items amplifications files, which are used in the
        tests.

        Args:
            itemsamps_file (str): items amplications file name
            n_items (int): number of unique item IDs
            formula (str): expression to evaluate when filling file
        """
        write_buffer = memoryview(bytearray(n_items * DATA_SIZE))
        item_amp_dtype = np.dtype([
            ('item_id', 'i4'), ('amplification_id', 'i4')
        ])
        event_item = np.ndarray(
            n_items, buffer=write_buffer, dtype=item_amp_dtype
        )
        it = np.nditer(event_item, op_flags=['writeonly'], flags=['c_index'])
        for row in it:
            row[...] = (eval('it.index' + formula), eval('it.index' + formula))
        with open(itemsamps_file, 'wb') as f:
            f.write(np.int32(0).tobytes())   # Empty header
            f.write(write_buffer[:])

    def write_gul_files(self, n_pairs, it, gul_file):
        """
        Class method to write temporary Ground Up Loss (GUL) input and Post Loss
        Amplification (PLA) output files. These files are used in the tests.

        Args:
            n_pairs (int): number of 8-bit tuples that should form write buffer
            it (numpy.nditer): iterator over array of losses
            gul_file (tempfile): temporary file object to be written
        """
        # Write file headers
        gul_file.write(np.int32(2 << 24).tobytes())   # FM stream type
        gul_file.write(np.int32(2).tobytes())   # number of samples

        # Write data
        write_buffer = memoryview(bytearray(n_pairs * DATA_SIZE))
        event_item = np.ndarray(
            n_pairs, buffer=write_buffer, dtype=event_item_dtype
        )
        sidx_loss = np.ndarray(
            n_pairs, buffer=write_buffer, dtype=sidx_loss_dtype
        )
        current_event_id = 0
        current_item_id = 0
        max_sidx = 3
        cursor = 0
        for entry in it:
            if current_event_id != it.multi_index[0] + 1 or current_item_id != it.multi_index[1] + 1:
                event_item[cursor]['event_id'] = it.multi_index[0] + 1
                current_event_id = it.multi_index[0] + 1
                event_item[cursor]['item_id'] = it.multi_index[1] + 1
                current_item_id = it.multi_index[1] + 1
                cursor += 1
            sidx_loss[cursor]['sidx'] = (it.multi_index[2] + 1) % max_sidx
            sidx_loss[cursor]['loss'] = entry
            cursor += 1
        gul_file.write(write_buffer[:])

    def setUp(self):
        """
        Set up and write items amplifications, loss factors, input Ground Up
        Loss (GUL) and expected Post Loss Amplification (PLA) files for tests.
        """

        n_events = 2
        n_items = 2

        # Write items amplfications file
        self.input_dir = Path('./input')
        self.input_dir.mkdir()
        itemsamps_file = os.path.join(
            self.input_dir, ITEMS_AMPLIFICATIONS_FILE_NAME
        )
        self.write_items_amplifications_file(
            n_items * DATA_SIZE, itemsamps_file, formula='+ 1'
        )

        # Write loss factors file
        self.static_dir = Path('./static')
        self.static_dir.mkdir()
        lossfactors_file = os.path.join(self.static_dir, LOSS_FACTORS_FILE_NAME)
        n_amplifications = 2
        factors = np.array([[1.1, 1.2], [1.0, 0.8]])
        n_pairs = n_events + sum(len(event) for event in factors)
        write_buffer = memoryview(bytearray(n_pairs * DATA_SIZE))
        event_count = np.ndarray(
            n_pairs, buffer=write_buffer, dtype=event_count_dtype
        )
        amp_factor = np.ndarray(
            n_pairs, buffer=write_buffer, dtype=amp_factor_dtype
        )
        it = np.nditer(factors, op_flags=['readonly'], flags=['multi_index'])
        current_event_id = 0
        cursor = 0
        for entry in it:
            if current_event_id != it.multi_index[0] + 1:
                event_count[cursor]['event_id'] = it.multi_index[0] + 1
                event_count[cursor]['count'] = n_amplifications
                current_event_id = it.multi_index[0] + 1
                cursor += 1
            amp_factor[cursor]['amplification_id'] = it.multi_index[1] + 1
            amp_factor[cursor]['factor'] = entry
            cursor += 1
        with open(lossfactors_file, 'wb') as f:
            f.write(np.int32(0).tobytes())   # Empty header
            f.write(write_buffer[:])

        # Write input GUL file
        self.gul_in = NamedTemporaryFile(prefix='gul')
        losses = np.array([
            [[100., 50., 0.0], [10., 30., 0.0]],
            [[5., 100., 0.0], [30., 50., 0.0]]
        ])
        n_pairs = n_events * n_items + sum(
            sum(len(sample) for sample in items) for items in losses
        )
        it = np.nditer(losses, op_flags=['readonly'], flags=['multi_index'])
        self.write_gul_files(n_pairs, it, self.gul_in)
        self.gul_in.seek(0)

        # Write expected (i.e. control) output PLA file
        self.ctrl_pla_out = NamedTemporaryFile(prefix='cpla')
        pla_losses = np.reshape(factors, (2, 2, 1)) * losses
        it = np.nditer(
            pla_losses, op_flags=['readonly'], flags=['multi_index']
        )
        self.write_gul_files(n_pairs, it, self.ctrl_pla_out)
        self.ctrl_pla_out.seek(0)

    def tearDown(self):
        """
        Delete items amplifications, loss factors, input Ground Up Loss (GUL)
        and expected Post Loss Amplification (PLA) files.
        """
        for f in self.input_dir.iterdir():
            f.unlink()
        self.input_dir.rmdir()

        for f in self.static_dir.iterdir():
            f.unlink()
        self.static_dir.rmdir()

        self.gul_in.close()
        self.ctrl_pla_out.close()

    def test_run_plapy(self):
        """
        Test plapy functionality.
        """
        pla_out = NamedTemporaryFile(prefix='pla')
        run(
            run_dir='.', file_in=self.gul_in.name, file_out=pla_out.name,
            input_path='input', static_path='static'
        )

        self.assertTrue(filecmp.cmp(
            self.ctrl_pla_out.name, pla_out.name, shallow=False
        ))

        pla_out.close()

    def test_structure__get_items_amplifications__first_item_id_not_1(self):
        """
        Test pla.structure.get_items_amplifications() raises SystemExit if the
        first item ID is not 1.
        """
        # Write items amplifications file with first item ID = 2
        itemsamps_file = os.path.join('.', ITEMS_AMPLIFICATIONS_FILE_NAME)
        self.write_items_amplifications_file(2, itemsamps_file, formula='+ 2')

        with pytest.raises(SystemExit) as e:
            get_items_amplifications('.')
        os.remove(itemsamps_file)
        assert e.type == SystemExit
        assert e.value.code == 1

    def test_structure__get_items_amplfications__non_contiguous_item_ids(self):
        """
        Test pla.structure.get_items_amplifications() raises SystemExit if the
        item IDs are not contiguous.
        """
        # Write items amplfications file where difference between item IDs is
        # not 1
        itemsamps_file = os.path.join('.', ITEMS_AMPLIFICATIONS_FILE_NAME)
        self.write_items_amplifications_file(
            4, itemsamps_file, formula='* 2 + 1'
        )

        with pytest.raises(SystemExit) as e:
            get_items_amplifications('.')
        os.remove(itemsamps_file)
        assert e.type == SystemExit
        assert e.value.code == 1
