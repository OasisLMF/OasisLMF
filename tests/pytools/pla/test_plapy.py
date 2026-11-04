import filecmp
from mock import patch
import numpy as np
import os
import pandas as pd
import pytest
from pathlib import Path
from oasislmf.pytools.common.input_files import AMPLIFICATIONS_FILE
from tempfile import NamedTemporaryFile
from unittest import TestCase

from oasislmf.pytools.pla.common import (
    DATA_SIZE,
    event_count_dtype,
    amp_factor_dtype,
    PLAFACTORS_FILE
)
from oasislmf.pytools.common.event_stream import (stream_info_to_bytes, FM_STREAM_ID, ITEM_STREAM,
                                                  mv_write_item_header, mv_write_sidx_loss,
                                                  CHANCE_OF_LOSS_IDX, decode_building, decode_local_sidx)

from oasislmf.pytools.pla.manager import run
from tests.pytools.utils import packed_gul_stream, read_gul_stream


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
        gul_file.write(stream_info_to_bytes(FM_STREAM_ID, ITEM_STREAM))   # FM stream type
        gul_file.write(np.int32(2).tobytes())   # number of samples

        # Write data
        write_buffer = memoryview(bytearray(n_pairs * DATA_SIZE))
        write_byte_mv = np.frombuffer(buffer=write_buffer, dtype='b')

        current_event_id = 0
        current_item_id = 0
        max_sidx = 3
        cursor = 0
        for entry in it:
            if current_event_id != it.multi_index[0] + 1 or current_item_id != it.multi_index[1] + 1:
                current_event_id = it.multi_index[0] + 1
                current_item_id = it.multi_index[1] + 1
                cursor = mv_write_item_header(write_byte_mv, cursor, current_event_id, current_item_id)
            cursor = mv_write_sidx_loss(write_byte_mv, cursor, (it.multi_index[2] + 1) % max_sidx, entry)
        gul_file.write(write_buffer[:cursor])

    def setUp(self):
        """
        Set up and write items amplifications, loss factors, input Ground Up
        Loss (GUL) and expected Post Loss Amplification (PLA) files for tests.
        """

        n_events = 2
        n_items = 2

        # Write items amplfications file
        self.input_dir = Path('./input')
        self.input_dir.mkdir(exist_ok=True)
        itemsamps_file = os.path.join(
            self.input_dir, AMPLIFICATIONS_FILE
        )
        self.write_items_amplifications_file(
            n_items * DATA_SIZE, itemsamps_file, formula='+ 1'
        )

        # Write loss factors file
        self.static_dir = Path('./static')
        self.static_dir.mkdir(exist_ok=True)
        lossfactors_file = os.path.join(self.static_dir, PLAFACTORS_FILE)
        n_amplifications = 2
        factors = np.array([[1.125, 1.25], [1.0, 0.75]])
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

        # Write expected output PLA file with secondary factor
        self.second_pla_out = NamedTemporaryFile(prefix='second')
        self.second_factor = 2   # Secondary PLA factor
        second_factors = 1 + (factors - 1) * self.second_factor
        second_losses = np.reshape(second_factors, (2, 2, 1)) * losses
        it = np.nditer(
            second_losses, op_flags=['readonly'], flags=['multi_index']
        )
        self.write_gul_files(n_pairs, it, self.second_pla_out)
        self.second_pla_out.seek(0)

        # Write expected output PLA file with large secondary factor
        # Negative losses should not be possible
        self.large_second_pla_out = NamedTemporaryFile(prefix='largesecond')
        self.large_second_factor = 6
        large_second_factors = np.clip(
            1 + (factors - 1) * self.large_second_factor, 0.0, None
        )
        large_second_losses = np.reshape(
            large_second_factors, (2, 2, 1)
        ) * losses
        it = np.nditer(
            large_second_losses, op_flags=['readonly'], flags=['multi_index']
        )
        self.write_gul_files(n_pairs, it, self.large_second_pla_out)
        self.large_second_pla_out.seek(0)

        # Write expected output PLA file with uniform factor
        self.uni_pla_out = NamedTemporaryFile(prefix='uni')
        self.uni_factor = 1.25   # Uniform PLA factor
        uni_losses = losses * self.uni_factor
        it = np.nditer(uni_losses, op_flags=['readonly'], flags=['multi_index'])
        self.write_gul_files(n_pairs, it, self.uni_pla_out)
        self.uni_pla_out.seek(0)

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
        self.second_pla_out.close()
        self.large_second_pla_out.close()
        self.uni_pla_out.close()

    def test_run_plapy(self):
        """
        Test plapy functionality.
        """
        pla_out = NamedTemporaryFile(prefix='pla')
        run(
            run_dir='.', file_in=self.gul_in.name, file_out=pla_out.name,
            input_path='input', static_path='static', secondary_factor=1,
            uniform_factor=0
        )

        self.assertTrue(filecmp.cmp(
            self.ctrl_pla_out.name, pla_out.name, shallow=False
        ))

        pla_out.close()

    def test_run_plapy__secondary_factor_provided(self):
        """
        Test plapy functionality if secondary factor provided.
        """
        second_out = NamedTemporaryFile(prefix='plasecond')
        run(
            run_dir='.', file_in=self.gul_in.name, file_out=second_out.name,
            input_path='input', static_path='static',
            secondary_factor=self.second_factor, uniform_factor=0
        )

        self.assertTrue(filecmp.cmp(
            self.second_pla_out.name, second_out.name, shallow=False
        ))

        second_out.close()

    def test_run_plapy__no_negative_losses(self):
        """
        Test plapy functionality if secondary factor is very large which should
        not lead to negative losses in the case of post loss reductions.
        """
        large_second_out = NamedTemporaryFile(prefix='plalargesecond')
        run(
            run_dir='.', file_in=self.gul_in.name,
            file_out=large_second_out.name, input_path='input',
            static_path='static', secondary_factor=self.large_second_factor,
            uniform_factor=0
        )

        self.assertTrue(filecmp.cmp(
            self.large_second_pla_out.name, large_second_out.name, shallow=False
        ))

        large_second_out.close()

    def test_run_plapy__uniform_factor_provided(self):
        """
        Test plapy functionality if uniform factor provided.
        """
        uni_out = NamedTemporaryFile(prefix='plauni')
        run(
            run_dir='.', file_in=self.gul_in.name, file_out=uni_out.name,
            input_path='input', static_path='static', secondary_factor=1,
            uniform_factor=self.uni_factor
        )

        self.assertTrue(filecmp.cmp(
            self.uni_pla_out.name, uni_out.name, shallow=False
        ))

        uni_out.close()

    def test_run_plapy__secondary_and_uniform_factors_provided(self):
        """
        Test plapy functionality if secondary and uniform factors are provided.
        """
        second_uni_out = NamedTemporaryFile(prefix='plaseconduni')
        run(
            run_dir='.', file_in=self.gul_in.name, file_out=second_uni_out.name,
            input_path='input_nowhere', static_path='static_nowhere',
            secondary_factor=self.second_factor, uniform_factor=self.uni_factor
        )

        # Secondary factor should be ignored
        self.assertTrue(filecmp.cmp(
            self.uni_pla_out.name, second_uni_out.name, shallow=False
        ))

        second_uni_out.close()


def _write_factors_by_event(run_dir, item_ids, event_ids):
    """Every item on amplification 1, at a factor that differs by event: 1 + event_id / 100."""
    with open(run_dir / 'input' / AMPLIFICATIONS_FILE, 'wb') as f:
        f.write(np.int32(0).tobytes())
        f.write(np.array([(i, 1) for i in item_ids], dtype=[('item_id', 'i4'), ('amplification_id', 'i4')]).tobytes())
    with open(run_dir / 'static' / PLAFACTORS_FILE, 'wb') as f:
        f.write(np.int32(0).tobytes())
        for event_id in event_ids:
            f.write(np.array([(event_id, 1)], dtype=event_count_dtype).tobytes())
            f.write(np.array([(1, 1 + event_id / 100)], dtype=amp_factor_dtype).tobytes())


@pytest.mark.parametrize('uniform_factor', [0.0, 1.25], ids=['factors_by_event', 'uniform_factor'])
def test_a_packed_item_is_amplified_whole(tmp_path, uniform_factor):
    """A factor applies to an item and event, so every building of a packed item is scaled by it,
    and the stream keeps its sample indices. The chance of a loss is a probability and is never
    scaled: there is one per building, at -4, -9, -14 ..."""
    sample_size, n_buildings = 10, 3
    run_dir, gul = packed_gul_stream(tmp_path / 'model', n_building=-n_buildings, sample_size=sample_size)
    before = read_gul_stream(gul)[2]
    if not uniform_factor:
        _write_factors_by_event(run_dir, pd.read_csv(run_dir / 'input' / 'items.csv')['item_id'],
                                before['event_id'].unique())

    out = run_dir / 'pla.bin'
    run(run_dir=str(run_dir), file_in=str(gul), file_out=str(out), input_path='input', static_path='static',
        secondary_factor=1.0, uniform_factor=uniform_factor)
    after = read_gul_stream(out)[2]

    pd.testing.assert_frame_equal(after[['event_id', 'item_id', 'sidx']], before[['event_id', 'item_id', 'sidx']])
    chance_of_loss = [decode_local_sidx(s, sample_size) == CHANCE_OF_LOSS_IDX for s in before['sidx']]
    assert sum(chance_of_loss) > 0 and decode_building(before['sidx'].min(), sample_size) == n_buildings
    factor = np.where(chance_of_loss, 1.0, uniform_factor or 1 + before['event_id'] / 100)
    np.testing.assert_allclose(after['loss'], before['loss'] * factor, rtol=1e-5)
