"""PlaStage against plapy: the same bytes for the same stream."""
import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.pla.manager import run
from oasislmf.pytools.pla.stage import PlaStage
from tests.pytools.utils import packed_gul_stream, read_gul_stream

from .test_plapy import _write_factors_by_event

HEADER_SIZE = 8


@pytest.mark.parametrize('n_building', [1, -3], ids=['one_building', 'packed'])
@pytest.mark.parametrize('options', [{}, {'secondary_factor': 0.5}, {'uniform_factor': 1.25}],
                         ids=['factors', 'secondary_factor', 'uniform_factor'])
def test_pla_stage_writes_the_plapy_stream(tmp_path, n_building, options):
    run_dir, gul = packed_gul_stream(tmp_path / 'model', n_building=n_building, sample_size=10)
    events = read_gul_stream(gul)[2]['event_id'].unique()
    _write_factors_by_event(run_dir, pd.read_csv(run_dir / 'input' / 'items.csv')['item_id'], events)
    expected = run_dir / 'plapy.bin'
    run(run_dir=str(run_dir), file_in=str(gul), file_out=str(expected), input_path='input', static_path='static',
        **{'secondary_factor': 1.0, 'uniform_factor': 0.0, **options})

    data = np.fromfile(gul, dtype=np.int8)
    stage = PlaStage(str(run_dir), data[:HEADER_SIZE].tobytes(), **options)
    block = data[HEADER_SIZE:].copy()
    stage.amplify(block)
    assert stage.header + block.tobytes() == expected.read_bytes()
    assert not np.array_equal(block, data[HEADER_SIZE:])


def test_pla_stage_leaves_an_empty_block(tmp_path):
    run_dir, gul = packed_gul_stream(tmp_path / 'model', n_building=1, sample_size=2)
    stage = PlaStage(str(run_dir), b'\0' * HEADER_SIZE, uniform_factor=2.0)
    block = np.empty(0, dtype=np.int8)
    stage.amplify(block)
    assert block.shape == (0,)
