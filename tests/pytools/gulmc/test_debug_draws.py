"""Under ``-d 1|2`` gulmc writes the random values it drew, not losses.

Each building draws its own, so an item whose buildings are summed at source would add N
uniforms together and report a value anywhere in [0, N]. Debug output keeps every building
separate instead: one value per draw, each in [0, 1].
"""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.gulmc.manager import run as run_gulmc
from tests.pytools.utils import set_coverage_buildings

SRC_MODEL = Path(__file__).parents[2].joinpath("assets", "test_model_1")
SAMPLE_SIZE = 10
N_BUILDINGS = 3


@pytest.mark.parametrize("debug", [1, 2], ids=["hazard draws", "damage draws"])
def test_debug_writes_every_buildings_draw_of_a_summed_item(tmp_path, debug):
    run_dir = tmp_path / 'assets'
    shutil.copytree(SRC_MODEL, run_dir)
    shutil.rmtree(run_dir / 'input' / 'gulmc_structure', ignore_errors=True)
    # positive: summed at source
    set_coverage_buildings(run_dir / 'input', N_BUILDINGS)

    run_gulmc(run_dir=run_dir, ignore_file_type=set(),
              file_in=run_dir / 'input' / 'events.bin', file_out=run_dir / 'o.bin',
              sample_size=SAMPLE_SIZE, loss_threshold=0., alloc_rule=0, debug=debug,
              random_generator=2, ignore_correlation=False, effective_damageability=False)
    bintocsv(run_dir / 'o.bin', run_dir / 'o.csv', 'gul')
    draws = pd.read_csv(run_dir / 'o.csv').query('sidx > 0')

    assert not draws.empty
    assert draws['loss'].between(0, 1).all(), f"debug values up to {draws['loss'].max():.3f}"
    per_item = draws.groupby(['event_id', 'item_id']).size()
    assert (per_item == N_BUILDINGS * SAMPLE_SIZE).all(), "a building's draws are missing"
