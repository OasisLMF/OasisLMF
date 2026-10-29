"""Building packing in gulpy, run as a model runs it: ``evepy | modelpy | gulpy``.

gulpy writes the same packed stream as gulmc, so what that stream holds is tested once, in
tests/pytools/gulmc/test_building_packing.py. What is here is what gulpy does on its own: its
refusals, its structure cache, and the inputs it reads differently.
"""
import subprocess

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.common.data import correlations_dtype, items_dtype
from oasislmf.pytools.common.event_stream import decode_building, decode_local_sidx
from tests.pytools.utils import copy_test_model, read_gul_stream, set_coverage_buildings

SAMPLE_SIZE = 10


def _gulpy(run_dir, *flags, sample_size=SAMPLE_SIZE, check=True):
    """Run gulpy behind evepy and modelpy in ``run_dir`` and return the finished process."""
    return subprocess.run(
        f'evepy 1 1 | modelpy | gulpy -a0 -S{sample_size} -L0 {" ".join(flags)} > out.bin',
        cwd=run_dir, shell=True, capture_output=True, text=True, check=check)


def _losses(run_dir, *flags, sample_size=SAMPLE_SIZE):
    _gulpy(run_dir, *flags, sample_size=sample_size)
    df = read_gul_stream(run_dir / 'out.bin')[2]
    df['building'] = [decode_building(s, sample_size) for s in df['sidx']]
    df['local_sidx'] = [decode_local_sidx(s, sample_size) for s in df['sidx']]
    return df


def _model(tmp_path, name, n_building=None):
    run_dir = copy_test_model(tmp_path / name)
    if n_building is not None:
        set_coverage_buildings(run_dir / 'input', n_building)
    return run_dir


def test_building_1_of_a_packed_run_is_the_unpacked_run(tmp_path):
    """Building 1 is the draw an unpacked run makes, record for record; building 2 draws its own."""
    unpacked = _losses(_model(tmp_path, 'unpacked'), '--random-generator=2')
    packed = _losses(_model(tmp_path, 'packed', -2), '--random-generator=2')

    keys = ['event_id', 'item_id', 'local_sidx', 'loss']
    building_1 = packed[packed['building'] == 1]
    pd.testing.assert_frame_equal(building_1[keys].reset_index(drop=True), unpacked[keys].reset_index(drop=True))

    building_2 = packed[packed['building'] == 2]
    assert len(building_2) == len(unpacked), "building 2 is incomplete"
    samples = building_2[building_2['local_sidx'] > 0].merge(
        building_1[building_1['local_sidx'] > 0], on=['event_id', 'item_id', 'local_sidx'], suffixes=('_2', '_1'))
    assert (samples['loss_2'] != samples['loss_1']).any(), "building 2 repeats building 1's samples"


@pytest.mark.parametrize('flags, sample_size, message', [
    (['--random-generator=1'], SAMPLE_SIZE, '--random-generator=2'),
    (['--random-generator=2'], 100_000, 'IsAggregate'),     # 30,000 buildings past the int32 sidx
], ids=['generator', 'sidx_overflow'])
def test_gulpy_refuses_what_it_cannot_write(tmp_path, flags, sample_size, message):
    """The same refusals as gulmc, made by gulpy itself."""
    n_building = -2 if message.startswith('--') else -30_000
    proc = _gulpy(_model(tmp_path, 'packed', n_building), *flags, sample_size=sample_size, check=False)
    assert proc.returncode != 0
    assert message in proc.stderr


def test_sparse_item_ids_read_their_own_correlation(tmp_path):
    """Correlations are looked up by item_id, so renumbering the items changes nothing.

    They were read by position, which lines up only for a dense 1..N: with gaps, an item took
    another's correlation, and an id past the end read out of bounds.
    """
    dense = _losses(_model(tmp_path, 'dense'), '--random-generator=2')

    run_dir = _model(tmp_path, 'sparse')
    renumber = 3     # item i becomes 3i
    items = np.fromfile(run_dir / 'input' / 'items.bin', dtype=items_dtype)
    items['item_id'] *= renumber
    items.tofile(run_dir / 'input' / 'items.bin')
    pd.DataFrame({k: items[k] for k in items_dtype.names}).to_csv(run_dir / 'input' / 'items.csv', index=False)
    corr = np.fromfile(run_dir / 'input' / 'correlations.bin', dtype=correlations_dtype)
    corr['item_id'] *= renumber
    corr.tofile(run_dir / 'input' / 'correlations.bin')
    pd.DataFrame({k: corr[k] for k in correlations_dtype.names}).to_csv(run_dir / 'input' / 'correlations.csv',
                                                                        index=False)
    sparse = _losses(run_dir, '--random-generator=2')
    sparse['item_id'] //= renumber

    pd.testing.assert_frame_equal(sparse, dense)


def test_a_structure_cache_gives_the_same_losses(tmp_path):
    """``gulpy --create-structures`` builds the cache every parallel gulpy then loads, and a cache
    left half written by an interrupted build is ignored rather than failing the run."""
    run_dir = _model(tmp_path, 'packed', -2)
    without = _losses(run_dir, '--random-generator=2')

    subprocess.run(['gulpy', '--create-structures', '--run-dir', str(run_dir)], check=True, capture_output=True)
    assert (run_dir / 'input' / 'gulpy_structure').is_dir()
    pd.testing.assert_frame_equal(_losses(run_dir, '--random-generator=2'), without)

    with open(run_dir / 'input' / 'gulpy_structure' / 'metadata.npy', 'r+b') as f:
        f.truncate(8)
    pd.testing.assert_frame_equal(_losses(run_dir, '--random-generator=2'), without)


def test_debug_writes_every_buildings_draw_of_a_summed_item(tmp_path):
    """Under -d gulpy writes the random values it drew, one per building. Summing a summed
    item's buildings would report a value in [0, N]; debug keeps them separate instead."""
    n_buildings = 3
    draws = _losses(_model(tmp_path, 'summed', n_buildings), '-d', '--random-generator=2')
    draws = draws[draws['sidx'] > 0]

    assert not draws.empty
    assert draws['loss'].between(0, 1).all(), f"debug values up to {draws['loss'].max():.3f}"
    per_item = draws.groupby(['event_id', 'item_id']).size()
    assert (per_item == n_buildings * SAMPLE_SIZE).all(), "a building's draws are missing"
