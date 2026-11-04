"""Building packing in gulmc: what a packed run writes.

A coverage's signed building count (``coverage_buildings``) decides how its items are written:

* negative -- the buildings are kept separate, in one stream item, each building with its own
  block of specials and samples: ``sidx = (b - 1) * S + s`` for a sample, and
  ``s - (b - 1) * 5`` for a special;
* positive -- the buildings are summed at source into an ordinary item.

Every building draws its own samples, so one run serves both, and building 1 is the draw an
unpacked run makes.
"""
import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.common.data import oasis_float
from oasislmf.pytools.common.event_stream import (CHANCE_OF_LOSS_IDX, ITEM_PACKED_STREAM, ITEM_STREAM, MAX_LOSS_IDX,
                                                  MEAN_IDX, PIPE_CAPACITY, TIV_IDX, decode_building, decode_local_sidx)
from oasislmf.pytools.gul.common import NUM_IDX
from oasislmf.pytools.gulmc.manager import run as run_gulmc
from oasislmf.utils.exceptions import OasisException
from tests.pytools.utils import copy_test_model, read_gul_stream, set_coverage_buildings

SAMPLE_SIZE = 20


def _run(tmp_path, n_building=None, *, name='run', sample_size=SAMPLE_SIZE, alloc_rule=0, loss_threshold=0.,
         random_generator=2, debug=0, ignore_correlation=False, one_event=False):
    """Run gulmc on a copy of test_model_1 and return ``(agg_type, records)``.

    ``n_building`` is the signed count for every coverage, or a list of one per coverage; None
    writes no coverage_buildings file. The records carry the decoded building and local sidx.
    """
    run_dir = copy_test_model(tmp_path / name)
    if n_building is not None:
        set_coverage_buildings(run_dir / 'input', n_building)
    if one_event:
        events = np.fromfile(run_dir / 'input' / 'events.bin', dtype='i4')
        events[:1].tofile(run_dir / 'input' / 'events.bin')
    out = run_dir / 'out.bin'
    run_gulmc(run_dir=run_dir, ignore_file_type=set(), file_in=run_dir / 'input' / 'events.bin', file_out=out,
              sample_size=sample_size, loss_threshold=loss_threshold, alloc_rule=alloc_rule, debug=debug,
              random_generator=random_generator, ignore_correlation=ignore_correlation,
              effective_damageability=False)
    agg_type, _, df = read_gul_stream(out)
    df['building'] = [decode_building(s, sample_size) for s in df['sidx']]
    df['local_sidx'] = [decode_local_sidx(s, sample_size) for s in df['sidx']]
    return agg_type, df


def _coverages(tmp_path):
    """item_id -> coverage_id, and the per-building TIV of each coverage."""
    items = pd.read_csv(copy_test_model(tmp_path / 'coverages').joinpath('input', 'items.csv'))
    tiv = np.fromfile(tmp_path / 'coverages' / 'input' / 'coverages.bin', dtype=oasis_float)
    return items.set_index('item_id')['coverage_id'], pd.Series(tiv, index=np.arange(1, len(tiv) + 1))


def test_building_1_of_a_packed_run_is_the_unpacked_run(tmp_path):
    """Building 1 is the draw an unpacked run makes, record for record; building 2 draws its own."""
    _, unpacked = _run(tmp_path, name='unpacked')
    _, packed = _run(tmp_path, -2, name='packed')

    keys = ['event_id', 'item_id', 'local_sidx']
    building_1 = packed[packed['building'] == 1][keys + ['loss']].reset_index(drop=True)
    pd.testing.assert_frame_equal(building_1, unpacked[keys + ['loss']].reset_index(drop=True))

    building_2 = packed[packed['building'] == 2]
    assert len(building_2) == len(building_1), "building 2 is incomplete"
    samples = building_2[building_2['local_sidx'] > 0].merge(
        building_1[building_1['local_sidx'] > 0], on=keys, suffixes=('_2', '_1'))
    assert (samples['loss_2'] != samples['loss_1']).any(), "building 2 repeats building 1's samples"


@pytest.mark.parametrize('n_building', [-2, 2], ids=['separate', 'summed'])
@pytest.mark.parametrize('random_generator', [0, 1], ids=['mersenne', 'latin_hypercube'])
def test_packing_needs_random_generator_2(tmp_path, random_generator, n_building):
    """Only Philox can address a building's draws without materialising all N * S of them first."""
    with pytest.raises(OasisException, match='--random-generator=2'):
        _run(tmp_path, n_building, random_generator=random_generator)


def test_a_packed_item_whose_sidx_would_overflow_is_refused(tmp_path):
    """A packed sidx is written as an int32. Past it, it would wrap negative and every reader
    would take the sample for a special, so the run is refused before anything is drawn."""
    with pytest.raises(OasisException, match='IsAggregate'):
        _run(tmp_path, -30_000, sample_size=100_000)


@pytest.mark.parametrize('n_building, agg_type', [
    (1, ITEM_STREAM),
    (4, ITEM_STREAM),                    # summed at source: an ordinary stream
    (-2, ITEM_PACKED_STREAM),
    ([-2] + [1] * 9, ITEM_PACKED_STREAM),  # one separated coverage is enough
], ids=['one_building', 'summed', 'separate', 'one_separate'])
def test_the_header_says_whether_the_stream_is_packed(tmp_path, n_building, agg_type):
    """The record layout is the same either way; what a reader needs to know is that a sidx can
    run past the sample size."""
    assert _run(tmp_path, n_building)[0] == agg_type


@pytest.mark.parametrize('alloc_rule', [0, 1, 2])
def test_an_item_larger_than_the_output_buffer_comes_out_whole(tmp_path, alloc_rule):
    """The buffer grows to the largest item rather than splitting one, and every building's
    block is there once, in the ascending sidx order the stream requires."""
    sample_size = 64
    bytes_per_block = 8 + (sample_size + NUM_IDX + 1) * 8
    n_buildings = 2 * PIPE_CAPACITY // bytes_per_block + 64
    _, df = _run(tmp_path, -n_buildings, sample_size=sample_size, alloc_rule=alloc_rule, one_event=True)

    assert (df.groupby(['event_id', 'item_id'])['header'].nunique() == 1).all(), \
        "an item was written in more than one piece"
    for _, item in df.groupby(['event_id', 'item_id']):
        assert item['sidx'].is_monotonic_increasing and item['sidx'].is_unique
        assert sorted(item.loc[item['sidx'] > 0, 'sidx']) == list(range(1, n_buildings * sample_size + 1))
        assert (item['sidx'] < 0).sum() == n_buildings * NUM_IDX


def test_alloc_rule_caps_each_building_at_its_share_of_the_tiv(tmp_path):
    """coverages.bin holds one building's share of the TIV, and the items of a coverage are capped
    building by building, so one building's losses cannot use another's insured value."""
    coverage_of, tiv = _coverages(tmp_path)

    def per_building_totals(alloc_rule):
        _, df = _run(tmp_path, -2, name=f'a{alloc_rule}', alloc_rule=alloc_rule)
        df = df[df['local_sidx'] > 0]
        df['coverage_id'] = df['item_id'].map(coverage_of)
        totals = df.groupby(['event_id', 'coverage_id', 'building', 'local_sidx'])['loss'].sum().reset_index()
        return totals, totals['coverage_id'].map(tiv)

    uncapped, cap = per_building_totals(0)
    assert (uncapped['loss'] > cap * 1.001).any(), "no building's losses exceed its TIV, so nothing is tested"
    capped, cap = per_building_totals(1)
    assert (capped['loss'] <= cap * (1 + 1e-5)).all()


@pytest.mark.parametrize('alloc_rule', [0, 1, 2])
def test_a_summed_item_is_the_sum_of_its_buildings(tmp_path, alloc_rule):
    """Summed at source or kept separate, the buildings are the same draws, capped the same way."""
    _, separate = _run(tmp_path, -3, name='separate', alloc_rule=alloc_rule)
    _, summed = _run(tmp_path, 3, name='summed', alloc_rule=alloc_rule)

    keys = ['event_id', 'item_id', 'local_sidx']
    expected = separate[separate['local_sidx'] > 0].groupby(keys)['loss'].sum()
    got = summed[summed['local_sidx'] > 0].set_index(keys)['loss']
    assert summed['building'].eq(1).all(), "a summed item carried a packed sidx"
    pd.testing.assert_series_equal(got.sort_index(), expected.sort_index(), check_dtype=False, rtol=1e-5)


def test_the_loss_threshold_drops_samples_but_keeps_every_buildings_specials(tmp_path):
    _, df = _run(tmp_path, -3, loss_threshold=1e12)
    assert not df.empty
    assert (df['sidx'] < 0).all()
    assert (df.groupby(['event_id', 'item_id']).size() == 3 * NUM_IDX).all()


def test_each_building_draws_a_latin_hypercube_of_its_own(tmp_path):
    """Not a slice of one larger sample, which would leave each building a random subset of the
    strata: each building's S draws fall one in each of S strata, and no two buildings share them.
    ``-d 2`` writes the damage draws."""
    _, df = _run(tmp_path, -3, debug=2, ignore_correlation=True)
    draws = df[df['local_sidx'] > 0]
    assert draws['building'].nunique() == 3
    for _, block in draws.groupby(['event_id', 'item_id', 'building']):
        strata = np.sort(np.floor(block['loss'].to_numpy() * SAMPLE_SIZE).astype(int))
        np.testing.assert_array_equal(strata, np.arange(SAMPLE_SIZE))
    # and they are not one Latin Hypercube repeated
    by_building = draws.pivot_table(index=['event_id', 'item_id', 'local_sidx'], columns='building', values='loss')
    assert (by_building[1] != by_building[2]).all() and (by_building[2] != by_building[3]).all()


def test_a_summed_items_specials_describe_the_sum(tmp_path):
    """Mean, TIV and maximum loss add up over the buildings; the chance of a loss is a property
    of the risk and is taken once. The standard deviation, which depends on how the buildings
    correlate, is tested in test_summed_specials."""
    _, separate = _run(tmp_path, -3, name='separate')
    _, summed = _run(tmp_path, 3, name='summed')

    keys = ['event_id', 'item_id', 'local_sidx']
    building_1 = separate[(separate['building'] == 1) & (separate['sidx'] < 0)].set_index(keys)['loss']
    got = summed[summed['sidx'] < 0].set_index(keys)['loss']
    for special, factor in ((MEAN_IDX, 3), (TIV_IDX, 3), (MAX_LOSS_IDX, 3), (CHANCE_OF_LOSS_IDX, 1)):
        one = building_1.xs(special, level='local_sidx')
        np.testing.assert_allclose(got.xs(special, level='local_sidx').loc[one.index], factor * one, rtol=1e-5,
                                   err_msg=f'special sidx {special}')
