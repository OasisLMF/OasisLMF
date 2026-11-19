"""Blended areaperils, end to end through gulmc's and gulpy's run entry points.

Every model here is test_model_1 with its items rewritten: one item per (location, building,
coverage), all on vulnerability 8, and the locations named in gul_summary_map. A blended item is
checked against the same item placed directly on each of its positions: those runs draw the same
random numbers, so a blended sample must equal, exactly, the loss of the position it was placed on.

The two cells are both in test_model_1's footprint and give different losses under vulnerability 8:
``A`` (154) has hazard in every event, ``B`` (101) has none in event 1. ``make_dynamic`` puts them in
two sections of a dynamic footprint, with return periods either side of the items' protection.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.common.blended_areaperil import read_csv, validate
from oasislmf.pytools.common.event_stream import MEAN_IDX
from oasislmf.pytools.getmodel.common import FootprintHeader
from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.gulmc.manager import run as run_gulmc
from oasislmf.pytools.gulmc.structure import create_gulmc_structure
from oasislmf.utils.exceptions import OasisException
from tests.pytools.utils import copy_test_model, read_gul_stream, write_correlations

A, B = 154, 101
BLENDED, BLENDED_2 = 9001, 9002
VULNERABILITY = 8
SAMPLES = 100

THREE_LOCATIONS = [  # item_id, coverage_id, areaperil_id, location, building_id: one building each
    (1, 1, BLENDED, 1, 1),
    (2, 2, BLENDED, 2, 1),
    (3, 3, BLENDED, 3, 1),
]


def blend_rows(positions, blended_id=BLENDED, scheme=1, placement_ids=None):
    """Blend table rows for ``positions``, a list of (weight, areaperil_id); weight None for equal weights.

    The placements are numbered 0, 1, 2... unless ``placement_ids`` gives their ids.
    """
    rows = pd.DataFrame({
        'blended_areaperil_id': blended_id,
        'placement_scheme_id': scheme,
        'placement_id': range(len(positions)) if placement_ids is None else placement_ids,
        'weight': [w for w, _ in positions],
        'areaperil_id': [ap for _, ap in positions],
    })
    return rows.drop(columns='weight') if rows['weight'].isna().all() else rows


def write_model(dst, items, blend=None, dependent_to_source=None, location_ids=None):
    """test_model_1 at ``dst`` with ``items`` and, if given, the blend table ``blend``.

    ``dependent_to_source`` maps a dependent item_id to its source item_id. ``location_ids`` maps a
    location to its LocNumber, which by default is the location's number.
    """
    run_dir = copy_test_model(dst)
    for name in ('items.bin', 'correlations.bin', 'correlations.csv'):
        (run_dir / 'input' / name).unlink(missing_ok=True)
    items = pd.DataFrame(items, columns=['item_id', 'coverage_id', 'areaperil_id', 'location', 'building_id'])
    items['vulnerability_id'] = VULNERABILITY
    items['group_id'] = items['location'] * 1000 + items['building_id']
    items[['item_id', 'coverage_id', 'areaperil_id', 'vulnerability_id', 'group_id']].to_csv(
        run_dir / 'input' / 'items.csv', index=False)
    location_ids = location_ids or {}
    pd.DataFrame({
        'item_id': items['item_id'],
        'PortNumber': '1',
        'AccNumber': 'ACC1',
        'LocNumber': [str(location_ids.get(loc, loc)) for loc in items['location']],
        'building_id': items['building_id'],
    }).to_csv(run_dir / 'input' / 'gul_summary_map.csv', index=False)
    dependent_to_source = dependent_to_source or {}
    write_correlations(run_dir / 'input', items['item_id'].to_numpy(), 0.,
                       source_item_id=[dependent_to_source.get(i, 0) for i in items['item_id']])
    if blend is not None:
        pd.concat(blend if isinstance(blend, list) else [blend]).to_csv(run_dir / 'static' / 'blended_areaperil_id.csv', index=False)
    return run_dir


def placed_on(items, areaperil_id):
    """The items with each blended one placed directly on ``areaperil_id``."""
    return [(i, c, areaperil_id if ap in (BLENDED, BLENDED_2) else ap, loc, b) for i, c, ap, loc, b in items]


def gul_losses(run_dir, sample_size=SAMPLES, effective_damageability=False):
    """Run gulmc on ``run_dir`` and return its losses indexed by (event_id, item_id, sidx)."""
    out = run_dir / 'gul.bin'
    run_gulmc(run_dir=run_dir, ignore_file_type=set(), file_in=run_dir / 'input' / 'events.bin', file_out=out,
              sample_size=sample_size, loss_threshold=0., alloc_rule=0, debug=0, random_generator=2,
              ignore_correlation=False, effective_damageability=effective_damageability)
    return read_gul_stream(out)[2].set_index(['event_id', 'item_id', 'sidx'])['loss']


def sampled(losses):
    return losses[losses.index.get_level_values('sidx') > 0]


def means(losses):
    return losses.xs(MEAN_IDX, level='sidx')


def placements(blended, references):
    """Position of every (item, sample) of a blended run, read from the per-position runs ``references``.

    Asserts that each (item, sample) takes, in every event, the loss of one and the same position.
    Returns a frame indexed by (item_id, sidx): ``position`` (the first that fits) and ``known``,
    False where more than one position fits every event (they all lose nothing, say).
    """
    blended = sampled(blended)
    fits = pd.DataFrame({
        k: (blended == sampled(ref).reindex(blended.index, fill_value=0.)).groupby(level=['item_id', 'sidx']).all()
        for k, ref in enumerate(references)
    })
    unplaced = ~fits.any(axis=1)
    assert not unplaced.any(), f"(item, sample) with no single position behind their losses:\n{fits[unplaced].head()}"
    return pd.DataFrame({'position': fits.to_numpy().argmax(axis=1), 'known': fits.sum(axis=1).to_numpy() == 1},
                        index=fits.index)


# --------------------------------------------------------------------------------------------------
# gulmc
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('weight', [None, 7, 1.], ids=['equal', 'count', 'float'])
@pytest.mark.parametrize('sample_size', [0, 10])
@pytest.mark.parametrize('effective_damageability', [False, True], ids=['eff=False', 'eff=True'])
def test_a_single_position_blend_is_its_areaperil(tmp_path, weight, sample_size, effective_damageability):
    """A blended areaperil of one certain position is the base case: the whole stream is identical."""
    blended = write_model(tmp_path / 'blended', THREE_LOCATIONS, blend_rows([(weight, A)]))
    direct = write_model(tmp_path / 'direct', placed_on(THREE_LOCATIONS, A))
    gul_losses(blended, sample_size, effective_damageability)
    gul_losses(direct, sample_size, effective_damageability)
    assert (blended / 'gul.bin').read_bytes() == (direct / 'gul.bin').read_bytes()


def test_each_sample_takes_one_position_for_every_event(tmp_path):
    """Each building stands at one position per sample, the same for every event, chosen by weight."""
    blend = blend_rows([(3, A), (1, B)])
    losses = gul_losses(write_model(tmp_path / 'blended', THREE_LOCATIONS, blend))
    at = [gul_losses(write_model(tmp_path / f'at_{ap}', placed_on(THREE_LOCATIONS, ap))) for ap in (A, B)]

    placed = placements(losses, at)
    assert placed['known'].all()
    share_on_a = (placed['position'] == 0).groupby(level='item_id').mean()
    np.testing.assert_allclose(share_on_a, 0.75, atol=0.05)


def test_items_of_a_building_share_its_placement(tmp_path):
    """Coverages and perils of one building read the same position, each through its own areaperil.

    Item 2 is another peril of item 1's building: its blended areaperil names the same scheme, with
    the cells the other way round, so the position that puts item 1 on A puts item 2 on B.
    """
    items = [(1, 1, BLENDED, 1, 1), (2, 2, BLENDED_2, 1, 1), (3, 3, BLENDED, 1, 1)]
    blend = [blend_rows([(1, A), (1, B)]), blend_rows([(1, B), (1, A)], blended_id=BLENDED_2)]
    losses = gul_losses(write_model(tmp_path / 'blended', items, blend))
    at_a, at_b = (gul_losses(write_model(tmp_path / f'at_{ap}', placed_on(items, ap))) for ap in (A, B))

    def item_2_from(first, second):
        return pd.concat([first.drop(2, level='item_id'), second.xs(2, level='item_id', drop_level=False)])

    position = placements(losses, [item_2_from(at_a, at_b), item_2_from(at_b, at_a)])['position'].unstack('item_id')
    assert (position[1] == position[2]).all() and (position[1] == position[3]).all()
    assert 0 < (position[1] == 0).mean() < 1


def test_buildings_and_locations_are_placed_independently(tmp_path):
    """Two buildings of one location, and the same building number at another location, each get their own draw."""
    items = [(1, 1, BLENDED, 1, 1), (2, 2, BLENDED, 1, 2), (3, 3, BLENDED, 2, 1)]
    blend = blend_rows([(1, A), (1, B)])
    losses = gul_losses(write_model(tmp_path / 'blended', items, blend))
    at = [gul_losses(write_model(tmp_path / f'at_{ap}', placed_on(items, ap))) for ap in (A, B)]

    position = placements(losses, at)['position'].unstack('item_id')
    assert (position[1] != position[2]).any()
    assert (position[1] != position[3]).any()


def test_placement_follows_the_location_not_the_item(tmp_path):
    """Renumbering the items and adding a location leaves each location's placements, and so its losses, unchanged."""
    blend = blend_rows([(3, A), (1, B)])
    first = [(1, 1, BLENDED, 1, 1), (2, 2, BLENDED, 2, 1)]
    second = [(1, 3, BLENDED, 3, 1), (2, 2, BLENDED, 2, 1), (3, 1, BLENDED, 1, 1)]

    def by_location(items, name):
        losses = gul_losses(write_model(tmp_path / name, items, blend, location_ids={1: 'L-001', 2: 'L-002', 3: 'L-003'}))
        location = {i: loc for i, _, _, loc, _ in items}
        frame = losses.reset_index()
        frame['location'] = frame['item_id'].map(location)
        return frame.set_index(['location', 'event_id', 'sidx'])['loss'].sort_index()

    before, after = by_location(first, 'first'), by_location(second, 'second')
    pd.testing.assert_series_equal(before, after.loc[[1, 2]])


@pytest.mark.parametrize('positions, shares', [
    ([(None, A), (None, B)], [0.5, 0.5]),
    ([(3, A), (1, B)], [0.75, 0.25]),
    ([(0.6, A), (0.3, B)], [0.6, 0.3]),
    ([(0.6, A)], [0.6, 0.]),
    ([(3, A), (1, 0)], [0.75, 0.]),
], ids=['equal', 'count', 'float with shortfall', 'single float', 'count with not-at-risk position'])
@pytest.mark.parametrize('effective_damageability', [False, True], ids=['eff=False', 'eff=True'])
def test_analytic_mean_is_the_weighted_mixture_of_the_positions(tmp_path, positions, shares, effective_damageability):
    """The mean loss is each position's mean by its share; a shortfall or areaperil 0 adds nothing."""
    losses = gul_losses(write_model(tmp_path / 'blended', THREE_LOCATIONS, blend_rows(positions)),
                        effective_damageability=effective_damageability)
    at_a, at_b = (means(gul_losses(write_model(tmp_path / f'at_{ap}', placed_on(THREE_LOCATIONS, ap)),
                                   effective_damageability=effective_damageability)) for ap in (A, B))
    expected = shares[0] * at_a + shares[1] * at_b.reindex(at_a.index, fill_value=0.)
    np.testing.assert_allclose(means(losses).reindex(expected.index), expected, rtol=1e-5)


@pytest.mark.parametrize('positions, share', [([(0.6, A)], 0.6), ([(3, A), (1, 0)], 0.75)],
                         ids=['float shortfall', 'not-at-risk position'])
def test_a_building_placed_outside_the_hazard_loses_nothing(tmp_path, positions, share):
    """A sample placed in the float shortfall, or on areaperil 0, loses nothing; the others lose as on A."""
    losses = gul_losses(write_model(tmp_path / 'blended', THREE_LOCATIONS, blend_rows(positions)))
    at_a = gul_losses(write_model(tmp_path / 'at_a', placed_on(THREE_LOCATIONS, A)))
    nothing = sampled(at_a) * 0.

    placed = placements(losses, [at_a, nothing])
    on_a = placed.loc[placed['known'], 'position'] == 0
    np.testing.assert_allclose(on_a.mean(), share, atol=0.05)


def test_a_dependent_coverage_follows_its_source_to_each_position(tmp_path):
    """A dependent coverage is driven by its source at the position the building was placed on.

    The identity conditional vulnerability puts the dependent in its source's damage bin, so on every
    sample both losses are those of the run where the building stands at that position.
    """
    items = [(1, 1, BLENDED, 1, 1), (2, 2, BLENDED, 1, 1)]
    blend = blend_rows([(1, A), (1, B)])

    def run(name, items_, blend_=None):
        return gul_losses(with_conditional_dependency(write_model(tmp_path / name, items_, blend_, dependent_to_source={2: 1}), 2))

    losses = run('blended', items, blend)
    at = [run(f'at_{ap}', placed_on(items, ap)) for ap in (A, B)]
    position = placements(losses, at)['position'].unstack('item_id')
    assert (position[1] == position[2]).all()
    assert 0 < (position[1] == 0).mean() < 1


def make_dynamic(run_dir, protection):
    """Turn a ``write_model`` model into a dynamic footprint one, with every item protected to ``protection``.

    A and B are in different sections: every event reaches A at a return period of 20 and B at 200,
    so the protection, between the two, holds at A and not at B.
    """
    static = run_dir / 'static'
    num_intensity_bins = int(np.fromfile(static / 'footprint.bin', dtype=FootprintHeader, count=1)['num_intensity_bins'][0])
    for footprint_file in static.glob('footprint.*'):
        footprint_file.unlink()
    pd.DataFrame({'section_id': [1, 1, 2, 2], 'areaperil_id': [A, A, B, B], 'return_period': [10, 20, 100, 200],
                  'intensity': [40, 45, 50, 55]}).to_parquet(static / 'hazard_case.parquet', index=False)
    event_ids = pd.read_csv(run_dir / 'input' / 'events.csv')['event_id']
    pd.concat([pd.DataFrame({'event_id': event_ids, 'section_id': 1, 'rp_from': 10, 'rp_to': 20, 'interpolation': 1.}),
               pd.DataFrame({'event_id': event_ids, 'section_id': 2, 'rp_from': 100, 'rp_to': 200, 'interpolation': 1.})]
              ).to_parquet(static / 'event_definition.parquet', index=False)
    pd.DataFrame({'peril_id': 'WTC', 'intensity_bin': np.arange(num_intensity_bins + 1),
                  'intensity': np.arange(num_intensity_bins + 1)}).to_csv(static / 'intensity_bin_dict.csv', index=False)
    (static / 'footprint_parquet_meta.json').write_text(
        f'{{"num_intensity_bins": {num_intensity_bins}, "has_intensity_uncertainty": 0}}')
    pd.DataFrame({'section_id': [1, 2]}).to_csv(run_dir / 'input' / 'sections.csv', index=False)
    summary_map = pd.read_csv(run_dir / 'input' / 'gul_summary_map.csv', dtype=str)
    summary_map.assign(peril_id='WTC').to_csv(run_dir / 'input' / 'gul_summary_map.csv', index=False)
    pd.DataFrame({'item_id': pd.read_csv(run_dir / 'input' / 'items.csv')['item_id'], 'intensity_adjustment': 0,
                  'return_period': protection}).to_csv(run_dir / 'input' / 'item_adjustments.csv', index=False)
    return run_dir


def with_conditional_dependency(run_dir, dependent_item_id):
    """Give the dependent item an identity conditional vulnerability: it lands in its source's damage bin."""
    items = pd.read_csv(run_dir / 'input' / 'items.csv')
    items.loc[items['item_id'] == dependent_item_id, 'vulnerability_id'] = 108
    items.to_csv(run_dir / 'input' / 'items.csv', index=False)
    n_bins = len(pd.read_csv(run_dir / 'static' / 'damage_bin_dict.csv'))
    pd.DataFrame({'vulnerability_id': 108, 'source_damage_bin': range(1, n_bins + 1),
                  'damage_bin': range(1, n_bins + 1), 'probability': 1.}).to_csv(
        run_dir / 'static' / 'conditional_vulnerability.csv', index=False)
    return run_dir


def test_protection_holds_or_fails_at_each_position(tmp_path, monkeypatch):
    """With a dynamic footprint, standard of protection is judged at the position the building stands
    on, for a dependent coverage as for its source: protected at A, the building loses nothing there,
    and on B it loses as if it stood on B."""
    items = [(1, 1, BLENDED, 1, 1), (2, 2, BLENDED, 1, 1), (3, 3, BLENDED, 2, 1)]
    blend = blend_rows([(1, A), (1, B)])

    def run(name, items_, blend_=None):
        run_dir = write_model(tmp_path / name, items_, blend_, dependent_to_source={2: 1})
        make_dynamic(with_conditional_dependency(run_dir, 2), protection=50)
        monkeypatch.chdir(run_dir)
        out = run_dir / 'gul.bin'
        run_gulmc(run_dir=run_dir, ignore_file_type=set(), file_in=run_dir / 'input' / 'events.bin', file_out=out,
                  sample_size=SAMPLES, loss_threshold=0., alloc_rule=0, debug=0, random_generator=2, dynamic_footprint=True)
        return read_gul_stream(out)[2].set_index(['event_id', 'item_id', 'sidx'])['loss']

    losses = run('blended', items, blend)
    at_a, at_b = (run(f'at_{ap}', placed_on(items, ap)) for ap in (A, B))
    assert (sampled(at_a) == 0).all() and (sampled(at_b) > 0).any()

    placed = placements(losses, [at_a, at_b])
    position = placed['position'].unstack('item_id')
    assert (position[1] == position[2]).all()
    assert 0 < (position[1] == 1).mean() < 1


def test_a_dependency_entirely_outside_the_hazard_loses_nothing(tmp_path):
    """A source and dependent whose every position is not at risk run, and lose nothing, beside a location that does."""
    items = [(1, 1, BLENDED_2, 1, 1), (2, 2, BLENDED_2, 1, 1), (3, 3, BLENDED, 2, 1)]
    blend = [blend_rows([(1, A), (1, B)]), blend_rows([(1, 0)], blended_id=BLENDED_2, scheme=2)]
    run_dir = with_conditional_dependency(write_model(tmp_path / 'blended', items, blend, dependent_to_source={2: 1}), 2)
    losses = sampled(gul_losses(run_dir))
    assert (losses.drop(3, level='item_id') == 0).all()
    assert (losses.xs(3, level='item_id') > 0).any()


def test_binary_and_csv_tables_give_the_same_losses(tmp_path):
    blend = blend_rows([(3, A), (1, B)])
    from_csv = write_model(tmp_path / 'csv', THREE_LOCATIONS, blend)
    from_bin = write_model(tmp_path / 'bin', THREE_LOCATIONS, blend)
    static = from_bin / 'static'
    csvtobin(static / 'blended_areaperil_id.csv', static / 'blended_areaperil_id.bin', 'blended_areaperil')
    (static / 'blended_areaperil_id.csv').unlink()
    pd.testing.assert_series_equal(gul_losses(from_csv), gul_losses(from_bin))


# --------------------------------------------------------------------------------------------------
# gulmc structure: placements are drawn once, when the structure is built
# --------------------------------------------------------------------------------------------------

def test_placements_need_the_sample_size_when_the_structure_is_built(tmp_path):
    run_dir = write_model(tmp_path / 'blended', THREE_LOCATIONS, blend_rows([(3, A), (1, B)]))
    with pytest.raises(OasisException, match='sample size is needed'):
        create_gulmc_structure(run_dir, set(), [], False, 'oasis_data_manager.df_reader.reader.OasisPandasReader')


@pytest.mark.parametrize('built_for', [SAMPLES, SAMPLES + 7], ids=['same sample size', 'other sample size'])
def test_a_prebuilt_structure_gives_the_losses_of_a_fresh_build(tmp_path, built_for):
    """Placements read from a saved structure are those a run draws itself, and a run with another
    sample size than the structure was built for draws its own rather than read past them."""
    blend = blend_rows([(3, A), (1, B)])
    prebuilt = write_model(tmp_path / 'prebuilt', THREE_LOCATIONS, blend)
    create_gulmc_structure(prebuilt, set(), [], False, 'oasis_data_manager.df_reader.reader.OasisPandasReader',
                           sample_size=built_for)
    fresh = write_model(tmp_path / 'fresh', THREE_LOCATIONS, blend)
    pd.testing.assert_series_equal(gul_losses(prebuilt), gul_losses(fresh))


def test_structures_built_by_the_cli_give_the_losses_of_a_fresh_build(tmp_path):
    """``gulmc --create-structures``, as the run scripts call it, draws the placements for the run's sample size."""
    blend = blend_rows([(3, A), (1, B)])
    prebuilt = write_model(tmp_path / 'prebuilt', THREE_LOCATIONS, blend)
    subprocess.run(['gulmc', '--create-structures', f'-S{SAMPLES}', '--run-dir', str(prebuilt)], check=True)
    fresh = write_model(tmp_path / 'fresh', THREE_LOCATIONS, blend)
    pd.testing.assert_series_equal(gul_losses(prebuilt), gul_losses(fresh))


def test_csv_and_parquet_summary_maps_place_buildings_alike(tmp_path):
    """The location identifiers are text: LocNumber '007' read from a csv is not the number 7."""
    blend = blend_rows([(3, A), (1, B)])
    location_ids = {1: '007', 2: '7', 3: '0.10'}
    from_csv = write_model(tmp_path / 'csv', THREE_LOCATIONS, blend, location_ids=location_ids)
    from_parquet = write_model(tmp_path / 'parquet', THREE_LOCATIONS, blend, location_ids=location_ids)
    summary_map = from_parquet / 'input' / 'gul_summary_map.csv'
    pd.read_csv(summary_map, dtype={'PortNumber': str, 'AccNumber': str, 'LocNumber': str}).to_parquet(
        summary_map.with_suffix('.parquet'), index=False)
    summary_map.unlink()
    pd.testing.assert_series_equal(gul_losses(from_csv), gul_losses(from_parquet))


def test_placement_ids_are_identifiers_not_positions(tmp_path):
    """Any distinct placement ids place as 0, 1, 2... would, the positions taken in ascending id order."""
    numbered = write_model(tmp_path / 'numbered', THREE_LOCATIONS, blend_rows([(3, A), (1, B)]))
    any_ids = write_model(tmp_path / 'any_ids', THREE_LOCATIONS,
                          blend_rows([(1, B), (3, A)], placement_ids=[900311, 70412]))
    pd.testing.assert_series_equal(gul_losses(numbered), gul_losses(any_ids))


@pytest.mark.parametrize('second_weights', [(1, 1), (3, 1)], ids=['other weights', 'same weights'])
def test_a_location_must_place_all_its_items_by_one_scheme(tmp_path, second_weights):
    """Two perils of one location in different schemes are refused, even when the weights happen to match."""
    items = [(1, 1, BLENDED, 1, 1), (2, 2, BLENDED_2, 1, 1)]
    blend = [blend_rows([(3, A), (1, B)]),
             blend_rows(list(zip(second_weights, (A, B))), blended_id=BLENDED_2, scheme=2, placement_ids=[10, 11])]
    with pytest.raises(OasisException, match='same placement scheme'):
        gul_losses(write_model(tmp_path / 'blended', items, blend))


def test_every_blended_item_needs_its_location(tmp_path):
    run_dir = write_model(tmp_path / 'blended', THREE_LOCATIONS, blend_rows([(3, A), (1, B)]))
    summary_map = pd.read_csv(run_dir / 'input' / 'gul_summary_map.csv')
    summary_map[summary_map['item_id'] != 2].to_csv(run_dir / 'input' / 'gul_summary_map.csv', index=False)
    with pytest.raises(OasisException, match='missing from gul_summary_map'):
        gul_losses(run_dir)


# --------------------------------------------------------------------------------------------------
# getmodel + gulpy: a blended item's cdf is the mixture of its positions'
# --------------------------------------------------------------------------------------------------

def gulpy_losses(run_dir, sample_size=10):
    out = run_dir / 'gulpy.bin'
    subprocess.run(f'evepy 1 1 | modelpy | gulpy -a0 -S{sample_size} -L0 --random-generator=1 > {out}',
                   cwd=run_dir, shell=True, check=True)
    return read_gul_stream(out)[2].set_index(['event_id', 'item_id', 'sidx'])['loss']


def test_gulpy_mean_is_the_mixture_gulmc_computes(tmp_path):
    """getmodel streams a blended item the mixture of its positions' cdfs, so gulpy's mean is gulmc's."""
    run_dir = write_model(tmp_path / 'blended', THREE_LOCATIONS, blend_rows([(0.6, A), (0.3, B)]))
    gulmc_mean = means(gul_losses(run_dir))
    gulpy_mean = means(gulpy_losses(run_dir))
    assert (gulmc_mean > 0).any()
    np.testing.assert_allclose(gulpy_mean.reindex(gulmc_mean.index), gulmc_mean, rtol=1e-5)


def test_gulpy_is_unchanged_for_a_model_without_blending(tmp_path):
    """A blend table no item reads changes nothing for gulpy."""
    without = write_model(tmp_path / 'without', placed_on(THREE_LOCATIONS, A))
    unread = write_model(tmp_path / 'unread', placed_on(THREE_LOCATIONS, A), blend_rows([(3, A), (1, B)]))
    gulpy_losses(without)
    gulpy_losses(unread)
    assert (without / 'gulpy.bin').read_bytes() == (unread / 'gulpy.bin').read_bytes()


# --------------------------------------------------------------------------------------------------
# The blend table: validation and the csv <-> bin converters
# --------------------------------------------------------------------------------------------------

def check_table(text):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'blended_areaperil_id.csv'
        path.write_text(text)
        with open(path) as f:
            return validate(read_csv(f))


HEADER = 'blended_areaperil_id,placement_scheme_id,placement_id,weight,areaperil_id\n'


@pytest.mark.parametrize('rows, match', [
    ('9,1,0,1,1\n9,2,1,1,2\n', 'more than one placement_scheme_id'),
    ('9,1,5,1,1\n9,1,5,1,2\n', 'more than once'),
    ('9,1,0,1,1\n9,1,1,1,2\n8,1,0,1,3\n', 'list different placement_id'),
    ('9,1,0,1,1\n9,1,1,1,2\n8,1,0,1,3\n8,1,7,1,4\n', 'list different placement_id'),
    ('9,1,0,1,1\n9,1,1,1,2\n8,1,0,1,3\n8,1,1,2,4\n', 'different weights'),
    ('9,1,0,-1,1\n9,1,1,2,2\n', 'negative or non-finite'),
    ('9,1,0,0,1\n9,1,1,0,2\n', 'total weight of 0'),
    ('9,1,0,0.7,1\n9,1,1,0.4,2\n', 'sum to more than 1'),
    ('9,1,0,1,1\n9,1,1,1,8\n8,2,0,1,3\n', 'also appear as areaperil_id'),
], ids=['two schemes', 'placement listed twice', 'scheme sizes differ', 'scheme placements differ', 'scheme weights differ', 'negative weight',
        'zero count total', 'float sum over 1', 'blended id used as a cell'])
def test_an_inconsistent_table_is_rejected(rows, match):
    with pytest.raises(OasisException, match=match):
        check_table(HEADER + rows)


def test_a_misspelt_weight_column_is_rejected():
    with pytest.raises(OasisException):
        check_table('blended_areaperil_id,placement_scheme_id,placement_id,wieght,areaperil_id\n9,1,0,1,1\n')


def test_placement_ids_need_not_be_numbered_from_0():
    """Placement ids from a model-wide numbering, with gaps, are accepted and sorted per blended id."""
    table = check_table(HEADER + '9,1,7002,1,1\n9,1,15,1,2\n8,1,15,1,3\n8,1,7002,1,4\n').table
    assert table['placement_id'].tolist() == [15, 7002, 15, 7002]


@pytest.mark.parametrize('weights', [None, [3, 1], [0.6, 0.25]], ids=['equal', 'count', 'float'])
def test_the_table_survives_a_round_trip_through_the_binary_form(tmp_path, weights):
    table = pd.DataFrame({'blended_areaperil_id': [9, 9, 8, 8], 'placement_scheme_id': [1, 1, 1, 1],
                          'placement_id': [7, 41, 7, 41], 'areaperil_id': [1, 2, 3, 4]})
    if weights is not None:
        table.insert(3, 'weight', weights * 2)
    table.to_csv(tmp_path / 'in.csv', index=False)
    csvtobin(tmp_path / 'in.csv', tmp_path / 'table.bin', 'blended_areaperil')
    bintocsv(tmp_path / 'table.bin', tmp_path / 'out.csv', 'blended_areaperil')

    expected = table.sort_values(['blended_areaperil_id', 'placement_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / 'out.csv'), expected, check_dtype=False)
