"""Run the dummy model end to end, as `oasislmf test model run` does."""

import json

import numpy as np
import pandas as pd
import pytest

from oasislmf.manager import OasisManager
from oasislmf.pytools.common.input_files import read_correlations
from oasislmf.pytools.gulmc.items import read_items
from oasislmf.pytools.summary.manager import load_summary_map

ORD_OUTPUT = {'elt_sample': True, 'elt_moment': True, 'plt_sample': True, 'alt_period': True,
              'ept_full_uncertainty_aep': True, 'return_period_file': True}
OUTPUT_FILES = ['S1_selt.csv', 'S1_melt.csv', 'S1_splt.csv', 'S1_palt.csv', 'S1_ept.csv']

NUM_LOCATIONS = 6
COVERAGES_PER_LOCATION = 2


def analysis_settings(il):
    """Analysis settings with one summary per loss type and no correlation settings."""
    settings = {
        'version': '3',
        'source_tag': 'dummy',
        'analysis_tag': 'dummy',
        'model_supplier_id': 'dummy',
        'model_name_id': 'dummy',
        'number_of_samples': 5,
        'model_settings': {},
        'gul_output': True,
        'gul_summaries': [{'id': 1, 'ord_output': dict(ORD_OUTPUT)}],
    }
    if il:
        settings['il_output'] = True
        settings['il_summaries'] = [{'id': 1, 'ord_output': dict(ORD_OUTPUT)}]
    return settings


def run_dummy_model(tmp_path, il):
    settings_fp = tmp_path / 'analysis_settings.json'
    settings_fp.write_text(json.dumps(analysis_settings(il)))
    target_dir = tmp_path / 'run'
    with pytest.warns(UserWarning, match='Return period file is not generated'):
        OasisManager().generate_losses_dummy_model(
            analysis_settings_json=str(settings_fp),
            target_dir=str(target_dir),
            kernel_num_processes=2,
            num_vulnerabilities=5,
            num_intensity_bins=5,
            num_damage_bins=5,
            num_events=20,
            num_areaperils=8,
            areaperils_per_event=4,
            num_periods=20,
            num_locations=NUM_LOCATIONS,
            coverages_per_location=COVERAGES_PER_LOCATION,
            random_seed=1234,
        )
    return target_dir


@pytest.mark.parametrize('il', [False, True], ids=['gul', 'gul_il'])
def test_dummy_model_run_writes_losses(tmp_path, il):
    target_dir = run_dummy_model(tmp_path, il)
    output_dir = target_dir / 'output'

    run_settings = json.loads((target_dir / 'input' / 'analysis_settings.json').read_text())
    assert run_settings['gul_summaries'][0]['ord_output']['return_period_file'] is False

    perspectives = ['gul', 'il'] if il else ['gul']
    for perspective in perspectives:
        for file_name in OUTPUT_FILES:
            output = pd.read_csv(output_dir / f'{perspective}_{file_name}')
            assert len(output) > 0, f'{perspective}_{file_name} has no rows'
    if not il:
        assert not (output_dir / 'il_S1_selt.csv').exists()

    gul_selt = pd.read_csv(output_dir / 'gul_S1_selt.csv')
    gul_samples = gul_selt[gul_selt['SampleId'] > 0]
    assert (gul_samples['Loss'] >= 0).all()
    assert gul_samples['Loss'].sum() > 0

    if il:
        il_selt = pd.read_csv(output_dir / 'il_S1_selt.csv')
        il_samples = il_selt[il_selt['SampleId'] > 0]
        assert (il_samples['Loss'] >= 0).all()
        assert 0 < il_samples['Loss'].sum() <= gul_samples['Loss'].sum()


def test_dummy_oasis_files_include_correlations_and_summary_maps(tmp_path):
    target_dir = tmp_path / 'files'
    OasisManager().generate_dummy_oasis_files(
        target_dir=str(target_dir),
        num_vulnerabilities=5,
        num_intensity_bins=5,
        num_damage_bins=5,
        num_events=20,
        num_areaperils=8,
        num_periods=20,
        num_locations=NUM_LOCATIONS,
        coverages_per_location=COVERAGES_PER_LOCATION,
        num_layers=2,
        random_seed=1234,
    )
    input_dir = str(target_dir / 'input')
    num_items = NUM_LOCATIONS * COVERAGES_PER_LOCATION
    loc_ids = np.repeat(np.arange(1, NUM_LOCATIONS + 1), COVERAGES_PER_LOCATION)

    correlations = read_correlations(input_dir)
    items = read_items(input_dir)
    np.testing.assert_array_equal(correlations['item_id'], items['item_id'])
    np.testing.assert_array_equal(correlations['hazard_group_id'], loc_ids)
    assert not correlations['peril_correlation_group'].any()
    assert not correlations['damage_correlation_value'].any()
    assert not correlations['hazard_correlation_value'].any()
    assert not correlations['source_item_id'].any()

    gul_summary_map = load_summary_map(input_dir, 'gul_summary_map')
    np.testing.assert_array_equal(gul_summary_map['item_id'], np.arange(1, num_items + 1))
    np.testing.assert_array_equal(gul_summary_map['loc_id'], loc_ids)
    assert (gul_summary_map['building_id'] == 1).all()

    fm_summary_map = load_summary_map(input_dir, 'fm_summary_map', col_map={'item_id': 'output_id'})
    np.testing.assert_array_equal(fm_summary_map['item_id'], np.arange(1, 2 * num_items + 1))
    np.testing.assert_array_equal(fm_summary_map['loc_id'], np.repeat(loc_ids, 2))
