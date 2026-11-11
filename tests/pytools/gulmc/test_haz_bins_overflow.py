"""
An areaperil whose dynamic footprint holds more hazard records than the model has intensity bins.

A dynamic footprint record is a raw intensity, so an areaperil can carry many records that fall in
the same intensity bin, more than the bins gulmc sizes its per-hazard buffers on. Splitting each
bin's record into several records of that bin, with its probability shared between them, describes
the same hazard, so gulmc must write the same losses, sample for sample.
"""
import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.gulmc.manager import run as run_gulmc

ONE_RECORD_PER_BIN = [6, 2]               # bin 3 then bin 2, as the footprint gives intensities largest first
MANY_RECORDS_PER_BIN = list(range(1, 9))   # 4 in each bin: above the 3 intensity and 3 damage bins the buffers are sized on
AREAPERIL_IDS = [1, 2]       # two items, so the second runs after the first has used the scratch


def write_model(run_dir, intensities):
    """A one-event, one-section dynamic footprint model whose areaperils share one hazard.

    Intensity 0 is bin 1 (no hazard), 1 to 4 are bin 2 and 5 to 8 are bin 3; the probability is
    shared evenly between the records given by `intensities`. Bin 2 damages into damage bin 2 and
    bin 3 into damage bin 3, so a record put in the wrong bin changes the losses.
    """
    static, inp = run_dir / 'static', run_dir / 'input'
    static.mkdir(parents=True)
    inp.mkdir()

    pd.DataFrame({'bin_index': [1, 2, 3], 'bin_from': [0., 0., .5], 'bin_to': [0., .5, 1.],
                  'interpolation': [0., .25, .75], 'damage_type': 0}).to_csv(static / 'damage_bin_dict.csv', index=False)
    pd.DataFrame({'vulnerability_id': 1, 'intensity_bin_id': [1, 2, 3], 'damage_bin_id': [1, 2, 3],
                  'probability': 1.}).to_csv(static / 'vulnerability.csv', index=False)
    pd.DataFrame({'peril_id': 'WTC', 'intensity': range(9), 'intensity_bin': [1] + [2] * 4 + [3] * 4}
                 ).to_csv(static / 'intensity_bin_dict.csv', index=False)
    (static / 'footprint_parquet_meta.json').write_text('{"num_intensity_bins": 3, "has_intensity_uncertainty": 1}')
    pd.DataFrame([{'section_id': 1, 'areaperil_id': ap, 'return_period': rp, 'intensity': i,
                   'probability': 1 / len(intensities)}
                  for ap in AREAPERIL_IDS for rp in (10, 100) for i in intensities]
                 ).to_parquet(static / 'hazard_case.parquet', index=False)
    pd.DataFrame({'event_id': [1], 'section_id': 1, 'rp_from': 10, 'rp_to': 100, 'interpolation': .5}
                 ).to_parquet(static / 'event_definition.parquet', index=False)

    item_ids = np.arange(1, len(AREAPERIL_IDS) + 1)
    pd.DataFrame({'item_id': item_ids, 'coverage_id': item_ids, 'areaperil_id': AREAPERIL_IDS,
                  'vulnerability_id': 1, 'group_id': item_ids}).to_csv(inp / 'items.csv', index=False)
    pd.DataFrame({'coverage_id': item_ids, 'tiv': 1000.}).to_csv(inp / 'coverages.csv', index=False)
    pd.DataFrame({'LocID': item_ids, 'PerilID': 'WTC', 'CoverageTypeID': 1, 'AreaPerilID': AREAPERIL_IDS,
                  'VulnerabilityID': 1}).to_csv(inp / 'keys.csv', index=False)
    pd.DataFrame({'item_id': item_ids, 'peril_id': 'WTC'}).to_csv(inp / 'gul_summary_map.csv', index=False)
    pd.DataFrame({'item_id': item_ids, 'peril_correlation_group': 0, 'damage_correlation_value': 0.,
                  'hazard_group_id': 0, 'hazard_correlation_value': 0., 'source_item_id': 0}
                 ).to_csv(inp / 'correlations.csv', index=False)
    pd.DataFrame({'section_id': [1]}).to_csv(inp / 'sections.csv', index=False)
    np.array([1], dtype=np.int32).tofile(inp / 'events.bin')


def gulmc_losses(tmp_path, monkeypatch, name, intensities, sample_size, effective_damageability):
    run_dir = tmp_path / name
    write_model(run_dir, intensities)
    monkeypatch.chdir(run_dir)  # the dynamic footprint reads input/ relative to the run directory
    run_gulmc(run_dir=str(run_dir), ignore_file_type=set(), file_in=str(run_dir / 'input' / 'events.bin'),
              file_out=str(run_dir / 'gul.bin'), sample_size=sample_size, loss_threshold=0., alloc_rule=1,
              debug=0, random_generator=1, effective_damageability=effective_damageability, dynamic_footprint=True)
    return (run_dir / 'gul.bin').read_bytes()


@pytest.mark.parametrize("effective_damageability", [False, True], ids=["full-monte-carlo", "effective-damageability"])
def test_records_of_a_bin_give_the_losses_of_one_record_per_bin(tmp_path, monkeypatch, effective_damageability):
    one_record = gulmc_losses(tmp_path, monkeypatch, 'one', ONE_RECORD_PER_BIN, 10, effective_damageability)
    many_records = gulmc_losses(tmp_path, monkeypatch, 'many', MANY_RECORDS_PER_BIN, 10, effective_damageability)

    assert many_records == one_record
    # both runs go through the same merge, so also check the losses against the hazard: after the
    # stream header every 8 bytes are an (event_id, item_id) or a (sidx, loss) pair, and the mean
    # loss (sidx -1) of each item is TIV x (.5 in bin 2 x damage .25 + .5 in bin 3 x damage .75)
    pairs = np.frombuffer(many_records[8:], dtype=[('sidx', np.int32), ('loss', np.float32)])
    np.testing.assert_allclose(pairs['loss'][pairs['sidx'] == -1], [500.] * len(AREAPERIL_IDS), rtol=1e-6)
