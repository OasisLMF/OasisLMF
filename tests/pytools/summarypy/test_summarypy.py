import filecmp
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.common.data import gul_summary_xref_dtype
from oasislmf.pytools.common.event_stream import (MAX_LOSS_IDX, MEAN_IDX, NUMBER_OF_AFFECTED_RISK_IDX, TIV_IDX,
                                                  decode_building, decode_local_sidx)
from oasislmf.pytools.summary.cli import parser, manager
from tests.pytools.utils import packed_gul_stream, read_gul_stream, read_summary_stream


TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_summarypy")

IDX_DTYPE = np.dtype([('summary_id', '<i4'), ('offset', '<i8')])  # fix dtype to reproduce test setup


def case_runner(test_name, test_case):

    base_path = Path(TESTS_ASSETS_DIR, test_name)
    with TemporaryDirectory() as tmp_result_dir_str:
        for run_type, summary_set_ids in test_case.items():
            if run_type == manager.RUNTYPE_REINSURANCE_LOSS:
                static_path = base_path.joinpath('RI_1')
                output_zeros = ' -z'
            else:
                static_path = base_path
                output_zeros = ''

            summary_sets_cmd = ' '.join(f" -{summary_set_id} "
                                        f"{Path(tmp_result_dir_str, f'{run_type}_S{summary_set_id}_summary.bin')}"
                                        for summary_set_id in summary_set_ids)
            cmd = (f"-m -t {run_type}{output_zeros} -p {static_path}"
                   f" -i {Path(TESTS_ASSETS_DIR, run_type + '.bin')}{summary_sets_cmd}").split()

            kwargs = vars(parser.parse_args(cmd))
            kwargs.pop('logging_level')
            manager.main(**kwargs)
            for summary_set_id in summary_set_ids:
                base_file_name = f"{run_type}_S{summary_set_id}_summary"
                try:
                    for file_extention in ['.bin', '.idx']:
                        assert filecmp.cmp(Path(tmp_result_dir_str, base_file_name + file_extention),
                                           Path(base_path, base_file_name + file_extention), shallow=True)
                    idx_path = Path(tmp_result_dir_str, base_file_name + '.idx')
                    assert idx_path.stat().st_size % IDX_DTYPE.itemsize == 0, \
                        f"{idx_path} size is not a multiple of IDX_DTYPE.itemsize ({IDX_DTYPE.itemsize})"
                    recs = np.fromfile(idx_path, dtype=IDX_DTYPE)
                    assert len(recs) > 0, f"{idx_path} parsed to zero records"
                    assert (recs['summary_id'] >= 1).all(), f"{idx_path} has summary_id < 1"
                    assert (recs['offset'] >= 0).all(), f"{idx_path} has negative offsets"
                    assert (np.diff(recs['offset']) > 0).all(), \
                        f"{idx_path} offsets are not strictly increasing"
                except Exception as e:
                    error_path = base_path.joinpath('error_files')
                    error_path.mkdir(exist_ok=True)
                    for file_extention in ['.bin', '.idx']:
                        shutil.copyfile(Path(tmp_result_dir_str, base_file_name + file_extention),
                                        Path(error_path, base_file_name + file_extention))
                    raise Exception(f"running 'summarypy {' '.join(cmd)}' led to diff, see files at {error_path}") from e


def test_single_summary_set():
    test_case = {
        manager.RUNTYPE_GROUNDUP_LOSS: [1, ],
        manager.RUNTYPE_INSURED_LOSS: [1, ],
        manager.RUNTYPE_REINSURANCE_LOSS: [1, ],
    }
    case_runner('single_summary_set', test_case)


def test_multiple_summary_set():
    test_case = {
        manager.RUNTYPE_GROUNDUP_LOSS: [1, 2],
        manager.RUNTYPE_INSURED_LOSS: [1, 2, 3],
        manager.RUNTYPE_REINSURANCE_LOSS: [1, 2],
    }
    case_runner('multiple_summary_set', test_case)


@pytest.mark.parametrize('sample_size', [10, 0], ids=['samples', 'mean_only'])
def test_a_packed_items_buildings_add_into_its_summary(tmp_path, sample_size):
    """A summary is per item, and a packed item stands for all its buildings, so every building's
    samples, mean, maximum and TIV go into the item's summary, and each building of an affected
    location counts as an affected risk, as it would as an item of its own. A mean-only run
    (number_of_samples defaults to 0) still carries a set of specials per building.

    The summary map is written here; that generation puts ``number_of_buildings`` on it is tested
    by the summary-info TIV test in tests/computation/test_generate_losses.py."""
    n_buildings = 3
    run_dir, gul = packed_gul_stream(tmp_path / 'model', n_building=-n_buildings, sample_size=sample_size)
    # a location per coverage, and a summary per parity of the location
    items = pd.read_csv(run_dir / 'input' / 'items.csv').set_index('item_id')['coverage_id']
    xref = np.zeros(len(items), dtype=gul_summary_xref_dtype)
    xref['item_id'], xref['summary_id'], xref['summaryset_id'] = items.index, 1 + items % 2, 1
    xref.tofile(tmp_path / 'gulsummaryxref.bin')
    pd.DataFrame({'loc_id': items, 'item_id': items.index, 'building_id': 1,
                  'number_of_buildings': n_buildings}).to_csv(tmp_path / 'gul_summary_map.csv', index=False)

    out = tmp_path / 'summary.bin'
    manager.run(files_in=[str(gul)], static_path=str(tmp_path), run_type=manager.RUNTYPE_GROUNDUP_LOSS,
                low_memory=False, output_zeros=False, summary_sets_output=['-1', str(out)])
    got = read_summary_stream(out)

    stream = read_gul_stream(gul)[2]
    assert decode_building(stream['sidx'].min(), sample_size) == n_buildings, "the stream is not packed"
    stream['loc_id'] = stream['item_id'].map(items)
    stream['summary_id'] = 1 + stream['loc_id'] % 2
    stream['sidx'] = [decode_local_sidx(s, sample_size) for s in stream['sidx']]
    expected = stream.groupby(['event_id', 'summary_id', 'sidx'])['loss'].sum()

    assert set(got['sidx']) == {MAX_LOSS_IDX, NUMBER_OF_AFFECTED_RISK_IDX, MEAN_IDX, *range(1, sample_size + 1)}
    summed = got[got['sidx'] != NUMBER_OF_AFFECTED_RISK_IDX].set_index(['event_id', 'summary_id', 'sidx'])['loss']
    summed_sidx = expected.index.get_level_values('sidx').isin([MAX_LOSS_IDX, MEAN_IDX, *range(1, sample_size + 1)])
    expected_summed = expected[summed_sidx & (expected != 0)]  # summarypy writes no zero loss
    assert summed.index.sort_values().equals(expected_summed.index.sort_values())
    np.testing.assert_allclose(summed, expected_summed.loc[summed.index], rtol=1e-5)
    risks = got[got['sidx'] == NUMBER_OF_AFFECTED_RISK_IDX].set_index(['event_id', 'summary_id'])['loss']
    affected_locations = stream[stream['loss'] != 0].groupby(['event_id', 'summary_id'])['loc_id'].nunique()
    pd.testing.assert_series_equal(risks.sort_index(), (n_buildings * affected_locations).sort_index(),
                                   check_names=False, check_dtype=False)
    exposure = got.groupby(['event_id', 'summary_id'])['exposure_value'].first()
    np.testing.assert_allclose(exposure, expected.xs(TIV_IDX, level='sidx').loc[exposure.index], rtol=1e-5)
