"""Intensity bin lookup for dynamic footprints with negative intensities (issue #2165)."""
import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.getmodel.manager import encode_peril_id, get_intensity_bin_dict
from oasislmf.pytools.gulmc.common import haz_arr_type
from oasislmf.pytools.gulmc.manager import compute_haz_bin_id

ITEM_DTYPE = np.dtype([('peril_id', np.int32)])


def _bin_dict(tmp_path, rows):
    pd.DataFrame(rows, columns=['peril_id', 'intensity', 'intensity_bin']).to_csv(
        tmp_path / 'intensity_bin_dict.csv', index=False)
    return get_intensity_bin_dict(str(tmp_path))


def _lookup(tables, peril, intensities, adjustment=0):
    """Resolve intensities to bin ids the way gulmc does under a dynamic footprint."""
    peril_ids, bins, bin_min = tables
    haz = np.zeros(len(intensities), dtype=haz_arr_type.dtype)
    haz['intensity'] = intensities
    item = np.zeros(1, dtype=ITEM_DTYPE)[0]
    item['peril_id'] = encode_peril_id(peril)
    return list(compute_haz_bin_id(haz, item, adjustment, True, peril_ids, bins, bin_min))


def test_non_negative_dict_keeps_its_layout(tmp_path):
    tables = _bin_dict(tmp_path, [['ORF', i, i + 1] for i in range(6)])
    peril_ids, bins, bin_min = tables
    assert list(bin_min) == [0]
    assert bins.tolist() == [[1, 2, 3, 4, 5, 6]]
    assert _lookup(tables, 'ORF', [0, 3, 5]) == [1, 4, 6]


def test_negative_intensities_get_their_own_bins(tmp_path):
    # -3..-1 are off-floodplain impacts, 0..3 fluvial; bins 1..7
    tables = _bin_dict(tmp_path, [['ORF', i, i + 4] for i in range(-3, 4)])
    _, bins, bin_min = tables
    assert list(bin_min) == [-3]
    # no wrap-around: the last column still holds intensity 3, not intensity -1
    assert bins.tolist() == [[1, 2, 3, 4, 5, 6, 7]]
    assert _lookup(tables, 'ORF', [-3, -1, 0, 3]) == [1, 3, 4, 7]


def test_adjustment_into_negative_range(tmp_path):
    tables = _bin_dict(tmp_path, [['ORF', i, i + 4] for i in range(-3, 4)])
    # a defence lowering intensity 1 by 3 gives -2, which is bin 2 (not the intensity-0 bin)
    assert _lookup(tables, 'ORF', [1, 3], adjustment=3) == [2, 4]


def test_intensities_beyond_the_table_take_the_nearest_edge_bin(tmp_path):
    tables = _bin_dict(tmp_path, [['ORF', i, i + 4] for i in range(-3, 4)])
    assert _lookup(tables, 'ORF', [-10, 50]) == [1, 7]
    tables = _bin_dict(tmp_path, [['WTC', i, i + 1] for i in range(5)])
    # previously both went to the intensity-0 bin
    assert _lookup(tables, 'WTC', [-2, 9]) == [1, 5]


def test_perils_with_different_ranges(tmp_path):
    rows = [['ORF', i, i + 4] for i in range(-3, 2)] + [['WTC', i, 20 + i] for i in range(0, 8)]
    tables = _bin_dict(tmp_path, rows)
    peril_ids, bins, bin_min = tables
    assert dict(zip(peril_ids, bin_min)) == {encode_peril_id('ORF'): -3, encode_peril_id('WTC'): 0}
    assert _lookup(tables, 'ORF', [-3, 1, 4]) == [1, 5, 5]
    assert _lookup(tables, 'WTC', [-1, 0, 7]) == [20, 20, 27]


def test_bin_id_zero_is_not_treated_as_missing(tmp_path):
    tables = _bin_dict(tmp_path, [['WTC', 0, 3], ['WTC', 1, 0], ['WTC', 2, 4]])
    assert _lookup(tables, 'WTC', [0, 1, 2]) == [3, 0, 4]


@pytest.mark.parametrize('rows, expected', [
    # gaps take the intensity-0 bin when the peril defines one
    ([['WTC', 0, 9], ['WTC', 1, 1], ['WTC', 4, 4]], [9, 1, 9, 9, 4]),
    # and the nearest lower bin when it does not
    ([['ORF', -2, 1], ['ORF', -1, 2], ['ORF', 2, 5]], [1, 2, 2, 2, 5]),
])
def test_gaps(tmp_path, rows, expected):
    tables = _bin_dict(tmp_path, rows)
    lo = min(r[1] for r in rows)
    assert _lookup(tables, rows[0][0], list(range(lo, lo + 5))) == expected


def test_dict_starting_above_zero_does_not_map_low_intensities_to_bin_zero(tmp_path):
    tables = _bin_dict(tmp_path, [['WTC', i, i] for i in range(1, 4)])
    assert _lookup(tables, 'WTC', [0, 1, 3]) == [1, 1, 3]
