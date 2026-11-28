"""
A dynamic footprint carries intensities, mapped to the model's bins by intensity_bin_dict.csv.

An intensity past the table takes the bin of the peril's largest intensity: the table's last row
stands for that much or more, not for no hazard. Intensities can be negative (#2165): the table then
starts at its lowest intensity, and an intensity below it takes the bin of the lowest one.
"""
import numpy as np
import pandas as pd
import pytest

from oasislmf.pytools.getmodel.manager import encode_peril_id, get_intensity_bin_dict
from oasislmf.pytools.gulmc.common import haz_arr_type
from oasislmf.pytools.gulmc.manager import merge_dynamic_haz_records_per_bin


@pytest.fixture
def intensity_bins(tmp_path):
    """OSF up to intensity 5 (with a gap at 3), WTC only up to 2, so its row is padded."""
    pd.DataFrame([('OSF', 0, 1), ('OSF', 1, 2), ('OSF', 2, 3), ('OSF', 4, 4), ('OSF', 5, 5),
                  ('WTC', 0, 11), ('WTC', 1, 12), ('WTC', 2, 13)],
                 columns=['peril_id', 'intensity', 'intensity_bin']).to_csv(tmp_path / 'intensity_bin_dict.csv', index=False)
    return get_intensity_bin_dict(str(tmp_path))


def bins_of(intensity_bins, peril_id, intensities, intensity_adjustment=0):
    """The bin of each intensity, given one record at a time so that none is merged with another."""
    item = np.zeros(1, dtype=[('peril_id', np.int32)])[0]
    item['peril_id'] = encode_peril_id(peril_id)
    haz_pdf_record = np.zeros(1, dtype=haz_arr_type)
    n_bin_ids = intensity_bins[1].max() + 1
    bin_slot = np.full(n_bin_ids, -1, dtype=np.int64)
    haz_bin_id = np.empty(n_bin_ids, dtype=np.int32)
    haz_prob = np.empty(n_bin_ids, dtype=haz_pdf_record['probability'].dtype)
    bins = []
    for intensity in intensities:
        haz_pdf_record['intensity'] = intensity
        merge_dynamic_haz_records_per_bin(haz_pdf_record, 0, 1, item, intensity_adjustment, *intensity_bins,
                                          bin_slot, haz_bin_id, haz_prob)
        bins.append(int(haz_bin_id[0]))
    return bins


def test_an_intensity_past_the_table_takes_the_top_bin(intensity_bins):
    # 3 is a gap in the table and keeps the bin of intensity 0, as before
    assert bins_of(intensity_bins, 'OSF', [0, 1, 3, 5, 6, 1000]) == [1, 2, 1, 5, 5, 5]


def test_a_peril_with_a_shorter_table_takes_its_own_top_bin(intensity_bins):
    assert bins_of(intensity_bins, 'WTC', [0, 2, 3, 5, 1000]) == [11, 13, 13, 13, 13]


def test_a_negative_intensity_takes_the_bin_of_intensity_0_without_negative_bins(intensity_bins):
    # an adjustment for defences can take the intensity below 0
    assert bins_of(intensity_bins, 'OSF', [2, 4], intensity_adjustment=3) == [1, 2]


def write_intensity_bin_dict(path, rows):
    pd.DataFrame(rows, columns=['peril_id', 'intensity', 'intensity_bin']).to_csv(path / 'intensity_bin_dict.csv', index=False)
    return get_intensity_bin_dict(str(path))


@pytest.fixture
def negative_intensity_bins(tmp_path):
    """ORF from -3 to 2 (with a gap at -1), as for flood depths below a datum; WTC only from 0 to 2."""
    return write_intensity_bin_dict(tmp_path, [
        ('ORF', -3, 1), ('ORF', -2, 2), ('ORF', 0, 4), ('ORF', 1, 5), ('ORF', 2, 6),
        ('WTC', 0, 11), ('WTC', 1, 12), ('WTC', 2, 13)])


def test_a_table_with_negative_intensities_starts_at_its_lowest_intensity(negative_intensity_bins):
    peril_ids, bins, start = negative_intensity_bins
    assert start == -3
    assert bins.shape == (2, 6)  # intensities -3 to 2
    # -1 is a gap and takes the bin of intensity 0, as a gap above 0 does
    assert bins[list(peril_ids).index(encode_peril_id('ORF'))].tolist() == [1, 2, 4, 4, 5, 6]
    # WTC has no negative intensity: below 0 it keeps taking the bin of intensity 0
    assert bins[list(peril_ids).index(encode_peril_id('WTC'))].tolist() == [11, 11, 11, 11, 12, 13]


def test_a_negative_intensity_takes_its_own_bin(negative_intensity_bins):
    assert bins_of(negative_intensity_bins, 'ORF', [-3, -2, -1, 0, 1, 2]) == [1, 2, 4, 4, 5, 6]


def test_an_intensity_below_the_table_takes_the_bottom_bin(negative_intensity_bins):
    assert bins_of(negative_intensity_bins, 'ORF', [-4, -1000, 3, 1000]) == [1, 1, 6, 6]
    assert bins_of(negative_intensity_bins, 'WTC', [-1, -3, -1000]) == [11, 11, 11]


def test_an_adjustment_can_take_an_intensity_into_the_negative_bins(negative_intensity_bins):
    # a defence lowering the intensity by 3: 1 -> -2, 0 -> -3, 2 -> -1 (a gap)
    assert bins_of(negative_intensity_bins, 'ORF', [1, 0, 2, -5], intensity_adjustment=3) == [2, 1, 4, 1]
    # a negative adjustment raises it
    assert bins_of(negative_intensity_bins, 'ORF', [-5, -4], intensity_adjustment=-2) == [1, 2]


def test_a_table_with_only_negative_intensities(tmp_path):
    """A temperature-like hazard need not reach 0 at all."""
    table = write_intensity_bin_dict(tmp_path, [('ORF', -30, 1), ('ORF', -20, 2), ('ORF', -10, 3)])
    assert table[2] == -30
    assert bins_of(table, 'ORF', [-40, -30, -20, -10, 0, 5]) == [1, 1, 2, 3, 3, 3]


def test_a_listed_bin_is_kept_whatever_its_id(tmp_path):
    """A slot is filled because the file lists it, not because its bin id is not 0."""
    _, bins, _ = write_intensity_bin_dict(tmp_path, [('OSF', 0, 3), ('OSF', 1, 0), ('OSF', 2, 4)])
    assert bins[0].tolist() == [3, 0, 4]


def legacy_intensity_bins(rows):
    """The table get_intensity_bin_dict built before negative intensities were supported, indexed from intensity 0."""
    data = np.array([(encode_peril_id(p), i, b) for p, i, b in rows], dtype=np.int32)
    peril_ids = np.unique(data[:, 0])
    bins = np.zeros((len(peril_ids), int(data[:, 1].max()) + 1), dtype=np.int32)
    for peril_id, intensity, bin_id in data:
        bins[np.searchsorted(peril_ids, peril_id), intensity] = bin_id
    for pi in range(len(peril_ids)):
        peril_max = int(data[data[:, 0] == peril_ids[pi], 1].max())
        for i in range(1, bins.shape[1]):
            if bins[pi, i] == 0:
                bins[pi, i] = bins[pi, 0] if i < peril_max else bins[pi, peril_max]
    return peril_ids, bins


@pytest.mark.parametrize('seed', range(20))
def test_a_table_without_negative_intensities_is_unchanged(tmp_path, seed):
    """Models with no negative intensity get the same table, indexed from 0, and the same bins, so the same losses."""
    rng = np.random.default_rng(seed)
    rows = []
    for peril_id in ['WTC', 'OSF', 'ORF'][:rng.integers(1, 4)]:
        intensities = np.sort(rng.choice(np.arange(0, 30), size=rng.integers(1, 15), replace=False))
        if rng.random() < 0.7:  # most tables list intensity 0
            intensities = np.union1d(intensities, [0])
        rows += [(peril_id, int(i), int(b)) for b, i in enumerate(intensities, start=1)]
    table = write_intensity_bin_dict(tmp_path, rows)
    legacy_peril_ids, legacy_bins = legacy_intensity_bins(rows)
    assert table[2] == 0
    np.testing.assert_array_equal(table[0], legacy_peril_ids)
    np.testing.assert_array_equal(table[1], legacy_bins)

    intensities = np.arange(-40, 40)
    for peril_id in {r[0] for r in rows}:
        legacy_row = legacy_bins[np.searchsorted(legacy_peril_ids, encode_peril_id(peril_id))]
        for adjustment in (0, 3, -3):
            expected = [int(legacy_row[min(max(i - adjustment, 0), legacy_bins.shape[1] - 1)]) for i in intensities]
            assert bins_of(table, peril_id, intensities, intensity_adjustment=adjustment) == expected
