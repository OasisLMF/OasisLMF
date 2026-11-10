"""
A dynamic footprint carries intensities, mapped to the model's bins by intensity_bin_dict.csv.

An intensity past the table takes the bin of the peril's largest intensity: the table's last row
stands for that much or more, not for no hazard.
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
        merge_dynamic_haz_records_per_bin(haz_pdf_record, item, intensity_adjustment, *intensity_bins,
                                          bin_slot, haz_bin_id, haz_prob)
        bins.append(int(haz_bin_id[0]))
    return bins


def test_an_intensity_past_the_table_takes_the_top_bin(intensity_bins):
    # 3 is a gap in the table and keeps the bin of intensity 0, as before
    assert bins_of(intensity_bins, 'OSF', [0, 1, 3, 5, 6, 1000]) == [1, 2, 1, 5, 5, 5]


def test_a_peril_with_a_shorter_table_takes_its_own_top_bin(intensity_bins):
    assert bins_of(intensity_bins, 'WTC', [0, 2, 3, 5, 1000]) == [11, 13, 13, 13, 13]


def test_a_negative_intensity_takes_the_bin_of_intensity_0(intensity_bins):
    # an adjustment for defences can take the intensity below 0
    assert bins_of(intensity_bins, 'OSF', [2, 4], intensity_adjustment=3) == [1, 2]
