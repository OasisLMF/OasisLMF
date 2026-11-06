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
from oasislmf.pytools.gulmc.manager import compute_haz_bin_id


@pytest.fixture
def intensity_bins(tmp_path):
    """OSF up to intensity 5 (with a gap at 3), WTC only up to 2, so its row is padded."""
    pd.DataFrame([('OSF', 0, 1), ('OSF', 1, 2), ('OSF', 2, 3), ('OSF', 4, 4), ('OSF', 5, 5),
                  ('WTC', 0, 11), ('WTC', 1, 12), ('WTC', 2, 13)],
                 columns=['peril_id', 'intensity', 'intensity_bin']).to_csv(tmp_path / 'intensity_bin_dict.csv', index=False)
    return get_intensity_bin_dict(str(tmp_path))


def bins_of(intensity_bins, peril_id, intensities, intensity_adjustment=0):
    haz_pdf_record = np.zeros(len(intensities), dtype=haz_arr_type)
    haz_pdf_record['intensity'] = intensities
    item = np.zeros(1, dtype=[('peril_id', np.int32)])[0]
    item['peril_id'] = encode_peril_id(peril_id)
    return compute_haz_bin_id(haz_pdf_record, item, intensity_adjustment, True, *intensity_bins).tolist()


def test_an_intensity_past_the_table_takes_the_top_bin(intensity_bins):
    # 3 is a gap in the table and keeps the bin of intensity 0, as before
    assert bins_of(intensity_bins, 'OSF', [0, 1, 3, 5, 6, 1000]) == [1, 2, 1, 5, 5, 5]


def test_a_peril_with_a_shorter_table_takes_its_own_top_bin(intensity_bins):
    assert bins_of(intensity_bins, 'WTC', [0, 2, 3, 5, 1000]) == [11, 13, 13, 13, 13]


def test_a_negative_intensity_takes_the_bin_of_intensity_0(intensity_bins):
    # an adjustment for defences can take the intensity below 0
    assert bins_of(intensity_bins, 'OSF', [2, 4], intensity_adjustment=3) == [1, 2]
