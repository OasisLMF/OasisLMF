from pathlib import Path

import numpy as np
import pandas as pd
from oasis_data_manager.filestore.backends.local import LocalStorage

from oasislmf.pytools.getmodel.footprint import Footprint, FootprintBin, FootprintCsv

footprints_path = Path(__file__).resolve().parent / "footprints"


def test_csv_footprint_matches_bin(tmp_path):
    with FootprintBin(LocalStorage(footprints_path)) as bin_footprint:
        event_ids = bin_footprint.footprint_index['event_id'].tolist()
        bin_events = {e: bin_footprint.get_event(e) for e in event_ids}

    df = pd.concat(pd.DataFrame(bin_events[e]).assign(event_id=e) for e in event_ids)
    df[['event_id', 'areaperil_id', 'intensity_bin_id', 'probability']].to_csv(tmp_path / "footprint.csv", index=False)

    with Footprint.load(LocalStorage(tmp_path)) as csv_footprint:
        assert isinstance(csv_footprint, FootprintCsv)
        assert csv_footprint.num_intensity_bins == df['intensity_bin_id'].max()
        assert csv_footprint.has_intensity_uncertainty == (df.groupby(['event_id', 'areaperil_id']).size().max() > 1)
        for e in event_ids:
            csv_event = csv_footprint.get_event(e)
            for col in ('areaperil_id', 'intensity_bin_id', 'probability'):
                np.testing.assert_array_equal(csv_event[col], bin_events[e][col])
        assert csv_footprint.get_event(max(event_ids) + 1) is None
