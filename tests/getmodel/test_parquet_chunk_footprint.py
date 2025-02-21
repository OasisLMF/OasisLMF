from oasislmf.pytools.getmodel.footprint import (
    FootprintParquetChunk, FootprintBin
)
import pytest
from oasis_data_manager.filestore.backends.local import LocalStorage
import os
import numpy as np
import pandas as pd
from oasislmf.pytools.data_layer.conversions.footprint import (
    convert_bin_to_parquet
)

script_dir = os.path.dirname(os.path.abspath(__file__))
footprints_path = os.path.join(script_dir, "footprints")
convert_bin_to_parquet(footprints_path, chunk_size=1)
path = os.path.join(footprints_path, "footprint_lookup.parquet")
footprint_lookup = pd.read_parquet(path)


@pytest.mark.parametrize("event_id", [
    1, 2, 3, 4, 5, 6, 7, 8
])
def test_get_event(event_id):
    with FootprintParquetChunk(LocalStorage(footprints_path)) as footprint:
        parquet_event = footprint.get_event(event_id)
    with FootprintBin(LocalStorage(footprints_path)) as footprint:
        bin_event = footprint.get_event(event_id)

    assert np.array_equal(parquet_event, bin_event)

    if event_id != 8:
        assert parquet_event is not None
    else:
        assert parquet_event is None


@pytest.mark.parametrize("event_id", [
    1, 2, 3, 4, 5, 6, 7
])
def test_get_events(event_id):
    with FootprintParquetChunk(LocalStorage(footprints_path)) as footprint:
        get_event_event = footprint.get_event(event_id)
        # import pdb; pdb.set_trace()
        row = footprint_lookup.loc[
            footprint_lookup["event_id"] == event_id
        ].iloc[0]

        partition = row["partition"]
        get_events_event = footprint.get_events(partition)[0]

        assert np.array_equal(get_event_event, get_events_event)
