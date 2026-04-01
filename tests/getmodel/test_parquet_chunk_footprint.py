import shutil

from oasislmf.pytools.getmodel.footprint import (
    FootprintParquetChunk, FootprintBin
)
import pytest
from oasis_data_manager.filestore.backends.local import LocalStorage
from pathlib import Path
import numpy as np
import pandas as pd
from oasislmf.pytools.data_layer.conversions.footprint import convert_bin_to_parquet_chunk

script_dir = Path(__file__).resolve().parent
footprints_path = script_dir / "footprints"


@pytest.fixture(scope="session")
def footprints_tmp(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("footprints")
    shutil.copy2(footprints_path / "footprint.bin", tmp / "footprint.bin")
    shutil.copy2(footprints_path / "footprint.idx", tmp / "footprint.idx")
    convert_bin_to_parquet_chunk(tmp, chunk_size=0)
    return tmp


@pytest.mark.parametrize("event_id", [
    1, 2, 3, 4, 5, 6, 7, 8
])
def test_get_event(footprints_tmp, event_id):
    with FootprintParquetChunk(LocalStorage(footprints_tmp)) as footprint:
        parquet_event = footprint.get_event(event_id)
    with FootprintBin(LocalStorage(footprints_tmp)) as footprint:
        bin_event = footprint.get_event(event_id)

    assert np.array_equal(parquet_event, bin_event)

    if event_id != 8:
        assert parquet_event is not None
    else:
        assert parquet_event is None


@pytest.mark.parametrize("event_id", [
    1, 2, 3, 4, 5, 6, 7
])
def test_get_events(footprints_tmp, event_id):
    with FootprintParquetChunk(LocalStorage(footprints_tmp)) as footprint:
        get_event_event = footprint.get_event(event_id)
        footprint_lookup = pd.read_parquet(footprints_tmp / "footprint_lookup.parquet")
        row = footprint_lookup.loc[footprint_lookup["event_id"] == event_id].iloc[0]

        partition = row["partition"]
        get_events_events = footprint.get_events(partition)

        assert any(np.array_equal(get_event_event, event) for event in get_events_events)
