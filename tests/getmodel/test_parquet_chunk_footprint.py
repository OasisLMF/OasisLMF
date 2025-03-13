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
path = footprints_path / "footprint_lookup.parquet"


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
        footprint_lookup = pd.read_parquet(path)
        row = footprint_lookup.loc[footprint_lookup["event_id"] == event_id].iloc[0]

        partition = row["partition"]
        get_events_events = footprint.get_events(partition)

        assert any(np.array_equal(get_event_event, event) for event in get_events_events)


@pytest.fixture(scope="session", autouse=True)
def cleanup_parquet_files():
    convert_bin_to_parquet_chunk(footprints_path, chunk_size=0)
    yield

    for file in footprints_path.glob("*.parquet"):
        file.unlink()
    for file in footprints_path.glob("*footprint_lookup*"):
        file.unlink()
    for file in footprints_path.glob("*.json"):
        file.unlink()
