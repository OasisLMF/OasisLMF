"""convert_bin_to_parquet_event writes each event sorted by areaperil_id in small, encoded row
groups, and FootprintParquet reads back exactly the bin footprint's rows."""
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from oasis_data_manager.filestore.backends.local import LocalStorage

from oasislmf.pytools.data_layer.conversions.footprint import convert_bin_to_parquet_event
from oasislmf.pytools.getmodel.common import Event_dtype, EventIndexBin_dtype
from oasislmf.pytools.getmodel.footprint import FootprintBin, FootprintParquet

footprints_path = Path(__file__).resolve().parent / "footprints"


def read_events(path, event_ids, cls, areaperil_ids=None):
    with cls(LocalStorage(path), areaperil_ids=areaperil_ids) as footprint:
        return {event_id: footprint.get_event(event_id) for event_id in event_ids}


def sorted_by_areaperil(event):
    return event[np.argsort(event['areaperil_id'], kind='stable')]


def write_bin_footprint(path, events, num_intensity_bins=50):
    """Writes events ({event_id: Event array}) as an uncompressed footprint.bin and footprint.idx."""
    index = np.zeros(len(events), dtype=EventIndexBin_dtype)
    offset = 8
    with open(path / 'footprint.bin', 'wb') as f:
        f.write(np.array([num_intensity_bins, 1], dtype=np.int32).tobytes())  # has intensity uncertainty
        for i, (event_id, event) in enumerate(sorted(events.items())):
            data = event.astype(Event_dtype).tobytes()
            index[i] = (event_id, offset, len(data))
            f.write(data)
            offset += len(data)
    index.tofile(path / 'footprint.idx')


@pytest.fixture
def fixture_footprint(tmp_path):
    shutil.copy2(footprints_path / "footprint.bin", tmp_path / "footprint.bin")
    shutil.copy2(footprints_path / "footprint.idx", tmp_path / "footprint.idx")
    return tmp_path


@pytest.fixture
def synthetic_footprint(tmp_path):
    """Events with several intensity bins per areaperil, the second one not sorted by areaperil_id."""
    rng = np.random.default_rng(0)
    events = {}
    for event_id, areaperil_ids in [(1, np.repeat(np.arange(1, 301), 2)), (2, rng.permutation(np.arange(1000, 1500))), (3, np.array([], dtype=int))]:
        event = np.zeros(len(areaperil_ids), dtype=Event_dtype)
        event['areaperil_id'] = areaperil_ids
        event['intensity_bin_id'] = rng.integers(1, 50, len(event))
        event['probability'] = rng.random(len(event))
        events[event_id] = event
    write_bin_footprint(tmp_path, events)
    return tmp_path


def test_fixture_events_match_bin(fixture_footprint):
    convert_bin_to_parquet_event(str(fixture_footprint))
    event_ids = range(1, 8)
    assert read_events(fixture_footprint, event_ids, FootprintParquet).keys() == read_events(fixture_footprint, event_ids, FootprintBin).keys()
    for event_id, event in read_events(fixture_footprint, event_ids, FootprintParquet).items():
        np.testing.assert_array_equal(event, read_events(fixture_footprint, [event_id], FootprintBin)[event_id])
    assert json.loads((fixture_footprint / 'footprint_parquet_meta.json').read_text()) == {
        'num_intensity_bins': 58, 'has_intensity_uncertainty': True}


@pytest.mark.parametrize('processes', [1, 2])
def test_synthetic_events_sorted_and_complete(synthetic_footprint, processes):
    convert_bin_to_parquet_event(str(synthetic_footprint), row_group_size=64, processes=processes)
    bin_events = read_events(synthetic_footprint, [1, 2, 3], FootprintBin)
    parquet_events = read_events(synthetic_footprint, [1, 2, 3], FootprintParquet)
    for event_id in [1, 2, 3]:
        np.testing.assert_array_equal(parquet_events[event_id], sorted_by_areaperil(bin_events[event_id]))


@pytest.mark.parametrize('areaperil_ids', [[5, 150, 300], [1000, 1250, 1499], [999, 2000]])
def test_filtered_read_matches_bin(synthetic_footprint, areaperil_ids):
    convert_bin_to_parquet_event(str(synthetic_footprint), row_group_size=64)
    bin_events = read_events(synthetic_footprint, [1, 2, 3], FootprintBin)
    parquet_events = read_events(synthetic_footprint, [1, 2, 3], FootprintParquet, areaperil_ids)
    for event_id, event in bin_events.items():
        expected = sorted_by_areaperil(event)
        expected = expected[np.isin(expected['areaperil_id'], areaperil_ids)]
        np.testing.assert_array_equal(parquet_events[event_id], expected)


def test_file_layout(synthetic_footprint):
    convert_bin_to_parquet_event(str(synthetic_footprint), row_group_size=64)
    metadata = pq.ParquetFile(synthetic_footprint / 'footprint.parquet' / 'event_id=1' / 'part-0.parquet').metadata
    assert metadata.num_rows == 600
    assert metadata.num_row_groups == 10  # 600 rows in groups of 64
    columns = {metadata.schema.column(i).name: metadata.row_group(0).column(i) for i in range(metadata.num_columns)}
    assert set(columns) == {'areaperil_id', 'intensity_bin_id', 'probability'}
    assert all(c.compression == 'ZSTD' and c.statistics.has_min_max for c in columns.values())
    assert 'DELTA_BINARY_PACKED' in columns['areaperil_id'].encodings
    assert 'BYTE_STREAM_SPLIT' in columns['probability'].encodings


def test_reconversion_replaces_previous_output(synthetic_footprint):
    convert_bin_to_parquet_event(str(synthetic_footprint), row_group_size=64)
    convert_bin_to_parquet_event(str(synthetic_footprint), row_group_size=128)
    event_dir = synthetic_footprint / 'footprint.parquet' / 'event_id=1'
    assert [p.name for p in event_dir.iterdir()] == ['part-0.parquet']
    assert pq.ParquetFile(event_dir / 'part-0.parquet').metadata.num_row_groups == 5
