"""FootprintParquet reads only the row groups that may hold the requested areaperil_ids, and
returns exactly the rows the previous "in" filter read returned."""
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from oasis_data_manager.filestore.backends.local import LocalStorage

from oasislmf.pytools.getmodel.common import Event_dtype
from oasislmf.pytools.getmodel.footprint import FootprintParquet, select_row_groups


def make_event(areaperil_ids, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'areaperil_id': np.asarray(areaperil_ids, dtype=Event_dtype['areaperil_id']),
        'intensity_bin_id': rng.integers(1, 50, len(areaperil_ids)).astype(np.int32),
        'probability': rng.random(len(areaperil_ids)).astype(Event_dtype['probability']),
    })


def write_footprint(root, events, row_group_size=10, write_statistics=True):
    """events: {event_id: DataFrame, or a list of DataFrames to write as separate part files}"""
    with open(os.path.join(root, 'footprint_parquet_meta.json'), 'w') as f:
        json.dump({'num_intensity_bins': 50, 'has_intensity_uncertainty': True}, f)
    for event_id, parts in events.items():
        event_dir = os.path.join(root, 'footprint.parquet', f'event_id={event_id}')
        os.makedirs(event_dir)
        for i, df in enumerate(parts if isinstance(parts, list) else [parts]):
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), os.path.join(event_dir, f'part-{i}.parquet'),
                           row_group_size=row_group_size, write_statistics=write_statistics)


def previous_read(root, event_id, areaperil_ids):
    """What FootprintParquet.get_event returned before row group selection: an "in" filtered read."""
    df = pd.read_parquet(os.path.join(root, 'footprint.parquet', f'event_id={event_id}'),
                         filters=[('areaperil_id', 'in', list(areaperil_ids))])
    return FootprintParquet.prepare_df_data(df)


def get_event(root, event_id, areaperil_ids=None):
    with FootprintParquet(LocalStorage(root_dir=str(root), cache_dir=None), areaperil_ids=areaperil_ids) as footprint:
        return footprint.get_event(event_id)


AREAPERIL_IDS = [5, 6, 47, 120, 121, 199]


@pytest.mark.parametrize('ids', [AREAPERIL_IDS, [1], [200], [1000], list(range(1, 201))])
def test_sorted_event_matches_previous_read(tmp_path, ids):
    write_footprint(tmp_path, {1: make_event(np.arange(1, 201))})
    np.testing.assert_array_equal(get_event(tmp_path, 1, ids), previous_read(tmp_path, 1, ids))


def test_sorted_event_skips_row_groups(tmp_path):
    write_footprint(tmp_path, {1: make_event(np.arange(1, 201))})  # 20 row groups of 10 ids
    metadata = pq.ParquetFile(tmp_path / 'footprint.parquet' / 'event_id=1' / 'part-0.parquet').metadata
    assert select_row_groups(metadata, np.array(AREAPERIL_IDS)) == [0, 4, 11, 12, 19]


def test_unsorted_event_matches_previous_read(tmp_path):
    ids = np.random.default_rng(1).permutation(np.arange(1, 201))
    write_footprint(tmp_path, {1: make_event(ids)})
    np.testing.assert_array_equal(get_event(tmp_path, 1, AREAPERIL_IDS), previous_read(tmp_path, 1, AREAPERIL_IDS))


def test_intensity_uncertainty_keeps_all_rows_of_an_areaperil(tmp_path):
    # several intensity bins per areaperil, some split across row groups
    write_footprint(tmp_path, {1: make_event(np.repeat(np.arange(1, 51), 3))}, row_group_size=7)
    result = get_event(tmp_path, 1, [3, 30])
    assert len(result) == 6
    np.testing.assert_array_equal(result, previous_read(tmp_path, 1, [3, 30]))


def test_without_statistics_reads_every_row_group(tmp_path):
    write_footprint(tmp_path, {1: make_event(np.arange(1, 201))}, write_statistics=False)
    metadata = pq.ParquetFile(tmp_path / 'footprint.parquet' / 'event_id=1' / 'part-0.parquet').metadata
    assert select_row_groups(metadata, np.array(AREAPERIL_IDS)) == list(range(20))
    np.testing.assert_array_equal(get_event(tmp_path, 1, AREAPERIL_IDS), previous_read(tmp_path, 1, AREAPERIL_IDS))


def test_several_part_files(tmp_path):
    write_footprint(tmp_path, {1: [make_event(np.arange(1, 101), seed=1), make_event(np.arange(101, 201), seed=2)]})
    np.testing.assert_array_equal(get_event(tmp_path, 1, AREAPERIL_IDS), previous_read(tmp_path, 1, AREAPERIL_IDS))


def test_no_areaperil_ids_returns_whole_event(tmp_path):
    event = make_event(np.arange(1, 201))
    write_footprint(tmp_path, {1: event})
    np.testing.assert_array_equal(get_event(tmp_path, 1), FootprintParquet.prepare_df_data(event))


def test_no_matching_areaperil_ids(tmp_path):
    write_footprint(tmp_path, {1: make_event(np.arange(1, 201))})
    result = get_event(tmp_path, 1, [500, 600])
    assert len(result) == 0 and result.dtype == Event_dtype


def test_missing_event(tmp_path):
    write_footprint(tmp_path, {1: make_event(np.arange(1, 201))})
    result = get_event(tmp_path, 2, AREAPERIL_IDS)
    assert len(result) == 0 and result.dtype == Event_dtype
