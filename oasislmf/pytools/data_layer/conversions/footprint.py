import json
import os
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from contextlib import ExitStack

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from oasis_data_manager.filestore.backends.local import LocalStorage
from oasislmf.pytools.getmodel.footprint import Footprint


default_row_group_size = 16384

# Written so that FootprintParquet can skip most of an event when a portfolio only needs some of its
# areaperils: rows sorted by areaperil_id in small row groups, whose min/max statistics it uses to
# skip row groups. Delta encoding the sorted areaperil_ids also makes the files much smaller.
EVENT_PARQUET_OPTIONS = {
    'compression': 'zstd',
    'compression_level': 9,
    'use_dictionary': ['intensity_bin_id'],
    'column_encoding': {'areaperil_id': 'DELTA_BINARY_PACKED', 'probability': 'BYTE_STREAM_SPLIT'},
}


def open_bin_footprint(static_path):
    """Opens the bin or compressed bin footprint in static_path (use as a context manager)."""
    storage = LocalStorage(root_dir=static_path, cache_dir=None)
    return Footprint.load(storage, ignore_file_type={'csv', 'parquet'})


def write_parquet_events(static_path, event_ids, row_group_size):
    """Writes each event to footprint.parquet/event_id=<event_id>/part-0.parquet, sorted by areaperil_id."""
    root_path = os.path.join(static_path, 'footprint.parquet')
    with open_bin_footprint(static_path) as footprint_obj:
        for event_id in event_ids:
            event = footprint_obj.get_event(event_id)
            # stable, so the order of an areaperil's intensity bins is kept, and sorted events are unchanged
            event = event[np.argsort(event['areaperil_id'], kind='stable')]
            event_dir = os.path.join(root_path, f'event_id={event_id}')
            os.makedirs(event_dir, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(pd.DataFrame(event), preserve_index=False),
                           os.path.join(event_dir, 'part-0.parquet'),
                           row_group_size=row_group_size, **EVENT_PARQUET_OPTIONS)
    return len(event_ids)


def convert_bin_to_parquet_event(static_path: str, row_group_size=default_row_group_size, processes=1, **kwargs) -> None:
    """Converts the data from a binary file to a parquet file, one file per event.

    Args:
        static_path: (str) the path to the static file
        row_group_size: (int) rows per row group; smaller lets FootprintParquet skip more of an event
            for a portfolio that only needs some of its areaperils, at a small cost in file size
        processes: (int) number of processes converting events in parallel
        **kwargs: additional keyword arguments, accepted and ignored so that the conversion
            entrypoints can share one signature

    Returns: None
    """
    root_path = os.path.join(static_path, 'footprint.parquet')
    try:
        shutil.rmtree(root_path)
    except FileNotFoundError:
        pass
    with open_bin_footprint(static_path) as footprint_obj:
        event_ids = footprint_obj.footprint_index['event_id'].copy()
        meta_data = {
            "num_intensity_bins": footprint_obj.num_intensity_bins,
            "has_intensity_uncertainty": True if footprint_obj.has_intensity_uncertainty == 1 else False
        }

    # many more batches than processes, as event sizes can vary a lot
    batches = np.array_split(event_ids, min(len(event_ids), max(processes, 1) * 16)) if len(event_ids) else []
    with tqdm(total=len(event_ids), desc="processing events") as progress:
        if processes > 1:
            with ProcessPoolExecutor(processes) as pool:
                futures = [pool.submit(write_parquet_events, static_path, batch, row_group_size) for batch in batches]
                for future in as_completed(futures):
                    progress.update(future.result())
        else:
            for batch in batches:
                progress.update(write_parquet_events(static_path, batch, row_group_size))

    with open(os.path.join(static_path, 'footprint_parquet_meta.json'), 'w') as outfile:
        json.dump(meta_data, outfile)


def convert_bin_to_parquet_chunk(static_path, chunk_size, **kwargs) -> None:
    """Converts the data from a binary file to a parquet file.

    Args:
        static_path: (str) the path to the static file
        chunk_size: target raw size of the partition
        **kwargs: additional keyword arguments, accepted and ignored so that the conversion
            entrypoints can share one signature

    Returns: None
    """
    root_path = os.path.join(static_path, 'footprint_chunk')
    try:
        shutil.rmtree(root_path)
    except FileNotFoundError:
        pass
    os.mkdir(root_path)
    density = (0.0, 0)
    with ExitStack() as stack:
        storage = LocalStorage(
            root_dir=static_path,
            cache_dir=None,
        )
        footprint_obj = stack.enter_context(
            Footprint.load(storage, ignore_file_type={'csv', 'parquet'})
        )
        index_data = footprint_obj.footprint_index
        meta_data = {
            "num_intensity_bins": footprint_obj.num_intensity_bins,
            "has_intensity_uncertainty": True
            if footprint_obj.has_intensity_uncertainty == 1
            else False
        }

        event_data = []
        for event_id in tqdm(index_data['event_id'], desc="parsing index file"):
            data_slice = footprint_obj.get_event(event_id)
            df = pd.DataFrame(data_slice)
            min_areaperil_id = min(df['areaperil_id'])
            max_areaperil_id = max(df['areaperil_id'])
            event_data.append((min_areaperil_id, max_areaperil_id, event_id))
            if df.shape[0]:
                if max_areaperil_id == min_areaperil_id:
                    cur_density = 1
                else:
                    cur_density = df.shape[0] / (max_areaperil_id - min_areaperil_id)
                density = ((density[0] * density[1] + cur_density) / (density[1] + 1),
                           density[1] + 1)

        event_data.sort(key=lambda x: x[0])

        current_chunk = []
        current_size = 0
        count = 1
        footprint_lookup = []

        for min_apid, max_apid, event_id in tqdm(event_data, desc="processing chunks"):
            footprint_lookup.append({
                'event_id': event_id,
                'partition': count,
                'min_areaperil_id': min_apid,
                'max_areaperil_id': max_apid
            })  # size?

            data_slice = footprint_obj.get_event(event_id)
            df = pd.DataFrame(data_slice)
            df["event_id"] = event_id

            current_chunk.append(df)
            current_size += df.memory_usage(deep=True).sum()

            if (current_size < chunk_size):
                continue

            pq.write_table(
                pa.Table.from_pandas(
                    pd.concat(current_chunk, ignore_index=True)
                ),
                os.path.join(root_path, f'footprint_{count}.parquet'),
                compression="BROTLI"
            )

            current_chunk = []
            current_size = 0
            count += 1

        if current_chunk:
            pq.write_table(
                pa.Table.from_pandas(
                    pd.concat(current_chunk, ignore_index=True)
                ),
                os.path.join(root_path, f'footprint_{count}.parquet'),
                compression='BROTLI'
            )

        footprint_lookup_df = pd.DataFrame(footprint_lookup)

        footprint_lookup_df.to_parquet(f'{static_path}/footprint_lookup.parquet', index=False)
        # Writing to other file types min and max ids
        no_partition = footprint_lookup_df.drop(columns=['partition'])
        no_partition.to_pickle(f'{static_path}/footprint_lookup.bin')
        no_partition.to_csv(f'{static_path}/footprint_lookup.csv', index=False)

        with open(f'{static_path}/footprint_parquet_meta.json', 'w') as outfile:
            json.dump(meta_data, outfile)
    print(f"event area_peril_id density is {density[0]}")


default_chunk_size = 1024 * 1024 * 8

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--static-path', help='path to the folder containing the footprint files', default='.')
parser.add_argument('--chunk', help='if flag is set will create chunk footprint with the target raw size of chunk_size', action='store_true')
parser.add_argument('--chunk-size', help=f'target size for chunk parquet, default {default_chunk_size / 1024 * 1024}MB',
                    type=int, default=default_chunk_size)
parser.add_argument('--row-group-size', help=f'rows per row group for the per-event footprint, default {default_row_group_size}',
                    type=int, default=default_row_group_size)
parser.add_argument('--processes', help='number of processes converting the per-event footprint in parallel, default 1',
                    type=int, default=1)


def main():
    kwargs = vars(parser.parse_args())
    if kwargs['static_path'] == '.':
        kwargs['static_path'] = str(os.getcwd())

    if kwargs['chunk']:
        convert_bin_to_parquet_chunk(**kwargs)
    else:
        convert_bin_to_parquet_event(**kwargs)


if __name__ == '__main__':
    main()
