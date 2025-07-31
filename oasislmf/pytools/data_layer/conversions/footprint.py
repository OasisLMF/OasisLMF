import json
import os
import argparse
import shutil
from tqdm.auto import tqdm
from contextlib import ExitStack

import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from oasis_data_manager.filestore.backends.local import LocalStorage
from oasislmf.pytools.getmodel.footprint import Footprint


def convert_bin_to_parquet_event(static_path: str, **kwargs) -> None:
    """
    Converts the data from a binary file to a parquet file.

    Args:
        static_path: (str) the path to the static file

    Returns: None
    """
    root_path = os.path.join(static_path, 'footprint.parquet')
    try:
        shutil.rmtree(root_path)
    except FileNotFoundError:
        pass
    with ExitStack() as stack:
        storage = LocalStorage(
            root_dir=static_path,
            cache_dir=None,
        )
        footprint_obj = stack.enter_context(Footprint.load(storage,
                                                           ignore_file_type={'csv', 'parquet'}))
        index_data = footprint_obj.footprint_index

        meta_data = {
            "num_intensity_bins": footprint_obj.num_intensity_bins,
            "has_intensity_uncertainty": True if footprint_obj.has_intensity_uncertainty == 1 else False
        }

        for event_id in tqdm(index_data.keys(), desc="processing events"):
            data_slice = footprint_obj.get_event(event_id)
            df = pd.DataFrame(data_slice)
            df["event_id"] = event_id
            pq.write_to_dataset(
                pa.Table.from_pandas(df),
                root_path=root_path,
                partition_cols=['event_id'],
                compression="BROTLI"
            )
        with storage.open('footprint_parquet_meta.json', 'w') as outfile:
            json.dump(meta_data, outfile)


def convert_bin_to_parquet_chunk(static_path, chunk_size, **kwargs) -> None:
    """
    Converts the data from a binary file to a parquet file.

    Args:
        static_path: (str) the path to the static file
        chunk_size: target raw size of the partition

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
        for event_id in tqdm(index_data.keys(), desc="parsing index file"):
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
