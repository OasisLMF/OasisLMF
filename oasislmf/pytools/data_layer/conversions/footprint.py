import json
import os
from contextlib import ExitStack

import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from oasis_data_manager.filestore.backends.local import LocalStorage
from oasislmf.pytools.getmodel.footprint import Footprint


def convert_bin_to_parquet(
        static_path: str,
        chunk_size=1024 * 1024 * 8
) -> None:
    """
    Converts the data from a binary file to a parquet file.

    Args:
        static_path: (str) the path to the static file

    Returns: None
    """
    with ExitStack() as stack:
        storage = LocalStorage(
            root_dir=static_path,
            cache_dir=None,
        )
        footprint_obj = stack.enter_context(
            Footprint.load(storage, ignore_file_type={'z', 'csv', 'parquet'})
        )
        index_data = footprint_obj.footprint_index

        meta_data = {
            "num_intensity_bins": footprint_obj.num_intensity_bins,
            "has_intensity_uncertainty": True
            if footprint_obj.has_intensity_uncertainty == 1
            else False
        }

        event_data = []
        for event_id in index_data.keys():
            data_slice = footprint_obj.get_event(event_id)
            df = pd.DataFrame(data_slice)
            min_areaperil_id = min(df['areaperil_id'])
            max_areaperil_id = max(df['areaperil_id'])
            event_data.append((min_areaperil_id, max_areaperil_id, event_id))

        event_data.sort(key=lambda x: x[0])

        current_chunk = []
        current_size = 0
        count = 1
        footprint_lookup = []

        for min_apid, max_apid, event_id in event_data:
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
                f'{static_path}/footprint_{count}.parquet',
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
                f'{static_path}/footprint_{count}.parquet',
                compression='BROTLI'
            )

        footprint_lookup_df = pd.DataFrame(footprint_lookup)

        footprint_lookup_df.to_parquet(
            f'{static_path}/footprint_lookup.parquet', index=False
        )

        with storage.open('footprint_parquet_meta.json', 'w') as outfile:
            json.dump(meta_data, outfile)


def main() -> None:
    """
    The entrypoint for convertbintoparquet command converting the f
    footprint.bin file in the current directory to a
    footprint.parquet file.

    Returns: None
    """
    convert_bin_to_parquet(static_path=str(os.getcwd()))
