import json
from contextlib import ExitStack

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oasislmf.pytools.getmodel.footprint import Footprint


def convert_bin_to_parquet(static_path):
    with ExitStack() as stack:
        footprint_obj = stack.enter_context(Footprint.load(static_path=static_path,
                                                           ignore_file_type={'z', 'csv', 'parquet'}))
        index_data = footprint_obj.footprint_index

        meta_data = {
            "num_intensity_bins": footprint_obj.num_intensity_bins,
            "has_intensity_uncertainty": True if footprint_obj.has_intensity_uncertainty == 1 else False
        }

        for event_id in index_data.keys():
            data_slice = footprint_obj.get_event(event_id)
            df = pd.DataFrame(data_slice)
            df["event_id"] = event_id
            pq.write_to_dataset(
                pa.Table.from_pandas(df),
                root_path=f'{static_path}/footprint.parquet',
                partition_cols=['event_id'],
                compression="gzip"
            )
        with open(f'{static_path}/footprint_parquet_meta.json', 'w') as outfile:
            json.dump(meta_data, outfile)
