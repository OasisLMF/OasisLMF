import msgpack
import numpy as np
import struct
from oasislmf.pytools.converters.data import TYPE_MAP
import pandas as pd


def complex_items_tobin(file_in, file_out, file_type):
    header_dtype = TYPE_MAP[file_type]["dtype"]
    with open(file_out, "wb") as output:
        s = struct.Struct('IIII')  # Matches dtype from complex_items_meta_output
        try:
            items_df = pd.read_csv(file_in)
        except pd.errors.EmptyDataError:
            np.empty(0, dtype=header_dtype).tofile(output)
            return
        for row in items_df.itertuples():
            # item_id,coverage_id,model_data,group_id
            packed_model_data = msgpack.packb(row.model_data)
            values = (
                int(row.item_id),
                int(row.coverage_id),
                int(float(row.group_id)),
                len(packed_model_data)
            )
            packed_data = s.pack(*values)
            output.write(packed_data)
            output.write(packed_model_data)
