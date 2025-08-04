import msgpack
import numpy as np
import struct
from oasislmf.pytools.common.data import resolve_file
from oasislmf.pytools.converters.data import TOOL_INFO
import pandas as pd


def complex_items_tobin(stack, file_in, file_out, file_type):
    header_dtype = TOOL_INFO[file_type]["dtype"]

    file_in = resolve_file(file_in, "r", stack)

    s = struct.Struct('IIII')  # Matches dtype from complex_items_meta_output
    try:
        items_df = pd.read_csv(file_in)
    except pd.errors.EmptyDataError:
        np.empty(0, dtype=header_dtype).tofile(file_out)
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
        file_out.write(packed_data)
        file_out.write(packed_model_data)
