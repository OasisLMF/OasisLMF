import csv
import msgpack
import numpy as np
from oasislmf.pytools.common.event_stream import mv_read
from oasislmf.pytools.converters.data import TYPE_MAP


def complex_items_tocsv(file_in, file_out, file_type, noheader):
    header_dtype = TYPE_MAP[file_type]["dtype"]

    with open(file_in, "rb") as f:
        byte_data = np.frombuffer(f.read(), dtype=np.uint8)

    csv_out_file = open(file_out, "w")
    writer = csv.writer(csv_out_file)

    # Write header line manually
    if not noheader:
        writer.writerow(("item_id", "coverage_id", "model_data", "group_id"))

    cursor = 0
    while cursor < byte_data.size:
        header_record = np.zeros((), dtype=header_dtype)
        header_record["item_id"], cursor = mv_read(byte_data, cursor, header_dtype["item_id"], header_dtype["item_id"].itemsize)
        header_record["coverage_id"], cursor = mv_read(byte_data, cursor, header_dtype["coverage_id"], header_dtype["coverage_id"].itemsize)
        header_record["group_id"], cursor = mv_read(byte_data, cursor, header_dtype["group_id"], header_dtype["group_id"].itemsize)
        header_record["model_data_len"], cursor = mv_read(byte_data, cursor, header_dtype["model_data_len"], header_dtype["model_data_len"].itemsize)

        model_data_len = header_record["model_data_len"]
        model_data_bytes = byte_data[cursor:cursor + model_data_len].tobytes()
        cursor += model_data_len

        # Unpack msgpack
        if msgpack.version >= (1, 0, 0):
            model_data = msgpack.unpackb(model_data_bytes, raw=False)
        else:
            model_data = msgpack.unpackb(model_data_bytes)
            if isinstance(model_data, bytes):
                model_data = model_data.decode("utf-8")

        writer.writerow((
            header_record["item_id"],
            header_record["coverage_id"],
            model_data,
            header_record["group_id"]
        ))

    csv_out_file.close()
