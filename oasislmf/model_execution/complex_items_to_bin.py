#!/usr/bin/env python
"""Implementation of ktool items conversion tool including complex item data serialized with msgpack."""

import sys
import msgpack
import struct

import pandas as pd


def items_to_bin(source, output):
    items_df = pd.read_csv(source)
    for row in items_df.itertuples():
        # item_id,coverage_id,model_data,group_id
        packed_model_data = msgpack.packb(row.model_data)
        values = (
            int(row.item_id),
            int(row.coverage_id),
            int(float(row.group_id)),
            len(packed_model_data)
        )
        s = struct.Struct('IIII')
        packed_data = s.pack(*values)
        output.write(packed_data)
        output.write(packed_model_data)


def main():

    PY3K = sys.version_info >= (3, 0)

    if PY3K:
        output = sys.stdout.buffer
    else:
        # Python 2 on Windows opens sys.stdin in text mode, and
        # binary data that read from it becomes corrupted on \r\n
        if sys.platform == "win32":
            # set sys.stdin to binary mode
            import os
            import msvcrt
            msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        output = sys.stdout

    items_to_bin(sys.stdin, output)


if __name__ == "__main__":

    main()
