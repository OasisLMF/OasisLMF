import sys
import msgpack
import struct
import csv

"""
Implementation of ktool items conversion tool including
complex item data serialized with msgpack.
"""


def items_to_csv(source, output):
    struct_fmt = 'IIII'
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    writer = csv.writer(output)
    writer.writerow(("item_id", "coverage_id", "model_data", "group_id"))

    while True:
        data = source.read(struct_len)
        if not data:
            break
        item_id, coverage_id, group_id, model_data_len = struct_unpack(data)

        # https://github.com/msgpack/msgpack-python#major-breaking-changes-in-msgpack-10
        if msgpack.version >= (1,0,0):
            model_data = msgpack.unpackb(source.read(model_data_len), raw=False)
            writer.writerow((item_id, coverage_id, model_data, group_id))
        else:
            model_data = msgpack.unpackb(source.read(model_data_len))
            writer.writerow((item_id, coverage_id, model_data.decode('utf-8'), group_id))


def main():
    PY3K = sys.version_info >= (3, 0)

    if PY3K:
        source = sys.stdin.buffer
    else:
        # Python 2 on Windows opens sys.stdin in text mode, and
        # binary data that read from it becomes corrupted on \r\n
        if sys.platform == "win32":
            # set sys.stdin to binary mode
            import os
            import msvcrt

            msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        source = sys.stdin

    items_to_csv(source, sys.stdout)


if __name__ == "__main__":
    main()
