#!/usr/bin/env python

import argparse
import csv
import logging
import msgpack
import numpy as np
from pathlib import Path

from oasislmf.pytools.common.event_stream import mv_read
from oasislmf.pytools.common.input_files import read_amplifications, read_coverages

from . import logger
from oasislmf.pytools.common.data import write_ndarray_to_fmt_csv
from oasislmf.pytools.converters.data import SUPPORTED_BINTOCSV, TYPE_MAP


def amplifications_tocsv(file_in, file_out, file_type, noheader):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    fmt = TYPE_MAP[file_type]["fmt"]

    amps_fp = Path(file_in)
    items_amps = read_amplifications(amps_fp.parent, amps_fp.name)
    items_amps = items_amps[1:]
    data = np.zeros(len(items_amps), dtype=dtype)
    data["item_id"] = np.arange(1, len(items_amps) + 1)
    data["amplification_id"] = items_amps

    csv_out_file = open(file_out, "w")
    if not noheader:
        csv_out_file.write(",".join(headers) + "\n")
    write_ndarray_to_fmt_csv(csv_out_file, data, headers, fmt)
    csv_out_file.close()


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


def coverages_tocsv(file_in, file_out, file_type, noheader):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    fmt = TYPE_MAP[file_type]["fmt"]

    cov_fp = Path(file_in)
    coverages = read_coverages(cov_fp.parent, filename=cov_fp.name)
    data = np.zeros(len(coverages), dtype=dtype)
    data["coverage_id"] = np.arange(1, len(coverages) + 1)
    data["tiv"] = coverages

    csv_out_file = open(file_out, "w")
    if not noheader:
        csv_out_file.write(",".join(headers) + "\n")
    write_ndarray_to_fmt_csv(csv_out_file, data, headers, fmt)
    csv_out_file.close()


def default_tocsv(file_in, file_out, file_type, noheader):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    fmt = TYPE_MAP[file_type]["fmt"]

    data = np.memmap(file_in, dtype=dtype)
    num_rows = data.shape[0]

    csv_out_file = open(file_out, "w")
    if not noheader:
        csv_out_file.write(",".join(headers) + "\n")

    buffer_size = 1000000
    for start in range(0, num_rows, buffer_size):
        end = min(start + buffer_size, num_rows)
        buffer_data = data[start:end]
        write_ndarray_to_fmt_csv(csv_out_file, buffer_data, headers, fmt)
    csv_out_file.close()


def bintocsv(file_in, file_out, file_type, noheader=False, **kwargs):
    """Convert bin file to csv file based on file type
    Args:
        file_in (str | os.PathLike): Input file path
        file_out (str | os.PathLike): Output file path
        file_type (str): File type str from SUPPORTED_BINTOCSV
        noheader (bool): Bool to not output header. Defaults to False.
    """
    tocsv_func = default_tocsv
    if file_type == "amplifications":
        tocsv_func = amplifications_tocsv
    elif file_type == "complex_items":
        tocsv_func = complex_items_tocsv
    elif file_type == "coverages":
        tocsv_func = coverages_tocsv

    tocsv_func(file_in, file_out, file_type, noheader)


def main():
    parser = argparse.ArgumentParser(description='Convert Binary to CSV for various file types.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest='file_type', required=True, help='Type of file to convert')
    for file_type in SUPPORTED_BINTOCSV:
        parser_curr = subparsers.add_parser(file_type, help=f'bin to csv tool for {file_type}')
        parser_curr.add_argument('-i', '--file_in', type=str, required=True, help='Input file path')
        parser_curr.add_argument('-o', '--file_out', type=str, required=True, help='Output file path')
        parser_curr.add_argument('-v', '--logging-level', type=int, default=30,
                                 help='logging level (debug:10, info:20, warning:30, error:40, critical:50)')
        parser_curr.add_argument('-H', '--noheader', action='store_true', help='Suppress header in output files')
    args = parser.parse_args()
    kwargs = vars(args)

    file_type = kwargs.pop('file_type')
    file_in = Path(kwargs.pop('file_in'))
    file_out = Path(kwargs.pop('file_out'))
    noheader = kwargs.pop('noheader')
    if file_in != "-" and file_in.suffix != '.bin':
        raise ValueError(f"Invalid file extension for Binary, expected .bin, got {file_in},")
    if file_out != "-" and file_out.suffix != '.csv':
        raise ValueError(f"Invalid file extension for CSV, expected .csv, got {file_out},")

    # Set up logging
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    bintocsv(file_in, file_out, file_type, noheader, **kwargs)


if __name__ == '__main__':
    main()
