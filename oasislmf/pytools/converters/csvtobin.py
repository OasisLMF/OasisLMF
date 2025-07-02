#!/usr/bin/env python

import argparse
import logging
import struct
import msgpack
import numpy as np
import pandas as pd
from pathlib import Path

from . import logger
from oasislmf.pytools.converters.data import SUPPORTED_CSVTOBIN, TYPE_MAP


def read_csv_as_ndarray(file_in, file_type):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    with open(file_in, "r") as fin:
        first_line = fin.readline()
        if not first_line.strip():
            return np.empty(0, dtype=dtype)
        first_line_elements = [header.strip() for header in first_line.strip().split(',')]
        has_header = first_line_elements == headers

    cvs_dtype = {key: col_dtype for key, (col_dtype, _) in dtype.fields.items()}
    try:
        df = pd.read_csv(file_in, delimiter=',', dtype=cvs_dtype, usecols=list(cvs_dtype.keys()))
    except pd.errors.EmptyDataError:
        return np.empty(0, dtype=dtype)

    data = np.empty(df.shape[0], dtype=dtype)
    for name in dtype.names:
        data[name] = df[name]
    return data


def amplifications_tobin(file_in, file_out, file_type):
    data = read_csv_as_ndarray(file_in, file_type)

    # Check item IDs start from 1 and are contiguous
    if len(data) > 0 and data["item_id"][0] != 1:
        raise ValueError(f'First item ID is {data["item_id"][0]}. Expected 1.')
    if len(data) > 0 and not np.all(data["item_id"][1:] - data["item_id"][:-1] == 1):
        raise ValueError(f'Item IDs in {file_in} are not contiguous')

    with open(file_out, "wb") as fout:
        # Write the 4-byte zero header
        np.array([0], dtype="i4").tofile(fout)
        data.tofile(fout)


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


def coverages_tobin(file_in, file_out, file_type):
    data = read_csv_as_ndarray(file_in, file_type)

    with open(file_out, "wb") as fout:
        data["tiv"].tofile(fout)


def returnperiods_tobin(file_in, file_out, file_type):
    data = read_csv_as_ndarray(file_in, file_type)
    data = np.sort(data, order="return_period")[::-1]
    data.tofile(file_out)


def default_tobin(file_in, file_out, file_type):
    data = read_csv_as_ndarray(file_in, file_type)
    data.tofile(file_out)


def csvtobin(file_in, file_out, file_type, **kwargs):
    """Convert csv file to bin file based on file type
    Args:
        file_in (str | os.PathLike): Input file path
        file_out (str | os.PathLike): Output file path
        file_type (str): File type str from SUPPORTED_CSVTOBIN
    """
    tobin_func = default_tobin
    if file_type == "amplifications":
        tobin_func = amplifications_tobin
    elif file_type == "complex_items":
        tobin_func = complex_items_tobin
    elif file_type == "coverages":
        tobin_func = coverages_tobin
    elif file_type == "returnperiods":
        tobin_func = returnperiods_tobin

    tobin_func(file_in, file_out, file_type)


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to Binary for various file types.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest='file_type', required=True, help='Type of file to convert')
    for file_type in SUPPORTED_CSVTOBIN:
        parser_curr = subparsers.add_parser(file_type, help=f'csv to bin tool for {file_type}')
        parser_curr.add_argument('-i', '--file_in', type=str, required=True, help='Input file path')
        parser_curr.add_argument('-o', '--file_out', type=str, required=True, help='Output file path')
        parser_curr.add_argument('-v', '--logging-level', type=int, default=30,
                                 help='logging level (debug:10, info:20, warning:30, error:40, critical:50)')
    args = parser.parse_args()
    kwargs = vars(args)

    file_type = kwargs.pop('file_type')
    file_in = Path(kwargs.pop('file_in'))
    file_out = Path(kwargs.pop('file_out'))
    if file_in != "-" and file_in.suffix != '.csv':
        raise ValueError(f"Invalid file extension for CSV, expected .csv, got {file_in},")
    if file_out != "-" and file_out.suffix != '.bin':
        raise ValueError(f"Invalid file extension for Binary, expected .bin, got {file_out},")

    # Set up logging
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    csvtobin(file_in, file_out, file_type, **kwargs)


if __name__ == '__main__':
    main()
