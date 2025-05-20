#!/usr/bin/env python

import argparse
import logging
import numpy as np
from pathlib import Path

from . import logger
from oasislmf.pytools.converters.data import SUPPORTED_TYPES


def read_csv_as_ndarray(file_in, type):
    headers = SUPPORTED_TYPES[type]["headers"]
    dtype = SUPPORTED_TYPES[type]["dtype"]
    with open(file_in, "r") as fin:
        first_line = fin.readline()
        first_line_elements = [header.strip() for header in first_line.strip().split(',')]
        has_header = first_line_elements == headers
        fin.seek(0)

        data = np.genfromtxt(
            fin,
            delimiter=',',
            dtype=dtype,
            skip_header=1 if has_header else 0
        )
    return data


def amplifications_tobin(file_in, file_out, type):
    data = read_csv_as_ndarray(file_in, type)

    # Check item IDs start from 1 and are contiguous
    if data["item_id"][0] != 1:
        raise ValueError(f'First item ID is {data["item_id"][0]}. Expected 1.')
    if not np.all(data["item_id"][1:] - data["item_id"][:-1] == 1):
        raise ValueError(f'Item IDs in {file_in} are not contiguous')

    with open(file_out, "wb") as fout:
        # Write the 4-byte zero header
        np.array([0], dtype="i4").tofile(fout)
        data.tofile(fout)


def coverages_tobin(file_in, file_out, type):
    data = read_csv_as_ndarray(file_in, type)

    with open(file_out, "wb") as fout:
        data["tiv"].tofile(fout)


def default_tobin(file_in, file_out, type):
    data = read_csv_as_ndarray(file_in, type)
    data.tofile(file_out)


def csvtobin(file_in, file_out, type):
    """Convert csv file to bin file based on file type
    Args:
        file_in (str | os.PathLike): Input file path
        file_out (str | os.PathLike): Output file path
        type (str): File type str from SUPPORTED_TYPES
    """
    tobin_func = default_tobin
    if type == "amplifications":
        tobin_func = amplifications_tobin
    elif type == "coverages":
        tobin_func = coverages_tobin

    tobin_func(file_in, file_out, type)


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to Binary for various file types.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--file_in', type=str, required=True, help='Input file path')
    parser.add_argument('-o', '--file_out', type=str, required=True, help='Output file path')
    parser.add_argument('-t', '--type', type=str, required=True, choices=SUPPORTED_TYPES.keys(),
                        help='Type of file to convert. Must be one of:\n' + '\n'.join(f'  - {key}' for key in SUPPORTED_TYPES.keys()))
    parser.add_argument('-v', '--logging-level', type=int, default=30,
                        help='logging level (debug:10, info:20, warning:30, error:40, critical:50)')
    args = parser.parse_args()

    file_in = Path(args.file_in)
    file_out = Path(args.file_out)
    if args.file_in != "-" and file_in.suffix != '.csv':
        raise ValueError(f"Invalid file extension for CSV, expected .csv, got {file_in},")
    if args.file_out != "-" and file_out.suffix != '.bin':
        raise ValueError(f"Invalid file extension for Binary, expected .bin, got {file_out},")
    kwargs = vars(args)

    # Set up logging
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    csvtobin(args.file_in, args.file_out, args.type)


if __name__ == '__main__':
    main()
