#!/usr/bin/env python

import argparse
import logging
import numpy as np
from pathlib import Path

from oasislmf.pytools.common.input_files import read_amplifications, read_coverages

from . import logger
from oasislmf.pytools.common.data import write_ndarray_to_fmt_csv
from oasislmf.pytools.converters.data import SUPPORTED_TYPES


def amplifications_tocsv(file_in, file_out, type, noheader):
    headers = SUPPORTED_TYPES[type]["headers"]
    dtype = SUPPORTED_TYPES[type]["dtype"]
    fmt = SUPPORTED_TYPES[type]["fmt"]

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


def coverages_tocsv(file_in, file_out, type, noheader):
    headers = SUPPORTED_TYPES[type]["headers"]
    dtype = SUPPORTED_TYPES[type]["dtype"]
    fmt = SUPPORTED_TYPES[type]["fmt"]

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


def default_tocsv(file_in, file_out, type, noheader):
    headers = SUPPORTED_TYPES[type]["headers"]
    dtype = SUPPORTED_TYPES[type]["dtype"]
    fmt = SUPPORTED_TYPES[type]["fmt"]

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


def bintocsv(file_in, file_out, type, noheader=False):
    """Convert bin file to csv file based on file type
    Args:
        file_in (str | os.PathLike): Input file path
        file_out (str | os.PathLike): Output file path
        type (str): File type str from SUPPORTED_TYPES
        noheader (bool): Bool to not output header. Defaults to False.
    """
    tocsv_func = default_tocsv
    if type == "amplifications":
        tocsv_func = amplifications_tocsv
    elif type == "coverages":
        tocsv_func = coverages_tocsv

    tocsv_func(file_in, file_out, type, noheader)


def main():
    parser = argparse.ArgumentParser(description='Convert Binary to CSV for various file types.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--file_in', type=str, required=True, help='Input file path')
    parser.add_argument('-o', '--file_out', type=str, required=True, help='Output file path')
    parser.add_argument('-t', '--type', type=str, required=True, choices=SUPPORTED_TYPES.keys(),
                        help='Type of file to convert. Must be one of:\n' + '\n'.join(f'  - {key}' for key in SUPPORTED_TYPES.keys()))
    parser.add_argument('-v', '--logging-level', type=int, default=30,
                        help='logging level (debug:10, info:20, warning:30, error:40, critical:50)')
    parser.add_argument('-H', '--noheader', action='store_true', help='Suppress header in output files')
    args = parser.parse_args()

    file_in = Path(args.file_in)
    file_out = Path(args.file_out)
    if args.file_in != "-" and file_in.suffix != '.bin':
        raise ValueError(f"Invalid file extension for Binary, expected .bin, got {file_in},")
    if args.file_out != "-" and file_out.suffix != '.csv':
        raise ValueError(f"Invalid file extension for CSV, expected .csv, got {file_out},")
    kwargs = vars(args)

    # Set up logging
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    bintocsv(args.file_in, args.file_out, args.type, args.noheader)


if __name__ == '__main__':
    main()
