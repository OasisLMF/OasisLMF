#!/usr/bin/env python

import argparse
import logging
import struct
import msgpack
import numba as nb
import numpy as np
import pandas as pd
from pathlib import Path

from . import logger
from oasislmf.pytools.common.data import generate_output_metadata, occurrence_dtype, occurrence_granular_dtype
from oasislmf.pytools.common.input_files import occ_get_date_id
from oasislmf.pytools.converters.data import SUPPORTED_CSVTOBIN, TYPE_MAP


def read_csv_as_ndarray(file_in, headers, dtype):
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
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    data = read_csv_as_ndarray(file_in, headers, dtype)

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
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    data = read_csv_as_ndarray(file_in, headers, dtype)

    with open(file_out, "wb") as fout:
        data["tiv"].tofile(fout)


def returnperiods_tobin(file_in, file_out, file_type):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    data = read_csv_as_ndarray(file_in, headers, dtype)
    data = np.sort(data, order="return_period")[::-1]
    data.tofile(file_out)


def occurrence_tobin(file_in, file_out, file_type, no_of_periods, no_date_alg=False, granular=False):
    @nb.njit(cache=True, error_model="numpy")
    def _get_occ_data_with_date_ids(occ_csv, occ_dtype):
        buffer_size = 1000000
        buffer = np.zeros(buffer_size, dtype=occ_dtype)

        idx = 0
        for row in occ_csv:
            if idx >= buffer_size:
                yield buffer[:idx]
                buffer = np.zeros(buffer_size, dtype=occ_dtype)
                idx = 0
            occ_date_id = occ_get_date_id(
                False,
                row["occ_year"],
                row["occ_month"],
                row["occ_day"],
            )
            buffer[idx]["event_id"] = row["event_id"]
            buffer[idx]["period_no"] = row["period_no"]
            buffer[idx]["occ_date_id"] = occ_date_id
            idx += 1
        yield buffer[:idx]

    @nb.njit(cache=True, error_model="numpy")
    def _get_occ_data_with_date_ids_gran(occ_csv, occ_dtype):
        buffer_size = 1000000
        buffer = np.zeros(buffer_size, dtype=occ_dtype)

        idx = 0
        for row in occ_csv:
            if idx >= buffer_size:
                yield buffer[:idx]
                buffer = np.zeros(buffer_size, dtype=occ_dtype)
                idx = 0
            occ_date_id = occ_get_date_id(
                True,
                row["occ_year"],
                row["occ_month"],
                row["occ_day"],
                row["occ_hour"],
                row["occ_minute"],
            )
            buffer[idx]["event_id"] = row["event_id"]
            buffer[idx]["period_no"] = row["period_no"]
            buffer[idx]["occ_date_id"] = occ_date_id
            idx += 1
        yield buffer[:idx]

    if no_date_alg and granular:
        raise RuntimeError("Cannot have an occurrence file with granular dates and no date algorithm. Use at most one of -D, -G, but not both")

    with open(file_out, "wb") as fout:
        # Write date opts
        date_opts = granular << 1 | (not no_date_alg)
        np.array([date_opts], dtype="i4").tofile(fout)
        np.array([no_of_periods], dtype="i4").tofile(fout)

        headers = TYPE_MAP[file_type]["headers"]
        dtype = TYPE_MAP[file_type]["dtype"]
        if no_date_alg:
            csv_data = read_csv_as_ndarray(file_in, headers, dtype)
            csv_data.tofile(fout)
        else:
            occ_csv_output = [
                ("event_id", 'i4', "%d"),
                ("period_no", 'i4', "%d"),
                ("occ_year", 'i4', "%d"),
                ("occ_month", 'i4', "%d"),
                ("occ_day", 'i4', "%d"),
            ]
            if granular:
                occ_csv_output += [
                    ("occ_hour", 'i4', "%d"),
                    ("occ_minute", 'i4', "%d"),
                ]
            headers, dtype, fmt = generate_output_metadata(occ_csv_output)
            csv_data = read_csv_as_ndarray(file_in, headers, dtype)
            gen = _get_occ_data_with_date_ids(csv_data, occurrence_dtype)
            if granular:
                gen = _get_occ_data_with_date_ids_gran(csv_data, occurrence_granular_dtype)

            for data in gen:
                if any(data["period_no"] > no_of_periods):
                    raise RuntimeError("FATAL: Period number exceeds maximum supplied")
                data.tofile(fout)


def default_tobin(file_in, file_out, file_type):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    data = read_csv_as_ndarray(file_in, headers, dtype)
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
    elif file_type == "occurrence":
        tobin_func = occurrence_tobin
    elif file_type == "returnperiods":
        tobin_func = returnperiods_tobin

    tobin_func(file_in, file_out, file_type, **kwargs)


def add_custom_args(file_type, parser):
    if file_type == "occurrence":
        parser.add_argument('-D', '--no_date_alg', action='store_true', help='No date algorithm in csv (use occ_date_id directly)')
        parser.add_argument('-G', '--granular', action='store_true', help='Use granular dates (occ_hour and occ_minute)')
        parser.add_argument('-P', '--no_of_periods', type=int, required=True, help='Number of periods')


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to Binary for various file types.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest='file_type', required=True, help='Type of file to convert')
    for file_type in SUPPORTED_CSVTOBIN:
        parser_curr = subparsers.add_parser(file_type, help=f'csv to bin tool for {file_type}')
        add_custom_args(file_type, parser_curr)
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
