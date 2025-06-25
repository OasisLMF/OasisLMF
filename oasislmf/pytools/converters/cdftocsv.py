#!/usr/bin/env python

import argparse
from contextlib import ExitStack
import logging
import numpy as np
from pathlib import Path
import sys

from . import logger
from oasislmf.pytools.common.data import generate_output_metadata, write_ndarray_to_fmt_csv
from oasislmf.pytools.common.input_files import read_coverages
from oasislmf.pytools.getmodel.manager import Item
from oasislmf.pytools.gul.common import coverage_type
from oasislmf.pytools.gul.manager import generate_item_map, gul_get_items, read_getmodel_stream


cdf_output = [
    ("event_id", 'i4', "%d"),
    ("areaperil_id", 'i4', "%d"),
    ("vulnerability_id", 'i4', "%d"),
    ("bin_index", 'i4', "%d"),
    ("prob_to", 'f4', "%f"),
    ("bin_mean", 'f4', "%f"),
]
cdf_headers, cdf_dtype, cdf_fmt = generate_output_metadata(cdf_output)


def get_cdf_data(event_id, damagecdfrecs, recs, rec_idx_ptr):
    """Get the cdf data produced by getmodel.
    Note that the input arrays are lists of cdf entries, namely
    the shape on axis=0 is the number of entries.
    Args:
        event_id (int): event_id
        damagecdfrecs (ndarray[damagecdfrec]): cdf record keys
        recs (ndarray[ProbMean]): cdf record values
        rec_idx_ptr (ndarray[int]): array with the indices of `rec` where each cdf record starts.
    Returns:
        data (ndarray[cdf_dtype]): cdf data extracted from recs/getmodel.
    """
    assert len(damagecdfrecs) == len(rec_idx_ptr) - 1, "Number of cdfrecs groups does not match number of cdf keys found"

    data = np.zeros(len(recs), dtype=cdf_dtype)
    Nbins = len(rec_idx_ptr) - 1
    idx = 0
    for group_idx in range(Nbins):
        areaperil_id, vulnerability_id = damagecdfrecs[group_idx]
        for bin_index, rec in enumerate(recs[rec_idx_ptr[group_idx]:rec_idx_ptr[group_idx + 1]]):
            data[idx]["event_id"] = event_id
            data[idx]["areaperil_id"] = areaperil_id
            data[idx]["vulnerability_id"] = vulnerability_id
            data[idx]["bin_index"] = bin_index + 1
            data[idx]["prob_to"] = rec["prob_to"]
            data[idx]["bin_mean"] = rec["bin_mean"]
            idx += 1
    return data


def cdftocsv(file_in, file_out, run_dir, noheader=False):
    """Run cdftocsv command: convert the cdf output from getmodel into csv format.
    The binary data is read from an input stream, and the csv file is streamed to stdout.
    Args:
        file_in (str | os.PathLike): Input file path
        file_out (str | os.PathLike): Output file path
        run_dir (str | os.PathLike): Run directory containing input files
        noheader (bool): Bool to not output header. Defaults to False.
    """

    with ExitStack() as stack:
        input_path = Path(run_dir, 'input')
        if file_in == "-":
            streams_in = sys.stdin.buffer
        else:
            streams_in = stack.enter_context(open(file_in, "rb"))

        if file_out == "-":
            stream_out = sys.stdout
        else:
            stream_out = stack.enter_context(open(file_out, "w"))

        if not noheader:
            stream_out.write(",".join(cdf_headers) + "\n")

        items = gul_get_items(input_path)
        coverages_tiv = read_coverages(input_path)
        coverages = np.zeros(coverages_tiv.shape[0] + 1, coverage_type)
        coverages[1:]['tiv'] = coverages_tiv
        item_map = generate_item_map(items, coverages)
        del coverages_tiv

        compute = np.zeros(coverages.shape[0] + 1, items.dtype['coverage_id'])
        seeds = np.zeros(len(np.unique(items['group_id'])), dtype=Item.dtype['group_id'])
        valid_area_peril_id = None

        for event_data in read_getmodel_stream(streams_in, item_map, coverages, compute, seeds, valid_area_peril_id):
            event_id, compute_i, items_data, damagecdfrecs, recs, rec_idx_ptr, rng_index = event_data
            data = get_cdf_data(event_id, damagecdfrecs, recs, rec_idx_ptr)
            write_ndarray_to_fmt_csv(stream_out, data, cdf_headers, cdf_fmt)


def main():
    parser = argparse.ArgumentParser(
        usage='use "%(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
    )
    parser.add_argument('-i', '--file_in', type=str, required=True, help='Input file path')
    parser.add_argument('-o', '--file_out', type=str, required=True, help='Output file path')
    parser.add_argument('--run-dir', help='path to the run directory (default: ".")', default='.')
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

    cdftocsv(**kwargs)


if __name__ == '__main__':
    main()
