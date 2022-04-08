#!/usr/bin/env python

import sys
import argparse
from contextlib import ExitStack

from oasislmf.pytools.gul.manager import read_getmodel_stream

parser = argparse.ArgumentParser(
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
)

parser.add_argument('-s', help='skip header (default: False).', default=False, action='store_true', dest='skip_header')
parser.add_argument('--run-dir', help='path to the run directory (default: ".")', default='.')


def print_cdftocsv(event_id, damagecdf, Nbins, rec):
    """Print the cdf produced by getmodel to csv file.
    Note that the input arrays are lists of cdf entries, namely
    the shape on axis=0 is the number of entries.

    Args:
        event_id (int): event_id
        damagecdf (array-like, damagecdf): damage cdf record
        Nbins (array-like, int): number of damage bins
        rec (array-like, rec): cdf record

    Returns:
        list[str]: list of csv lines
    """
    # TODO: accelerate this with numba when it will support string formatting

    # number of cdf entries in the input data
    Nentries = damagecdf.shape[0]

    # build the csv lines
    csv_lines = []
    for i in range(Nentries):
        csv_line_fixed = f"{event_id},{damagecdf[i]['areaperil_id']},{damagecdf[i]['vulnerability_id']},"
        for j in range(Nbins[i]):
            # note that bin index starts from 1 in the csv
            csv_lines.append(csv_line_fixed + f"{j + 1},{rec[i, j]['prob_to']:8.6f},{rec[i, j]['bin_mean']:8.6f}\n")

    return csv_lines


def run(run_dir, skip_header, file_in=None):
    """Run cdftocsv command: convert the cdf output from getmodel into csv format.
    The binary data is read from an input stream, and the csv file is streamed to stdout.

    Args:
        run_dir ([str]): Path to the run directory.

        skip_header ([bool]): If True, does not print the csv header.

    Raises:
        ValueError: If the stream type is not 1.

    """
    with ExitStack() as stack:
        if file_in is None:
            streams_in = sys.stdin.buffer
        else:
            streams_in = stack.enter_context(open(file_in, 'rb'))

        stream_out = sys.stdout

    if not skip_header:
        stream_out.write("event_id,areaperil_id,vulnerability_id,bin_index,prob_to,bin_mean\n")

    # TODO: try using np.savetxt for better performance
    for event_id, damagecdf, Nbins, rec in read_getmodel_stream(run_dir, streams_in):
        lines = print_cdftocsv(event_id, damagecdf, Nbins, rec)
        stream_out.writelines(lines)


def main():
    kwargs = vars(parser.parse_args())

    run(**kwargs)


if __name__ == '__main__':
    main()
