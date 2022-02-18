#!/usr/bin/env python

import sys
import argparse

from oasislmf.pytools.gul.manager import read_stream

parser = argparse.ArgumentParser(
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
)

parser.add_argument('-s', help='skip header (default: False).', default=False, action='store_true', dest='skip_header')
parser.add_argument('--run-dir', help='path to the run directory (default: ".")', default='.')


def print_cdftocsv(damagecdf, Nbins, rec):
    # print out in csv format

    csv_line_fixed = "{}".format(damagecdf['event_id'][0]) + ","
    csv_line_fixed += "{}".format(damagecdf['areaperil_id'][0]) + ","
    csv_line_fixed += "{}".format(damagecdf['vulnerability_id'][0]) + ","

    for i in range(Nbins[0]):
        csv_line = csv_line_fixed
        csv_line += "{}".format(i + 1) + ","   # bin index starts from 1
        csv_line += "{:8.6f},{:8.6f}".format(rec[i]["prob_to"], rec[i]["bin_mean"])

        yield csv_line


def run(run_dir, skip_header):
    """Run cdftocsv command: convert the cdf output from getmodel into csv format.
    The binary data is read from an input stream, and the csv file is streamed to stdout.

    Args:
        run_dir ([str]): Path to the run directory.

        skip_header ([bool]): If True, does not print the csv header.

    Raises:
        ValueError: If the stream type is not 1.

    """
    stream_out = sys.stdout

    if not skip_header:
        stream_out.write("event_id,areaperil_id,vulnerability_id,bin_index,prob_to,bin_mean\n")

    for damagecdf, Nbins, rec in read_stream(run_dir):
        for line in print_cdftocsv(damagecdf, Nbins, rec):
            stream_out.write(line + "\n")


def main():
    kwargs = vars(parser.parse_args())

    run(**kwargs)


if __name__ == '__main__':
    main()
