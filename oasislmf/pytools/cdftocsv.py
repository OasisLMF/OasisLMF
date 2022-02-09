#!/usr/bin/env python

import sys
import os
from contextlib import ExitStack
import argparse
import numpy as np

from oasislmf.pytools.getmodel.manager import get_mean_damage_bins
from oasislmf.pytools.gul.common import ProbMean, damagecdfrec

parser = argparse.ArgumentParser(
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
)

parser.add_argument('-s', help='skip header (default: False).', default=False, action='store_true', dest='skip_header')
parser.add_argument('--run-dir', help='path to the run directory (default: ".")', default='.')


def run(run_dir, skip_header):
    """Run cdftocsv command: convert the cdf output from getmodel into csv format.
    The binary data is read from an input stream, and the csv file is streamed to stdout.

    Args:
        run_dir ([str]): Path to the run directory.

        skip_header ([bool]): If True, does not print the csv header.

    Raises:
        ValueError: If the stream type is not 1.

    """
    # set up the streams
    streams_in = sys.stdin.buffer
    stream_out = sys.stdout

    # get damage bins from file
    static_path = os.path.join(run_dir, 'static')
    damage_bins = get_mean_damage_bins(static_path=static_path)

    # maximum number of damage bins (individual items can have up to `total_bins` bins)
    if damage_bins.shape[0] == 0:
        total_bins = 1000
    else:
        total_bins = damage_bins.shape[0]

    # determine stream type
    stream_type = np.frombuffer(streams_in.read(4), dtype='i4')

    if stream_type[0] != 1:
        raise ValueError(f"FATAL: Invalid stream type: expect 1, got {stream_type[0]}.")

    if not skip_header:
        stream_out.write("event_id,areaperil_id,vulnerability_id,bin_index,prob_to,bin_mean\n")

    # prepare all the data buffers
    damagecdf_mv = memoryview(bytearray(damagecdfrec.size))
    damagecdf = np.ndarray(1, buffer=damagecdf_mv, dtype=damagecdfrec.dtype)
    Nbins_mv = memoryview(bytearray(4))
    Nbins = np.ndarray(1, buffer=Nbins_mv, dtype='i4')
    rec_mv = memoryview(bytearray(total_bins * ProbMean.size))
    rec = np.ndarray(total_bins, buffer=rec_mv, dtype=ProbMean.dtype)

    # start reading the stream
    # each record from getmodel is expected to contain:
    # 1 damagecdfrec obj, 1 int (Nbins), a number `Nbins` of ProbMean objects
    while True:
        len_read = streams_in.readinto(damagecdf_mv)
        len_read = streams_in.readinto(Nbins_mv)
        len_read = streams_in.readinto(rec_mv[:Nbins[0] * ProbMean.size])

        # exit if the stream has ended
        if len_read == 0:
            break

        # print out in csv format
        csv_line_fixed = ",".join([f"{x}" for x in [damagecdf['event_id'][0],
                                                    damagecdf['areaperil_id'][0],
                                                    damagecdf['vulnerability_id'][0]]]) + ","

        for i in range(Nbins[0]):
            csv_line = csv_line_fixed
            csv_line += f"{i+1},"     # bin index starts from 1
            csv_line += ",".join([f"{x:8.6f}" for x in [rec[i]["prob_to"], rec[i]["bin_mean"]]])

            stream_out.write(csv_line + "\n")

    return


def main():
    kwargs = vars(parser.parse_args())

    run(**kwargs)


if __name__ == '__main__':
    main()
