#!/usr/bin/env python

import sys
import os
from contextlib import ExitStack
import argparse
import logging
import numpy as np

from oasislmf.pytools.getmodel.manager import get_mean_damage_bins
from oasislmf.pytools.gul.common import ProbMean, damagecdfrec

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
)

parser.add_argument('-i', '--file-in', help='input filename', action='store', type=str)
parser.add_argument('-s', help='skip header (default: False).', default=False, action='store_true', dest='skip_header')
parser.add_argument('--run-dir', help='path to the run directory', default='.')
parser.add_argument('--logging-level',
                    help='logging level (debug:10, info:20, warning:30, error:40, critical:50)', default=30, type=int)


def run(run_dir, skip_header, file_in=None, file_out=None):
    """Run cdftocsv command: convert the cdf output from getmodel into csv format.
    Presently, the csv file is streamed to stdout.

    Args:
        run_dir ([str]): Path to the run directory.
        skip_header ([bool]): If True, does not print the csv header.
        file_in ([str], optional): If provided, it reads the cdf from that file. Defaults to None.
        file_out ([str], optional): Placeholder for a future output file writer. Defaults to None.

    Raises:
        NotImplementedError: If file_out is not None.
        ValueError: If the stream type is not 1.

    """
    with ExitStack() as stack:
        if file_in is None:
            streams_in = sys.stdin.buffer

        else:
            streams_in = stack.enter_context(open(file_in, 'rb'))

        if file_out is None:
            stream_out = sys.stdout
        else:
            # implement here a file-writer (e.g., csv writer)
            raise NotImplementedError("file_out has not been implemented yet.")

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
            stream_out.write("event_id,areaperil_id,vulnerability_id,bin_index,prob_to,bin_mean")

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
            len_read = streams_in.readinto1(damagecdf_mv)
            len_read = streams_in.readinto1(Nbins_mv)
            len_read = streams_in.readinto1(rec_mv[:Nbins[0] * ProbMean.size])

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

    # add handler to cdftocsv logger
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    run(**kwargs)


if __name__ == '__main__':
    main()
