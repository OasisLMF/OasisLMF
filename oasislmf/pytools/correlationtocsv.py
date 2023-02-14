#!/usr/bin/env python

import argparse
import logging
from logging import NullHandler

from oasislmf import __version__ as oasis_version
from oasislmf.pytools.data_layer.conversions.correlations import \
    convert_bin_to_csv

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


def run(file_in_path, file_out_path):
    convert_bin_to_csv(file_in_path, file_out_path)


parser = argparse.ArgumentParser(
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
)

# arguments in alphabetical order (lower-case, then upper-case, then long arguments)
parser.add_argument('-i', action='store', type=str, dest='file_in_path', default="")
parser.add_argument('-o', action='store', type=str, dest='file_out_path', default="")
parser.add_argument('--logging-level',
                    help='logging level (debug:10, info:20, warning:30, error:40, critical:50). Default: 30',
                    default=30, type=int)
parser.add_argument('-V', '--version', action='version', version='{}'.format(oasis_version))


def main():
    # parse arguments to variables
    # note: the long flag name (e.g., '--opt-one') is used as variable name (i.e, the `dest`).
    # hyphens in the long flag name are parsed to underscores, e.g. '--opt-one' is stored in `opt_one``
    kwargs = vars(parser.parse_args())

    # add handler to gul logger
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    run(**kwargs)


if __name__ == '__main__':
    main()
