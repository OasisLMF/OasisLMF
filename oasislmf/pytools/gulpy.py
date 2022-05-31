#!/usr/bin/env python

import argparse
import logging

from oasislmf import __version__ as oasis_version
from oasislmf.pytools.gul import manager, logger

parser = argparse.ArgumentParser(
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
)

parser.add_argument('-a', help='back-allocation rule', default=0, type=int, dest='alloc_rule')
parser.add_argument('-d', help='output random numbers instead of gul (default: False).',
                    default=False, action='store_true', dest='debug')
parser.add_argument('-i', '--file-in', help='filename of input stream.', action='store', type=str, dest='file_in')
parser.add_argument('-o', '--file-out', help='filename of output stream.', action='store', type=str, dest='file_out')
parser.add_argument('-L', help='Loss treshold (default: 1e-6)', default=1e-6,
                    action='store', type=float, dest='loss_threshold')
parser.add_argument('-S', help='Sample size (default: 0).', default=0, action='store', type=int, dest='sample_size')
parser.add_argument('-V', '--version', action='version', version='{}'.format(oasis_version))
parser.add_argument('--ignore-file-type', nargs='*', help='the type of file to be loaded', default=set())
parser.add_argument('--random-generator',
                    help='random number generator\n0: numpy default (MT19937), 1: Latin Hypercube. Default: 1.',
                    default=1, type=int)
parser.add_argument('--run-dir', help='path to the run directory', default='.')
parser.add_argument('--logging-level',
                    help='logging level (debug:10, info:20, warning:30, error:40, critical:50). Default: 30.',
                    default=30, type=int)


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

    manager.run(**kwargs)


if __name__ == '__main__':
    main()
