#!/usr/bin/env python

import argparse
import logging

from oasislmf import __version__ as oasis_version
from oasislmf.pytools.gul import manager, logger

parser = argparse.ArgumentParser(
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
)

parser.add_argument('-a', help='back-allocation rule', default=0, action='store', type=int, dest='alloc_rule')
parser.add_argument('-d', help='output random numbers instead of gul (default: False).',
                    default=False, action='store_true', dest='debug')
parser.add_argument('-i', help='filename of items output', action='store', type=str, dest='items_outfile')
# parser.add_argument('-j', '--correlated-outfile', help='filename of correlated output', type=str, action='store', dest='corr_outfile')
parser.add_argument('-r', help='filename of random numbers', action='store', type=str, dest='random_numbers_file')
parser.add_argument('-s', help='skip header (default: False).', default=False, action='store_true', dest='skip_header')
parser.add_argument('-L', help='Loss treshold (default: 1e-6)', default=1e-6,
                    action='store', type=float, dest='loss_threshold')
parser.add_argument('-S', help='Sample size (default: 0).', default=0, action='store', type=int, dest='sample_size')
parser.add_argument('-V', '--version', action='version', version='{}'.format(oasis_version))
parser.add_argument('--file-in', action='store', type=str,)
parser.add_argument('--ignore-file-type', nargs='*', help='the type of file to be loaded', default=set())
parser.add_argument('--random-generator',
                    help='random number generator\n(0: numpy default (PCG64), 1: Latin Hypercube). Default: 1.',
                    default=1, type=int)
parser.add_argument('--run-dir', help='path to the run directory', default='.')
parser.add_argument('--logging-level', help='logging level (debug:10, info:20, warning:30, error:40, critical:50). Default: 30.',
                    default=30, action='store', type=int)

# TODO: continue here, implementing the --random-generator option

# [WIP] [TO BE REMOVED]
# args that were in gulcalc but are not currently implemented in gulpy.
# Remove when happy with this.
# "-b benchmark (in development)\n"
# "-R [max random numbers] used to allocate array for random numbers default 1,000,000\n"
# "-s seed for random number generation (used for debugging)\n"
# "-A automatically hashed seed driven random number generation (default)\n"
# "-l legacy mechanism driven by random numbers generated dynamically per group - will be removed in future\n"


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