#!/usr/bin/env python

import argparse
import logging

from oasislmf import __version__ as oasis_version
from oasislmf.pytools.gulmc import logger, manager

parser = argparse.ArgumentParser(
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
)

# arguments in alphabetical order (lower-case, then upper-case, then long arguments)
parser.add_argument('-a', help='back-allocation rule. Default: 0', default=0, type=int, dest='alloc_rule')
parser.add_argument('-d', help='output the ground up loss (0), the random numbers used for hazard sampling (1), '
                               'the random numbers used for damage sampling (2). Default: 0',
                    action='store', type=int, dest='debug', default=0)
parser.add_argument('-i', '--file-in', help='filename of input stream (list of events from `eve`).',
                    action='store', type=str, dest='file_in')
parser.add_argument('-o', '--file-out', help='filename of output stream (ground up losses).',
                    action='store', type=str, dest='file_out')
parser.add_argument('-L', help='Loss treshold. Default: 1e-6', default=1e-6,
                    action='store', type=float, dest='loss_threshold')
parser.add_argument('-S', help='Sample size. Default: 0', default=0, action='store', type=int, dest='sample_size')
parser.add_argument('-V', '--version', action='version', version='{}'.format(oasis_version))
parser.add_argument('--effective-damageability',
                    help='if passed true, the effective damageability is used to draw loss samples instead of full MC. Default: False',
                    action='store_true', dest='effective_damageability', default=False)
parser.add_argument('--ignore-correlation',
                    help='if passed true, peril correlation groups (if defined) are ignored for the generation of correlated samples. Default: False',
                    action='store_true', dest='ignore_correlation', default=False)
parser.add_argument('--ignore-haz-correlation',
                    help='if passed true, hazard correlation groups (if defined) are ignored for the generation of correlated samples. Default: False',
                    action='store_true', dest='ignore_haz_correlation', default=False)
parser.add_argument('--ignore-file-type', nargs='*',
                    help='the type of file to be loaded. Default: set()', default=set())
parser.add_argument('--data-server', help='=Use tcp/sockets for IPC data sharing.',
                    action='store_true', dest='data_server')
parser.add_argument('--logging-level',
                    help='logging level (debug:10, info:20, warning:30, error:40, critical:50). Default: 30',
                    default=30, type=int)
parser.add_argument('--vuln-cache-size', help='Size in MB of the in-memory cache to store and reuse vulnerability cdf. Default: 200',
                    default=200, action='store', type=int, dest='max_cached_vuln_cdf_size_MB')
parser.add_argument('--peril-filter', help='Id of the peril to keep, if empty take all perils',
                    nargs='+', dest='peril_filter')
parser.add_argument('--random-generator',
                    help='random number generator\n0: numpy default (MT19937), 1: Latin Hypercube. Default: 1',
                    default=1, type=int, dest='random_generator')
parser.add_argument('--run-dir', help='path to the run directory. Default: "."', default='.')
parser.add_argument('--model-df-engine', help='The engine to use when loading model dataframes',
                    default='oasis_data_manager.df_reader.reader.OasisPandasReader')


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
