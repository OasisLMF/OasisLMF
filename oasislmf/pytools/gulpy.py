#!/usr/bin/env python

import argparse
import logging

from oasislmf.pytools.gul import manager, logger

parser = argparse.ArgumentParser(
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
)

# parser.add_argument('-a', '--allocation-rule', help='back-allocation rule', default=0, type=int, action='store', dest='alloc_rule')
parser.add_argument('-i', '--items-outfile', help='filename of items output',
                    type=str, action='store', dest='items_outfile')
# parser.add_argument('-j', '--correlated-outfile', help='filename of correlated output', type=str, action='store', dest='corr_outfile')
parser.add_argument('-S', help='Sample size (default 0).', default=0, type=int, action='store', dest='sample_size')
# parser.add_argument('-v', '--version', help='gulpy version', action='store_true')
parser.add_argument('--ignore-file-type', nargs='*', help='the type of file to be loaded', default=set())
parser.add_argument('-L', '--loss-threshold', type=float, help='Loss treshold (default: 1e-6)', default=1e-6)
parser.add_argument('--run-dir', help='path to the run directory', default='.')
parser.add_argument('--logging-level', help='logging level (debug:10, info:20, warning:30, error:40, critical:50)', default=30, type=int)
parser.add_argument('-s', help='skip header (default: False).', default=False, action='store_true', dest='skip_header')


# [WIP] [TO BE REMOVED]
# args that were in gulcalc but are not currently implemented in gulpy.
# Remove when happy with this.
# "-b benchmark (in development)\n"
# "-c [output pipe] - coverage output\n"
# "-r use random number file\n"
# "-R [max random numbers] used to allocate array for random numbers default 1,000,000\n"
# "-h help\n"
# "-d debug (output random numbers instead of gul)\n"
# "-s seed for random number generation (used for debugging)\n"
# "-A automatically hashed seed driven random number generation (default)\n"
# "-l legacy mechanism driven by random numbers generated dynamically per group - will be removed in future\n"
# "-L Loss threshold (default 0)\n"


def main():
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
