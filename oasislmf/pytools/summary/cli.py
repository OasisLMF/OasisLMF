#!/usr/bin/env python

import argparse
import logging

from . import manager, logger

parser = argparse.ArgumentParser(
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
)

parser.add_argument('-i', '--files-in', help='names of the input file_path', nargs='?')
parser.add_argument('-t', '--run-type', help='stream run type', choices=manager.SUPPORTED_RUN_TYPE)
parser.add_argument('-m', '--low-memory', help='reduce downstream memory use with index file(s)', action='store_true')
parser.add_argument('-p', '--static-path', help='path to the folder containing the static files', default='input')
parser.add_argument('-v', '--logging-level', help='logging level (debug:10, info:20, warning:30, error:40, critical:50)',
                    default=30, type=int)
parser.add_argument('-z', '--output-zeros', help='if set, output zero values', action='store_true')
parser.add_argument('--create-summarypy-files', help='create summarypy files', action='store_true')
parser.add_argument('summary_sets_output', help='summary set output files: -summaryset_id file_path', nargs=argparse.REMAINDER)


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

    manager.main(**kwargs)


if __name__ == '__main__':
    main()
