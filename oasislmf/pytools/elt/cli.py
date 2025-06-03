#!/usr/bin/env python

import argparse
import logging

from . import manager, logger
from .data import VALID_EXT


def main():
    parser = argparse.ArgumentParser(description='Process event loss table stream')
    parser.add_argument('--run_dir', type=str, default='.', help='path to the run directory')
    parser.add_argument('-i', '--files_in', type=str, nargs='+', required=False, help='Input files')
    parser.add_argument('-s', '--selt', type=str, default=None, help='Output SELT CSV file')
    parser.add_argument('-m', '--melt', type=str, default=None, help='Output MELT CSV file')
    parser.add_argument('-q', '--qelt', type=str, default=None, help='Output QELT CSV file')
    parser.add_argument('-v', '--logging-level', type=int, default=30,
                        help='logging level (debug:10, info:20, warning:30, error:40, critical:50)')
    parser.add_argument('-H', '--noheader', action='store_true', help='Suppress header in output files')
    parser.add_argument('-E', '--ext', type=str, default='csv', choices=VALID_EXT, help='Output data format')

    args = parser.parse_args()
    kwargs = vars(args)

    # Set up logging
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    manager.main(**kwargs)


if __name__ == '__main__':
    main()
