#!/usr/bin/env python

import argparse
import logging

from . import manager, logger


def validate_flags(args):
    if args.binary and args.parquet:
        raise RuntimeError("Cannot output both parquet and binary flags at the same time.")


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
    parser.add_argument('-B', '--binary', action='store_true', help='Output data as binary files')
    parser.add_argument('-P', '--parquet', action='store_true', help='Output data as parquet files')

    args = parser.parse_args()
    validate_flags(args)
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
