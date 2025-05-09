#!/usr/bin/env python

import argparse
import logging

from . import manager, logger


def validate_flags(args):
    if args.binary and args.parquet:
        raise RuntimeError("Cannot output both parquet and binary flags at the same time.")


def main():
    parser = argparse.ArgumentParser(description='Process average annual loss and standard deviation')
    parser.add_argument('--run_dir', type=str, default='.', help='path to the run directory')
    parser.add_argument('-K', '--subfolder', type=str, default=None, help='workspace sub folder name, inside <run_dir>/work/<sub folder name>')
    parser.add_argument('-a', '--aal', type=str, default=None, help='Output Average Annual Loss Table (AAL)')
    parser.add_argument('-c', '--alct', type=str, default=None, help='Output Average Loss Convergence Table (ALCT)')
    parser.add_argument('-M', '--meanonly', action='store_true', help='Output AAL with mean only')
    parser.add_argument('-l', '--confidence', type=float, default=0.95, help='0 <= confidence level <= 1, default 0.95')
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
