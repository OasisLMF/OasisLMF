#!/usr/bin/env python

import argparse
import logging

from . import manager, logger


def main():
    parser = argparse.ArgumentParser(description='Process event loss table stream')
    parser.add_argument('--run_dir', type=str, default='.', help='path to the run directory')
    parser.add_argument('--files_in', type=str, nargs='+', required=True, help='Input files')
    parser.add_argument('--selt_output_file', type=str, default=None, help='Output SELT CSV file')
    parser.add_argument('--melt_output_file', type=str, default=None, help='Output MELT CSV file')
    parser.add_argument('--qelt_output_file', type=str, default=None, help='Output QELT CSV file')
    parser.add_argument('-v', '--logging-level', type=int, default=30,
                        help='logging level (debug:10, info:20, warning:30, error:40, critical:50)')

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