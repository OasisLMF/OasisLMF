#!/usr/bin/env python

from .pla import manager, logger

import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--file-in', help='name of the input file')
parser.add_argument('-o', '--file-out', help='name of the output file')
parser.add_argument(
    '-P', '--input-path', help='path to amplifications.bin',
    default='input'
)
parser.add_argument(
    '-p', '--static-path', help='path to lossfactors.bin', default='static'
)
parser.add_argument(
    '-r', '--run-dir', help='path to the run directory', default='.'
)
parser.add_argument(
    '-v', '--logging-level',
    help='logging level (debug:10, info:20, warning:30, error:40, critical:50)',
    default=30, type=int
)


def main():
    kwargs = vars(parser.parse_args())

    # Add handler to pla logger
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    manager.run(**kwargs)


if __name__ == '__main__':
    main()
