#!/usr/bin/env python

from .pla import manager, logger

import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument(
    '-f', '--secondary-factor',
    help='optional relative secondary factor within range [0, 1]', default=1.0,
    type=float
)
parser.add_argument(
    '-F', '--uniform-factor',
    help='optional uniform post loss amplification factor', default=0.0,
    type=float
)
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

    if kwargs['secondary_factor'] < 0.0 or kwargs['secondary_factor'] > 1.0:
        logger.error(f'Secondary factor {kwargs["secondary_factor"]} must lie within range [0, 1]')
        SystemExit(1)

    if kwargs['uniform_factor'] < 0.0:
        logger.error(f'Uniform factor {kwargs["uniform_factor"]} must be positive value')
        SystemExit(1)

    if kwargs['secondary_factor'] < 1.0 and kwargs['uniform_factor'] > 0.0:
        logger.warning('Relative secondary and uniform factors are incompatible')
        logger.info('Ignoring relative secondary factor')
        kwargs['secondary_factor'] = 1.0

    manager.run(**kwargs)


if __name__ == '__main__':
    main()
