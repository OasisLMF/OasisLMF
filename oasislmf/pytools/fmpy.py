#!/usr/bin/env python

from .fm import manager, logger

import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--allocation-rule', help='back-allocation rule', default=0, type=int)
parser.add_argument('-n', '--net-loss', help='output the net value instead of the gross value', action='store_true')
parser.add_argument('-p', '--static-path', help='path to the folder containing the static files', default='input')
parser.add_argument('-i', '--files-in', help='names of the input file_path', nargs='+')
parser.add_argument('-o', '--files-out', help='names of the output file_path', nargs='+')
parser.add_argument('-l', '--low-memory', help='in low memory mode, loss arrays are stored in memory map', action='store_true')
parser.add_argument('--sort-output', help='sort the output stream by item_id', action='store_true')
parser.add_argument('--storage-method', help='store data as "dense" or "sparse"', default='sparse')
parser.add_argument('--create-financial-structure-files', help='create financial structure', action='store_true')
parser.add_argument('-v', '--logging-level', help='logging level (debug:10, info:20, warning:30, error:40, critical:50)',
                    default=30, type=int)
parser.add_argument('-S', '--step-policies', help='not use, kept for backward compatibility with fmcalc', action='store_true')


def main():
    kwargs = vars(parser.parse_args())

    # add handler to fm logger
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    manager.run(**kwargs)


if __name__ == '__main__':
    main()
