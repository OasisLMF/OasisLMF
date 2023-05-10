#!/usr/bin/env python

from .pla import manager

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--file-in', help='name of the input file')
parser.add_argument('-o', '--file-out', help='name of the output file')
parser.add_argument(
    '-P', '--input-path', help='path to itemsamplifications.bin',
    default='input'
)
parser.add_argument(
    '-p', '--static-path', help='path to lossfactors.bin', default='static'
)
parser.add_argument(
    '-r', '--run-dir', help='path to the run directory', default='.'
)


def main():
    kwargs = vars(parser.parse_args())

    manager.run(**kwargs)


if __name__ == '__main__':
    main()
