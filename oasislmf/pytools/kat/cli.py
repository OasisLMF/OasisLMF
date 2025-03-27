#!/usr/bin/env python

import argparse
import logging

from . import manager, logger


def validate_flags(args):
    flags = [
        args.selt,
        args.melt,
        args.qelt,
        args.splt,
        args.mplt,
        args.qplt,
    ]
    if sum(flags) == 0:
        raise RuntimeError("ERROR: no file type flag provided to katpy, run with -h to see which flags are valid")
    if sum(flags) > 1:
        raise RuntimeError("ERROR: katpy cannot kat more than one file type flag")

    if args.files_in is None and args.dir_in is None:
        raise RuntimeError("Error: katpy must specify at least one of --files_in or --dir_in.")

    if args.file_type and args.file_type not in ["csv", "bin"]:
        raise RuntimeError("Error: katpy file_type must be [\"csv\", \"bin\"] or None")


def main():
    parser = argparse.ArgumentParser(description='Concatenate ELT/PLT CSV files')
    parser.add_argument('-o', '--out', type=str, required=True, help='Output Concatenated CSV file')
    parser.add_argument('-f', '--file_type', type=str, default=None, help='Input file type if not discernible from input file suffix.')
    parser.add_argument('-i', '--files_in', type=str, nargs='+', required=False, help='Individual input file paths to concatenate')
    parser.add_argument('-d', '--dir_in', type=str, default=None, help='Path to the directory containing files for concatenation')
    parser.add_argument('-s', '--selt', action='store_true', help='Concatenate SELT CSV file')
    parser.add_argument('-m', '--melt', action='store_true', help='Concatenate MELT CSV file')
    parser.add_argument('-q', '--qelt', action='store_true', help='Concatenate QELT CSV file')
    parser.add_argument('-S', '--splt', action='store_true', help='Concatenate SPLT CSV file')
    parser.add_argument('-M', '--mplt', action='store_true', help='Concatenate MPLT CSV file')
    parser.add_argument('-Q', '--qplt', action='store_true', help='Concatenate QPLT CSV file')
    parser.add_argument('-u', '--unsorted', action='store_true', help='Do not sort by event/period ID')
    parser.add_argument('-v', '--logging-level', type=int, default=30,
                        help='logging level (debug:10, info:20, warning:30, error:40, critical:50)')

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
