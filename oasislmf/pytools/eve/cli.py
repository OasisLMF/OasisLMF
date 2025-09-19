#!/usr/bin/env python

import argparse
import logging

from oasislmf.pytools.eve.manager import DEFAULT_EVENTS_FILE
from . import logger, manager


def validate_flags(args):
    if args.process_number <= 0:
        raise RuntimeError(f"ERROR: evepy process_number {args.process_number} is not valid.")

    if args.total_processes <= 0:
        raise RuntimeError(f"ERROR: evepy total_processes {args.total_processes} is not valid.")

    if args.total_processes < args.process_number:
        raise ValueError(f"ERROR: evepy process_number {args.process_number} cannot be larger than the total_processes {args.total_processes}")


def main():
    parser = argparse.ArgumentParser(description='Generate partitioned event IDs stream.')
    parser.add_argument('process_number', type=int,
                        help='Process number to receive a partition of events. Processes are one-indexed.')
    parser.add_argument('total_processes', type=int,
                        help='Total number of partitions of events to distribute to processes')
    parser.add_argument('-i', '--input_file', type=str, default=DEFAULT_EVENTS_FILE,
                        help=f'Input file containing the events binary (default: {DEFAULT_EVENTS_FILE})')

    parser.add_argument('-n', '--no_shuffle', action='store_true',
                        help='Disable shuffle. Events respect input ordering')
    parser.add_argument('-r', '--randomise', action='store_true',
                        help='Randomise events using the Fisher-Yates shuffle.')

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


if __name__ == "__main__":
    main()
