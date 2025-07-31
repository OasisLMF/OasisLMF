import argparse
import logging

from .manager import bintocsv, logger
from oasislmf.pytools.converters.data import SUPPORTED_BINTOCSV


def add_custom_args(file_type, parser):
    if file_type == "cdf":
        parser.add_argument('-d', '--run_dir', help='path to the run directory (default: ".")', default='.')
    if file_type == "footprint":
        parser.add_argument('-x', '--idx_file_in', required=True, type=str, help='Input index file path')
        parser.add_argument('-z', '--zip_files', action='store_true', help='Zip input files flag')
        parser.add_argument('-e', '--event_from_to', default=None, type=str, help='[event_id from]-[event_id to] extract an inclusive range of event')
    if file_type == "vulnerability":
        parser.add_argument('-z', '--zip_files', action='store_true', help='Zip input files flag')
        parser.add_argument('-x', '--idx_file_in', default=None, type=str, help='Input index file path')


def main():
    parser = argparse.ArgumentParser(description='Convert Binary to CSV for various file types.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest='file_type', required=True, help='Type of file to convert')
    for file_type in SUPPORTED_BINTOCSV:
        parser_curr = subparsers.add_parser(file_type, help=f'bin to csv tool for {file_type}')
        add_custom_args(file_type, parser_curr)
        parser_curr.add_argument('-i', '--file_in', default="-", type=str, help='Input file path')
        parser_curr.add_argument('-o', '--file_out', default="-", type=str, help='Output file path')
        parser_curr.add_argument('-v', '--logging-level', type=int, default=30,
                                 help='logging level (debug:10, info:20, warning:30, error:40, critical:50)')
        parser_curr.add_argument('-H', '--noheader', action='store_true', help='Suppress header in output files')
    args = parser.parse_args()
    kwargs = vars(args)

    file_type = kwargs.pop('file_type')
    file_in = kwargs.pop('file_in')
    file_out = kwargs.pop('file_out')
    noheader = kwargs.pop('noheader')

    # Set up logging
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    bintocsv(file_in, file_out, file_type, noheader, **kwargs)


if __name__ == '__main__':
    main()
