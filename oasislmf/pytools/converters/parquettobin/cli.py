import argparse
import logging

from .manager import parquettobin, logger
from oasislmf.pytools.converters.data import SUPPORTED_PARQUETTOBIN


def main():
    parser = argparse.ArgumentParser(description='Convert Parquet to Binary for various file types.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest='file_type', required=True, help='Type of file to convert')
    for file_type in SUPPORTED_PARQUETTOBIN:
        parser_curr = subparsers.add_parser(file_type, help=f'parquet to bin tool for {file_type}')
        parser_curr.add_argument('-i', '--file_in', type=str, default="-", help='Input file path')
        parser_curr.add_argument('-o', '--file_out', type=str, default="-", help='Output file path')
        parser_curr.add_argument('-v', '--logging-level', type=int, default=30,
                                 help='logging level (debug:10, info:20, warning:30, error:40, critical:50)')
    args = parser.parse_args()
    kwargs = vars(args)

    file_type = kwargs.pop('file_type')
    file_in = kwargs.pop('file_in')
    file_out = kwargs.pop('file_out')

    # Set up logging
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    parquettobin(file_in, file_out, file_type, **kwargs)


if __name__ == '__main__':
    main()
