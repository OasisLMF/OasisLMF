import argparse
import logging

from oasislmf.pytools.common.event_stream import LOSS_STREAM_ID

from .manager import csvtobin, logger
from oasislmf.pytools.converters.data import SUPPORTED_CSVTOBIN


def add_custom_args(file_type, parser):
    if file_type == "damagebin":
        parser.add_argument('-N', '--no_validation', action='store_true', help='No validation checks')
    if file_type == "occurrence":
        parser.add_argument('-D', '--no_date_alg', action='store_true', help='No date algorithm in csv (use occ_date_id directly)')
        parser.add_argument('-G', '--granular', action='store_true', help='Use granular dates (occ_hour and occ_minute)')
        parser.add_argument('-P', '--no_of_periods', type=int, required=True, help='Number of periods')
    if file_type == "fm" or file_type == "gul":
        parser.add_argument('-t', '--stream_type', type=int, default=LOSS_STREAM_ID, help=f'Stream Type. Default LOSS_STREAM_ID = {LOSS_STREAM_ID}')
        parser.add_argument('-S', '--max_sample_index', type=int, required=True, help='Maximum sample index')
    if file_type == "footprint":
        parser.add_argument('-x', '--idx_file_out', required=True, type=str, help='Output index file path')
        parser.add_argument('-z', '--zip_files', action='store_true', help='Zip input files flag')
        parser.add_argument('-m', '--max_intensity_bin_idx', type=int, required=True, help='Maximum intensity bin index')
        parser.add_argument('-n', '--no_intensity_uncertainty', action='store_true', help='No intensity uncertainty')
        parser.add_argument('-d', '--decompressed_size', action='store_true', help='If True, add the decompressed size to the index file')
        parser.add_argument('-N', '--no_validation', action='store_true', help='No validation checks')
    if file_type == "summarycalc":
        parser.add_argument('-t', '--summary_set_id', type=int, default=1, help='Summary Set Id. Default 1')
        parser.add_argument('-S', '--max_sample_index', type=int, required=True, help='Maximum sample index')
    if file_type == "vulnerability":
        parser.add_argument('-z', '--zip_files', action='store_true', help='Zip input files flag')
        parser.add_argument('-x', '--idx_file_out', default=None, type=str, help='Output index file path, if not set will not use idx file')
        parser.add_argument('-d', '--max_damage_bin_idx', type=int, required=True, help='Maximum damage bin index')
        parser.add_argument('-N', '--no_validation', action='store_true', help='No validation checks')
        parser.add_argument('-S', '--suppress_int_bin_checks', action='store_true',
                            help='Suppress all intensity bins present for each vulnerability ID validation checks (recommended for multiple peril models)')


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to Binary for various file types.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest='file_type', required=True, help='Type of file to convert')
    for file_type in SUPPORTED_CSVTOBIN:
        parser_curr = subparsers.add_parser(file_type, help=f'csv to bin tool for {file_type}')
        add_custom_args(file_type, parser_curr)
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

    csvtobin(file_in, file_out, file_type, **kwargs)


if __name__ == '__main__':
    main()
