#!/usr/bin/env python

import argparse
import logging

from . import manager, logger
from .data import VALID_EXT


def main():
    parser = argparse.ArgumentParser(description='Process loss exceedance data')
    parser.add_argument('--run_dir', type=str, default='.', help='path to the run directory')
    parser.add_argument('-K', '--subfolder', type=str, default=None, help='workspace sub folder name, inside <run_dir>/work/<sub folder name>')
    parser.add_argument('-O', '--ept', type=str, default=None, help='Output Exeedance Probability Table (EPT)')
    parser.add_argument('-o', '--psept', type=str, default=None, help='Output Per Sample Exeedance Probability Table (PSEPT)')
    parser.add_argument('-F', '--agg_full_uncertainty', action='store_true', help='Aggregate Full Uncertainty')
    parser.add_argument('-W', '--agg_wheatsheaf', action='store_true', help='Aggregate Wheatsheaf')
    parser.add_argument('-S', '--agg_sample_mean', action='store_true', help='Aggregate Sample Mean')
    parser.add_argument('-M', '--agg_wheatsheaf_mean', action='store_true', help='Aggregate Wheatsheaf Mean')
    parser.add_argument('-f', '--occ_full_uncertainty', action='store_true', help='Occurrence Full Uncertainty')
    parser.add_argument('-w', '--occ_wheatsheaf', action='store_true', help='Occurrence Wheatsheaf')
    parser.add_argument('-s', '--occ_sample_mean', action='store_true', help='Occurrence Sample Mean')
    parser.add_argument('-m', '--occ_wheatsheaf_mean', action='store_true', help='Occurrence Wheatsheaf Mean')
    parser.add_argument('-r', '--use_return_period', action='store_true', help='Use Return Period file')
    parser.add_argument('-v', '--logging-level', type=int, default=30,
                        help='logging level (debug:10, info:20, warning:30, error:40, critical:50)')
    parser.add_argument('-H', '--noheader', action='store_true', help='Suppress header in output files')
    parser.add_argument('-E', '--ext', type=str, default='csv', choices=VALID_EXT, help='Output data format')

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
