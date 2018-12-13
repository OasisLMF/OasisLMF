#!/usr/bin/env python
"""
Deterministic loss validation
"""
import argparse

from ..tests.data import (
    canonical_oed_accounts_profile,
    canonical_oed_exposures_profile,
    oed_fm_agg_profile,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Validate deterministic losses')
    parser.add_argument(
        '-n', '--output-dir', metavar='N', type=str, required=True,
        help='The output directory'
    parser.add_argument(
        '-o', '--input-dir', metavar='N', type=str, default=None, required=False,
        help='The input directory containing the set of OED exposure + accounts data + other assets required for MDK Oasis files generation')
    parser.add_argument(
        '-l', '--loss-factor', metavar='N', type=float, default=1.0,
        help='The loss factor to apply to TIVs.')
    parser.add_argument(
       '-d', '--debug', action='store', default=None,
       help='Store Debugging Logs under ./logs')

    args = parser.parse_args()