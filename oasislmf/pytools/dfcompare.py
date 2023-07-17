#!/usr/bin/env python

import argparse
from pathlib import Path
import pandas as pd
from oasislmf.pytools.utils import assert_allclose

parser = argparse.ArgumentParser(
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
)

# arguments in alphabetical order (lower-case, then upper-case, then long arguments)
parser.add_argument(help='filenames of the csv files to compare, e.g., file1.csv file2.csv', nargs='+', dest='files')
parser.add_argument('--col', help='column to compare. If not provided, all columns are compared.', default='', type=str, dest='col')
parser.add_argument('--rtol', help='relative tolerance. Default: 1e-6', default=1e-6, type=float, dest='rtol')
parser.add_argument('--atol', help='absolute tolerance. Default: 1e-8', default=1e-8, type=float, dest='atol')


def main():

    kwargs = vars(parser.parse_args())

    f1, f2 = kwargs['files']
    f1 = Path(f1)
    f2 = Path(f2)

    rtol = kwargs['rtol']
    atol = kwargs['atol']

    if f1.suffix == '.csv':
        df1 = pd.read_csv(f1)
    else:
        raise NotImplementedError(f"Cannot read file {f1} with extension {f1.suffix}.")

    if f2.suffix == '.csv':
        df2 = pd.read_csv(f2)
    else:
        raise NotImplementedError(f"Cannot read file {f2} with extension {f2.suffix}.")

    if any(df1.columns != df2.columns):
        raise ValueError(f"Expect {f1} and {f2} to have same columns, got {df1.columns} and {df2.columns}.")

    user_col = kwargs['col']
    if user_col:
        if user_col not in df1.columns:
            raise ValueError(f"Column '{user_col}' does not exist. Available columns are: {', '.join(list(df1.columns))}.")

        compare_cols = [user_col]
    else:
        compare_cols = df1.columns

    for col in compare_cols:
        try:
            assert_allclose(df1[col], df2[col], rtol=rtol, atol=atol, x_name=f1, y_name=f2)
        except AssertionError as e:
            print(e)
            print()


if __name__ == '__main__':

    main()
