#!/usr/bin/env python

import argparse

import pandas as pd

parser = argparse.ArgumentParser(
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter  # for multi-line help text
)

parser.add_argument('-p', '--parquet-path', help='path to parquet file', type=str, required=True)
parser.add_argument('-c', '--csv-path', help='path to csv file', type=str, required=True)
parser.add_argument('-s', '--sort', help='sort by string, int and categorical columns before check', action='store_true')
parser.add_argument('--check-exact', help='Whether to compare number exactly.', action='store_true')
parser.add_argument('--lower-case-cols', help='Lower cased column names before compare', action='store_true')
parser.add_argument('--rasie-error', help='Lower cased column names before compare', action='store_true')
parser.add_argument('--rtol', help='relative tolerance. Only used when check_exact is False', default=1e-05, action='store', type=float)
parser.add_argument('--atol', help='Absolute tolerance. Only used when check_exact is False', default=1e-08, action='store', type=float)


def is_sorting_type(dtype):
    return (pd.api.types.is_string_dtype(dtype) or
            pd.api.types.is_categorical_dtype(dtype) or
            pd.api.types.is_integer_dtype(dtype))


def compare(parquet_path, csv_path, sort, lower_case_cols, rasie_error, **kwargs):
    parquet_df = pd.read_parquet(parquet_path)
    dtypes = parquet_df.dtypes.apply(lambda x: x.name).to_dict()
    csv_df = pd.read_csv(csv_path, dtype=dtypes)

    if lower_case_cols:
        parquet_df.columns = parquet_df.columns.str.lower()
        csv_df.columns = csv_df.columns.str.lower()

    if sort:
        sort_column = [column for column, dtype in dtypes.items() if is_sorting_type(dtype)]
        parquet_df = parquet_df.sort_values(by=sort_column)
        csv_df = csv_df.sort_values(by=sort_column)

    try:
        pd.testing.assert_frame_equal(parquet_df, csv_df, **kwargs)
    except AssertionError as e:
        print(e)
        if rasie_error:
            raise


if __name__ == '__main__':
    # example:
    # python ~/OasisLMF/validation/utils/compare_parquet_csv.py -p ~/OasisPiWind/tests/inputs/SourceLocOEDPiWind.parquet  -c ~/OasisPiWind/tests/inputs/SourceLocOEDPiWind.csv

    kwargs = vars(parser.parse_args())
    compare(**kwargs)
