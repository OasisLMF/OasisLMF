from __future__ import absolute_import

import io

import pandas as pd


def read_csv(csv_filepath, csv_meta=None):
    """
    Filters the rows of a CSV file using validator functions defined in a
    custom meta dictionary, and returns dicts (one per row).
    """
    with io.open(csv_filepath, 'r', encoding='utf-8') as f:
        df = pd.read_csv(io.StringIO(f.read()), float_precision='high')
        df = df.where(df.notnull(), None)

    for i in range(len(df)):
        r = df.iloc[i].to_dict()
        if not csv_meta:
            yield r
        else:
            yield {
                k: csv_meta[k]['validator'](r[csv_meta[k]['csv_header']]) for k in csv_meta
            }
