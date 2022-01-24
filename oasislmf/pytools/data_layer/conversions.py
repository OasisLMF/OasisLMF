"""
This file houses simple isolated functions that convert data from one format to another.
"""


def footprint_csv_to_parquet(file_path: str) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path: str = file_path.replace(".csv", ".parquet")

    df = pd.read_csv(file_path)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)


def footprint_bin_to_parquet(file_path: str) -> None:
    pass

