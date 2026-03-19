"""Convert V2 footprint Parquet files (nested or flat) back to CSV format.

CSV columns:
  K=1: event_id, areaperil_id, intensity_bin_id, probability
  K=N: event_id, areaperil_id, intensity_bin_id_1, ..., intensity_bin_id_N, probability
"""
import os

import numpy as np
import pandas as pd
from pyarrow import parquet as pq


def _nested_table_to_df(table):
    """Expand a nested V2 footprint Parquet table to a flat DataFrame.

    Args:
        table (pa.Table): Nested Parquet table with probabilities and
            intensity_bin_ids list columns.

    Returns:
        pd.DataFrame: DataFrame with CSV-style columns.
    """
    df = table.to_pandas()

    rows = []
    for _, row in df.iterrows():
        eid = row["event_id"]
        apid = row["areaperils_id"]
        probs = row["probabilities"]
        bins = row["intensity_bin_ids"]
        n = len(probs)
        k = len(bins) // n

        for i in range(n):
            r = {"event_id": int(eid), "areaperil_id": int(apid)}
            if k == 1:
                r["intensity_bin_id"] = int(bins[i])
            else:
                # Column-major: type j values are at bins[j*n + i]
                for j in range(k):
                    r[f"intensity_bin_id_{j + 1}"] = int(bins[j * n + i])
            r["probability"] = float(probs[i])
            rows.append(r)

    if rows:
        k_sample = len(df.iloc[0]["intensity_bin_ids"]) // len(df.iloc[0]["probabilities"])
    else:
        k_sample = 1

    if k_sample == 1:
        cols = ["event_id", "areaperil_id", "intensity_bin_id", "probability"]
    else:
        cols = ["event_id", "areaperil_id"]
        cols += [f"intensity_bin_id_{j + 1}" for j in range(k_sample)]
        cols += ["probability"]

    out_df = pd.DataFrame(rows, columns=cols)

    # Match original CSV types
    out_df["event_id"] = out_df["event_id"].astype(np.int64)
    out_df["areaperil_id"] = out_df["areaperil_id"].astype(np.int64)
    if k_sample == 1:
        out_df["intensity_bin_id"] = out_df["intensity_bin_id"].astype(np.int64)
    else:
        for j in range(k_sample):
            col = f"intensity_bin_id_{j + 1}"
            out_df[col] = out_df[col].astype(np.int64)

    return out_df


def _flat_table_to_df(table):
    """Pivot a flat V2 footprint Parquet table back to CSV-style columns.

    Args:
        table (pa.Table): Flat Parquet table with scenario_index and
            intensity_index columns.

    Returns:
        pd.DataFrame: DataFrame with CSV-style columns.
    """
    df = table.to_pandas()

    k = int(df["intensity_index"].max()) + 1

    # Pivot intensity_index to columns
    pivoted = df.pivot_table(
        index=["event_id", "areaperils_id", "scenario_index"],
        columns="intensity_index",
        values="intensity_bin_id",
        aggfunc="first",
    ).reset_index()

    # Rename pivoted columns
    if k == 1:
        pivoted = pivoted.rename(columns={0: "intensity_bin_id"})
    else:
        pivoted = pivoted.rename(columns={i: f"intensity_bin_id_{i + 1}" for i in range(k)})

    # Get probability (same for all intensity_index rows in a scenario)
    prob_df = df[df["intensity_index"] == 0][
        ["event_id", "areaperils_id", "scenario_index", "probability"]
    ].copy()

    out_df = pivoted.merge(prob_df, on=["event_id", "areaperils_id", "scenario_index"])
    out_df = out_df.rename(columns={"areaperils_id": "areaperil_id"})
    out_df = out_df.sort_values(["event_id", "areaperil_id"]).reset_index(drop=True)

    # Build column order
    if k == 1:
        cols = ["event_id", "areaperil_id", "intensity_bin_id", "probability"]
    else:
        cols = ["event_id", "areaperil_id"]
        cols += [f"intensity_bin_id_{i + 1}" for i in range(k)]
        cols += ["probability"]

    out_df = out_df[cols]

    # Match original CSV types
    out_df["event_id"] = out_df["event_id"].astype(np.int64)
    out_df["areaperil_id"] = out_df["areaperil_id"].astype(np.int64)
    if k == 1:
        out_df["intensity_bin_id"] = out_df["intensity_bin_id"].astype(np.int64)
    else:
        for i in range(k):
            col = f"intensity_bin_id_{i + 1}"
            out_df[col] = out_df[col].astype(np.int64)

    return out_df


def nested_parquet_to_csv(parquet_path, csv_path):
    """Convert a nested V2 footprint Parquet file back to CSV.

    Reads nested lists, derives K from len(intensity_bin_ids)/len(probabilities),
    and expands each row into N scenario rows with K bin columns.

    Args:
        parquet_path (str): Path to input nested Parquet file.
        csv_path (str): Path to output CSV file.
    """
    table = pq.read_table(parquet_path)
    out_df = _nested_table_to_df(table)
    out_df.to_csv(csv_path, index=False)


def flat_parquet_to_csv(parquet_path, csv_path):
    """Convert a flat V2 footprint Parquet file back to CSV.

    Pivots intensity_index back into K bin columns and de-duplicates probability.

    Args:
        parquet_path (str): Path to input flat Parquet file.
        csv_path (str): Path to output CSV file.
    """
    table = pq.read_table(parquet_path)
    out_df = _flat_table_to_df(table)
    out_df.to_csv(csv_path, index=False)


def parquet_tree_to_csv(root_dir, csv_path):
    """Reassemble a directory tree of V2 footprint Parquet files into a CSV.

    Walks ``{root_dir}/footprint/`` collecting all ``.parquet`` files, reads
    each one, auto-detects schema type from ``oasis:schema`` metadata, and
    concatenates the results into a single sorted CSV.

    Args:
        root_dir (str): Root directory containing the ``footprint/`` tree.
        csv_path (str): Path to output CSV file.

    Raises:
        FileNotFoundError: If the footprint directory does not exist or
            contains no ``.parquet`` files.
        ValueError: If files have mixed schema types.
    """
    fp_dir = os.path.join(root_dir, "footprint")
    if not os.path.isdir(fp_dir):
        raise FileNotFoundError(f"Footprint directory not found: {fp_dir}")

    parquet_files = []
    for dirpath, _dirnames, filenames in os.walk(fp_dir):
        for fname in filenames:
            if fname.endswith(".parquet"):
                parquet_files.append(os.path.join(dirpath, fname))

    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found under {fp_dir}")

    parquet_files.sort()

    dfs = []
    schema_types = set()
    for pq_file in parquet_files:
        table = pq.read_table(pq_file)
        meta = table.schema.metadata or {}
        schema_type = meta.get(b"oasis:schema", b"").decode()
        schema_types.add(schema_type)

        if schema_type == "nested":
            dfs.append(_nested_table_to_df(table))
        elif schema_type == "flat":
            dfs.append(_flat_table_to_df(table))
        else:
            raise ValueError(f"Unknown oasis:schema {schema_type!r} in {pq_file}")

    if len(schema_types) > 1:
        raise ValueError(f"Mixed schema types across tree files: {schema_types}")

    out_df = pd.concat(dfs, ignore_index=True)
    sort_cols = [c for c in out_df.columns if c != "probability"]
    out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    out_df.to_csv(csv_path, index=False)
