"""Convert V2 footprint CSV files to nested or flat Parquet format.

CSV columns:
  K=1: event_id, areaperil_id, intensity_bin_id, probability
  K=N: event_id, areaperil_id, intensity_bin_id_1, ..., intensity_bin_id_N, probability
"""
import os

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq


def event_id_to_path(event_id, ext="parquet"):
    """Compute the directory-tree file path for a given event_id.

    The event_id is split into 4 bytes [B3][B2][B1][B0]. The file path is
    ``footprint/{B3:02X}/{B2:02X}/{B1:02X}.{ext}``. Each file holds up to
    256 events (sharing the same upper 3 bytes).

    Args:
        event_id (int): The event identifier.
        ext (str): File extension (default ``"parquet"``).

    Returns:
        str: Relative path, e.g. ``"footprint/00/00/01.parquet"``.
    """
    b3 = (event_id >> 24) & 0xFF
    b2 = (event_id >> 16) & 0xFF
    b1 = (event_id >> 8) & 0xFF
    return f"footprint/{b3:02X}/{b2:02X}/{b1:02X}.{ext}"


def event_id_to_slot(event_id):
    """Extract the slot index (lowest byte) from an event_id.

    Args:
        event_id (int): The event identifier.

    Returns:
        int: Slot index (0-255).
    """
    return event_id & 0xFF


def _detect_k(columns):
    """Derive K (number of intensity types) from CSV column names.

    Args:
        columns (list[str]): CSV column names.

    Returns:
        int: Number of intensity types.
        list[str]: Ordered list of bin column names.
    """
    bin_cols = [c for c in columns if c.startswith("intensity_bin_id")]
    if len(bin_cols) == 1 and bin_cols[0] == "intensity_bin_id":
        return 1, bin_cols
    # intensity_bin_id_1, intensity_bin_id_2, ...
    bin_cols = sorted(bin_cols, key=lambda c: int(c.rsplit("_", 1)[1]))
    return len(bin_cols), bin_cols


def _build_nested_schema():
    """Build the nested Parquet schema for V2 footprint.

    Returns:
        pa.Schema: The nested schema.
    """
    return pa.schema([
        pa.field("event_id", pa.int32(), nullable=False),
        pa.field("areaperils_id", pa.uint32(), nullable=False),
        pa.field("probabilities",
                 pa.list_(pa.field("item", pa.float32(), nullable=False)),
                 nullable=False),
        pa.field("intensity_bin_ids",
                 pa.list_(pa.field("item", pa.int32(), nullable=False)),
                 nullable=False),
    ])


def _build_flat_schema():
    """Build the flat Parquet schema for V2 footprint.

    Returns:
        pa.Schema: The flat schema.
    """
    return pa.schema([
        pa.field("event_id", pa.int32(), nullable=False),
        pa.field("areaperils_id", pa.uint32(), nullable=False),
        pa.field("scenario_index", pa.int32(), nullable=False),
        pa.field("intensity_index", pa.int32(), nullable=False),
        pa.field("intensity_bin_id", pa.int32(), nullable=False),
        pa.field("probability", pa.float32(), nullable=False),
    ])


def _build_metadata(df, k, bin_cols, has_uncertainty, schema_type):
    """Build Oasis file-level metadata for V2 parquet.

    Args:
        df (pd.DataFrame): Source CSV dataframe.
        k (int): Number of intensity types.
        bin_cols (list[str]): Bin column names.
        has_uncertainty (bool): Whether any group has N > 1.
        schema_type (str): "nested" or "flat".

    Returns:
        dict: Metadata key-value pairs (bytes).
    """
    max_bin = 0
    for col in bin_cols:
        col_max = int(df[col].max())
        if col_max > max_bin:
            max_bin = col_max

    n_events = int(df["event_id"].nunique())
    n_aps = int(df["areaperil_id"].nunique())

    return {
        b"oasis:format": b"footprint_v2_parquet",
        b"oasis:schema": schema_type.encode(),
        b"oasis:version": b"1",
        b"oasis:max_intensity_bin_id": str(max_bin).encode(),
        b"oasis:has_intensity_uncertainty": str(int(has_uncertainty)).encode(),
        b"oasis:areaperil_width": b"4",
        b"oasis:compression": b"zstd",
        b"oasis:max_num_intensity_types": str(k).encode(),
        b"oasis:total_events": str(n_events).encode(),
        b"oasis:total_areaperils": str(n_aps).encode(),
    }


def _df_to_nested_table(df, k, bin_cols):
    """Convert a DataFrame to a nested PyArrow Table.

    Groups by (event_id, areaperil_id) and packs probabilities and
    intensity_bin_ids into list columns.

    Args:
        df (pd.DataFrame): Source CSV dataframe with standard columns.
        k (int): Number of intensity types.
        bin_cols (list[str]): Bin column names.

    Returns:
        pa.Table: Nested table with schema and metadata attached.
    """
    grouped = df.groupby(["event_id", "areaperil_id"], sort=True)

    event_ids = []
    areaperils_ids = []
    probabilities_list = []
    intensity_bin_ids_list = []
    has_uncertainty = False

    for (eid, apid), grp in grouped:
        n = len(grp)
        if n > 1:
            has_uncertainty = True

        probs = grp["probability"].values.astype(np.float32).tolist()
        probabilities_list.append(probs)

        # Column-major: all values for type 0, then type 1, ...
        bins = []
        for col in bin_cols:
            bins.extend(grp[col].values.astype(np.int32).tolist())
        intensity_bin_ids_list.append(bins)

        event_ids.append(np.int32(eid))
        areaperils_ids.append(np.uint32(apid))

    schema = _build_nested_schema()
    metadata = _build_metadata(df, k, bin_cols, has_uncertainty, "nested")
    schema = schema.with_metadata(metadata)

    table = pa.table({
        "event_id": pa.array(event_ids, type=pa.int32()),
        "areaperils_id": pa.array(areaperils_ids, type=pa.uint32()),
        "probabilities": pa.array(probabilities_list, type=pa.list_(pa.field("item", pa.float32(), nullable=False))),
        "intensity_bin_ids": pa.array(intensity_bin_ids_list, type=pa.list_(pa.field("item", pa.int32(), nullable=False))),
    }, schema=schema)

    return table


def _df_to_flat_table(df, k, bin_cols):
    """Convert a DataFrame to a flat PyArrow Table.

    Each CSV row with K bin columns expands into K flat rows with
    scenario_index and intensity_index columns.

    Args:
        df (pd.DataFrame): Source CSV dataframe with standard columns.
        k (int): Number of intensity types.
        bin_cols (list[str]): Bin column names.

    Returns:
        pa.Table: Flat table with schema and metadata attached.
    """
    # Assign scenario_index per (event_id, areaperil_id) group
    df = df.copy()
    df["scenario_index"] = df.groupby(["event_id", "areaperil_id"]).cumcount().astype(np.int32)

    rows = []
    for _, row in df.iterrows():
        eid = row["event_id"]
        apid = row["areaperil_id"]
        sidx = row["scenario_index"]
        prob = row["probability"]
        for iidx, col in enumerate(bin_cols):
            rows.append((eid, apid, sidx, iidx, row[col], prob))

    flat_df = pd.DataFrame(rows, columns=[
        "event_id", "areaperils_id", "scenario_index",
        "intensity_index", "intensity_bin_id", "probability",
    ])
    flat_df = flat_df.sort_values(
        ["event_id", "areaperils_id", "scenario_index", "intensity_index"]
    ).reset_index(drop=True)

    flat_df["event_id"] = flat_df["event_id"].astype(np.int32)
    flat_df["areaperils_id"] = flat_df["areaperils_id"].astype(np.uint32)
    flat_df["scenario_index"] = flat_df["scenario_index"].astype(np.int32)
    flat_df["intensity_index"] = flat_df["intensity_index"].astype(np.int32)
    flat_df["intensity_bin_id"] = flat_df["intensity_bin_id"].astype(np.int32)
    flat_df["probability"] = flat_df["probability"].astype(np.float32)

    has_uncertainty = (df.groupby(["event_id", "areaperil_id"]).size() > 1).any()
    metadata = _build_metadata(df, k, bin_cols, bool(has_uncertainty), "flat")

    schema = _build_flat_schema().with_metadata(metadata)
    table = pa.Table.from_pandas(flat_df, schema=schema, preserve_index=False)

    return table


def csv_to_nested_parquet(csv_path, parquet_path):
    """Convert a V2 footprint CSV to nested Parquet format.

    One row per (event_id, areaperil_id) group. Lists hold per-scenario data.
    intensity_bin_ids uses column-major layout: all type-0 bins, then type-1, etc.

    Args:
        csv_path (str): Path to input CSV file.
        parquet_path (str): Path to output Parquet file.
    """
    df = pd.read_csv(csv_path)
    k, bin_cols = _detect_k(df.columns.tolist())
    table = _df_to_nested_table(df, k, bin_cols)
    pq.write_table(table, parquet_path, compression="ZSTD")


def csv_to_flat_parquet(csv_path, parquet_path):
    """Convert a V2 footprint CSV to flat Parquet format.

    One row per (event_id, areaperil_id, scenario, intensity_type). Each CSV row
    with K bin columns expands into K flat rows.

    Args:
        csv_path (str): Path to input CSV file.
        parquet_path (str): Path to output Parquet file.
    """
    df = pd.read_csv(csv_path)
    k, bin_cols = _detect_k(df.columns.tolist())
    table = _df_to_flat_table(df, k, bin_cols)
    pq.write_table(table, parquet_path, compression="ZSTD")


def csv_to_parquet_tree(csv_path, root_dir, schema_type="nested"):
    """Convert a V2 footprint CSV to a directory tree of Parquet files.

    The event_id is split into 4 bytes ``[B3][B2][B1][B0]``. Each unique
    combination of ``(B3, B2, B1)`` produces one file at
    ``{root_dir}/footprint/{B3:02X}/{B2:02X}/{B1:02X}.parquet``, holding
    up to 256 events.

    Args:
        csv_path (str): Path to input CSV file.
        root_dir (str): Root output directory.
        schema_type (str): ``"nested"`` or ``"flat"`` (default ``"nested"``).

    Raises:
        ValueError: If *schema_type* is not ``"nested"`` or ``"flat"``.
    """
    if schema_type not in ("nested", "flat"):
        raise ValueError(f"schema_type must be 'nested' or 'flat', got {schema_type!r}")

    df = pd.read_csv(csv_path)
    k, bin_cols = _detect_k(df.columns.tolist())

    # Path key = upper 3 bytes of event_id (B3/B2/B1)
    eid_arr = df["event_id"].values.astype(np.int64)
    df["_path_key"] = (eid_arr >> 8) & 0xFFFFFF

    build_table = _df_to_nested_table if schema_type == "nested" else _df_to_flat_table

    for path_key, group_df in df.groupby("_path_key"):
        group_df = group_df.drop(columns=["_path_key"])

        table = build_table(group_df, k, bin_cols)

        # Derive file path from any event_id in the group
        sample_eid = int(group_df["event_id"].iloc[0])
        rel_path = event_id_to_path(sample_eid)
        abs_path = os.path.join(root_dir, rel_path)

        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        pq.write_table(
            table, abs_path,
            compression="ZSTD",
            write_statistics=True,
            row_group_size=64 * 1024,  # ~1 MB target with typical row sizes
        )
