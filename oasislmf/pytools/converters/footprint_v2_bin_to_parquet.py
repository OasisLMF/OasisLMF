"""Convert V2 footprint OFPT binary files to nested or flat Parquet format.

Reads the OFPT directory tree (footprint/{B3:02X}/{B2:02X}/{B1:02X}.ofpt),
decodes each event's chunks, and writes Parquet files to a parallel tree
(footprint/{B3:02X}/{B2:02X}/{B1:02X}.parquet).

Usage:
    # V1 footprint.bin -> nested Parquet tree (output in same dir)
    python -m oasislmf.pytools.converters.footprint_v2_bin_to_parquet /path/to/model_data

    # V1 -> flat schema, custom output dir
    python -m oasislmf.pytools.converters.footprint_v2_bin_to_parquet /path/to/model_data -o /output --schema flat

    # OFPT -> Parquet
    python -m oasislmf.pytools.converters.footprint_v2_bin_to_parquet /path/to/ofpt_root --source ofpt
"""
import argparse
import os
import logging

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq

from oasislmf.pytools.common.ofpt import (
    NpMemMap, BinScanner, event_to_columnar,
    trailing_locator_dtype, file_footer_dtype,
    slot_table_dtype, slot_table_len,
    event_footer_bloc_header_dtype_u4,
    event_footer_bloc_chunk_dtype_ap_u4,
    magic_file, magic_footer, magic_event,
    complex_reverse_mv_read,
    BUFFER_READ_BYTES,
)
from oasislmf.pytools.converters.footprint_v2_csv_to_parquet import (
    event_id_to_path,
    _build_nested_schema,
    _build_flat_schema,
)

logger = logging.getLogger(__name__)


def _decode_chunk_py(raw, ap_width):
    """Decode a decompressed OFPT chunk into numpy arrays (pure Python).

    Args:
        raw (bytes): Decompressed chunk bytes.
        ap_width (int): Areaperil width in bytes (4 or 8).

    Returns:
        tuple: (areaperil_ids, cum_offsets, probabilities, intensity_bin_ids,
            num_intensity_types).
    """
    ap_dtype = np.dtype(f"<u{ap_width}")
    cursor = 0

    num_areaperils = int(np.frombuffer(
        raw[cursor:cursor + ap_width], dtype=ap_dtype)[0])
    cursor += ap_width

    areaperil_ids = np.frombuffer(
        raw[cursor:cursor + num_areaperils * ap_width], dtype=ap_dtype)
    cursor += num_areaperils * ap_width

    num_intensity_types = int(np.frombuffer(
        raw[cursor:cursor + 4], dtype=np.int32)[0])
    cursor += 4

    cum_offsets = np.frombuffer(
        raw[cursor:cursor + (num_areaperils + 1) * 4], dtype=np.int32)
    cursor += (num_areaperils + 1) * 4

    num_probabilities = int(cum_offsets[-1])
    probabilities = np.frombuffer(
        raw[cursor:cursor + num_probabilities * 4], dtype=np.float32)
    cursor += num_probabilities * 4

    intensity_bin_ids = np.frombuffer(
        raw[cursor:cursor + num_intensity_types * num_probabilities * 4],
        dtype=np.int32)

    return (areaperil_ids, cum_offsets, probabilities, intensity_bin_ids,
            num_intensity_types)


def _decode_ofpt_file(file_path, reader_interface=NpMemMap):
    """Read all events from a single OFPT binary file.

    Parses the file footer, slot table, and event footer block, then
    decodes each event's chunks into columnar arrays.

    Args:
        file_path (str): Path to the .ofpt file.
        reader_interface: Reader class (default NpMemMap).

    Yields:
        tuple: (event_id, list_of_chunk_data) where each chunk_data is
            (areaperil_ids, cum_offsets, probabilities, intensity_bin_ids,
             num_intensity_types).
    """
    reader = reader_interface(file_path, mode='r')
    buffer = reader.read_bytes_range(-BUFFER_READ_BYTES, None)
    backward_cursor = len(buffer)

    trailing_locator, backward_cursor = complex_reverse_mv_read(
        buffer, backward_cursor, trailing_locator_dtype)
    assert trailing_locator['magic'] == magic_file

    buffer_offset = trailing_locator['file_total_size'] - len(buffer)

    file_footer, backward_cursor = complex_reverse_mv_read(
        buffer, backward_cursor, file_footer_dtype)
    assert file_footer['magic'] == magic_footer

    ap_width = int(file_footer['areaperil_width'])

    slot_table, backward_cursor = complex_reverse_mv_read(
        buffer, backward_cursor, slot_table_dtype, slot_table_len)

    if backward_cursor < file_footer["event_footer_block_size"]:
        raise NotImplementedError(
            "Event footer block exceeds buffer — need multi-read (not yet supported)")

    event_footer_bloc = buffer[
        backward_cursor - file_footer["event_footer_block_size"]:backward_cursor
    ]

    for slot_i in range(slot_table_len):
        if slot_table[slot_i]['event_footer_size'] == 0:
            continue

        event_start = slot_table[slot_i]['event_footer_offset']
        event_end = event_start + slot_table[slot_i]['event_footer_size']
        event_footer = event_footer_bloc[event_start:event_end]

        header = np.frombuffer(
            event_footer[:event_footer_bloc_header_dtype_u4.itemsize],
            dtype=event_footer_bloc_header_dtype_u4
        )[0]
        assert header['magic'] == magic_event

        event_id = int(header['event_id'])
        num_chunks = int(header['num_chunks'])

        chunk_dir = np.frombuffer(
            event_footer[
                event_footer_bloc_header_dtype_u4.itemsize:
                event_footer_bloc_header_dtype_u4.itemsize
                + event_footer_bloc_chunk_dtype_ap_u4.itemsize * num_chunks
            ],
            dtype=event_footer_bloc_chunk_dtype_ap_u4,
        )

        chunks_data = []
        for chunk_entry in chunk_dir:
            chunk_start = int(chunk_entry['offset'])
            chunk_size = int(chunk_entry['compressed_size'])
            decompressed_size = int(chunk_entry['decompressed_size'])
            k = int(chunk_entry['num_intensity_types'])

            if chunk_start < buffer_offset or chunk_start + chunk_size > buffer_offset + len(buffer):
                raw_compressed = reader.read_bytes_range(chunk_start, chunk_start + chunk_size)
            else:
                buf_start = chunk_start - buffer_offset
                raw_compressed = buffer[buf_start:buf_start + chunk_size]

            raw = pa.decompress(
                raw_compressed, decompressed_size, codec='zstd')
            (areaperil_ids, cum_offsets, probabilities,
             intensity_bin_ids, _k) = _decode_chunk_py(raw, ap_width)
            chunks_data.append(
                (areaperil_ids, cum_offsets, probabilities, intensity_bin_ids, k))

        yield event_id, chunks_data


def _chunks_to_nested_arrays(event_id, chunks_data):
    """Convert decoded chunk data for one event into nested Parquet arrays.

    Args:
        event_id (int): The event identifier.
        chunks_data (list): List of (areaperil_ids, cum_offsets, probabilities,
            intensity_bin_ids, k) tuples.

    Returns:
        tuple: (event_ids, areaperils_ids, probabilities_list,
            intensity_bin_ids_list, has_uncertainty, max_bin, k_max).
    """
    event_ids = []
    areaperils_ids = []
    probabilities_list = []
    intensity_bin_ids_list = []
    has_uncertainty = False
    max_bin = 0

    for areaperil_ids, cum_offsets, probabilities, intensity_bin_ids, k in chunks_data:
        m = len(areaperil_ids)
        t = int(cum_offsets[-1])

        if len(intensity_bin_ids) > 0:
            chunk_max = int(np.max(intensity_bin_ids))
            if chunk_max > max_bin:
                max_bin = chunk_max

        for i in range(m):
            start = int(cum_offsets[i])
            end = int(cum_offsets[i + 1])
            n = end - start

            if n > 1:
                has_uncertainty = True

            probs = probabilities[start:end].astype(np.float32).tolist()
            probabilities_list.append(probs)

            bins = []
            for ki in range(k):
                bins.extend(
                    intensity_bin_ids[ki * t + start:ki * t + end]
                    .astype(np.int32).tolist()
                )
            intensity_bin_ids_list.append(bins)

            event_ids.append(np.int32(event_id))
            areaperils_ids.append(np.uint32(int(areaperil_ids[i])))

    return (event_ids, areaperils_ids, probabilities_list,
            intensity_bin_ids_list, has_uncertainty, max_bin)


def _chunks_to_flat_arrays(event_id, chunks_data):
    """Convert decoded chunk data for one event into flat Parquet arrays.

    Args:
        event_id (int): The event identifier.
        chunks_data (list): List of (areaperil_ids, cum_offsets, probabilities,
            intensity_bin_ids, k) tuples.

    Returns:
        tuple: (flat_event_ids, flat_areaperils_ids, flat_scenario_indices,
            flat_intensity_indices, flat_bin_ids, flat_probs,
            has_uncertainty, max_bin).
    """
    flat_event_ids = []
    flat_areaperils_ids = []
    flat_scenario_indices = []
    flat_intensity_indices = []
    flat_bin_ids = []
    flat_probs = []
    has_uncertainty = False
    max_bin = 0

    for areaperil_ids, cum_offsets, probabilities, intensity_bin_ids, k in chunks_data:
        m = len(areaperil_ids)
        t = int(cum_offsets[-1])

        if len(intensity_bin_ids) > 0:
            chunk_max = int(np.max(intensity_bin_ids))
            if chunk_max > max_bin:
                max_bin = chunk_max

        for i in range(m):
            start = int(cum_offsets[i])
            end = int(cum_offsets[i + 1])
            n = end - start

            if n > 1:
                has_uncertainty = True

            apid = int(areaperil_ids[i])
            for s in range(n):
                prob = float(probabilities[start + s])
                for ki in range(k):
                    bin_id = int(intensity_bin_ids[ki * t + start + s])
                    flat_event_ids.append(event_id)
                    flat_areaperils_ids.append(apid)
                    flat_scenario_indices.append(s)
                    flat_intensity_indices.append(ki)
                    flat_bin_ids.append(bin_id)
                    flat_probs.append(prob)

    return (flat_event_ids, flat_areaperils_ids, flat_scenario_indices,
            flat_intensity_indices, flat_bin_ids, flat_probs,
            has_uncertainty, max_bin)


def _build_parquet_metadata(n_events, n_aps, max_bin, has_uncertainty,
                            k_max, schema_type, areaperil_width=4):
    """Build Oasis file-level metadata for V2 Parquet from binary stats.

    Args:
        n_events (int): Number of unique events.
        n_aps (int): Number of unique areaperils.
        max_bin (int): Maximum intensity bin ID.
        has_uncertainty (bool): Whether any areaperil has N > 1 scenarios.
        k_max (int): Maximum number of intensity types across all chunks.
        schema_type (str): "nested" or "flat".
        areaperil_width (int): 4 or 8.

    Returns:
        dict: Metadata key-value pairs (bytes).
    """
    return {
        b"oasis:format": b"footprint_v2_parquet",
        b"oasis:schema": schema_type.encode(),
        b"oasis:version": b"1",
        b"oasis:max_intensity_bin_id": str(max_bin).encode(),
        b"oasis:has_intensity_uncertainty": str(int(has_uncertainty)).encode(),
        b"oasis:areaperil_width": str(areaperil_width).encode(),
        b"oasis:compression": b"zstd",
        b"oasis:max_num_intensity_types": str(k_max).encode(),
        b"oasis:total_events": str(n_events).encode(),
        b"oasis:total_areaperils": str(n_aps).encode(),
    }


def ofpt_to_nested_parquet(ofpt_root_dir, parquet_root_dir,
                           reader_interface=NpMemMap):
    """Convert an OFPT directory tree to nested Parquet files.

    One Parquet row per (event_id, areaperil_id) group. Probabilities and
    intensity_bin_ids are stored as list columns.

    Args:
        ofpt_root_dir (str): Root directory containing footprint/{B3}/{B2}/{B1}.ofpt.
        parquet_root_dir (str): Root output directory for .parquet files.
        reader_interface: Reader class (default NpMemMap).
    """
    _ofpt_to_parquet_tree(ofpt_root_dir, parquet_root_dir, "nested",
                          reader_interface)


def ofpt_to_flat_parquet(ofpt_root_dir, parquet_root_dir,
                         reader_interface=NpMemMap):
    """Convert an OFPT directory tree to flat Parquet files.

    One Parquet row per (event_id, areaperil_id, scenario, intensity_type).

    Args:
        ofpt_root_dir (str): Root directory containing footprint/{B3}/{B2}/{B1}.ofpt.
        parquet_root_dir (str): Root output directory for .parquet files.
        reader_interface: Reader class (default NpMemMap).
    """
    _ofpt_to_parquet_tree(ofpt_root_dir, parquet_root_dir, "flat",
                          reader_interface)


def _ofpt_to_parquet_tree(ofpt_root_dir, parquet_root_dir, schema_type,
                          reader_interface=NpMemMap):
    """Convert an OFPT directory tree to a Parquet directory tree.

    Walks the OFPT tree, reads each .ofpt file, decodes all events, and
    writes corresponding .parquet files with the same directory layout.

    Args:
        ofpt_root_dir (str): Root directory with footprint/ OFPT tree.
        parquet_root_dir (str): Root output directory for .parquet tree.
        schema_type (str): "nested" or "flat".
        reader_interface: Reader class (default NpMemMap).

    Raises:
        ValueError: If schema_type is not "nested" or "flat".
        FileNotFoundError: If no .ofpt files found.
    """
    if schema_type not in ("nested", "flat"):
        raise ValueError(
            f"schema_type must be 'nested' or 'flat', got {schema_type!r}")

    fp_dir = os.path.join(ofpt_root_dir, "footprint")
    if not os.path.isdir(fp_dir):
        raise FileNotFoundError(f"Footprint directory not found: {fp_dir}")

    ofpt_files = []
    for dirpath, _dirnames, filenames in os.walk(fp_dir):
        for fname in filenames:
            if fname.endswith(".ofpt"):
                ofpt_files.append(os.path.join(dirpath, fname))

    if not ofpt_files:
        raise FileNotFoundError(f"No .ofpt files found under {fp_dir}")

    ofpt_files.sort()
    total_files = len(ofpt_files)

    for file_idx, ofpt_path in enumerate(ofpt_files):
        rel_path = os.path.relpath(ofpt_path, ofpt_root_dir)
        parquet_rel = os.path.splitext(rel_path)[0] + ".parquet"
        parquet_path = os.path.join(parquet_root_dir, parquet_rel)

        logger.info("Converting %s (%d/%d)", ofpt_path, file_idx + 1,
                    total_files)

        if schema_type == "nested":
            _convert_file_nested(ofpt_path, parquet_path, reader_interface)
        else:
            _convert_file_flat(ofpt_path, parquet_path, reader_interface)


def _convert_file_nested(ofpt_path, parquet_path, reader_interface):
    """Convert a single OFPT file to nested Parquet.

    Args:
        ofpt_path (str): Path to input .ofpt file.
        parquet_path (str): Path to output .parquet file.
        reader_interface: Reader class.
    """
    all_event_ids = []
    all_areaperils_ids = []
    all_probabilities = []
    all_intensity_bin_ids = []
    has_uncertainty = False
    max_bin = 0
    k_max = 0
    n_events = 0
    ap_set = set()

    for event_id, chunks_data in _decode_ofpt_file(ofpt_path, reader_interface):
        n_events += 1
        for _, _, _, _, k in chunks_data:
            if k > k_max:
                k_max = k

        (eids, apids, probs_list, bins_list,
         chunk_unc, chunk_max) = _chunks_to_nested_arrays(event_id, chunks_data)

        all_event_ids.extend(eids)
        all_areaperils_ids.extend(apids)
        all_probabilities.extend(probs_list)
        all_intensity_bin_ids.extend(bins_list)

        if chunk_unc:
            has_uncertainty = True
        if chunk_max > max_bin:
            max_bin = chunk_max

        for apid in apids:
            ap_set.add(int(apid))

    if n_events == 0:
        return

    metadata = _build_parquet_metadata(
        n_events, len(ap_set), max_bin, has_uncertainty, k_max, "nested")

    schema = _build_nested_schema().with_metadata(metadata)

    table = pa.table({
        "event_id": pa.array(all_event_ids, type=pa.int32()),
        "areaperils_id": pa.array(all_areaperils_ids, type=pa.uint32()),
        "probabilities": pa.array(
            all_probabilities,
            type=pa.list_(pa.field("item", pa.float32(), nullable=False))),
        "intensity_bin_ids": pa.array(
            all_intensity_bin_ids,
            type=pa.list_(pa.field("item", pa.int32(), nullable=False))),
    }, schema=schema)

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    pq.write_table(
        table, parquet_path,
        compression="ZSTD",
        write_statistics=True,
        row_group_size=64 * 1024,
    )


def _convert_file_flat(ofpt_path, parquet_path, reader_interface):
    """Convert a single OFPT file to flat Parquet.

    Args:
        ofpt_path (str): Path to input .ofpt file.
        parquet_path (str): Path to output .parquet file.
        reader_interface: Reader class.
    """
    all_event_ids = []
    all_areaperils_ids = []
    all_scenario_indices = []
    all_intensity_indices = []
    all_bin_ids = []
    all_probs = []
    has_uncertainty = False
    max_bin = 0
    k_max = 0
    n_events = 0
    ap_set = set()

    for event_id, chunks_data in _decode_ofpt_file(ofpt_path, reader_interface):
        n_events += 1
        for _, _, _, _, k in chunks_data:
            if k > k_max:
                k_max = k

        (eids, apids, sidxs, iidxs, bids, prbs,
         chunk_unc, chunk_max) = _chunks_to_flat_arrays(event_id, chunks_data)

        all_event_ids.extend(eids)
        all_areaperils_ids.extend(apids)
        all_scenario_indices.extend(sidxs)
        all_intensity_indices.extend(iidxs)
        all_bin_ids.extend(bids)
        all_probs.extend(prbs)

        if chunk_unc:
            has_uncertainty = True
        if chunk_max > max_bin:
            max_bin = chunk_max

        ap_set.update(apids)

    if n_events == 0:
        return

    metadata = _build_parquet_metadata(
        n_events, len(ap_set), max_bin, has_uncertainty, k_max, "flat")

    schema = _build_flat_schema().with_metadata(metadata)

    table = pa.table({
        "event_id": pa.array(all_event_ids, type=pa.int32()),
        "areaperils_id": pa.array(all_areaperils_ids, type=pa.uint32()),
        "scenario_index": pa.array(all_scenario_indices, type=pa.int32()),
        "intensity_index": pa.array(all_intensity_indices, type=pa.int32()),
        "intensity_bin_id": pa.array(all_bin_ids, type=pa.int32()),
        "probability": pa.array(all_probs, type=pa.float32()),
    }, schema=schema)

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    pq.write_table(
        table, parquet_path,
        compression="ZSTD",
        write_statistics=True,
        row_group_size=64 * 1024,
    )


def ofpt_file_to_nested_parquet(ofpt_path, parquet_path,
                                reader_interface=NpMemMap):
    """Convert a single OFPT binary file to nested Parquet.

    Args:
        ofpt_path (str): Path to input .ofpt file.
        parquet_path (str): Path to output .parquet file.
        reader_interface: Reader class (default NpMemMap).
    """
    _convert_file_nested(ofpt_path, parquet_path, reader_interface)


def ofpt_file_to_flat_parquet(ofpt_path, parquet_path,
                              reader_interface=NpMemMap):
    """Convert a single OFPT binary file to flat Parquet.

    Args:
        ofpt_path (str): Path to input .ofpt file.
        parquet_path (str): Path to output .parquet file.
        reader_interface: Reader class (default NpMemMap).
    """
    _convert_file_flat(ofpt_path, parquet_path, reader_interface)


def v1_bin_to_nested_parquet(bin_root_dir, parquet_root_dir,
                             reader_interface=NpMemMap):
    """Convert V1 footprint.bin (+ index) directly to nested Parquet tree.

    First converts V1 binary to OFPT columnar in memory, then writes
    Parquet. Uses BinScanner to iterate events in sorted order.

    Args:
        bin_root_dir (str): Directory containing footprint.bin and footprint.idx.
        parquet_root_dir (str): Root output directory for .parquet tree.
        reader_interface: Reader class (default NpMemMap).
    """
    _v1_bin_to_parquet_tree(bin_root_dir, parquet_root_dir, "nested",
                            reader_interface)


def v1_bin_to_flat_parquet(bin_root_dir, parquet_root_dir,
                           reader_interface=NpMemMap):
    """Convert V1 footprint.bin (+ index) directly to flat Parquet tree.

    Args:
        bin_root_dir (str): Directory containing footprint.bin and footprint.idx.
        parquet_root_dir (str): Root output directory for .parquet tree.
        reader_interface: Reader class (default NpMemMap).
    """
    _v1_bin_to_parquet_tree(bin_root_dir, parquet_root_dir, "flat",
                            reader_interface)


def _v1_bin_to_parquet_tree(bin_root_dir, parquet_root_dir, schema_type,
                            reader_interface=NpMemMap):
    """Convert V1 footprint binary to a Parquet directory tree.

    Reads from footprint.bin / footprint.bin.z using BinScanner,
    converts each event's data to columnar form, groups events by
    path key (upper 3 bytes), and writes one Parquet file per group.

    Args:
        bin_root_dir (str): Directory with footprint.bin and footprint.idx.
        parquet_root_dir (str): Root output directory.
        schema_type (str): "nested" or "flat".
        reader_interface: Reader class.
    """
    if schema_type not in ("nested", "flat"):
        raise ValueError(
            f"schema_type must be 'nested' or 'flat', got {schema_type!r}")

    scanner = BinScanner(bin_root_dir, reader_interface)

    cur_path_key = None
    cur_events = []

    def flush_events(events, path_key):
        """Write accumulated events to a single Parquet file."""
        if not events:
            return

        sample_eid = events[0][0]
        rel_path = event_id_to_path(sample_eid, ext="parquet")
        abs_path = os.path.join(parquet_root_dir, rel_path)

        if schema_type == "nested":
            _flush_nested(events, abs_path)
        else:
            _flush_flat(events, abs_path)

    for event_info, event_data in scanner.sorted_iter():
        event_id = int(event_info['event_id'])
        path_key = (event_id >> 8) & 0xFFFFFF

        if cur_path_key is not None and path_key != cur_path_key:
            flush_events(cur_events, cur_path_key)
            cur_events = []

        cur_path_key = path_key
        areaperil_ids, cum_offsets, probabilities, intensity_bin_ids = \
            event_to_columnar(event_data)
        # event_to_columnar returns intensity_bin_ids as (T, K) row-major.
        # Convert to (K*T,) column-major to match the OFPT/Parquet convention.
        k = intensity_bin_ids.shape[1] if intensity_bin_ids.ndim == 2 else 1
        if intensity_bin_ids.ndim == 2:
            t = intensity_bin_ids.shape[0]
            # Column-major: all type-0 bins, then type-1, etc.
            intensity_bin_ids = intensity_bin_ids.T.ravel()
        cur_events.append(
            (event_id, areaperil_ids, cum_offsets, probabilities,
             intensity_bin_ids, k))

    flush_events(cur_events, cur_path_key)


def _flush_nested(events, parquet_path):
    """Write a list of events as a single nested Parquet file.

    Args:
        events (list): List of (event_id, areaperil_ids, cum_offsets,
            probabilities, intensity_bin_ids, k) tuples.
        parquet_path (str): Output path.
    """
    all_event_ids = []
    all_areaperils_ids = []
    all_probabilities = []
    all_intensity_bin_ids = []
    has_uncertainty = False
    max_bin = 0
    k_max = 0
    ap_set = set()

    for event_id, areaperil_ids, cum_offsets, probabilities, intensity_bin_ids, k in events:
        if k > k_max:
            k_max = k

        m = len(areaperil_ids)
        t = int(cum_offsets[-1])

        if len(intensity_bin_ids) > 0:
            chunk_max = int(np.max(intensity_bin_ids))
            if chunk_max > max_bin:
                max_bin = chunk_max

        for i in range(m):
            start = int(cum_offsets[i])
            end = int(cum_offsets[i + 1])
            n = end - start

            if n > 1:
                has_uncertainty = True

            probs = probabilities[start:end].astype(np.float32).tolist()
            all_probabilities.append(probs)

            bins = []
            for ki in range(k):
                bins.extend(
                    intensity_bin_ids[ki * t + start:ki * t + end]
                    .astype(np.int32).tolist()
                )
            all_intensity_bin_ids.append(bins)

            all_event_ids.append(np.int32(event_id))
            all_areaperils_ids.append(np.uint32(int(areaperil_ids[i])))
            ap_set.add(int(areaperil_ids[i]))

    metadata = _build_parquet_metadata(
        len(events), len(ap_set), max_bin, has_uncertainty, k_max, "nested")

    schema = _build_nested_schema().with_metadata(metadata)

    table = pa.table({
        "event_id": pa.array(all_event_ids, type=pa.int32()),
        "areaperils_id": pa.array(all_areaperils_ids, type=pa.uint32()),
        "probabilities": pa.array(
            all_probabilities,
            type=pa.list_(pa.field("item", pa.float32(), nullable=False))),
        "intensity_bin_ids": pa.array(
            all_intensity_bin_ids,
            type=pa.list_(pa.field("item", pa.int32(), nullable=False))),
    }, schema=schema)

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    pq.write_table(
        table, parquet_path,
        compression="ZSTD",
        write_statistics=True,
        row_group_size=64 * 1024,
    )


def _flush_flat(events, parquet_path):
    """Write a list of events as a single flat Parquet file.

    Args:
        events (list): List of (event_id, areaperil_ids, cum_offsets,
            probabilities, intensity_bin_ids, k) tuples.
        parquet_path (str): Output path.
    """
    all_event_ids = []
    all_areaperils_ids = []
    all_scenario_indices = []
    all_intensity_indices = []
    all_bin_ids = []
    all_probs = []
    has_uncertainty = False
    max_bin = 0
    k_max = 0
    ap_set = set()

    for event_id, areaperil_ids, cum_offsets, probabilities, intensity_bin_ids, k in events:
        if k > k_max:
            k_max = k

        m = len(areaperil_ids)
        t = int(cum_offsets[-1])

        if len(intensity_bin_ids) > 0:
            chunk_max = int(np.max(intensity_bin_ids))
            if chunk_max > max_bin:
                max_bin = chunk_max

        for i in range(m):
            start = int(cum_offsets[i])
            end = int(cum_offsets[i + 1])
            n = end - start

            if n > 1:
                has_uncertainty = True

            apid = int(areaperil_ids[i])
            ap_set.add(apid)
            for s in range(n):
                prob = float(probabilities[start + s])
                for ki in range(k):
                    bin_id = int(intensity_bin_ids[ki * t + start + s])
                    all_event_ids.append(event_id)
                    all_areaperils_ids.append(apid)
                    all_scenario_indices.append(s)
                    all_intensity_indices.append(ki)
                    all_bin_ids.append(bin_id)
                    all_probs.append(prob)

    metadata = _build_parquet_metadata(
        len(events), len(ap_set), max_bin, has_uncertainty, k_max, "flat")

    schema = _build_flat_schema().with_metadata(metadata)

    table = pa.table({
        "event_id": pa.array(all_event_ids, type=pa.int32()),
        "areaperils_id": pa.array(all_areaperils_ids, type=pa.uint32()),
        "scenario_index": pa.array(all_scenario_indices, type=pa.int32()),
        "intensity_index": pa.array(all_intensity_indices, type=pa.int32()),
        "intensity_bin_id": pa.array(all_bin_ids, type=pa.int32()),
        "probability": pa.array(all_probs, type=pa.float32()),
    }, schema=schema)

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    pq.write_table(
        table, parquet_path,
        compression="ZSTD",
        write_statistics=True,
        row_group_size=64 * 1024,
    )


def main():
    """CLI entry point for bin-to-parquet conversion."""
    parser = argparse.ArgumentParser(
        description="Convert V1 footprint.bin or OFPT binary to V2 Parquet tree.")
    parser.add_argument(
        "input_dir",
        help="Input directory (model_data root for V1, or OFPT root for --source ofpt)")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output root directory (default: same as input_dir)")
    parser.add_argument(
        "--schema", choices=["nested", "flat"], default="nested",
        help="Parquet schema type (default: nested)")
    parser.add_argument(
        "--source", choices=["v1", "ofpt"], default="v1",
        help="Source format: v1 (footprint.bin) or ofpt (default: v1)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir else input_dir

    if args.source == "v1":
        if args.schema == "nested":
            v1_bin_to_nested_parquet(input_dir, output_dir)
        else:
            v1_bin_to_flat_parquet(input_dir, output_dir)
    else:
        if args.schema == "nested":
            ofpt_to_nested_parquet(input_dir, output_dir)
        else:
            ofpt_to_flat_parquet(input_dir, output_dir)


if __name__ == "__main__":
    main()
