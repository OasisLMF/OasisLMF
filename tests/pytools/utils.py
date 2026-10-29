import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from oasislmf.pytools.common.data import (COVERAGE_BUILDINGS_FILE, correlations_dtype, coverage_buildings_dtype,
                                          oasis_float, summary_stream_index_dtype)
from oasislmf.pytools.common.event_stream import bytes_to_stream_types
from oasislmf.pytools.common.input_files import read_coverages

TEST_MODEL_1 = Path(__file__).parents[1].joinpath("assets", "test_model_1")


def make_idx_from_bin(bin_path: Path, idx_path: Path) -> None:
    """Scan a summary .bin and write a paired .idx recording each event block's byte offset.

    If the bin contains no data blocks (header only), creates a 0-byte idx to match
    summarypy --low-memory behaviour for partitions that received no events.
    """
    raw = np.fromfile(str(bin_path), dtype=np.int32)
    pos = 3  # skip 3-int stream header (stream_type, sample_size, summary_set_id)
    entries = []
    while pos < len(raw):
        byte_offset = pos * 4
        summary_id = int(raw[pos + 1])
        pos += 3  # event_id, summary_id, expval
        while pos < len(raw) and raw[pos] != 0:
            pos += 2  # sidx + loss pair (any non-zero sidx, including special negatives)
        pos += 2  # terminating (sidx=0, loss=0.0)
        entries.append((summary_id, byte_offset))
    if entries:
        np.array(entries, dtype=summary_stream_index_dtype).tofile(str(idx_path))
    else:
        idx_path.touch()  # empty partition → 0-byte idx


def set_coverage_buildings(input_dir, n_building, item_to_coverage=None):
    """Write a test input dir's coverage_buildings file with the given signed building count.

    The count lives on the COVERAGE and in a file of its own -- coverages.bin stays the published
    tiv-only format -- so a fixture that wants "N buildings on every item" has to say it per
    coverage. ``n_building`` is either one value for every coverage, or an array indexed by
    ``coverage_id - 1``. Pass ``item_to_coverage`` (the items table's ``coverage_id`` column,
    parallel to a per-item array) to spread per-item intent onto the coverages instead.

    Returns the signed per-coverage array that was written.
    """
    input_dir = Path(input_dir)
    n_coverages = read_coverages(input_dir).shape[0]
    out = np.zeros(n_coverages, dtype=coverage_buildings_dtype)
    out['coverage_id'] = np.arange(1, n_coverages + 1)
    out['n_building'] = 1
    if item_to_coverage is not None:
        wanted = np.asarray(n_building, dtype='i4')
        if wanted.ndim == 0:
            wanted = np.full(len(item_to_coverage), int(wanted), dtype='i4')
        out['n_building'][np.asarray(item_to_coverage, dtype='i8') - 1] = wanted
    else:
        out['n_building'] = np.asarray(n_building, dtype='i4')

    out.tofile(input_dir / f'{COVERAGE_BUILDINGS_FILE}.bin')
    return out['n_building']


def copy_test_model(dst, src=TEST_MODEL_1):
    """Copy a test model to ``dst`` without its cached gulmc/gulpy structures.

    A cached structure is loaded in preference to the input files, so a fixture that edits the
    inputs of a copy that kept one would run on the original inputs.
    """
    dst = Path(dst)
    shutil.copytree(src, dst)
    for cache in ('gulmc_structure', 'gulpy_structure'):
        shutil.rmtree(dst / 'input' / cache, ignore_errors=True)
    return dst


def write_correlations(input_dir, item_ids, damage_correlation_value, peril_correlation_group=1,
                       source_item_id=0):
    """Write correlations.bin and .csv for ``item_ids``, every other field at its default.

    The scalar arguments may also be arrays parallel to ``item_ids``.
    """
    corr = np.zeros(len(item_ids), dtype=correlations_dtype)
    corr['item_id'] = item_ids
    corr['peril_correlation_group'] = peril_correlation_group
    corr['damage_correlation_value'] = damage_correlation_value
    corr['hazard_group_id'] = 1
    corr['source_item_id'] = source_item_id
    corr.tofile(Path(input_dir) / 'correlations.bin')
    pd.DataFrame({k: corr[k] for k in corr.dtype.names}).to_csv(Path(input_dir) / 'correlations.csv', index=False)


def read_gul_stream(path):
    """Read a gul loss stream into its aggregation type and a frame of its records, in order.

    Returns ``(agg_type, sample_size, df)`` with ``df`` holding event_id, item_id, sidx and loss,
    one row per record and no delimiters, and ``header``: which item header the record follows,
    counted from 0, so that an item written in two pieces shows as two headers.
    """
    raw = Path(path).read_bytes()
    _, agg_type = bytes_to_stream_types(raw[:4])
    sample_size = int(np.frombuffer(raw[4:8], dtype='<i4')[0])
    ints = np.frombuffer(raw[8:], dtype='<i4')
    losses = np.frombuffer(raw[8:], dtype=oasis_float)
    rows, pos, header = [], 0, 0
    while pos < len(ints):
        event_id, item_id = int(ints[pos]), int(ints[pos + 1])
        pos += 2
        while ints[pos] != 0:
            rows.append((event_id, item_id, int(ints[pos]), float(losses[pos + 1]), header))
            pos += 2
        pos += 2
        header += 1
    return int(agg_type), sample_size, pd.DataFrame(rows, columns=['event_id', 'item_id', 'sidx', 'loss', 'header'])


def packed_gul_stream(run_dir, n_building, sample_size=10):
    """Run gulmc on a copy of test_model_1 whose every coverage has ``n_building`` buildings.

    Returns the copy and the path of the gul stream it wrote: building-packed when ``n_building``
    is negative, the form plapy and summarypy then read.
    """
    from oasislmf.pytools.gulmc.manager import run as run_gulmc

    run_dir = copy_test_model(run_dir)
    set_coverage_buildings(run_dir / 'input', n_building)
    out = run_dir / 'gul.bin'
    run_gulmc(run_dir=run_dir, ignore_file_type=set(), file_in=run_dir / 'input' / 'events.bin', file_out=out,
              sample_size=sample_size, loss_threshold=0., alloc_rule=1, debug=0, random_generator=2,
              ignore_correlation=False, effective_damageability=False)
    return run_dir, out


def read_summary_stream(path):
    """Read a summarypy output stream into a frame of event_id, summary_id, exposure_value, sidx
    and loss, one row per record."""
    raw = Path(path).read_bytes()
    ints = np.frombuffer(raw, dtype='<i4')
    floats = np.frombuffer(raw, dtype=oasis_float)
    rows, pos = [], 3   # stream type, sample size, summary set id
    while pos < len(ints):
        event_id, summary_id, exposure_value = int(ints[pos]), int(ints[pos + 1]), float(floats[pos + 2])
        pos += 3
        while ints[pos] != 0:
            rows.append((event_id, summary_id, exposure_value, int(ints[pos]), float(floats[pos + 1])))
            pos += 2
        pos += 2
    return pd.DataFrame(rows, columns=['event_id', 'summary_id', 'exposure_value', 'sidx', 'loss'])
