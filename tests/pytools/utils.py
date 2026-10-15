from pathlib import Path

import numpy as np
from oasislmf.pytools.common.data import (COVERAGE_BUILDINGS_FILE, coverage_buildings_dtype,
                                          summary_stream_index_dtype)
from oasislmf.pytools.common.input_files import read_coverages


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
