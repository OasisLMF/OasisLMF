from pathlib import Path

import numpy as np
from oasislmf.pytools.common.data import summary_stream_index_dtype


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
