import numpy as np

from oasislmf.utils.exceptions import OasisException


def check_index_bounds(data, idx, header_size, id_field, label, record_size=1):
    """Raise a clear error if any idx entry's byte range falls outside the data file's data
    (i.e. before the end of the header, or past the end of the file), instead of letting the
    read below silently clip, read mmap zero-padding, or -- for the batched njit loops, which
    index without bounds checking -- read past the mapped file entirely (undefined behaviour,
    up to and including a crash).

    Also rejects an entry whose offset (past the header) or size isn't a whole multiple of
    record_size: an in-bounds but misaligned offset doesn't raise on its own -- it gets silently
    floor-divided into the wrong row index downstream, reading another entry's rows without error.

    Args:
        data (np.ndarray): memory-mapped binary file content, viewed as bytes
        idx (np.ndarray): index entries, each with 'offset' and 'size' fields
        header_size (int): size in bytes of the fixed header preceding the data
        id_field (str): name of idx's own id field (e.g. 'event_id', 'vulnerability_id')
        label (str): human-readable name of the file type, used in the error message
        record_size (int): size in bytes of one atomic record in the data; defaults to 1 (no
            alignment check) for callers with no fixed record size to check against
    """
    starts = idx['offset'].astype(np.int64)
    sizes = idx['size'].astype(np.int64)
    ends = starts + sizes
    bad = np.flatnonzero(
        (starts < header_size) | (ends > len(data))
        | ((starts - header_size) % record_size != 0) | (sizes % record_size != 0)
    )
    if bad.size:
        i = int(bad[0])
        raise OasisException(
            f"Error: {label} index entry for {id_field} {int(idx[id_field][i])} references "
            f"bytes [{int(starts[i])}, {int(ends[i])}), which is outside the {label} file's data "
            f"[{header_size}, {len(data)}) or not aligned to its {record_size}-byte records; the "
            f"index and {label} files do not match, or the {label} file is truncated."
        )
