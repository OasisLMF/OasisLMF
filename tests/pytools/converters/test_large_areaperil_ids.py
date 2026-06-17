"""Regression tests for large AreaPeril IDs exceeding uint32 range.

The default OASIS_AREAPERIL_TYPE is u4 (uint32, max 4,294,967,295). Models with
AreaPeril IDs exceeding this limit produce silent data corruption: pandas 3.x
silently wraps the value mod 2^32 rather than raising an error.

Setting OASIS_AREAPERIL_TYPE=u8 (uint64) before importing oasislmf fixes this.
The carry-state scalars in the footprint validator must also use the correct type
so Numba JIT functions receive consistent argument types across chunk boundaries.

Reference: https://github.com/OasisLMF/OasisLMF/issues/2011
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

import oasislmf.pytools.converters.csvtobin.utils.footprint as footprint_utils
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.converters.data import TOOL_INFO

LARGE_AREAPERIL_ID = 50_776_441_987  # 50,776,441,987 > uint32 max (4,294,967,295)

FOOTPRINT_CSV = (
    b"event_id,areaperil_id,intensity_bin_id,probability\n"
    b"1,50776441987,1,1.0\n"
)

_CSVTOBIN_KWARGS = dict(
    file_type="footprint",
    zip_files=False,
    max_intensity_bin_idx=1,
    no_intensity_uncertainty=True,
    decompressed_size=False,
    no_validation=False,
)

_U8_FOOTPRINT_DTYPE = np.dtype([
    ("event_id", np.int32),
    ("areaperil_id", np.uint64),
    ("intensity_bin_id", np.int32),
    ("probability", np.float32),
])

_U8_EVENT_DTYPE = np.dtype([
    ("areaperil_id", np.uint64),
    ("intensity_bin_id", np.int32),
    ("probability", np.float32),
])


def test_uint32_silently_truncates_large_areaperil_id():
    """pandas silently wraps areaperil_id values beyond 4,294,967,295 when dtype=uint32.

    This is the root cause of issue #2011: pd.read_csv with dtype=uint32 silently
    truncates large integer values rather than raising an error.

    50,776,441,987 % 2^32 = 3,531,801,731
    """
    import io
    import pandas as pd

    df = pd.read_csv(
        io.BytesIO(FOOTPRINT_CSV),
        dtype={"areaperil_id": np.uint32},
    )
    stored = int(df["areaperil_id"][0])

    assert stored != LARGE_AREAPERIL_ID
    assert stored == LARGE_AREAPERIL_ID % (2 ** 32)


def test_large_areaperil_id_preserved_with_uint64():
    """uint64 preserves areaperil_id values > uint32 max through the full conversion.

    Simulates OASIS_AREAPERIL_TYPE=u8 by patching the three dtype objects that
    the footprint converter uses: the CSV read dtype, the carry-state areaperil
    type, and the binary output dtype. All must agree for Numba JIT validation to
    receive consistent argument types across chunk boundaries.
    """
    with (
        TemporaryDirectory() as tmp,
        mock.patch.dict(TOOL_INFO["footprint"], {"dtype": _U8_FOOTPRINT_DTYPE}),
        mock.patch.object(footprint_utils, "areaperil_int", np.dtype("u8")),
        mock.patch.object(footprint_utils, "Event_dtype", _U8_EVENT_DTYPE),
    ):
        csv_path = Path(tmp) / "footprint.csv"
        csv_path.write_bytes(FOOTPRINT_CSV)

        csvtobin(
            file_in=csv_path,
            file_out=Path(tmp) / "footprint.bin",
            idx_file_out=Path(tmp) / "footprint.idx",
            **_CSVTOBIN_KWARGS,
        )

        data = np.fromfile(Path(tmp) / "footprint.bin", dtype=_U8_EVENT_DTYPE, offset=8)
        stored_id = int(data["areaperil_id"][0])

    assert stored_id == LARGE_AREAPERIL_ID
