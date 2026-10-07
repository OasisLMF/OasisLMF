import struct
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import oasislmf.pytools.converters.csvtobin.utils.footprint as footprint_utils
from oasislmf.pytools.converters.bintocsv.manager import bintocsv
from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.converters.data import TOOL_INFO
from oasis_data_manager.filestore.backends.local import LocalStorage
from oasislmf.pytools.getmodel.common import Event_dtype, EventIndexBin_dtype, EventIndexBinZ_dtype
from oasislmf.pytools.getmodel.footprint import Footprint, FootprintBin
from oasislmf.utils.exceptions import OasisException
from tests.pytools.converters.helpers import TESTS_ASSETS_DIR, case_runner_tocsv_with_zip_and_idx, case_runner_tobin_with_zip_and_idx


def test_footprint():
    # zip_input = False
    case_runner_tocsv_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint",
        idx_file_in=Path(TESTS_ASSETS_DIR, "static", "footprint.idx"),
        event_from_to="1-3",
        zip_files=False
    )
    case_runner_tobin_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "footprint.idx"),
        max_intensity_bin_idx=3,
        no_intensity_uncertainty=True,
        decompressed_size=False,
        no_validation=False,
        zip_files=False
    )
    # no_validation=True must produce identical output to validated path on clean input
    case_runner_tobin_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "footprint.idx"),
        max_intensity_bin_idx=3,
        no_intensity_uncertainty=True,
        decompressed_size=False,
        no_validation=True,
        zip_files=False
    )

    # zip_input = True
    case_runner_tocsv_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint_zip",
        idx_file_in=Path(TESTS_ASSETS_DIR, "static", "footprint_zip.idx.z"),
        event_from_to="1-3",
        zip_files=True
    )
    case_runner_tobin_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint_zip",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "footprint_zip.idx.z"),
        max_intensity_bin_idx=58,
        no_intensity_uncertainty=False,
        decompressed_size=False,
        no_validation=False,
        zip_files=True
    )
    # no_validation=True must produce identical output to validated path on clean input
    case_runner_tobin_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint_zip",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "footprint_zip.idx.z"),
        max_intensity_bin_idx=58,
        no_intensity_uncertainty=False,
        decompressed_size=False,
        no_validation=True,
        zip_files=True
    )
    # decompressed_size=True: index includes uncompressed size field alongside compressed
    case_runner_tobin_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint_zip_dsize",
        idx_file_out=Path(TESTS_ASSETS_DIR, "static", "footprint_zip_dsize.idx.z"),
        max_intensity_bin_idx=58,
        no_intensity_uncertainty=False,
        decompressed_size=True,
        no_validation=False,
        zip_files=True
    )
    # decompressed_size=True tocsv: exercises pre-allocated buffer path (d_size in index)
    case_runner_tocsv_with_zip_and_idx(
        file_type="footprint",
        sub_dir="static",
        filename="footprint_zip_dsize",
        idx_file_in=Path(TESTS_ASSETS_DIR, "static", "footprint_zip_dsize.idx.z"),
        event_from_to="1-3",
        zip_files=True
    )


def test_footprint_no_validation_unsorted():
    """no_validation=False rejects unsorted input; no_validation=True accepts it and preserves order."""
    unsorted_csv = (
        b"event_id,areaperil_id,intensity_bin_id,probability\n"
        b"2,1,1,1.0\n"
        b"1,1,1,1.0\n"
    )
    with TemporaryDirectory() as tmp:
        csv_in = Path(tmp) / "footprint.csv"
        csv_in.write_bytes(unsorted_csv)

        # validated path must reject unsorted input
        with pytest.raises(OasisException, match="not in ascending order"):
            csvtobin(
                file_in=csv_in,
                file_out=Path(tmp) / "v.bin",
                file_type="footprint",
                idx_file_out=Path(tmp) / "v.idx",
                max_intensity_bin_idx=3,
                no_intensity_uncertainty=True,
                decompressed_size=False,
                no_validation=False,
                zip_files=False,
            )

        # no_validation path must accept unsorted input and preserve input order
        csvtobin(
            file_in=csv_in,
            file_out=Path(tmp) / "n.bin",
            file_type="footprint",
            idx_file_out=Path(tmp) / "n.idx",
            max_intensity_bin_idx=3,
            no_intensity_uncertainty=True,
            decompressed_size=False,
            no_validation=True,
            zip_files=False,
        )
        idx = np.fromfile(Path(tmp) / "n.idx", dtype=EventIndexBin_dtype)
        assert idx["event_id"].tolist() == [2, 1], (
            f"Expected events written in input order [2, 1], got {idx['event_id'].tolist()}"
        )


def test_footprint_unsorted_index():
    """bintocsv must sort events by event_id even when the index is out of order."""
    # Build a binary footprint: header + event 2 data (offset 8) + event 1 data (offset 20)
    header = struct.pack("<ii", 3, 1)  # num_intensity_bins=3, has_intensity_uncertainty=1
    ev2 = np.array([(1, 1, 1.0)], dtype=Event_dtype).tobytes()
    ev1 = np.array([(1, 1, 1.0)], dtype=Event_dtype).tobytes()
    bin_bytes = header + ev2 + ev1

    item_size = Event_dtype.itemsize  # 12
    header_size = 8
    # Index with event 2 first (unsorted order)
    idx = np.array(
        [(2, header_size, item_size), (1, header_size + item_size, item_size)],
        dtype=EventIndexBin_dtype,
    )

    with TemporaryDirectory() as tmp:
        bin_path = Path(tmp) / "footprint.bin"
        idx_path = Path(tmp) / "footprint.idx"
        out_path = Path(tmp) / "footprint.csv"
        bin_path.write_bytes(bin_bytes)
        idx.tofile(idx_path)

        bintocsv(
            file_in=bin_path,
            file_out=out_path,
            file_type="footprint",
            idx_file_in=idx_path,
            zip_files=False,
            event_from_to=None,
        )

        data = np.genfromtxt(out_path, delimiter=",", skip_header=1,
                             dtype=[("event_id", int), ("areaperil_id", int),
                                    ("intensity_bin_id", int), ("probability", float)])
        assert data["event_id"].tolist() == [1, 2], (
            f"Expected sorted output [1, 2], got {data['event_id'].tolist()}"
        )


def test_footprint_decompressed_size_without_zip_is_ignored(caplog):
    """decompressed_size only applies to zipped footprints: without zip_files it is ignored with a
    warning, so the output is identical to a plain uncompressed footprint."""
    kwargs = dict(
        file_in=Path(TESTS_ASSETS_DIR, "static", "footprint.csv"),
        file_type="footprint",
        max_intensity_bin_idx=3,
        no_intensity_uncertainty=True,
        no_validation=False,
        zip_files=False,
    )
    with TemporaryDirectory() as tmp:
        csvtobin(file_out=Path(tmp, "plain.bin"), idx_file_out=Path(tmp, "plain.idx"), decompressed_size=False, **kwargs)
        with caplog.at_level("WARNING"):
            csvtobin(file_out=Path(tmp, "dsize.bin"), idx_file_out=Path(tmp, "dsize.idx"), decompressed_size=True, **kwargs)

        assert "decompressed_size only applies to zipped footprints" in caplog.text
        assert Path(tmp, "dsize.bin").read_bytes() == Path(tmp, "plain.bin").read_bytes()
        assert Path(tmp, "dsize.idx").read_bytes() == Path(tmp, "plain.idx").read_bytes()


def test_footprint_uncompressed_with_decompressed_size_index():
    """Uncompressed footprints written with the decompressed size flag set (4-field index), as older
    csvtobin versions did with decompressed_size and no zip_files, are read by bintocsv and the runtime."""
    static = Path(TESTS_ASSETS_DIR, "static")
    with TemporaryDirectory() as tmp:
        # rebuild the static footprint in the legacy layout: header flag set, idx entries with d_size
        data = bytearray(Path(static, "footprint.bin").read_bytes())
        header = np.frombuffer(bytes(data[:8]), dtype=np.int32).copy()
        header[1] |= 1 << 1
        data[:8] = header.tobytes()
        Path(tmp, "footprint.bin").write_bytes(data)
        idx = np.fromfile(Path(static, "footprint.idx"), dtype=EventIndexBin_dtype)
        idx_dsize = np.empty(len(idx), dtype=EventIndexBinZ_dtype)
        for field in EventIndexBin_dtype.names:
            idx_dsize[field] = idx[field]
        idx_dsize["d_size"] = idx["size"]
        idx_dsize.tofile(Path(tmp, "footprint.idx"))

        bintocsv(
            file_in=Path(tmp, "footprint.bin"),
            file_out=Path(tmp, "footprint.csv"),
            file_type="footprint",
            noheader=False,
            idx_file_in=Path(tmp, "footprint.idx"),
            zip_files=False,
            event_from_to=None,
        )
        expected = pd.read_csv(Path(static, "footprint.csv"))
        pd.testing.assert_frame_equal(pd.read_csv(Path(tmp, "footprint.csv")), expected)

        with Footprint.load(LocalStorage(root_dir=tmp, cache_dir=None)) as footprint:
            assert isinstance(footprint, FootprintBin)
            for event_id, rows in expected.groupby("event_id"):
                event = footprint.get_event(event_id)
                assert event["areaperil_id"].tolist() == rows["areaperil_id"].tolist()
                assert event["intensity_bin_id"].tolist() == rows["intensity_bin_id"].tolist()
                np.testing.assert_allclose(event["probability"], rows["probability"], rtol=1e-6)


@pytest.mark.parametrize("filename", ["footprint_zip", "footprint_zip_dsize"])
def test_footprint_zipped_under_uncompressed_names(filename, caplog):
    """csvtobin does not add .z to zipped output names, so zipped footprints can arrive as
    footprint.bin / footprint.idx. FootprintBin must detect the compression and decompress the events
    instead of reading the compressed bytes as event rows."""
    static = Path(TESTS_ASSETS_DIR, "static")
    expected = pd.read_csv(Path(static, f"{filename}.csv"))
    with TemporaryDirectory() as tmp:
        Path(tmp, "footprint.bin").write_bytes(Path(static, f"{filename}.bin.z").read_bytes())
        Path(tmp, "footprint.idx").write_bytes(Path(static, f"{filename}.idx.z").read_bytes())

        with caplog.at_level("WARNING"), Footprint.load(LocalStorage(root_dir=tmp, cache_dir=None)) as footprint:
            assert isinstance(footprint, FootprintBin)
            assert footprint.compressed
            for event_id, rows in expected.groupby("event_id"):
                event = footprint.get_event(event_id)
                assert event["areaperil_id"].tolist() == rows["areaperil_id"].tolist()
                assert event["intensity_bin_id"].tolist() == rows["intensity_bin_id"].tolist()
                np.testing.assert_allclose(event["probability"], rows["probability"], rtol=1e-6)
        assert "holds zlib compressed events" in caplog.text


def test_footprint_uncompressed_not_detected_as_compressed():
    with Footprint.load(LocalStorage(root_dir=Path(TESTS_ASSETS_DIR, "static"), cache_dir=None)) as footprint:
        assert isinstance(footprint, FootprintBin)
        assert not footprint.compressed


@pytest.mark.parametrize("bin_name, idx_name, warns", [
    ("footprint.bin", "footprint.idx", True),
    ("footprint.bin.z", "footprint.idx", True),
    ("footprint.bin.z", "footprint.idx.z", False),
])
def test_footprint_zip_output_names_warning(bin_name, idx_name, warns, caplog):
    """Zipped output should use the .z names the runtime looks for; warn when it does not."""
    with TemporaryDirectory() as tmp, caplog.at_level("WARNING"):
        csvtobin(
            file_in=Path(TESTS_ASSETS_DIR, "static", "footprint.csv"),
            file_out=Path(tmp, bin_name),
            idx_file_out=Path(tmp, idx_name),
            file_type="footprint",
            max_intensity_bin_idx=3,
            no_intensity_uncertainty=True,
            decompressed_size=False,
            no_validation=False,
            zip_files=True,
        )
    assert ("should be named with a .z extension" in caplog.text) == warns


# --------------------------------------------------------------------------------------
# Regression tests for large AreaPeril IDs exceeding uint32 range.
#
# The default OASIS_AREAPERIL_TYPE is u4 (uint32, max 4,294,967,295). Models with
# AreaPeril IDs exceeding this limit produce silent data corruption: pandas 3.x
# silently wraps the value mod 2^32 rather than raising an error.
#
# Setting OASIS_AREAPERIL_TYPE=u8 (uint64) before importing oasislmf fixes this.
# The carry-state scalars in the footprint validator must also use the correct type
# so Numba JIT functions receive consistent argument types across chunk boundaries.
#
# Reference: https://github.com/OasisLMF/OasisLMF/issues/2011
# --------------------------------------------------------------------------------------
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


def _write_footprint(tmp, zip_files=False):
    csv_path = Path(tmp, "footprint.csv")
    csv_path.write_text("event_id,areaperil_id,intensity_bin_id,probability\n1,1,1,1.0\n")
    bin_path, idx_path = Path(tmp, "footprint.bin"), Path(tmp, "footprint.idx")
    csvtobin(csv_path, bin_path, "footprint", idx_file_out=idx_path, max_intensity_bin_idx=1,
             no_intensity_uncertainty=True, decompressed_size=False, no_validation=True, zip_files=zip_files)
    return bin_path, idx_path


@pytest.mark.parametrize("zip_files", [False, True])
def test_bintocsv_footprint_rejects_idx_past_end_of_file(zip_files):
    """A truncated footprint.bin or a footprint.idx from a different (larger) file must be
    rejected before the batched njit loop — which indexes without bounds checking — can read
    past the end of the mapped file."""
    with TemporaryDirectory() as tmp:
        bin_path, idx_path = _write_footprint(tmp, zip_files=zip_files)
        idx = np.fromfile(idx_path, dtype=EventIndexBin_dtype)
        idx["offset"][0] = 2_000_000_000
        idx.tofile(idx_path)
        with pytest.raises(OasisException, match="references bytes .* but the footprint file is only"):
            bintocsv(bin_path, Path(tmp, "out.csv"), "footprint", noheader=False,
                     idx_file_in=idx_path, zip_files=zip_files, event_from_to=None)


def test_bintocsv_footprint_rejects_truncated_bin_matching_a_real_mismatch():
    """Same check, exercised by directly truncating a real footprint.bin rather than editing the
    idx, matching how this surfaces in practice (a bin file cut short of its index)."""
    with TemporaryDirectory() as tmp:
        bin_path, idx_path = _write_footprint(tmp)
        bin_path.write_bytes(bin_path.read_bytes()[:-1])  # one byte short of the last event
        with pytest.raises(OasisException, match="references bytes .* but the footprint file is only"):
            bintocsv(bin_path, Path(tmp, "out.csv"), "footprint", noheader=False,
                     idx_file_in=idx_path, zip_files=False, event_from_to=None)


def test_bintocsv_footprint_rejects_footprint_file_too_short_for_header():
    """An empty or near-empty footprint.bin (e.g. empty stdin) used to fail inside
    _get_index_dtype with a bare 'can only convert an array of size 1 to a Python scalar'."""
    with TemporaryDirectory() as tmp:
        _, idx_path = _write_footprint(tmp)
        Path(tmp, "footprint.bin").write_bytes(b"\x00\x00")
        with pytest.raises(OasisException, match="too short to hold its"):
            bintocsv(Path(tmp, "footprint.bin"), Path(tmp, "out.csv"), "footprint", noheader=False,
                     idx_file_in=idx_path, zip_files=False, event_from_to=None)


def test_bintocsv_footprint_accepts_valid_file():
    with TemporaryDirectory() as tmp:
        bin_path, idx_path = _write_footprint(tmp)
        bintocsv(bin_path, Path(tmp, "out.csv"), "footprint", noheader=False,
                 idx_file_in=idx_path, zip_files=False, event_from_to=None)
        assert Path(tmp, "out.csv").read_text() == "event_id,areaperil_id,intensity_bin_id,probability\n1,1,1,1.000000\n"
