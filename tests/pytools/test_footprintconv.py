from contextlib import ExitStack
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from oasislmf.pytools.footprintconv import compress_footprint
from oasislmf.pytools.getmodel.common import (
    EventIndexBin_dtype, EventIndexBinZ_dtype, Event_dtype, FootprintHeader, zfootprint_index_filename,
)
from oasislmf.pytools.getmodel.footprint import FootprintBinZ


def _write_legacy_uncompressed_d_size_footprint(tmp):
    """Build a footprint.bin/.idx pair with the header's uncompressedMask bit set (meaning the
    idx entries carry d_size) while still uncompressed -- the layout csvtobin's own -d (without
    -z) used to produce before C1's fix forced decompressed_size off in that combination. Current
    csvtobin can no longer write this pair, but compress_footprint is a standalone tool that can
    be run against any pre-existing or externally-supplied footprint.bin/.idx, so it must still
    handle one correctly.
    """
    header = np.zeros(1, dtype=FootprintHeader)
    header['num_intensity_bins'] = 1
    header['has_intensity_uncertainty'] = 2  # uncompressedMask bit set

    event_1 = np.array([(1, 1, 1.0)], dtype=Event_dtype).tobytes()
    event_2 = np.array([(2, 1, 1.0)], dtype=Event_dtype).tobytes()
    Path(tmp, 'footprint.bin').write_bytes(header.tobytes() + event_1 + event_2)

    idx = np.zeros(2, dtype=EventIndexBinZ_dtype)
    idx[0] = (1, 8, len(event_1), len(event_1))
    idx[1] = (2, 8 + len(event_1), len(event_2), len(event_2))
    Path(tmp, 'footprint.idx').write_bytes(idx.tobytes())

    return event_1, event_2


def test_compress_footprint_reads_input_idx_dtype_from_its_own_header():
    """compress_footprint always memmapped the input footprint.idx as EventIndexBin (no d_size),
    regardless of what the footprint.bin header's own uncompressedMask bit says about the input's
    actual layout -- misaligning every field read once the input idx really carries d_size."""
    with TemporaryDirectory() as tmp:
        event_1, event_2 = _write_legacy_uncompressed_d_size_footprint(tmp)

        compress_footprint(tmp, decompressed_size=True)

        zidx = np.fromfile(Path(tmp, 'footprint.idx.z'), dtype=EventIndexBinZ_dtype)
        assert zidx['event_id'].tolist() == [1, 2]
        assert zidx['d_size'].tolist() == [len(event_1), len(event_2)]
        assert zidx['offset'][0] == 8
        assert zidx['offset'][1] == 8 + zidx['size'][0]


def test_compress_footprint_clears_the_d_size_bit_when_not_writing_d_size():
    """When decompressed_size=False, the output index is written without a d_size field
    (EventIndexBin, 20 bytes/entry). If the input footprint already had the uncompressedMask bit
    set (e.g. written by csvtobin -d without -z), that bit was passed through unchanged into the
    output header, which then claimed 28-byte entries while the index actually written used
    20-byte ones -- FootprintBinZ then fails to even parse the index."""
    with TemporaryDirectory() as tmp:
        _write_legacy_uncompressed_d_size_footprint(tmp)

        compress_footprint(tmp, decompressed_size=False)

        zidx_bytes = Path(tmp, zfootprint_index_filename).stat().st_size
        assert zidx_bytes == 2 * EventIndexBin_dtype.itemsize

        class FakeStorage:
            def with_fileno(self, name):
                return open(Path(tmp, name), 'rb')

        with ExitStack() as stack:
            reader = FootprintBinZ(FakeStorage(), stack)
            reader.__enter__()
            assert reader.footprint_index.dtype.itemsize == EventIndexBin_dtype.itemsize
            assert reader.footprint_index['event_id'].tolist() == [1, 2]
