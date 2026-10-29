from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from oasis_data_manager.filestore.backends.local import LocalStorage

from oasislmf.pytools.converters.csvtobin.manager import csvtobin
from oasislmf.pytools.getmodel.footprint import Footprint, FootprintBin, FootprintBinZ, FootprintCsv

footprints_path = Path(__file__).resolve().parent / "footprints"


def test_csv_footprint_matches_bin(tmp_path):
    with FootprintBin(LocalStorage(footprints_path)) as bin_footprint:
        event_ids = bin_footprint.footprint_index['event_id'].tolist()
        bin_events = {e: bin_footprint.get_event(e) for e in event_ids}

    df = pd.concat(pd.DataFrame(bin_events[e]).assign(event_id=e) for e in event_ids)
    df[['event_id', 'areaperil_id', 'intensity_bin_id', 'probability']].to_csv(tmp_path / "footprint.csv", index=False)

    with Footprint.load(LocalStorage(tmp_path)) as csv_footprint:
        assert isinstance(csv_footprint, FootprintCsv)
        assert csv_footprint.num_intensity_bins == df['intensity_bin_id'].max()
        assert csv_footprint.has_intensity_uncertainty == (df.groupby(['event_id', 'areaperil_id']).size().max() > 1)
        for e in event_ids:
            csv_event = csv_footprint.get_event(e)
            for col in ('areaperil_id', 'intensity_bin_id', 'probability'):
                np.testing.assert_array_equal(csv_event[col], bin_events[e][col])
        assert csv_footprint.get_event(max(event_ids) + 1) is None


def test_footprint_bin_handles_zero_event_footprint():
    """A footprint with zero events writes a literal 0-byte footprint.idx (no events to index).
    np.memmap on that 0-byte idx file raised "cannot mmap an empty file" -- a crash in the real
    getmodel runtime, not just the bintocsv debug tool."""
    with TemporaryDirectory() as tmp:
        csv_path = Path(tmp, "footprint.csv")
        csv_path.write_text("event_id,areaperil_id,intensity_bin_id,probability\n")
        bin_path, idx_path = Path(tmp, "footprint.bin"), Path(tmp, "footprint.idx")
        csvtobin(csv_path, bin_path, "footprint", idx_file_out=idx_path, max_intensity_bin_idx=1,
                 no_intensity_uncertainty=True, decompressed_size=False, no_validation=False, zip_files=False)
        assert idx_path.stat().st_size == 0

        with FootprintBin(LocalStorage(tmp)) as footprint:
            assert len(footprint.footprint_index) == 0
            assert footprint.get_event(1) is None


def test_footprint_binz_handles_zero_event_footprint():
    """FootprintBin.__enter__ guards against a 0-byte footprint.idx (a footprint with zero
    events), but FootprintBinZ.__enter__ -- the compressed-format reader -- had no such guard,
    so a zero-event footprint.idx.z crashed np.memmap even though the uncompressed path was
    already fixed."""
    with TemporaryDirectory() as tmp:
        csv_path = Path(tmp, "footprint.csv")
        csv_path.write_text("event_id,areaperil_id,intensity_bin_id,probability\n")
        bin_path, idx_path = Path(tmp, "footprint.bin.z"), Path(tmp, "footprint.idx.z")
        csvtobin(csv_path, bin_path, "footprint", idx_file_out=idx_path, max_intensity_bin_idx=1,
                 no_intensity_uncertainty=True, decompressed_size=False, no_validation=False, zip_files=True)
        assert idx_path.stat().st_size == 0

        with FootprintBinZ(LocalStorage(tmp)) as footprint:
            assert len(footprint.footprint_index) == 0
            assert footprint.get_event(1) is None
