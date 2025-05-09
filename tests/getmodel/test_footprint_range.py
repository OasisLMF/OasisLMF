from pathlib import Path
import pytest
from oasislmf.pytools.getmodel.footprint import FootprintBin
from oasis_data_manager.filestore.backends.local import LocalStorage
from oasislmf.pytools.data_layer.conversions.footprint import convert_bin_to_parquet_chunk

script_dir = Path(__file__).resolve().parent
footprints_path = script_dir / "footprints"


@pytest.mark.parametrize("areaperil_ids, expected", [
    ([20], [False, False, True, True, False, False, False]),
    ([21], [True, False, False, True, False, False, False]),
    ([2, 28], [False, True, False, True, False, True, False]),
    (None, [True, True, True, True, True, True, True]),
    ([300], [False, False, False, True, False, False, False])
])
def test_range(areaperil_ids, expected):
    with FootprintBin(LocalStorage(footprints_path), areaperil_ids=areaperil_ids) as footprint:
        for i in range(7):
            assert footprint.areaperil_in_range(i + 1, footprint.events_dict) == expected[i]


@pytest.fixture(scope="session", autouse=True)
def cleanup_parquet_files():
    convert_bin_to_parquet_chunk(footprints_path, chunk_size=0)
    yield

    for file in footprints_path.glob("*.parquet"):
        file.unlink()
    for file in footprints_path.glob("*footprint_lookup*"):
        file.unlink()
    for file in footprints_path.glob("*.json"):
        file.unlink()
