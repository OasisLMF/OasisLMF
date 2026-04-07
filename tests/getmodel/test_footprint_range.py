import shutil
from pathlib import Path
import pytest
from oasislmf.pytools.getmodel.footprint import FootprintBin
from oasis_data_manager.filestore.backends.local import LocalStorage
from oasislmf.pytools.data_layer.conversions.footprint import convert_bin_to_parquet_chunk

script_dir = Path(__file__).resolve().parent
footprints_path = script_dir / "footprints"


@pytest.fixture(scope="session")
def footprints_tmp(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("footprints")
    shutil.copy2(footprints_path / "footprint.bin", tmp / "footprint.bin")
    shutil.copy2(footprints_path / "footprint.idx", tmp / "footprint.idx")
    convert_bin_to_parquet_chunk(tmp, chunk_size=0)
    return tmp


@pytest.mark.parametrize("areaperil_ids, expected", [
    ([20], [False, False, True, True, False, False, False]),
    ([21], [True, False, False, True, False, False, False]),
    ([2, 28], [False, True, False, True, False, True, False]),
    (None, [True, True, True, True, True, True, True]),
    ([300], [False, False, False, True, False, False, False])
])
def test_range(footprints_tmp, areaperil_ids, expected):
    with FootprintBin(LocalStorage(footprints_tmp), areaperil_ids=areaperil_ids) as footprint:
        for i in range(7):
            assert footprint.areaperil_in_range(i + 1, footprint.events_dict) == expected[i]
