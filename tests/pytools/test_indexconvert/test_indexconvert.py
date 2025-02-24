import shutil
import filecmp
from pathlib import Path

from oasislmf.pytools.indexconvert import (
    change_footprint_apid_multi_peril, change_footprint_apid
)


def test_indexconvert():
    test_dir = Path(__file__).parent
    input_file = test_dir / "indexconvert_test_data.csv"
    temp_file = test_dir / "indexconvert_test_temp.csv"
    correct_file = test_dir / "indexconvert_test_correct1peril.csv"

    assert input_file.exists()
    shutil.copy(input_file, temp_file)
    change_footprint_apid(temp_file, 10)
    assert filecmp.cmp(temp_file, correct_file, shallow=False)

    temp_file.unlink()


def test_indexconvert_multiperil():
    test_dir = Path(__file__).parent
    input_file = test_dir / "indexconvert_test_data.csv"
    temp_file = test_dir / "indexconvert_test_temp.csv"
    correct_file = test_dir / "indexconvert_test_correct3peril.csv"

    assert input_file.exists()
    shutil.copy(input_file, temp_file)
    change_footprint_apid_multi_peril(temp_file, 10, 10, 3)
    assert filecmp.cmp(temp_file, correct_file, shallow=False)

    temp_file.unlink()
