import shutil
import filecmp
import os
from oasislmf.pytools.indexconvert import (
    change_footprint_apid_multi_peril, change_footprint_apid
)


def test_indexconvert():
    test_dir = os.path.dirname(__file__)
    input_file = os.path.join(test_dir, "indexconvert_test_data.csv")
    temp_file = os.path.join(test_dir, "indexconvert_test_temp.csv")
    correct_file = os.path.join(
        test_dir, "indexconvert_test_correct1peril.csv"
    )

    assert os.path.exists(input_file)
    shutil.copy(input_file, temp_file)
    change_footprint_apid(temp_file, 10)
    assert filecmp.cmp(temp_file, correct_file, shallow=False)

    os.remove(temp_file)


def test_indexconvert_multiperil():
    test_dir = os.path.dirname(__file__)
    input_file = os.path.join(test_dir, "indexconvert_test_data.csv")
    temp_file = os.path.join(test_dir, "indexconvert_test_temp.csv")
    correct_file = os.path.join(
        test_dir, "indexconvert_test_correct3peril.csv"
    )

    assert os.path.exists(input_file)
    shutil.copy(input_file, temp_file)
    change_footprint_apid_multi_peril(temp_file, 10, 10, 3)
    assert filecmp.cmp(temp_file, correct_file, shallow=False)

    os.remove(temp_file)
