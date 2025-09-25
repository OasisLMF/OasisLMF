from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import pytest

import numpy as np

from oasislmf.pytools.eve.manager import main
from oasislmf.pytools.common.data import oasis_int

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_evepy")
test_process_inputs = ((1, 3), (2, 3), (3, 3), (1, 1))


@pytest.fixture
def update_expected(request):
    return request.config.getoption('--update-expected')


def case_runner(input_file, output_file, process_number, total_processes,
                no_shuffle=False, randomise=False, randomise_builtin=False,
                update_expected=False):
    with TemporaryDirectory() as tmp_result_dir:
        input_file = Path(TESTS_ASSETS_DIR).joinpath("input").joinpath(input_file)
        expected_output = Path(TESTS_ASSETS_DIR).joinpath("expected").joinpath(output_file)
        actual_output = Path(tmp_result_dir).joinpath(output_file)

        kwargs = {
            "input_file": input_file,
            "output_file": actual_output,
            "process_number": process_number,
            "total_processes": total_processes,
            "no_shuffle": no_shuffle,
            "randomise": randomise,
            "randomise_builtin": randomise_builtin
        }

        main(**kwargs)

        if update_expected:
            shutil.copyfile(Path(actual_output), Path(expected_output))

        try:
            expected_data = np.fromfile(expected_output, dtype=oasis_int)
            actual_data = np.fromfile(actual_output, dtype=oasis_int)

            np.testing.assert_equal(expected_data, actual_data)
        except AssertionError as e:
            error_path = TESTS_ASSETS_DIR.joinpath('error_files')
            error_path.mkdir(exist_ok=True)
            shutil.copyfile(Path(actual_output), Path(error_path, output_file))

            arg_str = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
            raise Exception(f"running 'evepy {arg_str}' led to diff, see files at {error_path}") from e


@pytest.mark.parametrize("process_number,total_processes", test_process_inputs)
def test_evepy__default(process_number, total_processes, update_expected):
    """Test evepy with default settings (i.e. round robin shuffle)"""
    kwargs = {
        "input_file": Path("events.bin"),
        "output_file": f"output_evepy__default_{process_number}_{total_processes}.bin",
        "process_number": process_number,
        "total_processes": total_processes,
    }

    case_runner(update_expected=update_expected, **kwargs)


@pytest.mark.parametrize("process_number,total_processes", test_process_inputs)
def test_evepy__no_shuffle(process_number, total_processes, update_expected):
    """Test evepy with no shuffle"""
    kwargs = {
        "input_file": Path("events.bin"),
        "output_file": f"output_evepy__no_shuffle_{process_number}_{total_processes}.bin",
        "process_number": process_number,
        "total_processes": total_processes,
        "no_shuffle": True
    }

    case_runner(update_expected=update_expected, **kwargs)


@pytest.mark.parametrize("process_number,total_processes", test_process_inputs)
def test_evepy__randomise(process_number, total_processes, update_expected):
    """Test evepy with randomise"""
    kwargs = {
        "input_file": Path("events.bin"),
        "output_file": f"output_evepy__randomise_{process_number}_{total_processes}.bin",
        "process_number": process_number,
        "total_processes": total_processes,
        "randomise": True
    }

    case_runner(update_expected=update_expected, **kwargs)


@pytest.mark.parametrize("process_number,total_processes", test_process_inputs)
def test_evepy__randomise_builtin(process_number, total_processes, update_expected):
    """Test evepy with randomise"""
    kwargs = {
        "input_file": Path("events.bin"),
        "output_file": f"output_evepy__randomise_builtin_{process_number}_{total_processes}.bin",
        "process_number": process_number,
        "total_processes": total_processes,
        "randomise_builtin": True
    }

    case_runner(update_expected=update_expected, **kwargs)


@pytest.mark.parametrize("process_number,total_processes", test_process_inputs)
def test_evepy__no_shuffle_randomise(process_number, total_processes, update_expected):
    """Test evepy with no_shuffle and randomise. Should ignore randomise and
    output the same as no_shuffle."""
    kwargs = {
        "input_file": Path("events.bin"),
        "output_file": f"output_evepy__no_shuffle_{process_number}_{total_processes}.bin",
        "process_number": process_number,
        "total_processes": total_processes,
        "no_shuffle": True,
        "randomise": True
    }

    case_runner(update_expected=update_expected, **kwargs)


@pytest.mark.parametrize("process_number,total_processes", test_process_inputs)
def test_evepy__no_shuffle_randomise_randomise_builtin(process_number, total_processes, update_expected):
    """Test evepy with no_shuffle, randomise and randomise_builtin. Should output the same as no_shuffle."""
    kwargs = {
        "input_file": Path("events.bin"),
        "output_file": f"output_evepy__no_shuffle_{process_number}_{total_processes}.bin",
        "process_number": process_number,
        "total_processes": total_processes,
        "no_shuffle": True,
        "randomise": True,
        "randomise_builtin": True
    }

    case_runner(update_expected=update_expected, **kwargs)


@pytest.mark.parametrize("process_number,total_processes", test_process_inputs)
def test_evepy__randomise_randomise_builtin(process_number, total_processes, update_expected):
    """Test evepy with randomise and randomise_builtin. Should output the same as randomise."""
    kwargs = {
        "input_file": Path("events.bin"),
        "output_file": f"output_evepy__randomise_{process_number}_{total_processes}.bin",
        "process_number": process_number,
        "total_processes": total_processes,
        "randomise": True,
        "randomise_builtin": True
    }

    case_runner(update_expected=update_expected, **kwargs)


def test_less_events_than_processes(update_expected):
    """Test evepy with fewer events than processes"""
    total_processes = 3
    for process_number in range(1, 4):
        kwargs = {
            "input_file": Path("events_2.bin"),
            "output_file": f"output_test_less_events_than_processes_{process_number}_{total_processes}.bin",
            "process_number": process_number,
            "total_processes": total_processes
        }

        case_runner(update_expected=update_expected, **kwargs)
