from contextlib import ExitStack
import logging
from pathlib import Path
import sys
from numba import njit
from numba.core import types
from numba.typed import Dict
import numpy as np

from oasis_data_manager.filestore.backends.base import BaseStorage
from oasislmf.pytools.common.data import lossfactors_headers, lossfactors_dtype
from oasislmf.pytools.common.event_stream import mv_read
from .common import PLAFACTORS_FILE


logger = logging.getLogger(__name__)


def get_post_loss_amplification_factors(storage: BaseStorage, secondary_factor, uniform_factor, ignore_file_type=set()):
    """
    Get Post Loss Amplification (PLA) factors mapped to event ID-item ID pair.
    Returns empty dictionary if uniform factor to apply across all losses has
    been given.

    lossfactors.bin is binary file with layout:
        reserved header (4-byte int),
        event ID 1 (4-byte int), number of amplification IDs for event ID 1 (4-byte int),
        amplification ID 1 (4-byte int), loss factor for amplification ID 1 (4-byte float),
        ...
        amplification ID n (4-byte int), loss factor for amplification ID n (4-byte float),
        event ID 2 (4-byte int), number of amplification IDs for event ID 2 (4-byte int),
        ...
        event ID N (4-byte int), number of amplification IDs for event ID N (4-byte int),
        amplification ID 1 (4-byte int), loss factor for amplification ID 1 (4-byte float),
        ...
        amplification ID n (4-byte int), loss factor for amplification ID n (4-byte float)

    Args:
        storage: (BaseStorage) the storage connector for fetching the model data
        secondary_factor (float): secondary factor to apply to post loss
          amplification
        uniform_factor (float): uniform factor to apply across all losses
        ignore_file_type: set(str) file extension to ignore when loading

    Returns:
        plafactors (dict): event ID-item ID pairs mapped to amplification IDs
    """
    if uniform_factor > 0.0:
        return Dict.empty(
            key_type=types.UniTuple(types.int64, 2), value_type=types.float64
        )

    input_files = set(storage.listdir())
    if PLAFACTORS_FILE in input_files and 'bin' not in ignore_file_type:
        plafactors = read_lossfactors(storage.root_dir, set(["csv"]), PLAFACTORS_FILE)
        for key, value in plafactors.items():
            plafactors[key] = max(
                1 + (value - 1) * secondary_factor, 0.0
            )
        return plafactors
    else:
        raise FileNotFoundError(f"lossfactors.bin file not found at {storage.get_storage_url('', encode_params=False)[1]}")


def read_lossfactors(run_dir="", ignore_file_type=set(), filename=PLAFACTORS_FILE, use_stdin=False):
    """Load the correlations from the lossfactors file.
    Args:
        run_dir (str): path to lossfactors.bin file
        ignore_file_type (Set[str]): file extension to ignore when loading.
        filename (str | os.PathLike): lossfactors file name
        use_stdin (bool): Use standard input for file data, ignores run_dir/filename. Defaults to False.
    Returns:
        plafactors (dict): event ID-item ID pairs mapped to amplification IDs
    """
    int32_itemsize = np.dtype(np.int32).itemsize
    float32_itemsize = np.dtype(np.float32).itemsize

    @njit(cache=True, error_model="numpy")
    def _read_bin(lossfactors, plafactors):
        cursor = 0
        opts, cursor = mv_read(lossfactors, cursor, np.int32, int32_itemsize)

        valid_buf = len(lossfactors)
        while cursor + (2 * int32_itemsize) <= valid_buf:
            event_id, cursor = mv_read(lossfactors, cursor, np.int32, int32_itemsize)
            count, cursor = mv_read(lossfactors, cursor, np.int32, int32_itemsize)
            for _ in range(count):
                if cursor + (int32_itemsize + float32_itemsize) > valid_buf:
                    break
                amplification_id, cursor = mv_read(lossfactors, cursor, np.int32, int32_itemsize)
                factor, cursor = mv_read(lossfactors, cursor, np.float32, float32_itemsize)
                plafactors[(event_id, amplification_id)] = factor

    @njit(cache=True, error_model="numpy")
    def _read_csv(lossfactors, plafactors):
        for row in lossfactors:
            plafactors[(row["event_id"], row["amplification_id"])] = row["factor"]

    plafactors = Dict.empty(
        key_type=types.UniTuple(types.int64, 2), value_type=types.float64
    )
    for ext in ["bin", "csv"]:
        if ext in ignore_file_type:
            continue

        lossfactors_file = Path(run_dir, filename).with_suffix("." + ext)
        if lossfactors_file.exists():
            logger.debug(f"loading {lossfactors_file}")
            if ext == "bin":
                if use_stdin:
                    lossfactors = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8)
                else:
                    lossfactors = np.memmap(lossfactors_file, dtype=np.uint8, mode='r')
                _read_bin(lossfactors, plafactors)

            elif ext == "csv":
                with ExitStack() as stack:
                    if use_stdin:
                        fin = sys.stdin
                    else:
                        fin = stack.enter_context(open(lossfactors_file, "r"))

                    lines = fin.readlines()
                    # Check for header
                    first_line_elements = [header.strip() for header in lines[0].strip().split(',')]
                    has_header = first_line_elements == lossfactors_headers

                    data_lines = lines[1:] if has_header else lines
                    lossfactors = np.loadtxt(
                        data_lines,
                        dtype=lossfactors_dtype,
                        delimiter=",",
                        ndmin=1
                    )
                    _read_csv(lossfactors, plafactors)
            else:
                raise RuntimeError(f"Cannot read lossfactors file of type {ext}. Not Implemented.")
            return plafactors

    raise FileNotFoundError(f'lossfactors file not found at {run_dir}. Ignoring files with ext {ignore_file_type}.')
