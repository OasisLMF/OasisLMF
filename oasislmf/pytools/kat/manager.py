# katpy/manager.py

from collections import Counter
from contextlib import ExitStack
import glob
import logging
import shutil
from pathlib import Path

from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


def find_csv_with_header(
    stack,
    file_paths,
):
    """Find and check csv files for consistent and present headers
    Args:
        stack (ExitStack): Exit Stack.
        file_paths (List[str | os.PathLike]): List of csv file paths.
    Returns:
        files_with_header (List[bool]): Bool list of files with header present
        header (str): Header to write
    """
    headers = {}
    files_with_header = [False] * len(file_paths)
    for idx, fp in enumerate(file_paths):
        with stack.enter_context(fp.open("r", newline="", encoding="utf-8")) as f:
            first_row = f.readline().strip()
            if first_row:
                if "EventId" in first_row:
                    headers[fp] = first_row
                    files_with_header[idx] = True

    if not headers:
        raise ValueError("ERROR: katpy, no valid header found in any CSV file.")

    header_counts = Counter(headers.values())

    if len(header_counts) > 1:
        raise ValueError(f"ERROR: katpy, conflicting headers found: {header_counts}")

    return files_with_header, list(headers.values())[0]


def concat_unsorted(
    stack,
    file_paths,
    files_with_header,
    header,
    out_file,
):
    """Concats CSV files in order they are passed in.
    Args:
        stack (ExitStack): Exit Stack.
        file_paths (List[str | os.PathLike]): List of csv file paths.
        files_with_header (List[bool]): Bool list of files with header present
        header (str): Header to write
        out_file (str | os.PathLike): Output Concatenated CSV file.
    """
    first_header_written = False

    with stack.enter_context(out_file.open("wb")) as out:
        for i, fp in enumerate(file_paths):
            with stack.enter_context(fp.open("rb")) as csv_file:
                if files_with_header[i]:
                    # Read first line (header)
                    first_line = csv_file.readline()

                    # Write the expected header at the start of the file
                    if not first_header_written:
                        out.write(header.encode() + b'\n')
                        first_header_written = True
                shutil.copyfileobj(csv_file, out)


def run(
    out_file,
    files_in=None,
    dir_in=None,
    concatenate_selt=False,
    concatenate_melt=False,
    concatenate_qelt=False,
    concatenate_splt=False,
    concatenate_mplt=False,
    concatenate_qplt=False,
    unsorted=False,
):
    """Concatenate CSV files (optionally sorted)
    Args:
        out_file (str | os.PathLike): Output Concatenated CSV file.
        files_in (List[str | os.PathLike], optional): Individual CSV file paths to concatenate. Defaults to None.
        dir_in (str | os.PathLike, optional): Path to the directory containing files for concatenation. Defaults to None.
        concatenate_selt (bool, optional): Concatenate SELT CSV file. Defaults to False.
        concatenate_melt (bool, optional): Concatenate MELT CSV file. Defaults to False.
        concatenate_qelt (bool, optional): Concatenate QELT CSV file. Defaults to False.
        concatenate_splt (bool, optional): Concatenate SPLT CSV file. Defaults to False.
        concatenate_mplt (bool, optional): Concatenate MPLT CSV file. Defaults to False.
        concatenate_qplt (bool, optional): Concatenate QPLT CSV file. Defaults to False.
        unsorted (bool, optional): Do not sort by event/period ID. Defaults to False.
    """
    out_file = Path(out_file).resolve()
    if out_file.suffix.lower() != ".csv":
        raise ValueError(f"ERROR: File \'{out_file}\' is not a CSV file.")

    csv_files = []

    # Check and add files from dir_in
    if dir_in:
        dir_in = Path(dir_in)
        if not dir_in.exists():
            raise FileNotFoundError(f"ERROR: Directory \'{dir_in}\' does not exist")
        if not dir_in.is_dir():
            raise ValueError(f"ERROR: \'{dir_in}\' is not a directory.")

        dir_csv_files = glob.glob(str(dir_in / "*.csv"))
        if not dir_csv_files:
            logger.warning(f"Warning: No CSV files found in directory \'{dir_in}\'")
        csv_files += [Path(file).resolve() for file in dir_csv_files]

    csv_files.sort()

    # Check and add files from files_in
    if files_in:
        for file in files_in:
            path = Path(file).resolve()
            if not path.exists():
                raise FileNotFoundError(f"ERROR: File \'{path}\' does not exist.")
            if not path.is_file():
                raise FileNotFoundError(f"ERROR: File \'{path}\' is not a valid file.")
            if path.suffix.lower() != ".csv":
                raise ValueError(f"ERROR: File \'{path}\' is not a CSV file.")
            csv_files.append(path)

    if not csv_files:
        raise RuntimeError("ERROR: katpy has no input CSV files to join")

    sort_by_event = any([concatenate_selt, concatenate_melt, concatenate_qelt])
    sort_by_period = any([concatenate_splt, concatenate_mplt, concatenate_qplt])
    assert sort_by_event + sort_by_period == 1, "incorrect flag config in katpy"

    with ExitStack() as stack:
        files_with_header, header = find_csv_with_header(stack, csv_files)
        if unsorted:
            concat_unsorted(stack, csv_files, files_with_header, header, out_file)
        else:
            print("NOT IMPLEMENTED")


@redirect_logging(exec_name='katpy')
def main(
    out=None,
    files_in=None,
    dir_in=None,
    selt=False,
    melt=False,
    qelt=False,
    splt=False,
    mplt=False,
    qplt=False,
    unsorted=False,
    **kwargs
):
    run(
        out_file=out,
        files_in=files_in,
        dir_in=dir_in,
        concatenate_selt=selt,
        concatenate_melt=melt,
        concatenate_qelt=qelt,
        concatenate_splt=splt,
        concatenate_mplt=mplt,
        concatenate_qplt=qplt,
        unsorted=unsorted,
    )
