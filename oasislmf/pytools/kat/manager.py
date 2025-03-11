# katpy/manager.py

from collections import Counter
from contextlib import ExitStack
import csv
import glob
import heapq
import logging
import shutil
from pathlib import Path

from oasislmf.pytools.elt.data import MELT_headers, QELT_headers, SELT_headers
from oasislmf.pytools.plt.data import MPLT_headers, QPLT_headers, SPLT_headers
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)

OUT_NAMES = [
    "SELT",
    "MELT",
    "QELT",
    "SPLT",
    "MPLT",
    "QPLT",
]
OUT_SELT = 0
OUT_MELT = 1
OUT_QELT = 2
OUT_SPLT = 3
OUT_MPLT = 4
OUT_QPLT = 5


def check_file_extensions(file_paths):
    """Check file path extensions are all identical
    Args:
        file_paths (List[str | os.PathLike]): List of csv file paths.
    Returns:
        ext (str): file extension as a str
    """
    first_ext = file_paths[0].suffix

    if all(fp.suffix == first_ext for fp in file_paths):
        return first_ext
    raise RuntimeError("ERROR: katpy has input files with different file extensions. Make sure all input files are of the same type.")


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


def check_correct_headers(headers, file_type):
    """Checks headers found in csv file matches excpected headers for file type
    Args:
        headers (List[str]): Headers
        file_type (int): File type int matching OUT_NAMES index
    """
    if file_type == OUT_SELT:
        expected_headers = SELT_headers
    elif file_type == OUT_MELT:
        expected_headers = MELT_headers
    elif file_type == OUT_QELT:
        expected_headers = QELT_headers
    elif file_type == OUT_SPLT:
        expected_headers = SPLT_headers
    elif file_type == OUT_MPLT:
        expected_headers = MPLT_headers
    elif file_type == OUT_QPLT:
        expected_headers = QPLT_headers
    else:
        raise ValueError(f"ERROR: katpy, unknown file_type {file_type}, not in {OUT_NAMES}")
    if headers != expected_headers:
        raise RuntimeError(f"ERROR: katpy, incorrect headers found in csv file, expected {expected_headers} but got {headers}")


def get_header_idxs(
    headers,
    headers_to_search,
):
    """Search for index of headers_to_search in headers of csv file
    Args:
        headers (List[str]): Headers
        headers_to_search (List[str]): Headers to search
    Returns:
        idxs (List[int]): Indexes of searched headers
    """
    idxs = []
    for header in headers_to_search:
        try:
            idxs.append(headers.index(header))
        except ValueError as e:
            raise ValueError(f"ERROR: katpy, cannot sort by header {header}, not found. {e}")
    return idxs


def csv_concat_unsorted(
    stack,
    file_paths,
    files_with_header,
    headers,
    out_file,
):
    """Concats CSV files in order they are passed in.
    Args:
        stack (ExitStack): Exit Stack.
        file_paths (List[str | os.PathLike]): List of csv file paths.
        files_with_header (List[bool]): Bool list of files with header present
        headers (List[str]): Headers to write
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
                        out.write(",".join(headers).encode() + b"\n")
                        first_header_written = True
                shutil.copyfileobj(csv_file, out)


def csv_concat_sort_by_headers(
    stack,
    file_paths,
    files_with_header,
    headers,
    header_idxs,
    sort_fn,
    out_file,
    **sort_kwargs
):
    """Concats CSV files in order determined by the header_idxs and sort_fn
    Args:
        stack (ExitStack): Exit Stack.
        file_paths (List[str | os.PathLike]): List of csv file paths.
        files_with_header (List[bool]): Bool list of files with header present
        headers (List[str]): Headers to write
        header_idxs (List[int]): Indices of headers to sort by
        sort_fn (Callable[[List[int]], Any]): Sort function to apply to header_idxs
        out_file (str | os.PathLike): Output Concatenated CSV file.
    """
    # Open all csv files
    csv_files = [stack.enter_context(open(fp, "r", newline="")) for fp in file_paths]
    readers = [csv.reader(f) for f in csv_files]

    # Skip headers
    for idx, reader in enumerate(readers):
        if files_with_header[idx]:
            next(reader, None)

    def row_generator(reader, file_idx):
        for row in reader:
            sort_key = sort_fn([row[i] for i in header_idxs], **sort_kwargs)
            yield (sort_key, file_idx, row)

    # Create iterator for each file
    iterators = [row_generator(reader, idx) for idx, reader in enumerate(readers)]

    # Merge all iterators
    merged_iterator = heapq.merge(*iterators, key=lambda x: x[0])

    buffer_size = 1000000
    with stack.enter_context(out_file.open("w")) as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(headers)

        buffer = []
        for _, _, row in merged_iterator:
            buffer.append(row)
            if len(buffer) > buffer_size:
                writer.writerows(buffer)
                buffer.clear()

        if buffer:
            writer.writerows(buffer)

    for f in csv_files:
        f.close()


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

    input_files = []

    # Check and add files from dir_in
    if dir_in:
        dir_in = Path(dir_in)
        if not dir_in.exists():
            raise FileNotFoundError(f"ERROR: Directory \'{dir_in}\' does not exist")
        if not dir_in.is_dir():
            raise ValueError(f"ERROR: \'{dir_in}\' is not a directory.")

        dir_input_files = glob.glob(str(dir_in / "*.csv"))
        if not dir_input_files:
            logger.warning(f"Warning: No CSV files found in directory \'{dir_in}\'")
        input_files += [Path(file).resolve() for file in dir_input_files]

    input_files.sort()

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
            input_files.append(path)

    if not input_files:
        raise RuntimeError("ERROR: katpy has no input CSV files to join")

    input_type = check_file_extensions(input_files + [out_file])

    output_flags = [
        concatenate_selt,
        concatenate_melt,
        concatenate_qelt,
        concatenate_splt,
        concatenate_mplt,
        concatenate_qplt,
    ]

    sort_by_event = any(output_flags[OUT_SELT:OUT_QELT + 1])
    sort_by_period = any(output_flags[OUT_SPLT:OUT_QPLT + 1])
    assert sort_by_event + sort_by_period == 1, "incorrect flag config in katpy"

    with ExitStack() as stack:
        files_with_header, header = find_csv_with_header(stack, input_files)
        headers = header.strip().split(",")
        check_correct_headers(headers, output_flags.index(True))

        if unsorted:
            if input_type == ".csv":
                csv_concat_unsorted(stack, input_files, files_with_header, headers, out_file)
            elif input_type == ".bin":
                print("NOT IMPLEMENTED")
            else:
                raise RuntimeError(f"ERROR: katpy, file type {input_type} not supported.")
        else:
            if sort_by_event:
                if input_type == ".csv":
                    header_idxs = get_header_idxs(headers, ["EventId"])
                    csv_concat_sort_by_headers(
                        stack,
                        input_files,
                        files_with_header,
                        headers,
                        header_idxs,
                        lambda values: int(values[0]),
                        out_file,
                    )
                elif input_type == ".bin":
                    print("NOT IMPLEMENTED")
            else:
                if input_type == ".csv":
                    header_idxs = get_header_idxs(headers, ["EventId", "Period"])
                    csv_concat_sort_by_headers(
                        stack,
                        input_files,
                        files_with_header,
                        headers,
                        header_idxs,
                        lambda values: (int(values[0]), int(values[1])),
                        out_file,
                    )
                elif input_type == ".bin":
                    print("NOT IMPLEMENTED")
                else:
                    raise RuntimeError(f"ERROR: katpy, file type {input_type} not supported.")


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
