# katpy/manager.py

from collections import Counter
from contextlib import ExitStack
import csv
import glob
import heapq
import logging
import numba as nb
import numpy as np
import shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile

from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, write_ndarray_to_fmt_csv
from oasislmf.pytools.common.utils.nb_heapq import heap_pop, heap_push, init_heap
from oasislmf.pytools.elt.data import MELT_dtype, MELT_fmt, MELT_headers, QELT_dtype, QELT_fmt, QELT_headers, SELT_dtype, SELT_fmt, SELT_headers
from oasislmf.pytools.plt.data import MPLT_dtype, MPLT_fmt, MPLT_headers, QPLT_dtype, QPLT_fmt, QPLT_headers, SPLT_dtype, SPLT_fmt, SPLT_headers
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


KAT_SELT = 0
KAT_MELT = 1
KAT_QELT = 2
KAT_SPLT = 3
KAT_MPLT = 4
KAT_QPLT = 5

KAT_MAP = {
    KAT_SELT: {
        "name": "SELT",
        "headers": SELT_headers,
        "dtype": SELT_dtype,
        "fmt": SELT_fmt,
    },
    KAT_MELT: {
        "name": "MELT",
        "headers": MELT_headers,
        "dtype": MELT_dtype,
        "fmt": MELT_fmt,
    },
    KAT_QELT: {
        "name": "QELT",
        "headers": QELT_headers,
        "dtype": QELT_dtype,
        "fmt": QELT_fmt,
    },
    KAT_SPLT: {
        "name": "SPLT",
        "headers": SPLT_headers,
        "dtype": SPLT_dtype,
        "fmt": SPLT_fmt,
    },
    KAT_MPLT: {
        "name": "MPLT",
        "headers": MPLT_headers,
        "dtype": MPLT_dtype,
        "fmt": MPLT_fmt,
    },
    KAT_QPLT: {
        "name": "QPLT",
        "headers": QPLT_headers,
        "dtype": QPLT_dtype,
        "fmt": QPLT_fmt,
    },
}


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
        file_type (int): File type int matching KAT_NAMES index
    """
    if file_type in KAT_MAP:
        expected_headers = KAT_MAP[file_type]["headers"]
    else:
        file_type_names = [v["name"] for v in KAT_MAP.values()]
        raise ValueError(f"ERROR: katpy, unknown file_type {file_type}, not in {file_type_names}")
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

    buffer_size = DEFAULT_BUFFER_SIZE
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


def bin_concat_unsorted(
    stack,
    file_paths,
    out_file,
):
    """Concats Binary files in order they are passed in.
    Args:
        stack (ExitStack): Exit Stack.
        file_paths (List[str | os.PathLike]): List of bin file paths.
        out_file (str | os.PathLike): Output Concatenated Binary file.
    """
    with stack.enter_context(out_file.open('wb')) as out:
        for fp in file_paths:
            with fp.open('rb') as bin_file:
                shutil.copyfileobj(bin_file, out)


@nb.njit(cache=True, error_model="numpy")
def merge_elt_data(memmaps):
    """Merge sorted chunks using a k-way merge algorithm
    Args:
        memmaps (List[np.memmap]): List of temporary file memmaps
    Yields:
        buffer (ndarray): yields sorted buffer from memmaps
    """
    min_heap = init_heap(num_compare=1)
    size = 0
    # Initialize the min_heap with the first row of each memmap
    for i, mmap in enumerate(memmaps):
        if len(mmap) > 0:
            first_row = mmap[0]
            min_heap, size = heap_push(min_heap, size, np.array(
                [first_row["EventId"], i, 0], dtype=np.int32
            ))

    buffer_size = DEFAULT_BUFFER_SIZE
    buffer = np.empty(buffer_size, dtype=memmaps[0].dtype)
    bidx = 0

    # Perform the k-way merge
    while size > 0:
        # The min heap will store the smallest row at the top when popped
        element, min_heap, size = heap_pop(min_heap, size)
        file_idx = element[-2]
        row_num = element[-1]
        smallest_row = memmaps[file_idx][row_num]

        # Add to buffer and yield when full
        buffer[bidx] = smallest_row
        bidx += 1
        if bidx >= buffer_size:
            yield buffer[:bidx]
            bidx = 0

        # Push the next row from the same file into the heap if there are any more rows
        if row_num + 1 < len(memmaps[file_idx]):
            next_row = memmaps[file_idx][row_num + 1]
            min_heap, size = heap_push(min_heap, size, np.array(
                [next_row["EventId"], file_idx, row_num + 1], dtype=np.int32
            ))
    yield buffer[:bidx]


@nb.njit(cache=True, error_model="numpy")
def merge_plt_data(memmaps):
    """Merge sorted chunks using a k-way merge algorithm
    Args:
        memmaps (List[np.memmap]): List of temporary file memmaps
    Yields:
        buffer (ndarray): yields sorted buffer from memmaps
    """
    min_heap = init_heap(num_compare=2)
    size = 0
    # Initialize the min_heap with the first row of each memmap
    for i, mmap in enumerate(memmaps):
        if len(mmap) > 0:
            first_row = mmap[0]
            min_heap, size = heap_push(min_heap, size, np.array(
                [first_row["EventId"], first_row["Period"], i, 0], dtype=np.int32
            ))

    buffer_size = DEFAULT_BUFFER_SIZE
    buffer = np.empty(buffer_size, dtype=memmaps[0].dtype)
    bidx = 0

    # Perform the k-way merge
    while size > 0:
        # The min heap will store the smallest row at the top when popped
        element, min_heap, size = heap_pop(min_heap, size)
        file_idx = element[-2]
        row_num = element[-1]
        smallest_row = memmaps[file_idx][row_num]

        # Add to buffer and yield when full
        buffer[bidx] = smallest_row
        bidx += 1
        if bidx >= buffer_size:
            yield buffer[:bidx]
            bidx = 0

        # Push the next row from the same file into the heap if there are any more rows
        if row_num + 1 < len(memmaps[file_idx]):
            next_row = memmaps[file_idx][row_num + 1]
            min_heap, size = heap_push(min_heap, size, np.array(
                [next_row["EventId"], next_row["Period"], file_idx, row_num + 1], dtype=np.int32
            ))
    yield buffer[:bidx]


def bin_concat_sort_by_headers(
    stack,
    file_paths,
    file_type,
    out_type,
    out_file,
):
    """Concats Binary files in order determined out_type and their respective merge functions
    Args:
        stack (ExitStack): Exit Stack.
        file_paths (List[str | os.PathLike]): List of bin file paths.
        file_type (int): File type int matching KAT_NAMES index
        out_type (str): Out type str between "elt" and "plt"
        out_file (str | os.PathLike): Output Concatenated Binary file.
    """
    files = [np.memmap(fp, dtype=KAT_MAP[file_type]["dtype"]) for fp in file_paths]

    if out_type == "elt":
        gen = merge_elt_data(files)
    elif out_type == "plt":
        gen = merge_plt_data(files)
    else:
        raise RuntimeError(f"ERROR: katpy, unknown out_type {out_type}")

    with stack.enter_context(out_file.open("wb")) as out:
        for data in gen:
            data.tofile(out)


def parquet_concat_unsorted(
    file_paths,
    out_file,
):
    """Concats Parquet files in order they are passed in.
    Args:
        file_paths (List[str | os.PathLike]): List of parquet file paths.
        out_file (str | os.PathLike): Output Concatenated Parquet file.
    """
    writer = None
    for fp in file_paths:
        pq_file = pq.ParquetFile(fp)
        for rg in range(pq_file.num_row_groups):
            table = pq_file.read_row_group(rg)
            if writer is None:
                writer = pq.ParquetWriter(out_file, table.schema)
            writer.write_table(table)

    if writer:
        writer.close()


def parquet_kway_merge(
    file_paths,
    keys,
    chunk_size=100000,
):
    """Merge sorted chunks using a k-way merge algorithm
    Args:
        file_paths (List[str | os.PathLike]): List of parquet file paths.
        keys (List[str]): List of keys to sort by
        chunk_size (int): Chunk size for reading parquet files. Defaults to 100000.
    Yields:
        buffer (pa.Table): yields sorted pyarrow table from input files
    """
    # Helper to read batches and manage current state for each file
    class FileStream:
        def __init__(self, path):
            self.reader = pq.ParquetFile(str(path)).iter_batches(batch_size=chunk_size)
            self.chunk = None
            self.table = None
            self.index = 0
            self._load_next_chunk()

        def _load_next_chunk(self):
            try:
                self.chunk = next(self.reader)
                self.table = self.chunk.to_pydict()
                self.index = 0
            except StopIteration:
                self.chunk = None
                self.table = None

        def has_data(self):
            return self.table is not None

        def current_key(self, keys):
            return tuple(self.table[k][self.index] for k in keys)

        def current_row(self):
            return {k: self.table[k][self.index] for k in self.table}

        def advance(self):
            self.index += 1
            if self.index >= len(next(iter(self.table.values()))):
                self._load_next_chunk()

    streams = [FileStream(fp) for fp in file_paths]
    heap = [
        (stream.current_key(keys), i)
        for i, stream in enumerate(streams) if stream.has_data()
    ]
    heapq.heapify(heap)

    buffer = {}
    schema = None

    while heap:
        _, i = heapq.heappop(heap)
        stream = streams[i]

        # Initialize schema and buffer
        if schema is None:
            schema = stream.chunk.schema
            buffer = {name: [] for name in schema.names}

        # Append current row to buffer
        row = stream.current_row()
        for col in buffer:
            buffer[col].append(row[col])

        stream.advance()
        if stream.has_data():
            heapq.heappush(heap, (stream.current_key(keys), i))

        # Output buffer if full
        if len(buffer[keys[0]]) >= chunk_size:
            yield pa.table(buffer, schema=schema)
            buffer = {name: [] for name in schema.names}

    # Yield remaining buffer
    if buffer and buffer[keys[0]]:
        yield pa.table(buffer, schema=schema)


def parquet_concat_sorted(
    file_paths,
    out_type,
    out_file,
    chunk_size=100000,
):
    """Concats Parquet files in order determined out_type and their respective merge functions
    Args:
        file_paths (List[str | os.PathLike]): List of parquet file paths.
        out_type (str): Out type str between "elt" and "plt"
        out_file (str | os.PathLike): Output Concatenated Parquet file.
        chunk_size (int): Chunk size for reading parquet files. Defaults to 100000.
    """
    if out_type == "elt":
        keys = ["EventId"]
    elif out_type == "plt":
        keys = ["EventId", "Period"]
    else:
        raise RuntimeError(f"Unknown out_type: {out_type}")

    writer = None

    for table in parquet_kway_merge(file_paths, keys, chunk_size):
        if writer is None:
            writer = pq.ParquetWriter(out_file, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()


def run(
    out_file,
    file_type=None,
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
        file_type (str, optional): Input file type suffix, if not discernible from input files. Defaults to None.
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
    input_files = []

    # Check and add files from dir_in
    if dir_in:
        dir_in = Path(dir_in)
        if not dir_in.exists():
            raise FileNotFoundError(f"ERROR: Directory \'{dir_in}\' does not exist")
        if not dir_in.is_dir():
            raise ValueError(f"ERROR: \'{dir_in}\' is not a directory.")

        dir_csv_input_files = glob.glob(str(dir_in / "*.csv"))
        dir_bin_input_files = glob.glob(str(dir_in / "*.bin"))
        dir_parquet_input_files = glob.glob(str(dir_in / "*.parquet"))
        if not dir_csv_input_files and not dir_bin_input_files and not dir_parquet_input_files:
            logger.warning(f"Warning: No valid files found in directory \'{dir_in}\'")
        input_files += [Path(file).resolve() for file in dir_csv_input_files + dir_bin_input_files + dir_parquet_input_files]

    input_files.sort()

    # Check and add files from files_in
    if files_in:
        for file in files_in:
            path = Path(file).resolve()
            if not path.exists():
                raise FileNotFoundError(f"ERROR: File \'{path}\' does not exist.")
            if not path.is_file():
                raise FileNotFoundError(f"ERROR: File \'{path}\' is not a valid file.")
            input_files.append(path)

    if not input_files:
        raise RuntimeError("ERROR: katpy has no input CSV files to join")

    out_file = Path(out_file).resolve()
    input_type = check_file_extensions(input_files)
    output_type = out_file.suffix

    if input_type == "":
        if not file_type:
            raise RuntimeError("ERROR: katpy, no discernible file type suffix found from input files, please provide a file_type")
        input_type = "." + file_type

    # If out_file is a csv and input_files are not csvs, then output to temporary outfile
    # of type input_type, and convert to csv after
    bin_to_csv = False
    bin_to_parquet = False
    if output_type != input_type:
        if input_type == ".bin":
            if output_type not in [".csv", ".parquet"]:
                raise RuntimeError(f"ERROR: katpy does not support concatenating input files of type {input_type} to output type {output_type}")
            final_out_file_path = out_file
            temp_file = tempfile.NamedTemporaryFile(suffix=input_type, delete=False)
            out_file = Path(temp_file.name)
            bin_to_csv = output_type == ".csv"
            bin_to_parquet = output_type == ".parquet"
        else:
            raise RuntimeError(f"ERROR: katpy does not support concatenating input files of type {input_type} to output type {output_type}")

    output_flags = [
        concatenate_selt,
        concatenate_melt,
        concatenate_qelt,
        concatenate_splt,
        concatenate_mplt,
        concatenate_qplt,
    ]

    sort_by_event = any(output_flags[KAT_SELT:KAT_QELT + 1])
    sort_by_period = any(output_flags[KAT_SPLT:KAT_QPLT + 1])
    assert sort_by_event + sort_by_period == 1, "incorrect flag config in katpy"
    out_type = output_flags.index(True)

    with ExitStack() as stack:
        if input_type == ".csv":
            files_with_header, header = find_csv_with_header(stack, input_files)
            headers = header.strip().split(",")
            check_correct_headers(headers, out_type)

            if unsorted:
                csv_concat_unsorted(stack, input_files, files_with_header, headers, out_file)
            elif sort_by_event:
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
            elif sort_by_period:
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
            if unsorted:
                bin_concat_unsorted(stack, input_files, out_file)
            else:
                sort_type = "elt" if sort_by_event else "plt"
                bin_concat_sort_by_headers(
                    stack,
                    input_files,
                    out_type,
                    sort_type,
                    out_file,
                )
        elif input_type == ".parquet":
            if unsorted:
                parquet_concat_unsorted(
                    input_files,
                    out_file,
                )
            else:
                sort_type = "elt" if sort_by_event else "plt"
                parquet_concat_sorted(
                    input_files,
                    sort_type,
                    out_file,
                )
        else:
            raise RuntimeError(f"ERROR: katpy, file type {input_type} not supported.")

    if bin_to_csv or bin_to_parquet:
        data = np.memmap(out_file, dtype=KAT_MAP[out_type]["dtype"])
        headers = KAT_MAP[out_type]["headers"]
        fmt = KAT_MAP[out_type]["fmt"]
        num_rows = data.shape[0]
        if bin_to_csv:
            csv_out_file = open(final_out_file_path, "w")
            csv_out_file.write(",".join(headers) + "\n")

            buffer_size = DEFAULT_BUFFER_SIZE
            for start in range(0, num_rows, buffer_size):
                end = min(start + buffer_size, num_rows)
                buffer_data = data[start:end]
                write_ndarray_to_fmt_csv(csv_out_file, buffer_data, headers, fmt)
            csv_out_file.close()
        if bin_to_parquet:
            parquet_writer = None
            buffer_size = DEFAULT_BUFFER_SIZE
            for start in range(0, num_rows, buffer_size):
                end = min(start + buffer_size, num_rows)
                buffer_data = data[start:end]

                arrays = []
                fields = []
                for name in buffer_data.dtype.names:
                    array = pa.array(buffer_data[name])
                    arrays.append(array)
                    fields.append((name, array.type))

                schema = pa.schema(fields)
                table = pa.Table.from_arrays(arrays, schema=schema)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(final_out_file_path, schema)
                parquet_writer.write_table(table)

            if parquet_writer is not None:
                parquet_writer.close()


@redirect_logging(exec_name='katpy')
def main(
    out=None,
    file_type=None,
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
        file_type=file_type,
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
