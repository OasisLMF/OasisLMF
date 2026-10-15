import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE


class BufferedParquetWriter:
    """Parquet writer that buffers rows and writes them in fixed-size row groups.

    Args:
        file_path (str | os.PathLike): Output parquet file path.
        dtype (np.dtype): Structured numpy dtype of the rows to write.
        row_group_size (int): Number of rows per row group. Defaults to DEFAULT_BUFFER_SIZE.
    """

    def __init__(self, file_path, dtype, row_group_size=DEFAULT_BUFFER_SIZE):
        self.schema = pa.schema([(name, pa.from_numpy_dtype(dtype[name])) for name in dtype.names])
        self.writer = pq.ParquetWriter(file_path, self.schema)
        self.buffer = np.empty(row_group_size, dtype=dtype)
        self.size = 0

    def write(self, data):
        """Copy rows into the buffer, writing a row group each time it fills.

        Args:
            data (np.ndarray): Structured array of rows with the writer's dtype.
        """
        start = 0
        while start < len(data):
            n = min(len(data) - start, len(self.buffer) - self.size)
            self.buffer[self.size:self.size + n] = data[start:start + n]
            self.size += n
            start += n
            if self.size == len(self.buffer):
                self.flush()

    def flush(self):
        if self.size == 0:
            return
        rows = self.buffer[:self.size]
        arrays = [pa.array(rows[name]) for name in rows.dtype.names]
        self.writer.write_table(pa.Table.from_arrays(arrays, schema=self.schema))
        self.size = 0

    def close(self):
        try:
            self.flush()
        finally:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
