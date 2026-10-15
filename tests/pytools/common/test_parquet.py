import numpy as np
import pyarrow.parquet as pq
import pytest

from oasislmf.pytools.common.parquet import BufferedParquetWriter

dtype = np.dtype([("event_id", "i4"), ("loss", "f4")])


def test_buffered_parquet_writer_batches_row_groups(tmp_path):
    fp = tmp_path / "out.parquet"
    chunks = [np.array([(e, e * 1.5)] * (e % 4), dtype=dtype) for e in range(1, 40)]
    with BufferedParquetWriter(fp, dtype, row_group_size=7) as writer:
        for chunk in chunks:
            writer.write(chunk)

    expected = np.concatenate(chunks)
    pq_file = pq.ParquetFile(fp)
    assert [pq_file.metadata.row_group(i).num_rows for i in range(pq_file.num_row_groups)] == [7] * 8 + [4]
    table = pq_file.read()
    np.testing.assert_array_equal(table["event_id"].to_numpy(), expected["event_id"])
    np.testing.assert_array_equal(table["loss"].to_numpy(), expected["loss"])


def test_buffered_parquet_writer_empty(tmp_path):
    fp = tmp_path / "out.parquet"
    with BufferedParquetWriter(fp, dtype):
        pass
    assert pq.read_table(fp).num_rows == 0


def test_buffered_parquet_writer_large_write(tmp_path):
    fp = tmp_path / "out.parquet"
    data = np.array([(e, e * 1.5) for e in range(25)], dtype=dtype)
    with BufferedParquetWriter(fp, dtype, row_group_size=10) as writer:
        writer.write(data)

    pq_file = pq.ParquetFile(fp)
    assert [pq_file.metadata.row_group(i).num_rows for i in range(pq_file.num_row_groups)] == [10, 10, 5]
    table = pq_file.read()
    np.testing.assert_array_equal(table["event_id"].to_numpy(), data["event_id"])
    np.testing.assert_array_equal(table["loss"].to_numpy(), data["loss"])


def test_buffered_parquet_writer_keeps_rows_on_error(tmp_path):
    fp = tmp_path / "out.parquet"
    data = np.array([(e, e * 1.5) for e in range(5)], dtype=dtype)
    with pytest.raises(RuntimeError):
        with BufferedParquetWriter(fp, dtype, row_group_size=10) as writer:
            writer.write(data)
            raise RuntimeError("boom")

    np.testing.assert_array_equal(pq.read_table(fp)["event_id"].to_numpy(), data["event_id"])
