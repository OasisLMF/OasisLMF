# join-summary-info/manager.py

import logging
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import tempfile
from contextlib import ExitStack

from oasislmf.pytools.utils import redirect_logging


logger = logging.getLogger(__name__)


def load_summary_info(stack, summaryinfo_file):
    """Load summary-info data into an array as strings to maintain sigfigs/formatting
    Args:
        stack (ExitStack): Exit Stack
        summaryinfo_file (str | os.PathLike): Path to summary-info csv file
    Returns:
        full_summary_data (ndarray[object]): Array of strings, indexed by Summary Id
        headers (List[str]): List of strings for summary info headers to add to data
        max_summary_id (int): Max Summary ID
    """
    summaryinfo_file = Path(summaryinfo_file)

    if summaryinfo_file.suffix == ".csv":
        summary_ids = []
        summary_data = []

        with stack.enter_context(open(summaryinfo_file, "r")) as fin:
            headers = fin.readline().strip().split(",")
            summary_id_col_idx = headers.index("summary_id")

            for line in fin:
                row = line.strip().split(",")
                summary_ids.append(row[summary_id_col_idx])
                summary_data.append(",".join(row[:summary_id_col_idx] + row[summary_id_col_idx + 1:]))

        headers = headers[:summary_id_col_idx] + headers[summary_id_col_idx + 1:]

        summary_ids = np.array(summary_ids, dtype=np.int64)
        summary_data = np.array(summary_data, dtype=object)
    elif summaryinfo_file.suffix == ".parquet":
        table = pq.read_table(summaryinfo_file)
        df = table.to_pandas()

        if "summary_id" not in df.columns:
            raise ValueError("Missing 'summary_id' column in summary info file.")

        summary_ids = df["summary_id"].to_numpy(dtype=np.int64)
        summary_data = df.drop(columns=["summary_id"]).astype(str).agg(",".join, axis=1).to_numpy(dtype=object)
        headers = [col for col in df.columns if col != "summary_id"]
    else:
        raise ValueError(f"Unsupported file format {summaryinfo_file.suffix}.")

    max_summary_id = summary_ids.max()
    full_summary_data = np.full((max_summary_id + 1,), "," * (len(headers) - 1), dtype=object)
    for i in range(len(summary_ids)):
        full_summary_data[summary_ids[i]] = summary_data[i]

    return full_summary_data, headers, max_summary_id


def run(
    summaryinfo_file,
    data_file,
    output_file,
):
    """Join the summary-info file to the given ORD data file based on SummaryId
    Args:
        summaryinfo_file (str | os.PathLike): Path to summary-info file
        data_file (str | os.PathLike): Path to ORD output data file (e.g. SELT, MPLT, AAL, PSEPT) 
        output_file (str | os.PathLike): Path to combined output file
    """
    summaryinfo_file = Path(summaryinfo_file)
    data_file = Path(data_file)
    output_file = Path(output_file)

    valid_file_suffix = [".csv", ".parquet"]

    if summaryinfo_file.suffix not in valid_file_suffix or \
       data_file.suffix not in valid_file_suffix or \
       output_file.suffix not in valid_file_suffix:
        raise ValueError(f"All files must be in one of {valid_file_suffix} formats.")

    if not (summaryinfo_file.suffix == data_file.suffix == output_file.suffix):
        raise ValueError("Summary info, data, and output files must all have the same format.")

    with ExitStack() as stack:
        summary_data, summary_headers, max_summary_id = load_summary_info(stack, summaryinfo_file)

        # Use a temporary file if input and output are the same
        same_file = data_file.resolve() == output_file.resolve()
        temp_output = None

        if same_file:
            temp_output = tempfile.NamedTemporaryFile(mode="w", delete=False)
            temp_output_file = Path(temp_output.name)
        else:
            temp_output_file = output_file

        if data_file.suffix == ".csv":
            with stack.enter_context(open(data_file, "r")) as data_fin, stack.enter_context(open(temp_output_file, "w")) as fout:
                data_headers = data_fin.readline().strip().split(",")
                fout.write(",".join(data_headers + summary_headers) + "\n")

                summary_id_col_idx = data_headers.index("SummaryId")
                for line in data_fin:
                    row = line.strip().split(",")
                    summary_id = int(row[summary_id_col_idx])
                    if summary_id > max_summary_id:
                        fout.write(line.strip() + ("," * len(summary_headers)) + "\n")
                    else:
                        fout.write(line.strip() + "," + summary_data[summary_id] + "\n")
        elif data_file.suffix == ".parquet":
            table = pq.read_table(data_file)
            df = table.to_pandas()

            if "SummaryId" not in df.columns:
                raise ValueError("Missing 'SummaryId' column in data file.")

            df[summary_headers] = df["SummaryId"].apply(
                lambda sid: summary_data[sid] if sid <= max_summary_id else ""
            ).str.split(",", expand=True)
            pq.write_table(pa.Table.from_pandas(df), temp_output_file)
        else:
            raise ValueError(f"Unsupported file format {data_file.suffix}.")

        # Replace the original file if needed
        if same_file and temp_output:
            temp_output.close()
            temp_output_file.replace(output_file)
            temp_output_file.unlink(missing_ok=True)


@redirect_logging(exec_name='join-summary-info')
def main(summaryinfo=None, data=None, output=None, **kwargs):
    run(
        summaryinfo_file=summaryinfo,
        data_file=data,
        output_file=output,
    )
