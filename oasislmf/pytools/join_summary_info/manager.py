# join-summary-info/manager.py

import logging
import numpy as np
from pathlib import Path
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
        summaryinfo_file (str | os.PathLike): Path to summary-info csv file
        data_file (str | os.PathLike): Path to ORD output data file (e.g. SELT, MPLT, AAL, PSEPT) 
        output_file (str | os.PathLike): Path to combined output file
    """
    summaryinfo_file = Path(summaryinfo_file)
    data_file = Path(data_file)
    output_file = Path(output_file)

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

        # Replace the original file if needed
        if same_file and temp_output:
            temp_output.close()
            temp_output_file.replace(output_file)


@redirect_logging(exec_name='join-summary-info')
def main(summaryinfo=None, data=None, output=None, **kwargs):
    run(
        summaryinfo_file=summaryinfo,
        data_file=data,
        output_file=output,
    )
