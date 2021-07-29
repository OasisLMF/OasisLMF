import os
import sys
from io import StringIO
from typing import Optional

from pandas import read_csv, DataFrame

from .getmodel.get_model_process import GetModelProcess


def _process_input_data() -> Optional[DataFrame]:
    """
    Gets the input from the STDin and converts it to

    Returns: (Optional[DataFrame])
    """
    data = sys.stdin.buffer.read()
    if data == "":
        return None
    return read_csv(StringIO(data.decode()), sep=",")


def main() -> None:
    """
    Entry point of the 'new-model' command building the module and then piping it out as bytes.

    Returns: None
    """
    data_path: str = str(os.getcwd())
    process: GetModelProcess = GetModelProcess(data_path=data_path, events=_process_input_data())
    process.run()
    process.print_stream()
