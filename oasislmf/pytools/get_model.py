import os
import sys
from typing import Optional

from io import StringIO
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


def main():
    data_path: str = str(os.getcwd())
    process: GetModelProcess = GetModelProcess(data_path=data_path, events=_process_input_data())
    process.run()
    for i in process.stream:
        print(i, end="")

