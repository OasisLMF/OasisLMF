import os
import sys

from .getmodel.get_model_process import GetModelProcess


def main():
    print("the get model is firing")
    # takes in stream for events
    data_path: str = str(os.getcwd())
    data = sys.stdin.buffer.read()
    print("")
    print(f"\n\n\n\nhere is the data: {data.decode()}\n\n\n\n\n")
    process: GetModelProcess = GetModelProcess(data_path=data_path)
    process.run()
    print(process.stream)

