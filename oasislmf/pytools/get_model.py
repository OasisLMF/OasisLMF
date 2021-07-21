import os

from .getmodel.get_model_process import GetModelProcess


def main():
    print("the get model is firing")
    data_path: str = str(os.getcwd())
    process: GetModelProcess = GetModelProcess(data_path=data_path)
    process.run()
    process.stream()

