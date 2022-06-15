from typing import Optional

import numpy as np
import pandas as pd

from oasislmf.pytools.getmodel.common import Correlation


class CorrelationsData:

    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        self.data: Optional[pd.DataFrame] = data

    @staticmethod
    def from_csv(file_path: str) -> "CorrelationsData":
        return CorrelationsData(data=pd.read_csv(file_path))

    @staticmethod
    def from_bin(file_path: str) -> "CorrelationsData":
        data = pd.DataFrame(np.fromfile(file_path, dtype=Correlation))
        return CorrelationsData(data=data)

    def to_csv(self, file_path: str) -> None:
        self.data.to_csv(file_path, index=False)

    def to_bin(self, file_path: str) -> None:
        data = np.array(list(self.data.itertuples(index=False)), dtype=Correlation)
        data.tofile(file_path)
