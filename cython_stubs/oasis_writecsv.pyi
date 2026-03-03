from typing import IO, List
import numpy as np

def write_rows(
    output_file: IO[str],
    data: np.ndarray,
    headers: List[str],
    row_fmt: str,
) -> None: ...
