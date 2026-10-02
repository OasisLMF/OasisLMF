import numpy as np
import pandas as pd
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, resolve_file
from oasislmf.pytools.common.data import coverages_bin_dtype


def coverages_tobin(stack, file_in, file_out, file_type):
    """Write coverages.bin from csv: (tiv, n_building) per coverage, coverage_id implied by position."""
    tiv_dtype = coverages_bin_dtype.fields["tiv"][0]
    nb_dtype = coverages_bin_dtype.fields["n_building"][0]
    f = resolve_file(file_in, "r", stack)
    try:
        for chunk in pd.read_csv(f, usecols=["tiv", "n_building"],
                                 dtype={"tiv": tiv_dtype, "n_building": nb_dtype},
                                 chunksize=DEFAULT_BUFFER_SIZE):
            out = np.empty(len(chunk), dtype=coverages_bin_dtype)
            out["tiv"] = chunk["tiv"].to_numpy(dtype=tiv_dtype)
            out["n_building"] = chunk["n_building"].to_numpy(dtype=nb_dtype)
            # .write(tobytes()), not .tofile(): file_out may be a non-seekable pipe (#2140)
            file_out.write(out.tobytes())
    except pd.errors.EmptyDataError:
        pass
