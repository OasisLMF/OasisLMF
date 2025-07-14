from pathlib import Path
import numpy as np
from oasislmf.pytools.pla.common import amp_factor_dtype
from oasislmf.pytools.converters.data import TYPE_MAP
from oasislmf.pytools.pla.structure import read_lossfactors


def lossfactors_tobin(file_in, file_out, file_type):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]

    lossfactors_fp = Path(file_in)
    plafactors = read_lossfactors(lossfactors_fp.parent, set(["bin"]), filename=lossfactors_fp.name)

    with open(file_out, "wb") as fout:
        # Write the 4-byte zero header
        np.array([0], dtype="i4").tofile(fout)

        current_event_id = 0
        counter = 0
        factors = []
        for k, v in plafactors.items():
            if k[0] != current_event_id:
                if current_event_id != 0:
                    np.array([counter], dtype=np.int32).tofile(fout)
                    for af in factors:
                        np.array(af, dtype=amp_factor_dtype).tofile(fout)
                np.array(k[0], dtype=np.int32).tofile(fout)
                current_event_id = k[0]
                counter = 0
                factors = []
            factors.append((k[1], v))
            counter += 1
