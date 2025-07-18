from pathlib import Path
import numpy as np
from oasislmf.pytools.pla.common import amp_factor_dtype
from oasislmf.pytools.pla.structure import read_lossfactors


def lossfactors_tobin(stack, file_in, file_out, file_type):
    if str(file_in) == "-":
        plafactors = read_lossfactors(
            ignore_file_type=set(["bin"]),
            use_stdin=True
        )
    else:
        lossfactors_fp = Path(file_in)
        plafactors = read_lossfactors(
            run_dir=lossfactors_fp.parent,
            ignore_file_type=set(["bin"]),
            filename=lossfactors_fp.name
        )

    # Write the 4-byte zero header
    np.array([0], dtype="i4").tofile(file_out)

    current_event_id = -1
    counter = 0
    factors = []
    for k, v in plafactors.items():
        if k[0] != current_event_id:
            if current_event_id != -1:
                np.array([current_event_id], dtype=np.int32).tofile(file_out)
                np.array([counter], dtype=np.int32).tofile(file_out)
                for af in factors:
                    np.array(af, dtype=amp_factor_dtype).tofile(file_out)
            current_event_id = k[0]
            counter = 0
            factors = []
        factors.append((k[1], v))
        counter += 1
    if current_event_id != -1:
        np.array([current_event_id], dtype=np.int32).tofile(file_out)
        np.array([counter], dtype=np.int32).tofile(file_out)
        for af in factors:
            np.array(af, dtype=amp_factor_dtype).tofile(file_out)
