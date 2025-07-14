import numpy as np
from oasislmf.pytools.converters.csvtobin.utils.common import read_csv_as_ndarray
from oasislmf.pytools.converters.data import TYPE_MAP


def returnperiods_tobin(file_in, file_out, file_type):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    data = read_csv_as_ndarray(file_in, headers, dtype)
    data = np.sort(data, order="return_period")[::-1]
    data.tofile(file_out)
