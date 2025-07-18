import numpy as np
from oasislmf.pytools.converters.csvtobin.utils.common import read_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO


def returnperiods_tobin(stack, file_in, file_out, file_type):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    data = read_csv_as_ndarray(stack, file_in, headers, dtype)
    data = np.sort(data, order="return_period")[::-1]
    data.tofile(file_out)
