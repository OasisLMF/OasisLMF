import numpy as np

from oasislmf.pytools.common.data import oasis_int, oasis_float


VALID_EXT = ["csv", "bin", "parquet"]

AAL_output = [
    ('SummaryId', oasis_int, '%d'),
    ('SampleType', oasis_int, '%d'),
    ('MeanLoss', oasis_float, '%.6f'),
    ('SDLoss', oasis_float, '%.6f'),
]

AAL_meanonly_output = [
    ('SummaryId', oasis_int, '%d'),
    ('SampleType', oasis_int, '%d'),
    ('MeanLoss', oasis_float, '%.6f'),
]

ALCT_output = [
    ("SummaryId", oasis_float, '%d'),
    ("MeanLoss", oasis_float, '%.6f'),
    ("SDLoss", oasis_float, '%.6f'),
    ("SampleSize", oasis_float, '%d'),
    ("LowerCI", oasis_float, '%.6f'),
    ("UpperCI", oasis_float, '%.6f'),
    ("StandardError", oasis_float, '%.6f'),
    ("RelativeError", oasis_float, '%.6f'),
    ("VarElementHaz", oasis_float, '%.6f'),
    ("StandardErrorHaz", oasis_float, '%.6f'),
    ("RelativeErrorHaz", oasis_float, '%.6f'),
    ("VarElementVuln", oasis_float, '%.6f'),
    ("StandardErrorVuln", oasis_float, '%.6f'),
    ("RelativeErrorVuln", oasis_float, '%.6f'),
]


AAL_headers = [c[0] for c in AAL_output]
AAL_meanonly_headers = [c[0] for c in AAL_meanonly_output]

AAL_dtype = np.dtype([(c[0], c[1]) for c in AAL_output])
AAL_meanonly_dtype = np.dtype([(c[0], c[1]) for c in AAL_meanonly_output])

AAL_fmt = ','.join([c[2] for c in AAL_output])
AAL_meanonly_fmt = ','.join([c[2] for c in AAL_meanonly_output])

ALCT_headers = [c[0] for c in ALCT_output]
ALCT_dtype = np.dtype([(c[0], c[1]) for c in ALCT_output])
ALCT_fmt = ','.join([c[2] for c in ALCT_output])
