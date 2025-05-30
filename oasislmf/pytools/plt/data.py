import numpy as np

from oasislmf.pytools.common.data import oasis_int, oasis_float


VALID_EXT = ["csv", "bin", "parquet"]

SPLT_output = [
    ('Period', oasis_int, '%d'),
    ('PeriodWeight', oasis_float, '%.6f'),
    ('EventId', oasis_int, '%d'),
    ('Year', oasis_int, '%d'),
    ('Month', oasis_int, '%d'),
    ('Day', oasis_int, '%d'),
    ('Hour', oasis_int, '%d'),
    ('Minute', oasis_int, '%d'),
    ('SummaryId', oasis_int, '%d'),
    ('SampleId', oasis_int, '%d'),
    ('Loss', oasis_float, '%.2f'),
    ('ImpactedExposure', oasis_float, '%.2f'),
]

MPLT_output = [
    ('Period', oasis_int, '%d'),
    ('PeriodWeight', oasis_float, '%.6f'),
    ('EventId', oasis_int, '%d'),
    ('Year', oasis_int, '%d'),
    ('Month', oasis_int, '%d'),
    ('Day', oasis_int, '%d'),
    ('Hour', oasis_int, '%d'),
    ('Minute', oasis_int, '%d'),
    ('SummaryId', oasis_int, '%d'),
    ('SampleType', oasis_int, '%d'),
    ('ChanceOfLoss', oasis_float, '%.4f'),
    ('MeanLoss', oasis_float, '%.2f'),
    ('SDLoss', oasis_float, '%.2f'),
    ('MaxLoss', oasis_float, '%.2f'),
    ('FootprintExposure', oasis_float, '%.2f'),
    ('MeanImpactedExposure', oasis_float, '%.2f'),
    ('MaxImpactedExposure', oasis_float, '%.2f'),
]

QPLT_output = [
    ('Period', oasis_int, '%d'),
    ('PeriodWeight', oasis_float, '%.6f'),
    ('EventId', oasis_int, '%d'),
    ('Year', oasis_int, '%d'),
    ('Month', oasis_int, '%d'),
    ('Day', oasis_int, '%d'),
    ('Hour', oasis_int, '%d'),
    ('Minute', oasis_int, '%d'),
    ('SummaryId', oasis_int, '%d'),
    ('Quantile', oasis_float, '%.2f'),
    ('Loss', oasis_float, '%.2f'),
]

SPLT_headers = [c[0] for c in SPLT_output]
MPLT_headers = [c[0] for c in MPLT_output]
QPLT_headers = [c[0] for c in QPLT_output]
SPLT_dtype = np.dtype([(c[0], c[1]) for c in SPLT_output])
MPLT_dtype = np.dtype([(c[0], c[1]) for c in MPLT_output])
QPLT_dtype = np.dtype([(c[0], c[1]) for c in QPLT_output])
SPLT_fmt = ','.join([c[2] for c in SPLT_output])
MPLT_fmt = ','.join([c[2] for c in MPLT_output])
QPLT_fmt = ','.join([c[2] for c in QPLT_output])
