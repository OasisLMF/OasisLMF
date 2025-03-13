import numpy as np

from oasislmf.pytools.common.data import oasis_int, oasis_float


SELT_output = [
    ('EventId', oasis_int, '%d'),
    ('SummaryId', oasis_int, '%d'),
    ('SampleId', oasis_int, '%d'),
    ('Loss', oasis_float, '%.2f'),
    ('ImpactedExposure', oasis_float, '%.2f'),
]

MELT_output = [
    ('EventId', oasis_int, '%d'),
    ('SummaryId', oasis_int, '%d'),
    ('SampleType', oasis_int, '%d'),
    ('EventRate', oasis_float, '%.6f'),
    ('ChanceOfLoss', oasis_float, '%.6f'),
    ('MeanLoss', oasis_float, '%.6f'),
    ('SDLoss', oasis_float, '%.6f'),
    ('MaxLoss', oasis_float, '%.6f'),
    ('FootprintExposure', oasis_float, '%.6f'),
    ('MeanImpactedExposure', oasis_float, '%.6f'),
    ('MaxImpactedExposure', oasis_float, '%.6f'),
]

QELT_output = [
    ('EventId', oasis_int, '%d'),
    ('SummaryId', oasis_int, '%d'),
    ('Quantile', oasis_float, '%.6f'),
    ('Loss', oasis_float, '%.6f'),
]

SELT_headers = [c[0] for c in SELT_output]
MELT_headers = [c[0] for c in MELT_output]
QELT_headers = [c[0] for c in QELT_output]
SELT_dtype = np.dtype([(c[0], c[1]) for c in SELT_output])
MELT_dtype = np.dtype([(c[0], c[1]) for c in MELT_output])
QELT_dtype = np.dtype([(c[0], c[1]) for c in QELT_output])
SELT_fmt = ','.join([c[2] for c in SELT_output])
MELT_fmt = ','.join([c[2] for c in MELT_output])
QELT_fmt = ','.join([c[2] for c in QELT_output])
