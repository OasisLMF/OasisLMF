from oasislmf.pytools.common.data import generate_output_metadata, oasis_int, oasis_float


VALID_EXT = ["csv", "bin", "parquet"]

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

SELT_headers, SELT_dtype, SELT_fmt = generate_output_metadata(SELT_output)
MELT_headers, MELT_dtype, MELT_fmt = generate_output_metadata(MELT_output)
QELT_headers, QELT_dtype, QELT_fmt = generate_output_metadata(QELT_output)
