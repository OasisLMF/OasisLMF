from oasislmf.pytools.common.data import aggregatevulnerability_headers, aggregatevulnerability_dtype, aggregatevulnerability_fmt
from oasislmf.pytools.common.data import amplifications_headers, amplifications_dtype, amplifications_fmt


SUPPORTED_TYPES = {
    "aggregatevulnerability": {
        "headers": aggregatevulnerability_headers,
        "dtype": aggregatevulnerability_dtype,
        "fmt": aggregatevulnerability_fmt,
    },
    "amplifications": {
        "headers": amplifications_headers,
        "dtype": amplifications_dtype,
        "fmt": amplifications_fmt,
    },
}
