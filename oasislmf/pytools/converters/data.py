from oasislmf.pytools.common.data import aggregatevulnerability_headers, aggregatevulnerability_dtype, aggregatevulnerability_fmt
from oasislmf.pytools.common.data import damagebin_headers, damagebin_dtype, damagebin_fmt
from oasislmf.pytools.common.data import random_headers, random_dtype, random_fmt

from oasislmf.pytools.common.data import amplifications_headers, amplifications_dtype, amplifications_fmt
from oasislmf.pytools.common.data import complex_items_meta_headers, complex_items_meta_dtype, complex_items_meta_fmt
from oasislmf.pytools.common.data import correlations_headers, correlations_dtype, correlations_fmt
from oasislmf.pytools.common.data import coverages_headers, coverages_dtype, coverages_fmt
from oasislmf.pytools.common.data import eve_headers, eve_dtype, eve_fmt
from oasislmf.pytools.common.data import fm_policytc_headers, fm_policytc_dtype, fm_policytc_fmt
from oasislmf.pytools.common.data import fm_profile_headers, fm_profile_dtype, fm_profile_fmt
from oasislmf.pytools.common.data import fm_profile_step_headers, fm_profile_step_dtype, fm_profile_step_fmt
from oasislmf.pytools.common.data import fm_programme_headers, fm_programme_dtype, fm_programme_fmt
from oasislmf.pytools.common.data import fm_summary_xref_headers, fm_summary_xref_dtype, fm_summary_xref_fmt
from oasislmf.pytools.common.data import fm_xref_headers, fm_xref_dtype, fm_xref_fmt
from oasislmf.pytools.common.data import gul_summary_xref_headers, gul_summary_xref_dtype, gul_summary_xref_fmt
from oasislmf.pytools.common.data import items_headers, items_dtype, items_fmt
from oasislmf.pytools.common.data import occurrence_headers, occurrence_dtype, occurrence_fmt
from oasislmf.pytools.common.data import periods_headers, periods_dtype, periods_fmt
from oasislmf.pytools.common.data import quantile_headers, quantile_dtype, quantile_fmt
from oasislmf.pytools.common.data import returnperiods_headers, returnperiods_dtype, returnperiods_fmt

from oasislmf.pytools.aal.data import AAL_headers, AAL_dtype, AAL_fmt
from oasislmf.pytools.aal.data import AAL_meanonly_headers, AAL_meanonly_dtype, AAL_meanonly_fmt
from oasislmf.pytools.aal.data import ALCT_headers, ALCT_dtype, ALCT_fmt

from oasislmf.pytools.elt.data import SELT_headers, SELT_dtype, SELT_fmt
from oasislmf.pytools.elt.data import MELT_headers, MELT_dtype, MELT_fmt
from oasislmf.pytools.elt.data import QELT_headers, QELT_dtype, QELT_fmt

from oasislmf.pytools.lec.data import EPT_headers, EPT_dtype, EPT_fmt
from oasislmf.pytools.lec.data import PSEPT_headers, PSEPT_dtype, PSEPT_fmt

from oasislmf.pytools.plt.data import SPLT_headers, SPLT_dtype, SPLT_fmt
from oasislmf.pytools.plt.data import MPLT_headers, MPLT_dtype, MPLT_fmt
from oasislmf.pytools.plt.data import QPLT_headers, QPLT_dtype, QPLT_fmt


SUPPORTED_CSVTOBIN = [
    "aggregatevulnerability",
    "damagebin",
    "random",
    "amplifications",
    "complex_items",
    "correlations",
    "coverages",
    "eve",
    "fm_policytc",
    "fm_profile",
    "fm_profile_step",
    "fm_programme",
    "fm_summary_xref",
    "fm_xref",
    "gul_summary_xref",
    "items",
    "occurrence",
    "periods",
    "quantile",
    "returnperiods",
    "aal",
    "aalmeanonly",
    "alct",
    "selt",
    "melt",
    "qelt",
    "ept",
    "psept",
    "splt",
    "mplt",
    "qplt",
]


SUPPORTED_BINTOCSV = [
    "aggregatevulnerability",
    "damagebin",
    "random",
    "amplifications",
    "complex_items",
    "correlations",
    "coverages",
    "eve",
    "fm_policytc",
    "fm_profile",
    "fm_profile_step",
    "fm_programme",
    "fm_summary_xref",
    "fm_xref",
    "gul_summary_xref",
    "items",
    "occurrence",
    "periods",
    "quantile",
    "returnperiods",
    "aal",
    "aalmeanonly",
    "alct",
    "selt",
    "melt",
    "qelt",
    "ept",
    "psept",
    "splt",
    "mplt",
    "qplt",
]


TYPE_MAP = {
    # Static
    "aggregatevulnerability": {
        "headers": aggregatevulnerability_headers,
        "dtype": aggregatevulnerability_dtype,
        "fmt": aggregatevulnerability_fmt,
    },
    "damagebin": {
        "headers": damagebin_headers,
        "dtype": damagebin_dtype,
        "fmt": damagebin_fmt,
    },
    "random": {
        "headers": random_headers,
        "dtype": random_dtype,
        "fmt": random_fmt,
    },
    # Input
    "amplifications": {
        "headers": amplifications_headers,
        "dtype": amplifications_dtype,
        "fmt": amplifications_fmt,
    },
    "complex_items": {
        "headers": complex_items_meta_headers,
        "dtype": complex_items_meta_dtype,
        "fmt": complex_items_meta_fmt,
    },
    "correlations": {
        "headers": correlations_headers,
        "dtype": correlations_dtype,
        "fmt": correlations_fmt,
    },
    "coverages": {
        "headers": coverages_headers,
        "dtype": coverages_dtype,
        "fmt": coverages_fmt,
    },
    "eve": {
        "headers": eve_headers,
        "dtype": eve_dtype,
        "fmt": eve_fmt,
    },
    "fm_policytc": {
        "headers": fm_policytc_headers,
        "dtype": fm_policytc_dtype,
        "fmt": fm_policytc_fmt,
    },
    "fm_profile": {
        "headers": fm_profile_headers,
        "dtype": fm_profile_dtype,
        "fmt": fm_profile_fmt,
    },
    "fm_profile_step": {
        "headers": fm_profile_step_headers,
        "dtype": fm_profile_step_dtype,
        "fmt": fm_profile_step_fmt,
    },
    "fm_programme": {
        "headers": fm_programme_headers,
        "dtype": fm_programme_dtype,
        "fmt": fm_programme_fmt,
    },
    "fm_summary_xref": {
        "headers": fm_summary_xref_headers,
        "dtype": fm_summary_xref_dtype,
        "fmt": fm_summary_xref_fmt,
    },
    "fm_xref": {
        "headers": fm_xref_headers,
        "dtype": fm_xref_dtype,
        "fmt": fm_xref_fmt,
    },
    "gul_summary_xref": {
        "headers": gul_summary_xref_headers,
        "dtype": gul_summary_xref_dtype,
        "fmt": gul_summary_xref_fmt,
    },
    "items": {
        "headers": items_headers,
        "dtype": items_dtype,
        "fmt": items_fmt,
    },
    "periods": {
        "headers": periods_headers,
        "dtype": periods_dtype,
        "fmt": periods_fmt,
    },
    "occurrence": {
        "headers": occurrence_headers,
        "dtype": occurrence_dtype,
        "fmt": occurrence_fmt,
    },
    "quantile": {
        "headers": quantile_headers,
        "dtype": quantile_dtype,
        "fmt": quantile_fmt,
    },
    "returnperiods": {
        "headers": returnperiods_headers,
        "dtype": returnperiods_dtype,
        "fmt": returnperiods_fmt,
    },
    # Output AAL
    "aal": {
        "headers": AAL_headers,
        "dtype": AAL_dtype,
        "fmt": AAL_fmt,
    },
    "aalmeanonly": {
        "headers": AAL_meanonly_headers,
        "dtype": AAL_meanonly_dtype,
        "fmt": AAL_meanonly_fmt,
    },
    "alct": {
        "headers": ALCT_headers,
        "dtype": ALCT_dtype,
        "fmt": ALCT_fmt,
    },
    # Output ELT
    "selt": {
        "headers": SELT_headers,
        "dtype": SELT_dtype,
        "fmt": SELT_fmt,
    },
    "melt": {
        "headers": MELT_headers,
        "dtype": MELT_dtype,
        "fmt": MELT_fmt,
    },
    "qelt": {
        "headers": QELT_headers,
        "dtype": QELT_dtype,
        "fmt": QELT_fmt,
    },
    # Output LEC
    "ept": {
        "headers": EPT_headers,
        "dtype": EPT_dtype,
        "fmt": EPT_fmt,
    },
    "psept": {
        "headers": PSEPT_headers,
        "dtype": PSEPT_dtype,
        "fmt": PSEPT_fmt,
    },
    # Output PLT
    "splt": {
        "headers": SPLT_headers,
        "dtype": SPLT_dtype,
        "fmt": SPLT_fmt,
    },
    "mplt": {
        "headers": MPLT_headers,
        "dtype": MPLT_dtype,
        "fmt": MPLT_fmt,
    },
    "qplt": {
        "headers": QPLT_headers,
        "dtype": QPLT_dtype,
        "fmt": QPLT_fmt,
    },
}
