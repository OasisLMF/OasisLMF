
from tests.pytools.converters.helpers import case_runner


def test_lossfactors():
    case_runner("bintocsv", "lossfactors", "static")
    case_runner("csvtobin", "lossfactors", "static")


def test_random():
    case_runner("bintocsv", "random", "static")
    case_runner("csvtobin", "random", "static")


def test_weights():
    case_runner("bintocsv", "weights", "static")
    case_runner("csvtobin", "weights", "static")


def test_amplifications():
    case_runner("bintocsv", "amplifications", "input")
    case_runner("csvtobin", "amplifications", "input")


def test_correlations_items():
    case_runner("bintocsv", "correlations", "input")
    case_runner("csvtobin", "correlations", "input")


def test_complex_items():
    case_runner("bintocsv", "complex_items", "input")
    case_runner("csvtobin", "complex_items", "input")


def test_coverages():
    case_runner("bintocsv", "coverages", "input")
    case_runner("csvtobin", "coverages", "input")


def test_eve():
    case_runner("bintocsv", "eve", "input")
    case_runner("csvtobin", "eve", "input")


def test_items():
    case_runner("bintocsv", "items", "input")
    case_runner("csvtobin", "items", "input")


def test_occurrence():
    case_runner("bintocsv", "occurrence", "input", "occurrence")
    case_runner("bintocsv", "occurrence", "input", "occurrence_gran")
    case_runner("bintocsv", "occurrence", "input", "occurrence_noalg")
    case_runner("csvtobin", "occurrence", "input", "occurrence", no_of_periods=9)
    case_runner("csvtobin", "occurrence", "input", "occurrence_gran", no_of_periods=9, granular=True)
    case_runner("csvtobin", "occurrence", "input", "occurrence_noalg", no_of_periods=9, no_date_alg=True)


def test_periods():
    case_runner("bintocsv", "periods", "input")
    case_runner("csvtobin", "periods", "input")


def test_quantile():
    case_runner("bintocsv", "quantile", "input")
    case_runner("csvtobin", "quantile", "input")


def test_returnperiods():
    case_runner("bintocsv", "returnperiods", "input")
    case_runner("csvtobin", "returnperiods", "input")
