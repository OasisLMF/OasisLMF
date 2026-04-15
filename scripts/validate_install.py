#!/usr/bin/env python3
"""
Post-install validation for oasislmf.

Checks that the package and its key dependencies are importable and
reports platform/architecture information for debugging.

Usage:
    python -m scripts.validate_install
    # or after install:
    python scripts/validate_install.py

Exit codes:
    0 - all checks passed
    1 - one or more checks failed
"""
import importlib
import platform
import sys

REQUIRED_MODULES = [
    ("oasislmf", "oasislmf core"),
    ("numpy", "numerical computing"),
    ("pandas", "data manipulation"),
    ("scipy", "scientific computing"),
    ("numba", "JIT compilation"),
    ("pyarrow", "columnar data / Parquet"),
    ("fastparquet", "Parquet file support"),
    ("numexpr", "numerical expressions"),
    ("msgpack", "binary serialization"),
]

OPTIONAL_MODULES = [
    ("shapely", "geometry operations (extra)"),
    ("geopandas", "geospatial data (extra)"),
    ("sklearn", "machine learning (extra)"),
    ("rtree", "spatial indexing (extra)"),
]


def check_module(name, description):
    """Try to import a module and return (success, version_or_error)."""
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", getattr(mod, "version", "unknown"))
        return True, version
    except ImportError as exc:
        return False, str(exc)


def main():
    print("=" * 60)
    print("OasisLMF Installation Validation")
    print("=" * 60)
    print()
    print(f"Python:       {sys.version}")
    print(f"Platform:     {platform.platform()}")
    print(f"Machine:      {platform.machine()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print()

    failed = False

    print("--- Required Dependencies ---")
    for name, desc in REQUIRED_MODULES:
        ok, info = check_module(name, desc)
        status = "OK" if ok else "FAIL"
        print(f"  [{status:>4}] {name:<16} {info:<24} ({desc})")
        if not ok:
            failed = True

    print()
    print("--- Optional Dependencies (extras) ---")
    for name, desc in OPTIONAL_MODULES:
        ok, info = check_module(name, desc)
        status = "OK" if ok else "SKIP"
        print(f"  [{status:>4}] {name:<16} {info:<24} ({desc})")

    print()

    # Check oasislmf version specifically
    try:
        import oasislmf
        print(f"oasislmf version: {oasislmf.__version__}")
    except Exception:
        pass

    # Check for ktools binaries (informational)
    import shutil
    ktools_cmds = ["eve", "getmodel", "gulcalc", "fmcalc", "summarycalc"]
    ktools_found = [cmd for cmd in ktools_cmds if shutil.which(cmd)]
    ktools_missing = [cmd for cmd in ktools_cmds if not shutil.which(cmd)]

    print()
    print("--- ktools binaries (external, optional) ---")
    if ktools_found:
        print(f"  Found:   {', '.join(ktools_found)}")
    if ktools_missing:
        print(f"  Missing: {', '.join(ktools_missing)}")
    if not ktools_found:
        print("  (ktools not found on PATH — needed only for C-based execution kernels)")
        print("  (Python-based kernels gulpy/fmpy/gulmc work without ktools)")

    print()
    if failed:
        print("RESULT: FAIL — some required dependencies could not be imported.")
        print("        Check the errors above and install missing packages.")
        return 1
    else:
        print("RESULT: PASS — oasislmf is correctly installed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
