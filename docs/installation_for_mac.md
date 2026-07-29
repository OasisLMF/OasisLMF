# Installation on macOS (including Apple Silicon)

## Overview

OasisLMF is a pure Python package. Starting from version 2.5.x, `pip install oasislmf`
works natively on Apple Silicon (M1/M2/M3/M4) Macs without Rosetta 2 emulation.

All compiled dependencies (numpy, scipy, numba, pandas, pyarrow, etc.) publish
pre-built ARM64 macOS wheels on PyPI.

## Prerequisites

### Python 3.10+

OasisLMF requires Python 3.10 or later. The system Python on macOS is typically 3.9,
so you will need to install a newer version.

**Option A — Homebrew (recommended):**

```bash
brew install python@3.12
```

**Option B — pyenv:**

```bash
brew install pyenv
pyenv install 3.12
pyenv local 3.12
```

Verify your Python version:

```bash
python3 --version
# Must show 3.10 or later
```

### System Libraries (for optional extras only)

If you plan to use the `[extra]` dependencies (geopandas, shapely, rtree, scikit-learn),
install the required C libraries:

```bash
brew install spatialindex geos
```

These are **not required** for the core `oasislmf` package.

## Installation

### Standard install

```bash
# Create a virtual environment (recommended)
python3 -m venv oasis-env
source oasis-env/bin/activate

# Install oasislmf
pip install oasislmf
```

### Install with optional geospatial extras

```bash
pip install oasislmf[extra]
```

### Install from source (development)

```bash
git clone https://github.com/OasisLMF/OasisLMF.git
cd OasisLMF

python3 -m venv .venv
source .venv/bin/activate

pip install pip-tools
pip-compile requirements.in -o requirements.txt
pip install -r requirements.txt
pip install -e .
```

## Post-install Validation

Run the validation script to confirm everything is working:

```bash
python scripts/validate_install.py
```

Or verify with a quick import check:

```bash
python -c "import oasislmf; print(f'oasislmf {oasislmf.__version__} installed successfully')"
```

Run the CLI to confirm entry points work:

```bash
oasislmf --help
```

## ktools (External C Binaries)

As of OasisLMF 2.5.x, ktools binaries are **no longer bundled** with the Python
package. OasisLMF includes Python-based replacements for the core ktools kernels:

| Python tool | Replaces (ktools) | Purpose |
|-------------|-------------------|---------|
| `gulpy`     | `gulcalc`         | Ground-up loss calculation |
| `gulmc`     | `gulcalc`         | GUL with Monte Carlo sampling |
| `fmpy`      | `fmcalc`          | Financial module calculation |
| `summarypy` | `summarycalc`     | Summary calculation |
| `plapy`     | `placalc`         | Post-loss amplification |
| `katpy`     | `kat`             | Concatenation tool |
| `eltpy`     | `eltcalc`         | Event loss table |
| `pltpy`     | `pltcalc`         | Period loss table |
| `aalpy`     | `aalcalc`         | Average annual loss |
| `lecpy`     | `leccalc`         | Loss exceedance curve |

For most workflows, the Python-based tools are sufficient and no ktools
installation is needed.

### Building ktools from source (optional)

If you need the original C-based ktools for performance or compatibility:

```bash
# Install build dependencies
brew install autoconf automake libtool zlib-ng

# Clone and build ktools
git clone https://github.com/OasisLMF/ktools.git
cd ktools
./autogen.sh
./configure --enable-osx --enable-o3 --prefix=$HOME/ktools-bin
make check
make install

# Add to PATH
export PATH="$HOME/ktools-bin/bin:$PATH"
```

## Troubleshooting

### "No matching distribution found for oasislmf"

Your Python version is likely too old. OasisLMF requires Python 3.10+:

```bash
python3 --version
```

### numba / llvmlite installation failure

If numba fails to install, ensure you're using Python 3.10-3.13 and pip 23+:

```bash
pip install --upgrade pip
pip install numba
```

If building from source is attempted (shouldn't happen with pip ≥23 on arm64),
you may need:

```bash
brew install llvm
```

### scipy requires macOS 12+

SciPy's ARM64 wheels require macOS 12.0 (Monterey) or later. If you're on an
older macOS version, upgrade your OS or use Rosetta 2 emulation.

### rtree / libspatialindex not found

This only affects the `[extra]` optional dependencies:

```bash
brew install spatialindex
pip install oasislmf[extra]
```

### Multiprocessing errors

OasisLMF uses `fork` context for multiprocessing, which differs from macOS's
default `spawn`. This is handled automatically since version 2.5.x. If you
encounter multiprocessing issues with custom model lookups, ensure your lookup
functions are defined at module level (not as class static methods or lambdas).

## Known Limitations

- **ktools C binaries** are not distributed via pip for macOS ARM64. Use the
  Python-based kernel tools or build ktools from source.
- **macOS < 12 (Monterey)**: scipy ARM64 wheels require macOS 12+.
- **Python 3.14+**: not yet tested. Use Python 3.10–3.13.

## Architecture Verification

To confirm you're running native ARM64 (not Rosetta 2 emulation):

```bash
python3 -c "import platform; print(platform.machine())"
# Should print: arm64
```

If it prints `x86_64`, your Python is running under Rosetta 2. Install a native
ARM64 Python via Homebrew or pyenv.
