"""JIT Compilation Test Module.

Pre-compiles all ~191 Numba JIT functions across the pytools suite by running
minimal integration tests for each tool. Uses multiprocessing to compile all
tools in parallel, reducing cold-start wall time significantly.

Usage:
    # Pre-compile all JIT functions in parallel (recommended)
    pytest tests/pytools/test_jit_compilation.py::test_jit_compile_all -v

    # Run a single tool's compilation (for debugging)
    pytest tests/pytools/test_jit_compilation.py::test_jit_fmpy -v

    # Run only JIT compilation tests by marker
    pytest -m jit_compile -v
"""
import os

import pytest

from oasislmf.warmup import (
    warmup,
    _compile_fmpy, _compile_modelpy_gulpy_gulmc,
    _compile_summarypy, _compile_eltpy, _compile_pltpy,
    _compile_aalpy, _compile_lecpy, _compile_katpy, _compile_plapy,
)

# Skip entire module when JIT is disabled (e.g. NUMBA_DISABLE_JIT=1 for coverage runs)
pytestmark = [
    pytest.mark.jit_compile,
    pytest.mark.skipif(
        os.environ.get("NUMBA_DISABLE_JIT", "0") == "1",
        reason="JIT compilation tests require Numba JIT enabled"
    ),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_jit_compile_all():
    """Compile all ~191 JIT functions in parallel across worker processes.

    Submits each tool's compilation to a process pool, waits for all to
    finish, and reports any failures by tool name.
    """
    errors = warmup()
    if errors:
        msg = "\n".join(f"  {name}: {err}" for name, err in errors.items())
        pytest.fail(f"JIT compilation failed for {len(errors)} tool(s):\n{msg}")


# Individual tests — for selective debugging of a single tool.

def test_jit_fmpy():
    """FM pipeline — normal calcrules (100/101) + stepped calcrules (27-38)."""
    _compile_fmpy()


def test_jit_modelpy_gulpy_gulmc():
    """evepy | modelpy | gulpy + gulmc subprocess pipelines."""
    _compile_modelpy_gulpy_gulmc()


def test_jit_summarypy():
    """summarypy manager on single_summary_set."""
    _compile_summarypy()


def test_jit_eltpy():
    """eltpy manager — event loss table."""
    _compile_eltpy()


def test_jit_pltpy():
    """pltpy manager — period loss table."""
    _compile_pltpy()


def test_jit_aalpy():
    """aalpy manager — annual aggregate loss."""
    _compile_aalpy()


def test_jit_lecpy():
    """lecpy manager — all 8 report flags."""
    _compile_lecpy()


def test_jit_katpy():
    """katpy manager — sorted mode for nb_heapq."""
    _compile_katpy()


def test_jit_plapy():
    """plapy — post-loss amplification."""
    _compile_plapy()
