#!/usr/bin/env python3
"""Warm the Numba JIT cache for all pytools modules.

Runs each tool's compilation in a parallel process pool so that all ~191
Numba-compiled functions are cached in __pycache__ before the first real
model run.  This eliminates the 163-365 s cold-start overhead.

Test assets are bundled in oasislmf/_data/warmup/ (~400 KB) so this works
both from a source checkout and after ``pip install oasislmf`` from PyPI.

Usage (CLI):
    oasislmf warmup

Usage (standalone):
    python -m oasislmf.warmup

Usage (Dockerfile):
    RUN pip install oasislmf && oasislmf warmup

Usage (pytest):
    pytest tests/pytools/test_jit_compilation.py::test_jit_compile_all -v

Disable (when using NUMBA_DISABLE_JIT):
    NUMBA_DISABLE_JIT=1 oasislmf warmup
"""

import functools
import os
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory


# ---------------------------------------------------------------------------
# Locate bundled warmup assets inside the installed package.
# ---------------------------------------------------------------------------

def _get_data_dir():
    """Return path to oasislmf/_data/warmup/, whether installed or in-tree."""
    return Path(__file__).resolve().parent / "_data" / "warmup"


_DATA_DIR = _get_data_dir()


occurrence_rel_path = Path("input", "occurrence.bin")
periods_rel_path = Path("input", "periods.bin")
returnperiods_rel_path = Path("input", "returnperiods.bin")
quantile_rel_path = Path("input", "quantile.bin")
correlations_rel_path = Path("input", "correlations.bin")
coverages_rel_path = Path("input", "coverages.bin")
events_rel_path = Path("input", "events.bin")
items_rel_path = Path("input", "items.bin")
amplifications_rel_path = Path("input", "amplifications.bin")
damage_bin_dict_rel_path = Path("static", "damage_bin_dict.bin")
footprint_rel_path = Path("static", "footprint.bin")
footprint_idx_rel_path = Path("static", "footprint.idx")
vulnerability_rel_path = Path("static", "vulnerability.bin")
lossfactors_rel_path = Path("static", "lossfactors.bin")

summary_rel_path = Path("work", "gul", "summarypy.bin")

def _copy_rel_path(src, dest):
    os.makedirs(Path(dest).parent, exist_ok=True)
    shutil.copyfile(src, dest)

# ---------------------------------------------------------------------------
# Worker functions — top-level for pickling by ProcessPoolExecutor.
# Each does all imports locally so child processes start clean.
# ---------------------------------------------------------------------------

def _compile_fmpy():
    """FM pipeline — normal + stepped calcrules (sequential to avoid cache races)."""
    from oasislmf.computation.run.exposure import RunExposure

    for subdir, perils in [
        ("fmpy/Q1_1", ["WTC"]),
        ("fmpy/fm54", ["WTC"]),
    ]:
        src = _DATA_DIR / subdir
        if not src.exists():
            continue
        with TemporaryDirectory() as tmpdir:
            RunExposure(
                src_dir=str(src),
                run_dir=tmpdir,
                loss_factor=[1.0],
                output_level='port',
                output_file=str(Path(tmpdir) / "loc_summary.csv"),
                fmpy_sort_output=True,
                kernel_alloc_rule_il=2,
                kernel_alloc_rule_ri=2,
                intermediary_csv=True,
                model_perils_covered=perils,
            ).run()


def _compile_modelpy_gulpy_gulmc():
    """modelpy + gulpy + gulmc via subprocess pipelines.

    Runs sequentially to avoid Numba cache races on shared modelpy JIT functions.
    """
    with TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        needed_files = [correlations_rel_path, coverages_rel_path, events_rel_path, items_rel_path,
                        damage_bin_dict_rel_path, footprint_rel_path, footprint_idx_rel_path, vulnerability_rel_path]
        for rel_path in needed_files:
            _copy_rel_path((_DATA_DIR / rel_path), workspace / rel_path)

        # gulpy pipeline
        out_file = Path(tmpdir) / "gulpy_out.bin"
        cmd = (
            f"evepy 1 1 | modelpy | gulpy -a1 -S1 -L0 "
            f"--random-generator=1 > '{out_file}'"
        )
        result = subprocess.run(
            cmd, cwd=str(workspace), shell=True,
            capture_output=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"gulpy pipeline failed (rc={result.returncode}):\n"
                f"stderr: {result.stderr.decode()}"
            )

        # gulmc pipeline (sequential — shares modelpy JIT with gulpy)
        out_file = Path(tmpdir) / "gulmc_out.bin"
        cmd = (
            f"evepy 1 1 | modelpy | gulmc -a1 -S1 -L0 "
            f"--ignore-correlation --random-generator=1 > '{out_file}'"
        )
        result = subprocess.run(
            cmd, cwd=str(workspace), shell=True,
            capture_output=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"gulmc pipeline failed (rc={result.returncode}):\n"
                f"stderr: {result.stderr.decode()}"
            )


def _compile_summarypy():
    """summarypy manager on single_summary_set."""
    from oasislmf.pytools.summary.cli import manager
    with TemporaryDirectory() as tmpdir:
        manager.main(
            create_summarypy_files=False,
            low_memory=True,
            output_zeros=False,
            static_path=_DATA_DIR / "summarypy" / "single_summary_set",
            run_type="gul",
            files_in=[_DATA_DIR / "input" / "gul.bin"],
            summary_sets_output=["-1", str(Path(tmpdir) / 'gul_S1_summary.bin')]
        )


def _compile_eltpy():
    """eltpy manager — event loss table."""
    import numpy as np
    from unittest.mock import patch
    from oasislmf.pytools.common.data import oasis_int, oasis_float
    from oasislmf.pytools.elt.manager import main as elt_main
    with TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "selt.csv"
        with patch(
            'oasislmf.pytools.elt.manager.read_event_rates',
            return_value=(np.array([], dtype=oasis_int), np.array([], dtype=oasis_float))
        ):
            elt_main(
                run_dir=Path(tmpdir),
                files_in=_DATA_DIR / summary_rel_path,
                ext="csv",
                selt=out_file,
            )


def _compile_pltpy():
    """pltpy manager — period loss table (with occurrence for occ JIT)."""
    from oasislmf.pytools.plt.manager import main as plt_main
    with TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "splt.csv"
        plt_main(
            run_dir=_DATA_DIR,
            files_in=_DATA_DIR / summary_rel_path,
            ext="csv",
            splt=out_file,
        )


def _compile_aalpy():
    """aalpy manager — annual aggregate loss."""
    from oasislmf.pytools.aal.manager import main as aal_main
    with TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        for rel_path in [occurrence_rel_path, summary_rel_path]:
            _copy_rel_path((_DATA_DIR / rel_path), workspace / rel_path)
        out_dir = workspace / "out"
        out_dir.mkdir()
        out_file = out_dir / "aal.csv"
        aal_main(
            run_dir=workspace,
            subfolder="gul",
            aal=out_file,
            ext="csv",
            meanonly=False,
        )


def _compile_lecpy():
    """lecpy manager — all 8 report flags for max JIT coverage."""
    from oasislmf.pytools.lec.manager import main as lec_main
    with TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        needed_files = [occurrence_rel_path, periods_rel_path, returnperiods_rel_path, summary_rel_path]
        for rel_path in needed_files:
            _copy_rel_path((_DATA_DIR / rel_path), workspace / rel_path)
        out_dir = workspace / "out"
        out_dir.mkdir()
        ept_file = out_dir / "ept.csv"
        psept_file = out_dir / "psept.csv"
        lec_main(
            run_dir=workspace,
            subfolder="gul",
            use_return_period=True,
            agg_full_uncertainty=True,
            agg_wheatsheaf=True,
            agg_sample_mean=True,
            agg_wheatsheaf_mean=True,
            occ_full_uncertainty=True,
            occ_wheatsheaf=True,
            occ_sample_mean=True,
            occ_wheatsheaf_mean=True,
            ept=ept_file,
            psept=psept_file,
            ext="csv",
        )


def _compile_katpy():
    """katpy manager — sorted mode for nb_heapq JIT."""
    from oasislmf.pytools.kat.manager import main as kat_main
    with TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "katpy_qplt.csv"
        kat_main(
            dir_in=_DATA_DIR / "katpy",
            qplt=True,
            out=out_file,
            unsorted=False,
        )


def _compile_plapy():
    """plapy manager — post-loss amplification."""
    from tempfile import NamedTemporaryFile
    from oasislmf.pytools.pla.manager import run

    with TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        for rel_path in [amplifications_rel_path, lossfactors_rel_path]:
            _copy_rel_path((_DATA_DIR / rel_path), workspace / rel_path)

        with NamedTemporaryFile(prefix='pla', dir=str(tmpdir)) as pla_out:
            run(
                run_dir=str(workspace), file_in=str(_DATA_DIR / "input" / "gul.bin"), file_out=pla_out.name,
                input_path='input', static_path='static',
                secondary_factor=1, uniform_factor=0
            )

# ---------------------------------------------------------------------------
# Silence helper — suppresses all logging and stdout/stderr in worker processes.
# ---------------------------------------------------------------------------

class _silence:
    """Context manager that suppresses all logging and stdout/stderr.

    Safe to use both in worker processes and in the main process —
    restores original state on exit.
    """

    def __enter__(self):
        import logging
        self._prev_disable = logging.root.manager.disable
        self._prev_stdout = sys.stdout
        self._prev_stderr = sys.stderr
        logging.disable(logging.CRITICAL)
        self._devnull = open(os.devnull, 'w')
        sys.stdout = self._devnull
        sys.stderr = self._devnull
        return self

    def __exit__(self, *exc):
        import logging
        sys.stdout = self._prev_stdout
        sys.stderr = self._prev_stderr
        logging.disable(self._prev_disable)
        self._devnull.close()
        return False

def _silence_func(func):
    """silence decorator"""
    @functools.wraps(func)
    def silenced_func(*args, **kwargs):
        with _silence():
            return func(*args, **kwargs)
    return silenced_func



def _make_silent(fn, name):
    """Wrap fn with _silence_func and fix __name__/__qualname__ for pickling."""
    silent = _silence_func(fn)
    silent.__name__ = name
    silent.__qualname__ = name
    return silent


_compile_fmpy_silent = _make_silent(_compile_fmpy, '_compile_fmpy_silent')
_compile_modelpy_gulpy_gulmc_silent = _make_silent(_compile_modelpy_gulpy_gulmc, '_compile_modelpy_gulpy_gulmc_silent')
_compile_lecpy_silent = _make_silent(_compile_lecpy, '_compile_lecpy_silent')
_compile_aalpy_silent = _make_silent(_compile_aalpy, '_compile_aalpy_silent')
_compile_eltpy_silent = _make_silent(_compile_eltpy, '_compile_eltpy_silent')
_compile_pltpy_silent = _make_silent(_compile_pltpy, '_compile_pltpy_silent')
_compile_katpy_silent = _make_silent(_compile_katpy, '_compile_katpy_silent')
_compile_summarypy_silent = _make_silent(_compile_summarypy, '_compile_summarypy_silent')
_compile_plapy_silent = _make_silent(_compile_plapy, '_compile_plapy_silent')


# ---------------------------------------------------------------------------
# Task registry — ordered heaviest-first so the pool starts slow tasks early.
# ---------------------------------------------------------------------------
ALL_SILENT_TASKS = {
    "fmpy": _compile_fmpy_silent,
    "modelpy_gulpy_gulmc": _compile_modelpy_gulpy_gulmc_silent,
    "lecpy": _compile_lecpy_silent,
    "aalpy": _compile_aalpy_silent,
    "eltpy": _compile_eltpy_silent,
    "pltpy": _compile_pltpy_silent,
    "katpy": _compile_katpy_silent,
    "summarypy": _compile_summarypy_silent,
    "plapy": _compile_plapy_silent,
}

def warmup(max_workers=None):
    """Run all JIT compilations in parallel.

    Args:
        max_workers: Max parallel processes. Defaults to cpu_count.

    Returns:
        dict of task_name -> error for any failures (empty on success).
    """
    if not _DATA_DIR.is_dir():
        print(f"  Warmup assets not found at {_DATA_DIR} — skipping.")
        return {}

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(ALL_SILENT_TASKS))

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    errors = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn): name for name, fn in ALL_SILENT_TASKS.items()}
        pbar = tqdm(total=len(futures), desc="warmup", unit="task",
                    bar_format="{desc}: {bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") if has_tqdm else None
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                errors[name] = e
            if pbar:
                pbar.set_postfix_str(name)
                pbar.update(1)
        if pbar:
            pbar.close()
    return errors


def main():
    if os.environ.get("NUMBA_DISABLE_JIT") == "1":
        print("NUMBA_DISABLE_JIT=1 — skipping JIT warmup.")
        return

    print(f"Warming Numba JIT cache ({len(ALL_SILENT_TASKS)} tasks, "
          f"max_workers={os.cpu_count()}) ...")
    errors = warmup()
    if errors:
        print(f"\nFAILED — {len(errors)} task(s):", file=sys.stderr)
        for name, err in errors.items():
            print(f"  {name}:", file=sys.stderr)
            traceback.print_exception(type(err), err, err.__traceback__,
                                      file=sys.stderr)
        sys.exit(1)
    else:
        pkg_root = Path(__file__).resolve().parent
        cache_count = sum(1 for _ in pkg_root.rglob("*.nbi"))
        print(f"Done — {cache_count} Numba cache files written.")


if __name__ == "__main__":
    main()
