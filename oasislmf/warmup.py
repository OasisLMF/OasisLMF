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


# ---------------------------------------------------------------------------
# Worker functions — top-level for pickling by ProcessPoolExecutor.
# Each does all imports locally so child processes start clean.
# ---------------------------------------------------------------------------

def _compile_fmpy():
    """FM pipeline — normal + stepped calcrules (sequential to avoid cache races)."""
    with _silence():
        from oasislmf.warmup import _DATA_DIR as dd
        from oasislmf.computation.run.exposure import RunExposure

        for subdir, perils in [
            ("fmpy/Q1_1", ["WTC"]),
            ("fmpy/fm54", ["WTC"]),
        ]:
            src = dd / subdir
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
    with _silence():
        from oasislmf.warmup import _DATA_DIR as dd
        with TemporaryDirectory() as tmpdir:
            # gulpy pipeline
            out_file = Path(tmpdir) / "gulpy_out.bin"
            cmd = (
                f"evepy 1 1 | modelpy | gulpy -a1 -S1 -L0 "
                f"--random-generator=1 > '{out_file}'"
            )
            result = subprocess.run(
                cmd, cwd=str(dd / "model"), shell=True,
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
                cmd, cwd=str(dd / "model"), shell=True,
                capture_output=True, timeout=300
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"gulmc pipeline failed (rc={result.returncode}):\n"
                    f"stderr: {result.stderr.decode()}"
                )


def _compile_summarypy():
    """summarypy manager on single_summary_set."""
    with _silence():
        from oasislmf.warmup import _DATA_DIR as dd
        from oasislmf.pytools.summary.cli import manager
        with TemporaryDirectory() as tmpdir:
            manager.main(
                create_summarypy_files=False,
                low_memory=True,
                output_zeros=False,
                static_path=dd / "summarypy" / "single_summary_set",
                run_type=manager.RUNTYPE_GROUNDUP_LOSS,
                files_in=[dd / "summarypy" / f"{manager.RUNTYPE_GROUNDUP_LOSS}.bin"],
                summary_sets_output=["-1", str(Path(tmpdir) / 'gul_S1_summary.bin')]
            )


def _compile_eltpy():
    """eltpy manager — event loss table."""
    with _silence():
        from oasislmf.warmup import _DATA_DIR as dd
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
                    files_in=dd / "summarypy1.bin",
                    ext="csv",
                    selt=out_file,
                )


def _compile_pltpy():
    """pltpy manager — period loss table (with occurrence for occ JIT)."""
    with _silence():
        from oasislmf.warmup import _DATA_DIR as dd
        from oasislmf.pytools.plt.manager import main as plt_main
        with TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "splt.csv"
            plt_main(
                run_dir=dd / "pltpy",
                files_in=dd / "summarypy1.bin",
                ext="csv",
                splt=out_file,
            )


def _compile_aalpy():
    """aalpy manager — annual aggregate loss."""
    with _silence():
        from oasislmf.warmup import _DATA_DIR as dd
        from oasislmf.pytools.aal.manager import main as aal_main
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            shutil.copytree(dd / "aalpy", workspace)
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
    with _silence():
        from oasislmf.warmup import _DATA_DIR as dd
        from oasislmf.pytools.lec.manager import main as lec_main
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            shutil.copytree(dd / "lecpy", workspace)
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
    with _silence():
        from oasislmf.warmup import _DATA_DIR as dd
        from oasislmf.pytools.kat.manager import main as kat_main
        with TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "katpy_qplt.csv"
            kat_main(
                dir_in=dd / "katpy",
                qplt=True,
                out=out_file,
                unsorted=False,
            )


def _compile_plapy():
    """plapy — post-loss amplification (generates its own test data)."""
    with _silence():
        _compile_plapy_inner()


def _compile_plapy_inner():
    """Inner implementation of plapy compilation (separated for indentation clarity)."""
    from tempfile import NamedTemporaryFile
    import numpy as np
    from oasislmf.pytools.pla.common import (
        DATA_SIZE, event_count_dtype, amp_factor_dtype, PLAFACTORS_FILE
    )
    from oasislmf.pytools.common.input_files import AMPLIFICATIONS_FILE
    from oasislmf.pytools.common.event_stream import (
        stream_info_to_bytes, FM_STREAM_ID, ITEM_STREAM,
        mv_write_item_header, mv_write_sidx_loss
    )
    from oasislmf.pytools.pla.manager import run

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        input_dir.mkdir()
        static_dir = tmpdir / "static"
        static_dir.mkdir()

        n_events = 2
        n_items = 2

        # Write items amplifications file
        itemsamps_file = input_dir / AMPLIFICATIONS_FILE
        n_entries = n_items * DATA_SIZE
        write_buffer = memoryview(bytearray(n_entries * DATA_SIZE))
        item_amp_dtype = np.dtype([('item_id', 'i4'), ('amplification_id', 'i4')])
        event_item = np.ndarray(n_entries, buffer=write_buffer, dtype=item_amp_dtype)
        it = np.nditer(event_item, op_flags=['writeonly'], flags=['c_index'])
        for row in it:
            row[...] = (it.index + 1, it.index + 1)
        with open(itemsamps_file, 'wb') as f:
            f.write(np.int32(0).tobytes())
            f.write(write_buffer[:])

        # Write loss factors file
        lossfactors_file = static_dir / PLAFACTORS_FILE
        factors = np.array([[1.125, 1.25], [1.0, 0.75]])
        n_amplifications = 2
        n_pairs = n_events + sum(len(event) for event in factors)
        write_buffer = memoryview(bytearray(n_pairs * DATA_SIZE))
        event_count = np.ndarray(n_pairs, buffer=write_buffer, dtype=event_count_dtype)
        amp_factor = np.ndarray(n_pairs, buffer=write_buffer, dtype=amp_factor_dtype)
        it_f = np.nditer(factors, op_flags=['readonly'], flags=['multi_index'])
        current_event_id = 0
        cursor = 0
        for entry in it_f:
            if current_event_id != it_f.multi_index[0] + 1:
                event_count[cursor]['event_id'] = it_f.multi_index[0] + 1
                event_count[cursor]['count'] = n_amplifications
                current_event_id = it_f.multi_index[0] + 1
                cursor += 1
            amp_factor[cursor]['amplification_id'] = it_f.multi_index[1] + 1
            amp_factor[cursor]['factor'] = entry
            cursor += 1
        with open(lossfactors_file, 'wb') as f:
            f.write(np.int32(0).tobytes())
            f.write(write_buffer[:])

        # Write input GUL stream and run PLA
        losses = np.array([
            [[100., 50., 0.0], [10., 30., 0.0]],
            [[5., 100., 0.0], [30., 50., 0.0]]
        ])
        n_pairs_gul = n_events * n_items + sum(
            sum(len(sample) for sample in items) for items in losses
        )
        with NamedTemporaryFile(prefix='gul', dir=str(tmpdir)) as gul_in, \
                NamedTemporaryFile(prefix='pla', dir=str(tmpdir)) as pla_out:
            gul_in.write(stream_info_to_bytes(FM_STREAM_ID, ITEM_STREAM))
            gul_in.write(np.int32(2).tobytes())
            write_buffer = memoryview(bytearray(n_pairs_gul * DATA_SIZE))
            write_byte_mv = np.frombuffer(buffer=write_buffer, dtype='b')
            current_event_id = 0
            current_item_id = 0
            max_sidx = 3
            cursor = 0
            it_l = np.nditer(losses, op_flags=['readonly'], flags=['multi_index'])
            for entry in it_l:
                if current_event_id != it_l.multi_index[0] + 1 or current_item_id != it_l.multi_index[1] + 1:
                    current_event_id = it_l.multi_index[0] + 1
                    current_item_id = it_l.multi_index[1] + 1
                    cursor = mv_write_item_header(write_byte_mv, cursor, current_event_id, current_item_id)
                cursor = mv_write_sidx_loss(write_byte_mv, cursor, (it_l.multi_index[2] + 1) % max_sidx, entry)
            gul_in.write(write_buffer[:cursor])
            gul_in.seek(0)

            run(
                run_dir=str(tmpdir), file_in=gul_in.name, file_out=pla_out.name,
                input_path='input', static_path='static',
                secondary_factor=1, uniform_factor=0
            )


# ---------------------------------------------------------------------------
# Task registry — ordered heaviest-first so the pool starts slow tasks early.
# ---------------------------------------------------------------------------

ALL_TASKS = {
    "fmpy": _compile_fmpy,
    "modelpy_gulpy_gulmc": _compile_modelpy_gulpy_gulmc,
    "lecpy": _compile_lecpy,
    "aalpy": _compile_aalpy,
    "eltpy": _compile_eltpy,
    "pltpy": _compile_pltpy,
    "katpy": _compile_katpy,
    "summarypy": _compile_summarypy,
    "plapy": _compile_plapy,
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
        max_workers = min(os.cpu_count() or 4, len(ALL_TASKS))

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    errors = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn): name for name, fn in ALL_TASKS.items()}
        completed = 0
        pbar = tqdm(total=len(futures), desc="warmup", unit="task",
                    bar_format="{desc}: {bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") if has_tqdm else None
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                errors[name] = e
            completed += 1
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

    print(f"Warming Numba JIT cache ({len(ALL_TASKS)} tasks, "
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
