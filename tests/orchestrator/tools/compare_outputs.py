"""Compare the ``work/`` and ``output/`` trees of two full-kernel runs.

Two comparisons are provided:

* **bytes** (:func:`compare_trees_bytes`): both runs hold the same regular files under the
  compared sub-directories (``work`` and ``output`` by default) and each pair is identical byte
  for byte. This is the test for two runs with the same number of processes K: every
  ``work/kat/*_P<k>`` file, every ``work/*_summary_*/P<k>.bin`` file and every output.
* **rows** (:func:`compare_outputs_rows`): only ``output/``. Each CSV or parquet table is read
  into a DataFrame and the two must have the same columns and, once sorted, the same rows:
  integer and text columns exactly, float columns within a tolerance. This is the test for two
  runs with a different K, where ELT/PLT rows come in another order and AAL/EPT sums are taken
  in another order. Other files (JSON) are compared byte for byte.

FIFOs, symlinks and directories are not compared, so a reference directory's ``input`` and
``static`` links and its ``fifo/`` directory do not matter. ``DEFAULT_EXCLUDE`` leaves out the
scratch files aalpy and lecpy write next to their inputs (``work/*/aal_files``, ``lec_files``):
they are the tools' own intermediate state, not something the kernel hands from one stage to
the next.

Command line::

    python3 tests/orchestrator/tools/compare_outputs.py bytes RUN_A RUN_B [--subdirs work,output] [--details N]
    python3 tests/orchestrator/tools/compare_outputs.py rows RUN_A RUN_B [--rtol 1e-6] [--atol 0.01] [--details N]

The exit status is 0 when the runs are equal and 1 otherwise.
"""
import argparse
import fnmatch
import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SUBDIRS = ('work', 'output')
DEFAULT_EXCLUDE = ('work/*/aal_files/*', 'work/*/lec_files/*')
TABLE_SUFFIXES = ('.csv', '.parquet')
READ_CHUNK = 1 << 20
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 0.01


@dataclass
class TreeReport:
    """Result of comparing two run trees.

    Attributes:
        compared (int): files present in both runs and compared.
        only_in_a (list[str]): relative paths found only in run A.
        only_in_b (list[str]): relative paths found only in run B.
        different (list[str]): one line per differing file, with what differs.
    """
    compared: int = 0
    only_in_a: list = field(default_factory=list)
    only_in_b: list = field(default_factory=list)
    different: list = field(default_factory=list)

    @property
    def ok(self):
        """True when no file is missing or different."""
        return not (self.only_in_a or self.only_in_b or self.different)

    def summary(self, details=20):
        """Human readable report.

        Args:
            details (int): at most this many paths listed per category.

        Returns:
            str: the report.
        """
        lines = [f'{self.compared} files compared, {len(self.different)} different, '
                 f'{len(self.only_in_a)} only in A, {len(self.only_in_b)} only in B: {"EQUAL" if self.ok else "DIFFERENT"}']
        for title, items in (('only in A', self.only_in_a), ('only in B', self.only_in_b), ('different', self.different)):
            for item in items[:details]:
                lines.append(f'  {title}: {item}')
            if len(items) > details:
                lines.append(f'  ... {len(items) - details} more {title}')
        return '\n'.join(lines)


def list_files(root, subdirs=DEFAULT_SUBDIRS, exclude=DEFAULT_EXCLUDE):
    """Regular files under ``root/<subdir>`` for each subdir, without following symlinks.

    Args:
        root (str | Path): run directory.
        subdirs (iterable[str]): sub-directories to list.
        exclude (iterable[str]): fnmatch patterns of relative paths left out.

    Returns:
        dict: relative POSIX path -> Path.
    """
    root = Path(root)
    found = {}
    for sub in subdirs:
        top = root / sub
        if top.is_symlink() or not top.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            for name in filenames:
                path = Path(dirpath) / name
                rel = path.relative_to(root).as_posix()
                if stat.S_ISREG(path.lstat().st_mode) and not any(fnmatch.fnmatchcase(rel, p) for p in exclude):
                    found[rel] = path
    return found


def first_difference(path_a, path_b):
    """Offset of the first differing byte of two files, None if they are identical.

    Args:
        path_a (Path): first file.
        path_b (Path): second file.

    Returns:
        int | None: the offset (the shorter file's size when one is a prefix of the other).
    """
    offset = 0
    with open(path_a, 'rb') as fa, open(path_b, 'rb') as fb:
        while True:
            a = fa.read(READ_CHUNK)
            b = fb.read(READ_CHUNK)
            if a != b:
                n = min(len(a), len(b))
                for i in range(n):
                    if a[i] != b[i]:
                        return offset + i
                return offset + n
            if not a:
                return None
            offset += len(a)


def _missing(files_a, files_b, report):
    report.only_in_a = sorted(set(files_a) - set(files_b))
    report.only_in_b = sorted(set(files_b) - set(files_a))
    return sorted(set(files_a) & set(files_b))


def compare_trees_bytes(run_a, run_b, subdirs=DEFAULT_SUBDIRS, exclude=DEFAULT_EXCLUDE):
    """Compare every regular file under the sub-directories of two runs byte for byte.

    Args:
        run_a (str | Path): first run directory.
        run_b (str | Path): second run directory.
        subdirs (iterable[str]): sub-directories compared.
        exclude (iterable[str]): fnmatch patterns of relative paths left out.

    Returns:
        TreeReport: the result.
    """
    files_a, files_b = list_files(run_a, subdirs, exclude), list_files(run_b, subdirs, exclude)
    report = TreeReport()
    for rel in _missing(files_a, files_b, report):
        report.compared += 1
        size_a, size_b = files_a[rel].stat().st_size, files_b[rel].stat().st_size
        offset = first_difference(files_a[rel], files_b[rel])
        if offset is not None:
            report.different.append(f'{rel}: first difference at byte {offset} (sizes {size_a} and {size_b})')
    return report


def read_table(path):
    """Read an output table, CSV or parquet.

    Args:
        path (Path): the file.

    Returns:
        pd.DataFrame: the table.
    """
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def sort_rows(df):
    """Rows sorted by the non-float columns, then the float columns, in column order.

    Args:
        df (pd.DataFrame): a table.

    Returns:
        pd.DataFrame: sorted, with a fresh index.
    """
    keys = [c for c in df.columns if not pd.api.types.is_float_dtype(df[c])]
    floats = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    if not keys + floats:
        return df.reset_index(drop=True)
    return df.sort_values(keys + floats, kind='stable').reset_index(drop=True)


def compare_tables(df_a, df_b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """Compare two tables as sorted rows.

    Args:
        df_a (pd.DataFrame): first table.
        df_b (pd.DataFrame): second table.
        rtol (float): relative tolerance of float columns.
        atol (float): absolute tolerance of float columns.

    Returns:
        str | None: what differs, None if equal.
    """
    if list(df_a.columns) != list(df_b.columns):
        return f'columns {list(df_a.columns)} != {list(df_b.columns)}'
    if len(df_a) != len(df_b):
        return f'{len(df_a)} rows != {len(df_b)} rows'
    df_a, df_b = sort_rows(df_a), sort_rows(df_b)
    for column in df_a.columns:
        a, b = df_a[column], df_b[column]
        if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
            bad = ~np.isclose(a.to_numpy(), b.to_numpy(), rtol=rtol, atol=atol, equal_nan=True)
        else:
            bad = (a.astype(str) != b.astype(str)).to_numpy()
        if bad.any():
            row = int(np.flatnonzero(bad)[0])
            return f'{int(bad.sum())} sorted rows differ in column {column!r}, first at row {row}: {a.iloc[row]!r} != {b.iloc[row]!r}'
    return None


def compare_outputs_rows(run_a, run_b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """Compare the ``output/`` trees of two runs as sorted rows, with a float tolerance.

    Args:
        run_a (str | Path): first run directory.
        run_b (str | Path): second run directory.
        rtol (float): relative tolerance of float columns.
        atol (float): absolute tolerance of float columns.

    Returns:
        TreeReport: the result.
    """
    files_a, files_b = list_files(run_a, ('output',)), list_files(run_b, ('output',))
    report = TreeReport()
    for rel in _missing(files_a, files_b, report):
        report.compared += 1
        path_a, path_b = files_a[rel], files_b[rel]
        if path_a.suffix in TABLE_SUFFIXES:
            if path_a.stat().st_size == 0 or path_b.stat().st_size == 0:
                problem = None if path_a.stat().st_size == path_b.stat().st_size else 'one file is empty'
            else:
                problem = compare_tables(read_table(path_a), read_table(path_b), rtol, atol)
        else:
            offset = first_difference(path_a, path_b)
            problem = None if offset is None else f'first difference at byte {offset}'
        if problem:
            report.different.append(f'{rel}: {problem}')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    by_bytes = sub.add_parser('bytes', help='every file under the sub-directories, byte for byte')
    by_bytes.add_argument('--subdirs', default=','.join(DEFAULT_SUBDIRS), help='comma separated sub-directories')
    by_rows = sub.add_parser('rows', help='output/ tables as sorted rows with a float tolerance')
    by_rows.add_argument('--rtol', type=float, default=DEFAULT_RTOL)
    by_rows.add_argument('--atol', type=float, default=DEFAULT_ATOL)
    for p in (by_bytes, by_rows):
        p.add_argument('run_a')
        p.add_argument('run_b')
        p.add_argument('--details', type=int, default=20, help='paths listed per category')
    args = parser.parse_args(argv)
    if args.mode == 'bytes':
        report = compare_trees_bytes(args.run_a, args.run_b, tuple(s for s in args.subdirs.split(',') if s))
    else:
        report = compare_outputs_rows(args.run_a, args.run_b, args.rtol, args.atol)
    print(report.summary(args.details))
    return 0 if report.ok else 1


if __name__ == '__main__':
    sys.exit(main())
