# TEST MODEL 8

This is a small model useful for unit testing of **coverage dependency**, where a dependent
coverage's damage is driven by its source coverage's per-sample sampled damage bin through a
conditional (damage-transition) vulnerability, rather than by the footprint hazard.

The model has:
 - 4 events
 - 2 areaperil_ids (154, 54)
 - 4 coverages, 2 items each (8 items)
 - 2 perils
 - correlation (damage and hazard correlation groups)

## Purpose

It provides end-to-end regression cover for the coverage-dependency feature across the whole
`test_gulmc` parameter matrix (sample sizes, back-allocation rules, `ignore_correlation`, all
three random generators, and both effective-damageability modes). The targeted behavioural tests
live in `tests/pytools/gulmc/test_coverage_dependency.py`; this model exists so that a change to
the sampling kernel, the DFS ordering or the conditional-CDF assembly shows up as a loss diff.

### Coverage layout

| coverage | tiv | vulnerability | source | role |
|---|---|---|---|---|
| 1 | **0** | 8 / 2 (hazard-indexed) | — | root, **uninsured** retained driver |
| 2 | 790000 | 101 / 102 (conditional A) | 1 | dependent at depth 1, and source of coverage 3 |
| 3 | 160000 | 103 / 104 (conditional B) | 2 | dependent at depth 2 (chain) |
| 4 | 250000 | 8 / 2 (hazard-indexed) | — | independent, computed alongside the chain |

Within each coverage the two items sit at areaperils 154 and 54, so source and dependent share
the same areaperil multiset and pair up item by item.

Together this exercises: a dependency chain two levels deep (so the DFS ordering and the
depth-indexed source stacks are used non-trivially), an insured coverage that is both a dependent
and a source, a zero-TIV source that must drive its dependents while reporting no loss of its
own, and a mixed run where an independent coverage is computed alongside dependent ones.

### Conditional vulnerabilities

`static/conditional_vulnerability.csv` reuses the vulnerability schema, with `intensity_bin_id`
read as the **source damage bin**:

 - **Matrix A** (ids 101, 102, used by coverage 2): source damage bin `k` maps to
   `{k-1: 0.3, k: 0.7}` — the dependent tracks its source, one bin lower 30% of the time. Every
   source damage bin is defined.
 - **Matrix B** (ids 103, 104, used by coverage 3): source damage bin `k` maps to
   `{1: 0.5, k: 0.5}` — half the samples take no damage. Source damage bin 12 is deliberately
   left **undefined**, covering the documented "the source never reaches this bin, so no
   dependent damage" case.

The conditional vulnerability ids are absent from `vulnerability.csv`, as they must be: they are
indexed by damage bin, not by hazard intensity.

## Relation to test_model_1

The footprint, damage bin dictionary, intensity bin dictionary and vulnerability file are copied
from `test_model_1` unchanged, so the dependency configuration is the only variable between the
two models. The modifications are:

 - `input/items.csv`: 8 items over 4 coverages; coverages 2 and 3 use conditional vulnerability ids
 - `input/coverages.csv`: 4 coverages, the first with zero TIV
 - `input/correlations.csv`: `source_coverage_id` set to build the chain 1 -> 2 -> 3
 - `static/conditional_vulnerability.csv`: new file (see above)

## Generating the binary files
The `static/` and `input/` directories contain a specialised `Makefile` each.
By running `make` inside those directories, the binary files are created from the `.csv` files.

### Note on `correlations.bin` and `conditional_vulnerability.bin`
At the time of writing there is no command line tool to convert `correlations.csv` or
`conditional_vulnerability.csv` to binary format, so the `Makefile`s do not produce them.
`correlations.bin` can be created by executing the following Python code within the `input/`
directory:
```py
from oasislmf.pytools.data_layer.conversions.correlations import CorrelationsData

CorrelationsData.from_csv("correlations.csv").to_bin("correlations.bin")
```
`conditional_vulnerability.bin` is a fixed 4-byte `int32` header holding the number of damage
bins, followed by `vulnerability_dtype` records:
```py
import numpy as np, pandas as pd
from oasislmf.pytools.common.data import vulnerability_dtype

df = pd.read_csv("conditional_vulnerability.csv")
recs = np.array(list(df.itertuples(index=False, name=None)), dtype=vulnerability_dtype)
with open("conditional_vulnerability.bin", "wb") as f:
    np.array([12], dtype=np.int32).tofile(f)
    recs.tofile(f)
```
