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

Within each coverage the two items sit at areaperils 154 and 54, and each dependent item names the
source item at its own areaperil (`source_item_id` on the correlations file).

### Damage group ids

The group ids deliberately **mix**, so the expected results cover both cases:

| coverages | group id | effect |
|---|---|---|
| 1 and 2 | shared (`833720067`) | source and dependent draw from the same damage random stream, as the default `damage_group_id_cols` produces (they do not include coverage type). Their draws are rank-coupled, so the realised conditional probabilities differ from those authored in the matrix. |
| 3 | distinct (`335506702`) | coverage 3's draw is independent of its source's, so the matrix it was authored with is reproduced. |
| 4 | distinct (`2030714556`) | independent coverage. |

A model with a single group id everywhere would hide the coupling; one with all-distinct ids would
not represent what the MDK actually generates. Mixing them means a change to either behaviour shows
up as a loss diff. See the "Random draws and the conditional probabilities" section of
`docs/source/explanation/coverage-dependency.rst`.

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
 - `input/correlations.csv`: `source_item_id` set to build the chain 1 -> 2 -> 3, per item (each
   dependent item names the source item at its own areaperil)
 - `input/items.csv`: damage group ids mixed as described above
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
import numpy as np, pandas as pd
from oasislmf.pytools.common.data import correlations_dtype, correlations_headers

df = pd.read_csv("correlations.csv")
rec = np.zeros(len(df), dtype=correlations_dtype)
for col in correlations_headers:
    rec[col] = df[col].to_numpy()
rec.tofile("correlations.bin")
```
`conditional_vulnerability.bin` is produced by the standard converter:
```sh
csvtobin conditionalvulnerability \
    -i conditional_vulnerability.csv -o conditional_vulnerability.bin -d 12
```
`-d` is the maximum damage bin index. The converter validates that each source damage bin's
probabilities sum to 1 (within 1e-6), the same check it applies to `vulnerability.csv`; a source
damage bin left out entirely is allowed and is read by the engine as "that source damage produces
no dependent damage". `bintocsv conditionalvulnerability` converts back.
