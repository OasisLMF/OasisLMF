# Core components

The core kernel components run the calculation as a stream: events → ground-up loss
→ insured loss → summary. They are implemented in **pytools** (the `oasislmf.pytools`
package); the binary stream formats between stages are unchanged from the original
ktools design.

| Stage | pytools tool | Replaces (deprecated ktools binary) |
|-------|--------------|-------------------------------------|
| Event partitioning | `evepy` | `eve` |
| Effective-damageability CDFs | `modelpy` | `getmodel` |
| Ground-up loss | `gulpy` / `gulmc` | `gulcalc` |
| Financial module (insured loss) | `fmpy` | `fmcalc` |
| Summary aggregation | `summarypy` | `summarycalc` |

`gulpy` is the standard ground-up engine (it consumes `modelpy`'s effective-damage
CDF stream). `gulmc` is the full Monte-Carlo engine: it reads the model data directly
(doing the `modelpy` step internally) and is the **default** in generated runs.

## evepy

`evepy` reads a list of event ids and emits a partition of them as a binary stream.
Events are "shuffled" — assigned to processes cyclically rather than in contiguous
blocks — so the workload is evened out when large events are clustered in the id range.

**Output stream** — a simple list of `event_id`s (4-byte integers).

**Parameters**

- `process_number` `total_processes` (positional, required) — this process's partition
  and the total number of partitions.
- `-i, --input_file` — input events file (default `input/events.bin`).
- `-o, --output_file` — output file (default stdout).
- `-n, --no_shuffle` — keep input ordering (distribute in blocks).
- `-r, --randomise` — randomise with a Fisher-Yates shuffle.

**Usage**

```bash
evepy <p> <N> -o events.bin
evepy <p> <N> | modelpy | gulpy -S100 -a1
```

**Example**

```bash
evepy 1 2 -o events1_2.bin           # partition 1 of 2, shuffled
evepy 1 2 -n -o events1_2.bin        # unshuffled
evepy 1 1 | gulmc -S100 -a0          # full Monte-Carlo pipeline
```

**Internal data** — `input/events.bin` (a list of 4-byte event ids).

## modelpy

`modelpy` (the `getmodel` step) generates a stream of **effective damageability**
distributions (CDFs). It combines the model's `footprint` (hazard intensity
distributions) and `vulnerability` (conditional damage distributions) for the exposures
in `items`, convolving them into an effective damage CDF per areaperil/vulnerability.

**Output stream** — a CDF stream (stream type `0/1`).

**Parameters**

- `-i, --file-in` / `-o, --file-out` — input event stream / output CDF stream.
- `-r, --run-dir` — run directory (default `.`).
- `--peril-filter` — restrict to specific perils.
- `--data-server` — share model data over TCP sockets (for multi-process runs).

**Usage**

```bash
evepy 1 1 | modelpy | gulpy -S100 -a1 -o gul.bin
modelpy --run-dir . -i events.bin -o cdf.bin
```

**Internal data** (relative to the run directory)

- `static/footprint.bin`, `static/footprint.idx`
- `static/vulnerability.bin`
- `static/damage_bin_dict.bin`
- `input/items.bin`

**Calculation** — `modelpy` filters the footprint for areaperils and the vulnerability
for vulnerability ids that appear in `items`, convolves the intensity and conditional-
damage distributions per event/areaperil/vulnerability, and outputs the resulting
cumulative distributions (with the damage-bin mean used for interpolation downstream).

## gulpy / gulmc

Both compute **ground-up loss** by Monte-Carlo sampling; they assign the special
statistics below to negative sample indices.

- **`gulpy`** samples from the effective-damage CDF stream produced by `modelpy`
  (the classic `getmodel → gulcalc` split).
- **`gulmc`** is the full Monte-Carlo engine: it reads the model data directly, samples
  the hazard intensity and then the damage (so it does not need a separate `modelpy`
  step), and supports coverage dependency and separate hazard/damage correlation. It is
  the default engine in generated runs.

**Output stream** — a loss stream (stream type `2/1`).

**Parameters** (common)

- `-S SAMPLE_SIZE` — number of samples.
- `-a ALLOC_RULE` — back-allocation rule (see below; default `0`).
- `-L LOSS_THRESHOLD` — drop losses below the threshold (default `1e-6`).
- `-i, --file-in` / `-o, --file-out` — input / output.
- `--run-dir` — run directory (default `.`).
- `--random-generator` — `0` Mersenne-Twister, `1` Latin Hypercube, `2` Latin
  Hypercube on Philox4x32-7 (**default `2`**). See {doc}`RandomNumbers`.
- `--ignore-correlation` (and `--ignore-haz-correlation` for `gulmc`) — ignore the
  peril correlation groups.
- `gulmc` also: `--effective-damageability` (draw from the effective-damage
  distribution instead of full MC).

**Usage**

```bash
# full Monte-Carlo (default engine)
evepy 1 1 | gulmc -S100 -a1 | fmpy -a2 > il.bin

# standard engine via modelpy CDFs
evepy 1 1 | modelpy | gulpy -S100 -a1 -o gul.bin
```

**Internal data** — `static/damage_bin_dict.bin`, `input/items.bin`,
`input/coverages.bin` (plus the model data read via `modelpy`/directly).

**Random sampling** — for each item CDF and each sample, a uniform random number is
drawn and used to sample a damage factor by interpolation (linear, quadratic, or point-
value depending on the damage-bin definitions), which is multiplied by the item TIV.
Random numbers are reproducible; the generator is selected with `--random-generator`
(see {doc}`RandomNumbers`), replacing the ktools `-R`/`-r`/`-s` flags.

**Special samples** — negative sample indices carry statistics rather than samples:

| sidx | description |
|------|-------------|
| -1 | numerical integration mean |
| -2 | numerical integration standard deviation |
| -3 | impacted exposure |
| -4 | chance of loss |
| -5 | maximum loss |

**Allocation rule** (`-a`) — how item losses are adjusted when a coverage is hit by
multiple perils (total loss to a coverage cannot exceed its TIV):

| `-a` | description |
|------|-------------|
| 0 | pass losses through unadjusted (single-peril models) |
| 1 | sum losses, cap to TIV, back-allocate to items in proportion to unadjusted losses |
| 2 | keep the maximum sub-peril loss, others zero; back-allocate equally on ties |

## fmpy

`fmpy` is the Oasis **Financial Module**: it applies policy terms and conditions to the
ground-up losses, producing insured-loss samples. It reads a loss stream from `gulpy`/
`gulmc` (or from another `fmpy`) and can be chained to apply successive sets of terms
(e.g. direct insurance then reinsurance).

**Output stream** — a loss stream (stream type `2/1`).

**Parameters**

- `-a, --allocation-rule` — back-allocation rule: `0` none, `1` ground-up basis,
  `2` prior-level basis (default `0`).
- `-n, --net-loss` — output net losses (input minus calculated) instead of gross.
- `-p, --static-path` — location of the FM input files (default `input/`).
- `-i, --files-in` / `-o, --files-out`.
- `--create-financial-structure-files` — pre-build the shared FM structure.

**Usage**

```bash
evepy 1 1 | gulmc -S100 -a1 | fmpy -a2 | summarypy -t il -1 il_summary.bin
fmpy -p ri1 -a2 -n -i gul.bin -o ri1_net.bin        # reinsurance, net losses
```

**Internal data** — `input/items.bin`, `input/coverages.bin`, `input/fm_programme.bin`,
`input/fm_policytc.bin`, `input/fm_profile.bin` (or `fm_profile_step.bin`),
`input/fm_xref.bin`. For a loss-stream input only the four `fm_*` files are needed. Use
`-p` to point at a different set (e.g. `-p ri1`).

**Calculation** — `fmpy` passes the loss samples (including the mean, sidx -1, and
impacted exposure, sidx -3) through the financial calculation defined by the input
files; special samples -2, -4, -5 are dropped. See
{doc}`../../explanation/financial-module`.

## summarypy

`summarypy` aggregates loss samples to a reporting **summary level** — reducing stream
volume, unifying the `gulpy`/`gulmc` and `fmpy` stream shapes for downstream outputs,
and producing one or more summary sets in a single pass.

**Output stream** — a summary stream (stream type `3/1`).

**Parameters**

- `-t, --run-type {gul,il,ri}` — the input stream type (replaces the ktools `-i`/`-f`
  distinction).
- `-i, --files-in` — input stream.
- `-p, --static-path` — location of the summary-xref files.
- `-m, --low-memory` — reduce downstream memory with index files.

**Usage**

```bash
evepy 1 1 | gulmc -S100 -a1 | summarypy -t gul -1 gul_summary.bin
fmpy -a2 -i gul.bin | summarypy -t il -1 il_summary.bin
```

**Internal data** — `input/gulsummaryxref.bin` (for `-t gul`) or `input/fmsummaryxref.bin`
(for `-t il`/`-t ri`), which map the input identifier to a user-defined `summary_id`.

**Calculation** — losses are summed to each `summary_id`. The mean (sidx -1), impacted
exposure (sidx -3) and maximum loss (sidx -5) are summed as normal; the standard
deviation (sidx -2) is dropped; the chance of loss (sidx -4, gul input only) is combined
by the law of total probability, `1 − Π(1 − Cᵢ)` over the items in each summary.

---

See also: {doc}`OutputComponents` · {doc}`Specification` (stream formats). A worked,
pytools-correct pipeline walkthrough lives with the example models (OasisModels) and is
linked from the aggregated Oasis documentation.
