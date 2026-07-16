# Output components

The output components turn the summary stream (from `summarypy`) into **ORD** result
tables. They are implemented in pytools and are ORD-native — each tool writes the
relevant ORD tables directly (CSV, binary, or parquet via `-E`/`-f`).

| Result | pytools tool | ORD tables produced | Replaces (deprecated ktools) |
|--------|--------------|---------------------|------------------------------|
| Event loss table | `eltpy` | SELT, MELT, QELT | `eltcalc` |
| Period loss table | `pltpy` | SPLT, MPLT, QPLT | `pltcalc` |
| Loss exceedance (EP) | `lecpy` | EPT, PSEPT | `leccalc` / `ordleccalc` |
| Average annual loss | `aalpy` | ALT (AAL), ALCT | `aalcalc` |
| Concatenate partitions | `katpy` | — | `kat` / `katparquet` |

The exact column layout of each ORD table is defined by the ORD standard (the
`ODS_OpenResultsData` repository); see also the worked EP-curve analysis in the example
notebooks. This page documents what each tool does and how it is run.

## eltpy

`eltpy` produces **event loss tables** from a summary stream: the **SELT** (sample ELT),
**MELT** (moment ELT — mean and standard deviation by event/summary) and **QELT**
(quantile ELT).

**Parameters** — `--run_dir`; `-i, --files_in`; `-s SELT`, `-m MELT`, `-q QELT` (output
files for each table); `-E {csv,bin,parquet}`; `-H` (no header).

**Usage**

```bash
summarypy -t gul -1 - < gul.bin | eltpy -s gul_selt.csv -m gul_melt.csv
eltpy -i gul_summary.bin -s gul_selt.csv
```

## pltpy

`pltpy` produces **period loss tables** — **SPLT** (sample PLT), **MPLT** (moment PLT)
and **QPLT** (quantile PLT) — with event occurrence dates from the occurrence file.

**Parameters** — `--run_dir`; `-i, --files_in`; `-s SPLT`, `-m MPLT`, `-q QPLT`;
`-E {csv,bin,parquet}`; `-H`.

**Internal data** — `input/occurrence.bin` (to assign events to periods and dates).

**Usage**

```bash
summarypy -t il -1 - < il.bin | pltpy -s il_splt.csv -m il_mplt.csv
```

## lecpy

`lecpy` computes **loss exceedance (EP) curves** — the ORD **EPT** (Exceedance
Probability Table) and **PSEPT** (Per-Sample EPT). Losses are assigned to periods
(typically years) via the occurrence file, then rank-ordered by period; the relative
frequency of ranked period losses is the exceedance probability, expressed as a return
period. Only non-zero loss periods are returned.

Losses within a period are combined per curve basis:

- **Aggregate (AEP)** — sum of event losses in the period.
- **Occurrence (OEP)** — maximum event loss in the period.

and the EP curve is computed by one of four methods (each available as an *aggregate*
and an *occurrence* variant):

- **Full uncertainty** — all sampled period losses rank-ordered into a single curve.
- **Wheatsheaf / per-sample** — period losses rank-ordered *per sample*, giving one
  curve per sample (shows the variation due to damage uncertainty).
- **Sample mean** — losses averaged across samples per period, then a single curve.
- **Wheatsheaf mean** — the per-sample curves averaged at each return period.

The analytical mean loss (`sidx = -1`) is always output as its own curve; with zero
samples only that curve is produced.

**Parameters**

- `-K, --subfolder` — the `work/` sub-directory holding the summary binaries.
- `-O, --ept` / `-o, --psept` — output files for EPT / PSEPT.
- Aggregate methods: `-F` full-uncertainty, `-W` wheatsheaf, `-S` sample-mean,
  `-M` wheatsheaf-mean.
- Occurrence methods: `-f`, `-w`, `-s`, `-m` (same order).
- `-r` — use a return-period file (`input/returnperiods.bin`) if present.

**Internal data** — `input/occurrence.bin` (required); optionally
`input/returnperiods.bin` and `input/periods.bin` (period weighting). `lecpy` does not
read a stream: it reads all summary binaries for a summary set from `work/<subfolder>/`
(the full event set must be written there first, since EP curves are not valid on an
event subset).

**Usage**

```bash
# summary binaries for the set are first written to work/summary1/
lecpy -K summary1 -O ept.csv -F -f          # full-uncertainty AEP + OEP
lecpy -K summary1 -O ept.csv -o psept.csv -W -w   # wheatsheaf EPT + PSEPT
```

**Period weightings** — if `input/periods.bin` is present, per-period weights vary each
period's reoccurrence rate (neutral weight = 1 ÷ number of periods); zero-weight periods
contribute nothing. All periods `1..P` must appear (no gaps); the sum is unconstrained.

## aalpy

`aalpy` computes the **average annual loss** table (**ALT**/AAL) and, optionally, the
**Average Loss Convergence Table** (**ALCT**), which estimates the simulation error in
the sample AAL. Analytical (type 1) and sample (type 2) statistics are produced; with
zero samples only type 1 is returned.

**Parameters** — `-K, --subfolder` (the `work/` sub-directory); `-a, --aal` (AAL output);
`-c, --alct` (ALCT output); `-M, --meanonly`; `-l, --confidence` (ALCT confidence level);
`-E {csv,bin,parquet}`.

**Internal data** — `input/occurrence.bin` (required); reads summary binaries from
`work/<subfolder>/` (like `lecpy`, not a stream). Optionally uses `input/periods.bin`
for period weightings.

**Usage**

```bash
aalpy -K summary1 -a aal.csv                       # AAL
aalpy -K summary1 -a aal.csv -c alct.csv -l 0.95   # + ALCT at 95% confidence
```

**Calculation** — event losses are assigned to periods and summed by period and sample
("annual loss samples"). The **AAL** is the mean of annual losses over periods (type 1
from the numerically-integrated means, type 2 across all period×sample annual losses),
with the standard deviation from the squared errors about the mean over the degrees of
freedom.

The **ALCT** partitions the variance of the AAL estimate into hazard and vulnerability
components with a one-factor ANOVA on the annual loss samples:

```text
L(i,m) = AAL + h(i) + v(i,m)      # year i, sample m
Var(L) = Var(h) + Var(v)
Var(AAL estimate) = Var(v) / (I · M)   # events fixed across years → error is vulnerability-only
```

It reports the mean/SD, confidence interval, standard/relative error and the hazard vs
vulnerability variance contributions, and repeats the statistics over increasing sample
subsets (1, 2, 4, …) to show convergence of the AAL estimate with sample size.

## katpy

`katpy` concatenates the per-partition output files (produced when the run is split
across processes) into a single result file — CSV, parquet or binary.

**Parameters** — `-o, --out` (output file); `-f, --file_type {csv,parquet,bin}`;
`-i, --files_in` (explicit files) or `-d, --dir_in` (a directory); one flag per ORD
table type (`-s` SELT, `-m` MELT, …) to select what to concatenate.

**Usage**

```bash
katpy -s -d work/kat/gul_S1_elt_sample -o gul_S1_selt.csv          # concat SELT partitions (csv)
katpy -m -f parquet -i mplt_P1.parquet mplt_P2.parquet -o MPLT.parquet
```

---

See also: {doc}`CoreComponents` · {doc}`ORDOutputComponents` (the ORD tables in detail)
· {doc}`DataConversionComponents`.
