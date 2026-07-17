# Stream conversion components

The binary calculation streams can be converted to/from CSV for inspection and
debugging. In pytools this is done by two multi-purpose tools (one sub-command per
stream type), replacing the individual ktools utilities:

| ktools utility (deprecated) | pytools |
|-----------------------------|---------|
| `cdftocsv` | `bintocsv cdf` |
| `gultocsv` | `bintocsv gul` |
| `fmtocsv` | `bintocsv fm` |
| `summarycalctocsv` | `bintocsv summarycalc` |
| `gultobin` | `csvtobin gul` |

General form: `bintocsv <type> -i in.bin -o out.csv` and
`csvtobin <type> -i in.csv -o out.bin` (both also read stdin / write stdout). The
stream field layouts below are unchanged.

## bintocsv cdf

Converts the `modelpy` (getmodel) output stream — effective-damageability CDFs — to CSV.

**Usage**

```bash
evepy 1 1 | modelpy | bintocsv cdf -o cdf.csv
bintocsv cdf -i cdf.bin -o cdf.csv
```

**Output fields** — `event_id`, `areaperil_id`, `vulnerability_id`, `bin_index`,
`prob_to` (cumulative probability at the upper damage-bin threshold), `bin_mean`
(conditional mean of the damage bin).

## bintocsv gul

Converts the ground-up loss stream from `gulmc`/`gulpy` (item or coverage stream) to CSV.

**Usage**

```bash
evepy 1 1 | gulmc -S100 -a1 | bintocsv gul -o gul.csv
bintocsv gul -i gul.bin -o gul.csv
```

**Output fields** — item stream: `event_id`, `item_id`, `sidx`, `loss`; coverage
stream: `event_id`, `coverage_id`, `sidx`, `loss`.

## bintocsv fm

Converts the `fmpy` insured-loss stream to CSV.

**Usage**

```bash
evepy 1 1 | gulmc -S100 -a1 | fmpy -a2 | bintocsv fm -o il.csv
bintocsv fm -i il.bin -o il.csv
```

**Output fields** — `event_id`, `output_id`, `sidx`, `loss`.

## bintocsv summarycalc

Converts the `summarypy` summary stream to CSV.

**Usage**

```bash
summarypy -t il -1 - < il.bin | bintocsv summarycalc -o summary.csv
bintocsv summarycalc -i summary.bin -o summary.csv
```

**Output fields** — `event_id`, `summary_id`, `sidx`, `loss`.

## csvtobin gul

Converts a ground-up loss CSV back into the binary loss stream (e.g. to pipe into
`fmpy`). The CSV has `event_id`, `item_id`, `sidx`, `loss`; for each event/item a
`sidx = -1` (mean) and `sidx = -2` (std dev) record must be present (even if zero),
while zero-loss positive samples may be omitted.

**Usage**

```bash
csvtobin gul -i gul.csv -o gul.bin        # sample size / stream-type options as needed
csvtobin gul -i gul.csv | fmpy -a2 -o il.bin
```

---

See also: {doc}`DataConversionComponents` (data-file converters) ·
{doc}`ValidationComponents` · {doc}`Specification` (full stream layouts).
