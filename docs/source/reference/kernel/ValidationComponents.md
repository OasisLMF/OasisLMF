# Validation

Oasis validates two kinds of data: the **model static data** (damage bin dictionary,
footprint, vulnerability) and the **exposure / analysis inputs** (OED). There are no
standalone `validate*` binaries in pytools — validation is built into the tools that
read the data.

## Model-data validation

### Inline, during CSV → binary conversion (`csvtobin`)

`csvtobin` validates the model data **by default** as it converts it; pass `-N,
--no_validation` to skip. The checks (implemented in
`oasislmf/pytools/converters/csvtobin/utils/`) are:

- **`csvtobin damagebin`** — the first `bin_index` is 1; bin indices are contiguous; each
  interpolation (mean) damage lies within its bin range. *(Warning if the lower limit of
  the first bin is not 0.)*
- **`csvtobin footprint`** — the probabilities for each `(event_id, areaperil_id)` group
  sum to 1; rows are sorted by `event_id` then `areaperil_id` ascending.
- **`csvtobin vulnerability`** — `vulnerability_id`s are ascending; the probabilities for
  each `(vulnerability_id, intensity_bin_id)` group sum to 1; damage bins are contiguous.

```bash
csvtobin footprint -i footprint.csv -o footprint.bin        # validates by default
csvtobin footprint -N -i footprint.csv -o footprint.bin     # skip validation
```

### Standalone model-data check

`oasislmf.validation.model_data.csv_validity_test(model_data_dir)` runs the per-file
checks plus **cross-checks** across the three files:

- damage-bin ids used in the vulnerability data are a subset of the damage bin
  dictionary;
- intensity-bin ids in the footprint are a subset of those in the vulnerability data.

```{note}
This helper currently shells out to the legacy ktools binaries
(`validatedamagebin` / `validatefootprint` / `validatevulnerability` / `crossvalidation`)
via subprocess — a remaining ktools dependency pending a pytools cutover. The equivalent
per-file checks already run inline in `csvtobin` (above); the cross-file checks are the
piece still provided only by the ktools `crossvalidation`.
```

## Exposure / OED validation

OED exposure files are validated by **`ods_tools`** against the OED standard (required and
conditionally-required fields, valid code lists, peril codes, …). Use it directly
(`OedExposure(..., check_oed=True)` or `OedExposure.check(...)`), via the CLI
(`ods_tools check`), or as part of a model run with the `--check-oed` option
(`check_oed`). See the *load & validate OED* worked example in the ODS Tools docs.

## Analysis input checks

- **`--check-oed`** (`check_oed`) — validate the input OED files during a run.
- **`--check-missing-inputs`** (`check_missing_inputs`) — fail the run if IL/RI output is
  requested without the required generated input files.

## Mapping from the legacy ktools validators

| ktools validator (deprecated) | now |
|-------------------------------|-----|
| `validatedamagebin` | inline in `csvtobin damagebin` (and the legacy `csv_validity_test`) |
| `validatefootprint` | inline in `csvtobin footprint` |
| `validatevulnerability` | inline in `csvtobin vulnerability` |
| `crossvalidation` | `csv_validity_test` cross-checks (still ktools; pytools cutover pending) |
| `validateoasisfiles` | OED validation via `ods_tools` + the `--check-oed` / `--check-missing-inputs` run options |

---

See also: {doc}`DataConversionComponents` (the converters) ·
{doc}`../../explanation/financial-module`.
