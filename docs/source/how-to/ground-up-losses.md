# Compute ground-up losses with gulmc

Task recipes for running ground-up loss (GUL) calculations with the full
Monte-Carlo engine, **gulmc** (`oasislmf.pytools.gulmc`). For the *why* behind
these options see the {doc}`../explanation/index` pages; for the full option list
see {doc}`../options_config_file` and {doc}`../generated_options`.

Most of these are flags on `oasislmf model run`, and can equally be set in the run
configuration JSON passed with `-C` (same name, with underscores). Two things are not:
the sample count lives in the **analysis settings** JSON, and the correlation switches
exist only on the `gulmc` component itself — both noted where they come up below. See
{doc}`building-and-running-models` for the base run command.

## The ground-up engine (gulmc is the default)

`gulmc`, the full Monte-Carlo Python engine, is the **default** — no flag is needed
to select it. The sample count is set in the **analysis settings** file, not on the
command line:

```json
{
  "number_of_samples": 100
}
```

```bash
oasislmf model run --analysis-settings-json analysis_settings.json -C oasislmf.json
```

There is no `--number-of-samples` flag. If the key is absent, the run falls back to
`model_default_samples` from the model settings, and fails if that is missing too.

To opt out and fall back to the CDF-based `gulpy` engine, pass `--gulmc False`
(config `"gulmc": false`).

## Choose the random number generator

`--gul-random-generator` (config `gul_random_generator`) selects the sampler:

| Value | Generator |
|-------|-----------|
| `0` | Mersenne-Twister |
| `1` | Latin Hypercube |
| `2` | Latin Hypercube on Philox4x32-7 **(default)** |

```bash
oasislmf model run --gul-random-generator 1 -C oasislmf.json
```

See {doc}`../explanation/sampling-methodology` for what these do.

## Enable / disable correlation

Damage and hazard correlation are driven by the peril **correlation groups** in
the model's `correlations` input — they are active by default when that data is
present. They can be switched off only when invoking `gulmc` directly, as in
[the kernel pipeline below](#run-gulmc-directly-in-a-kernel-pipeline):

- `--ignore-correlation` — ignore damage correlation groups
- `--ignore-haz-correlation` — ignore hazard correlation groups

Neither is available on `oasislmf model run`, and neither is read from the run
configuration or analysis settings.

See {doc}`../explanation/correlation` for the model-data setup and the difference
between damage and hazard correlation.

## Disaggregation

Disaggregation — splitting aggregate locations into individual buildings before
sampling — is **on by default**. To turn it off:

```bash
oasislmf model run --do-disaggregation False -C oasislmf.json
```

See {doc}`../explanation/disaggregation`.

## Speed up large runs

- **Effective damageability** — draw from the effective damage distribution
  instead of full Monte-Carlo (faster, different sampling semantics):

  ```bash
  oasislmf model run --gulmc-effective-damageability -C oasislmf.json
  ```

- **Vulnerability cache** — size (MB) of the in-memory vulnerability-CDF cache
  (`--gulmc-vuln-cache-size`, config `gulmc_vuln_cache_size`, default `200`):

  ```bash
  oasislmf model run --gulmc-vuln-cache-size 500 -C oasislmf.json
  ```

## Run gulmc directly in a kernel pipeline

For low-level runs, `gulmc` reads an event stream and writes a GUL stream, like
the other kernel components (see {doc}`../reference/kernel/CoreComponents`):

```bash
evepy 1 1 | gulmc -S 100 -a 0 --random-generator 2 -o gulmc.bin
```

Key `gulmc` flags: `-S` sample size, `-a` back-allocation rule, `-L` loss
threshold, `--random-generator`, `--effective-damageability`,
`--ignore-correlation` / `--ignore-haz-correlation`, `--vuln-cache-size`,
`--peril-filter`. Run `gulmc --help` for the complete list.
```
