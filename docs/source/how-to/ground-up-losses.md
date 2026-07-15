# Compute ground-up losses with gulmc

Task recipes for running ground-up loss (GUL) calculations with the full
Monte-Carlo engine, **gulmc** (`oasislmf.pytools.gulmc`). For the *why* behind
these options see the {doc}`../explanation/index` pages; for the full option list
see {doc}`../options_config_file` and {doc}`../generated_options`.

These options can be passed as CLI flags to `oasislmf model run` **or** set in the
run configuration / analysis settings JSON (same names, with underscores). See
{doc}`../building-and-running-models` for the base run command.

## Select gulmc as the ground-up engine

Turn on the full Monte-Carlo Python engine:

```bash
oasislmf model run --gulmc --number-of-samples 100 -C oasislmf.json
```

Or in the config JSON:

```json
{
  "gulmc": true,
  "number_of_samples": 100
}
```

## Choose the random number generator

`--gul-random-generator` (config `gul_random_generator`) selects the sampler:

| Value | Generator |
|-------|-----------|
| `0` | Mersenne-Twister |
| `1` | Latin Hypercube |
| `2` | Latin Hypercube on Philox4x32-7 **(default)** |

```bash
oasislmf model run --gulmc --gul-random-generator 1 -C oasislmf.json
```

See {doc}`../explanation/sampling-methodology` for what these do.

## Enable / disable correlation

Damage and hazard correlation are driven by the peril **correlation groups** in
the model's `correlations` input — they are active by default when that data is
present. To ignore them for a run, use the gulmc engine flags:

- `--ignore-correlation` — ignore damage correlation groups
- `--ignore-haz-correlation` — ignore hazard correlation groups

See {doc}`../explanation/correlation` for the model-data setup and the difference
between damage and hazard correlation.

## Enable disaggregation

Split aggregate locations into individual buildings before sampling:

```bash
oasislmf model run --gulmc --do-disaggregation -C oasislmf.json
```

See {doc}`../explanation/disaggregation`.

## Enable coverage dependency

Coverage dependency (a dependent coverage's damage conditioned on a source
coverage) activates automatically when its inputs are present — no run flag is
needed. You must provide:

1. `coverage_dependency_settings` in `model_settings.json`,
2. the `source_coverage_id` column in the `correlations` input (populated during
   GUL input generation), and
3. a `conditional_vulnerability` static file.

See {doc}`../explanation/coverage-dependency` for the full configuration and rules.

## Speed up large runs

- **Effective damageability** — draw from the effective damage distribution
  instead of full Monte-Carlo (faster, different sampling semantics):

  ```bash
  oasislmf model run --gulmc --gulmc-effective-damageability -C oasislmf.json
  ```

- **Vulnerability cache** — size (MB) of the in-memory vulnerability-CDF cache
  (`--gulmc-vuln-cache-size`, config `gulmc_vuln_cache_size`, default `200`):

  ```bash
  oasislmf model run --gulmc --gulmc-vuln-cache-size 500 -C oasislmf.json
  ```

## Run gulmc directly in a kernel pipeline

For low-level runs, `gulmc` reads an event stream and writes a GUL stream, like
the other kernel components (see {doc}`../reference/kernel/CoreComponents`):

```bash
eve 1 1 | modelpy | gulmc -S 100 -a 0 --random-generator 2 -i - -o gulmc.bin
```

Key `gulmc` flags: `-S` sample size, `-a` back-allocation rule, `-L` loss
threshold, `--random-generator`, `--effective-damageability`,
`--ignore-correlation` / `--ignore-haz-correlation`, `--vuln-cache-size`,
`--peril-filter`. Run `gulmc --help` for the complete list.
```
