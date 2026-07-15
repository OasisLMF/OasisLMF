# Coverage dependency (gulmc)

```{note}
Documents the coverage-dependency feature in the ground-up Monte Carlo engine
(`oasislmf/pytools/gulmc`). Co-located with the code so behaviour and docs change
together. First-draft explanation; cross-links will be tightened in the full docs
rework.
```

## What it does

Coverage dependency lets a **dependent** coverage have its sampled damage *driven
by* a **source** coverage at the same location and areaperil. The source's sampled
damage conditions the dependent's loss distribution — for example, **contents**
damage conditioned on how severely the **building** was damaged.

This produces realistic correlated damage between coverages, beyond what damage- or
hazard-correlation alone give you (those correlate the *sampling*; coverage
dependency makes one coverage's outcome an *input* to another's).

## The conditional mechanism

The dependent coverage's vulnerability is authored as a **damage-transition
matrix** — `P(dependent_damage_bin | source_damage_bin)` — rather than the usual
hazard-intensity-indexed matrix. During computation:

1. The **source** coverage samples its damage normally (hazard intensity → damage
   bin via its standard vulnerability).
2. For the **dependent**, the source's per-sample damage bin plays the role its
   hazard bin would normally play: it indexes the conditional matrix directly.
3. The dependent's damage is therefore drawn from
   `P(dependent bin | the source's sampled bin)`.

A source damage bin may be left undefined in the matrix (an all-zero column), which
is sampled as "this source damage → no dependent damage" — a valid modelling choice.

```{note}
An earlier **percentile** dependency mode was removed; the single remaining
behaviour is this **conditional** (damage-bin → damage-bin) mechanism.
```

## How to enable it

Three pieces of configuration:

### 1. Model settings — declare the coverage-type pairing

In `model_settings.json`, list which coverage types depend on which:

```json
{
  "model_settings": {
    "coverage_dependency_settings": [
      {"source_coverage_type": 1, "dependent_coverage_type": 3}
    ]
  }
}
```

(Here coverage type 1 = building drives type 3 = contents.)

### 2. Correlations input — the per-item source link

The correlations input gains a **`source_coverage_id`** column. For each item it
carries the `coverage_id` of its source coverage, or `0` if independent. This is
populated automatically during GUL input preparation from
`coverage_dependency_settings` + the keys, so you normally don't hand-edit it.

### 3. Conditional vulnerability — the damage-transition matrix

A new optional static file, `conditional_vulnerability.csv` (or `.bin`), holds the
transition matrices for dependent vulnerabilities:

```text
vulnerability_id, source_damage_bin, damage_bin, probability
```

- `source_damage_bin` — the source's damage bin (1..num_damage_bins)
- `damage_bin` — the dependent's damage bin
- `probability` — `P(dependent damage bin | source damage bin)`

No new CLI flags are needed — the feature activates when this configuration and data
are present.

## Rules the data must satisfy

- A **dependent** coverage **must** use a conditional vulnerability; an
  **independent** coverage **must not**.
- **Per-location activation:** a dependency is active only when the keys return both
  the source and dependent coverages at the **same areaperil** at that location, with
  one-to-one aligned items. If the areaperils differ, or the source/dependent items
  don't align, the dependent is **demoted to independent** (logged with a warning).
- **Zero-TIV sources still drive dependents:** an uninsured (zero-TIV) source is
  retained purely to drive its dependent; its own damage is physical and independent
  of insurance.

## How it works (compute order)

Source→dependent links form a **directed acyclic forest** (a source can drive many
dependents; chains are allowed; cycles are rejected at build time). The engine walks
the forest **depth-first**, computing each root and its entire dependent subtree
before the next root, so a source's per-sample result is always available when its
dependents are computed. Aggregate locations split by disaggregation are handled
transparently — a building's contents links to *that building's* structure.

## Related

- {doc}`correlation` — damage/hazard correlation composes with coverage dependency
  (they are orthogonal: correlation seeds the sampling, dependency conditions one
  coverage on another).
- {doc}`disaggregation` — how aggregate locations split into buildings, which is
  transparent to the dependency logic.
- {doc}`../reference/index` — the generated API for `oasislmf.pytools.gulmc`
  (`build_coverage_dependency_forest`, `get_conditional_vulns`,
  `validate_coverage_dependency`, and the `compute_event_losses` loop).
```
