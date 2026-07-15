# Explanation

Understanding-oriented material: the *why* and *how it works* behind OasisLMF.

These pages are **co-located with the code they describe** so that a change in
behaviour and its explanation travel together in the same pull request. This is
the core of the documentation strategy — see [`DOCS_STRATEGY.md`](https://github.com/OasisLMF/OasisLMF/blob/main/DOCS_STRATEGY.md).

## Financial Module

The conceptual model and the Python/NumPy implementation of the Financial Module
(`oasislmf/pytools/fm`):

```{toctree}
:maxdepth: 1
:caption: Financial Module

financial-module
fm-architecture
```

## Ground-up loss modelling

How ground-up losses are simulated by the Monte-Carlo engine
(`oasislmf/pytools/gulmc`): the modelling methodology, sampling, correlation,
disaggregation, and coverage dependency.

```{toctree}
:maxdepth: 1
:caption: Ground-up loss

modelling-methodology
sampling-methodology
correlation
disaggregation
coverage-dependency
```

## Adjustments & methodology

Adjustments applied around the core loss calculation — vulnerability and damage
adjustments, post-loss amplification, pre-analysis exposure adjustments, and
currency conversion:

```{toctree}
:maxdepth: 1
:caption: Adjustments & methodology

vulnerability-adjustments
absolute-damage
post-loss-amplification
pre-analysis-adjustments
currency_conversion
```

## Models

What an Oasis model is made of — its components and structure:

```{toctree}
:maxdepth: 1
:caption: Models

Oasis-models
```

## Keys & lookup

How exposure is matched to model data (areaperil and vulnerability IDs) via the
keys / lookup service (`oasislmf/lookup`):

```{toctree}
:maxdepth: 1
:caption: Keys & lookup

keys-service
```

The corresponding **API reference** for these subsystems is generated directly from
the source in {doc}`../reference/index`, so explanation and reference sit side by
side.
