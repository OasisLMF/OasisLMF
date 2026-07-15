# Explanation

Understanding-oriented material: the *why* and *how it works* behind OasisLMF.

These pages are **co-located with the code they describe** so that a change in
behaviour and its explanation travel together in the same pull request. This is
the core of the documentation strategy — see [`DOCS_STRATEGY.md`](https://github.com/OasisLMF/OasisLMF/blob/main/DOCS_STRATEGY.md).

```{toctree}
:maxdepth: 2

financial-module
fm-architecture
```

## Financial Module

The Financial Module is the highest-value, highest-drift subsystem, and is used
here as the worked example for the documentation restructure:

- **{doc}`financial-module`** — the conceptual model: supported OED terms,
  profiles, programme hierarchy, back-allocation, reinsurance. *(Migrated from
  the separate GenerateDocs repository, where it was divorced from the code.)*
- **{doc}`fm-architecture`** — how the Python/NumPy implementation is
  structured. *(Wired in from `oasislmf/pytools/fm_architecture.md`, which was
  previously orphaned.)*

The corresponding **API reference** for the Financial Module is generated
directly from the source in {doc}`../reference/index`, so explanation and
reference for the same subsystem sit side by side.
