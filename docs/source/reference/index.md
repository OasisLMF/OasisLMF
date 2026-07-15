# Reference

Information-oriented material: dry, look-it-up facts. Most of this section is
**generated from the source**, so it stays in step with the code.

## Configuration & CLI

```{toctree}
:maxdepth: 1

../options_config_file
../environment-variables
```

## Python API

The API reference is generated directly from the source docstrings with
[`sphinx-autoapi`](https://sphinx-autoapi.readthedocs.io/) (AST-based, so it
never imports the package).

```{note}
For this proof-of-concept the API reference is **scoped to the Financial
Module** (`oasislmf.pytools.fm`) so it sits directly beside its
{doc}`../explanation/financial-module` and {doc}`../explanation/fm-architecture`
pages, and so the build is fast (seconds rather than ~4 minutes for the previous
whole-package dump). The plan is to widen this per-subsystem — see
`DOCS_STRATEGY.md`.
```

```{toctree}
:maxdepth: 2

api/oasislmf/pytools/fm/index
```
