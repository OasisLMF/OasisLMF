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
The API reference is **scoped per-subsystem** — currently the Financial Module
(`oasislmf.pytools.fm`) and the ground-up Monte-Carlo engine
(`oasislmf.pytools.gulmc`) — so it sits directly beside the matching explanation
pages and the build stays fast (seconds rather than the ~4 minutes of the previous
whole-package dump). The plan is to widen this per-subsystem as each module is
migrated — see `DOCS_STRATEGY.md` / `MIGRATION_PLAN.md`.
```

```{toctree}
:maxdepth: 2

api/oasislmf/pytools/fm/index
api/oasislmf/pytools/gulmc/index
```

## Calculation kernel

Component, stream-format, converter, validation and calc-rule reference for the
Oasis calculation kernel, migrated from ktools and being updated to the pytools
implementation.

```{toctree}
:maxdepth: 1

kernel/index
```
