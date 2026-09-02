# Reference

Information-oriented material: dry, look-it-up facts. Most of this section is
**generated from the source**, so it stays in step with the code.

## Configuration & CLI

```{toctree}
:maxdepth: 1

../options_config_file
../environment-variables
```

## Model data & package

File formats for model and Oasis input/output data, and the `oasislmf` Python
package reference:

```{toctree}
:maxdepth: 1

OasisLMF-package
Oasis-model-data-formats
Oasis-file-formats
```

## Python API

The API reference is generated directly from the source docstrings with
[`sphinx-autoapi`](https://sphinx-autoapi.readthedocs.io/) (AST-based, so it
never imports the package).

```{note}
The API reference is **scoped per-subsystem** rather than covering the whole
package, so each subsystem sits beside the explanation pages that describe it and the
build stays fast. Currently included: the Financial Module (`oasislmf.pytools.fm`), the
ground-up Monte-Carlo engine (`oasislmf.pytools.gulmc`), the shared stream/data layer
(`oasislmf.pytools.common`), the output modules (`elt`, `lec`, `plt`, `aal`, `summary`,
`pla`) and the keys/lookup framework (`oasislmf.lookup`). Coverage widens as more
subsystems are documented.
```

```{toctree}
:maxdepth: 2

api/oasislmf/pytools/fm/index
api/oasislmf/pytools/gulmc/index
api/oasislmf/pytools/common/index
api/oasislmf/pytools/elt/index
api/oasislmf/pytools/lec/index
api/oasislmf/pytools/plt/index
api/oasislmf/pytools/aal/index
api/oasislmf/pytools/summary/index
api/oasislmf/pytools/pla/index
api/oasislmf/lookup/index
```

## Outputs & results

The Oasis loss outputs (standard and ORD result tables), their formats and
file-naming conventions.

```{toctree}
:maxdepth: 1

outputs/index
```

## Calculation kernel

Component, stream-format, converter, validation and calc-rule reference for the
Oasis calculation kernel, migrated from ktools and being updated to the pytools
implementation.

```{toctree}
:maxdepth: 1

kernel/index
```
