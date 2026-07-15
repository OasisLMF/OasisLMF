# Calculation kernel

Reference for the Oasis calculation kernel — the components, binary stream formats,
data converters, validation tools, and calc rules.

```{note}
These pages were **migrated from the ktools repository** (`docs/md/`) as part of the
ktools decommission — the components are reimplemented in `oasislmf/pytools`
(gulpy/gulmc, fmpy, summarypy, the output modules, and the converters). The
**stream/file-format and calc-rule reference is stable**, but some command-line
syntax and workflow examples still reflect the legacy C++ binaries and are being
**updated to the pytools implementation** — see `MIGRATION_PLAN.md` §3.3.

- The Financial Module *design* is documented in
  {doc}`../../explanation/financial-module`.
- The ORD *standard* is single-sourced in the `ODS_OpenResultsData` repository;
  the page here documents how the kernel *emits* ORD outputs.
```

```{toctree}
:maxdepth: 1

Contents
Introduction
Overview
ReferenceModelOverview
CoreComponents
Specification
fmprofiles
OutputComponents
ORDOutputComponents
DataConversionComponents
StreamConversionComponents
ValidationComponents
MultiPeril
RandomNumbers
Workflows
```
