# Outputs & results

Reference for Oasis loss outputs — the standard result outputs and the ORD (Open
Results Data) result tables (ELT, PLT, EPT/PSEPT, ALT, …), their formats and
file-naming conventions.

```{note}
The ORD **standard** itself is single-sourced in the `ODS_OpenResultsData`
repository and will be pulled into the aggregated site via intersphinx. The pages
here document the Oasis **implementation** — the output formats and the Python
modules that emit them (`oasislmf.pytools.{elt,lec,plt,aal,summary,pla}`, whose
generated API is in the Python API section above). The kernel components that
produce these outputs are documented under {doc}`../kernel/index`
(`OutputComponents`, `ORDOutputComponents`).
```

```{toctree}
:maxdepth: 2

results
```
