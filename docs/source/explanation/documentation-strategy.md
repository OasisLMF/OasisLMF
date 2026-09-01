# How the documentation is organised

This page is for people **writing** OasisLMF documentation. It records how the docs
are structured, built and published, and the rules to follow when adding a page, so
that decisions already taken don't have to be rediscovered.

## Ownership: each repository documents its own code

**Every code repository owns the documentation for its own code, in-repo, next to that
code. One thin orchestrator aggregates them into the published site.**

```
OasisLMF/docs/          MDK, pytools, CLI, the financial module, modelling
                        methodology, correlation, disaggregation, the kernel
                        component reference
OasisPlatform/docs/     platform, deployment, distributed execution, REST API
ODS_Tools/docs/         OED loading and validation, the settings schemas
ODS_OpenExposureData/   the OED standard          } spec-generated reference
ODS_OpenResultsData/    the ORD standard          }
OasisModels/docs/       worked, end-to-end model examples

GenerateDocs            orchestrator: pins each repository to a ref, builds each
                        one, resolves cross-references between them, and publishes
                        the combined site to oasislmf.github.io
```

Why this shape:

- **It kills drift.** Conceptual pages sit beside the code, so a behaviour change and
  its documentation travel in the same pull request and the same review.
- **One source of truth per topic.** The orchestrator holds no prose; it only assembles.
- **Versioned.** The orchestrator pins refs, so each published build maps to real
  releases and the site can offer a version selector.
- **Lower barrier.** Documentation lives where contributors already are.

The trade-off, accepted deliberately: contributors must know which repository owns a
page. If you are unsure, the owner is whichever repository contains the code the page
describes.

ktools is being decommissioned and does **not** own documentation. Its component docs
were drained into this repository and rewritten against the `oasislmf/pytools`
implementations; ktools is a content source, not a documentation home.

## Content model: Diátaxis

Four modes, and **never two on the same page**:

| Mode | Purpose | Examples here | Maintained as |
|------|---------|---------------|---------------|
| **Tutorials** | Learning by doing | "Run your first analysis" | Hand-written, ideally executable |
| **How-to** | Task recipes | "Generate Oasis files", "Configure distributed execution" | Hand-written |
| **Reference** | Dry facts | Python API, CLI options, settings schemas, stream formats | **Generated** where possible (autoapi, argparse, schema) |
| **Explanation** | Understanding | Financial module, sampling, correlation, disaggregation | Hand-written, **co-located with the algorithms** |

The reference layer is largely automatable, which frees effort for the explanation
layer — where both the value and the drift risk concentrate.

Each audience gets a landing page: analysts and end users, model developers, platform
operators, and contributors.

## Tooling

In place:

| Concern | Choice | Why |
|---------|--------|-----|
| Engine | Sphinx | Already in use; the right tool |
| Theme | Furo | Branded (Oasis colours, Raleway) |
| Authoring | MyST Markdown alongside reStructuredText | Lower barrier; both compile, so migration is incremental |
| API reference | `sphinx-autoapi`, **scoped per subsystem** | AST-based, so no heavy imports; avoids a whole-package dump |
| Landing pages | `sphinx-design` cards and grids | Per-audience entry points |
| Executable docs | `myst-nb` | Tutorials run at build time, so they cannot silently rot |
| Copy buttons | `sphinx-copybutton` | Contributor experience |
| Cross-repo links | `sphinx.ext.intersphinx`, driven by the orchestrator | See the note below |

Agreed but **not yet implemented** — worth knowing before you assume a safety net exists:

- **`linkcheck` in CI.** Dead external links build perfectly cleanly. There is no job
  running it today.
- **Docstring-coverage gate** (e.g. `interrogate`), to keep the generated reference
  honest as the autoapi scope widens.
- **Mermaid diagrams.** Diagrams-as-text would diff cleanly and avoid stale binary
  assets, but the extension is not configured.
- **A render check.** See the first gotcha below: "builds clean" is not "renders clean".

## Authoring rules and traps

These are concrete, recurring problems found while building these docs. Where a rule
exists it is because something broke.

1. **Do not use `.. contents::`.** Furo renders its own "On this page" sidebar, and a
   docutils `.. contents::` directive becomes a red error box visible to readers.

2. **"Builds clean" is not "renders clean".** That `.. contents::` failure produced
   **exit 0 and no Sphinx warning** — the theme injects the error into the HTML, not
   the build log. When you change something structural, look at the built page.

3. **Cross-repository references need the orchestrator.** An `{external+...}` role only
   resolves when the orchestrator supplies the inventories, and an unresolved one does
   not degrade to plain text — Sphinx drops the link text and silently mangles the
   sentence. For prose a reader must follow, either route the target through a
   per-build-mode substitution in `conf.py` or write an ordinary link; the orchestrator
   rewrites in-site URLs to page-relative either way.

4. **Google-style docstrings: mind the colon.** In a `Returns:` block napoleon treats
   everything before the first colon on the first line as the return *type*. A named
   return (`intervals (np.ndarray): …`) therefore turns the name into a type and emits a
   cross-reference; write the description first instead.

5. **Docstrings are RST.** Bare indented code or algorithm sketches need a `::` literal
   block; lists need a blank line before them; `*args`-style text and identifiers ending
   in an underscore (`numpy.str_`) must be inline literals or they are read as markup.

6. **Never paste code.** Use `literalinclude` or an executable notebook cell, so samples
   come from sources that are actually run.

7. **Keep the build at zero warnings.** Every component currently builds clean, standalone
   and aggregated. That is only useful as a gate if it stays true.
