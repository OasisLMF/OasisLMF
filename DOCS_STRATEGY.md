# OasisLMF Documentation Strategy

**Status:** Draft for team review · **Author:** initial proposal · **Scope:** how OasisLMF documentation is authored, built, and published

> This document proposes a target workflow for OasisLMF documentation and a
> phased path to get there. A working proof-of-concept accompanies it in this
> branch (`docs/restructure-poc`) — see [What this branch changes](#what-this-branch-changes).

---

## 1. Executive summary

OasisLMF documentation is currently produced by **two independent, overlapping
systems** that were never reconciled. The high-value conceptual content lives in
a *separate repository* from the code it describes, so it drifts; the in-repo
system auto-generates an unstructured API dump with no narrative spine; and both
have, at different times, deployed to the same GitHub Pages target.

The proposal: **consolidate on a single, in-repo, Diátaxis-structured Sphinx
project per code repository, aggregated into one published site**, with the
conceptual/explanation content co-located next to the code it describes so that
docs and behaviour change in the same pull request.

We keep Sphinx — it is the right tool. What changes is *where content lives*,
*how it is structured*, and *how it is kept honest*.

---

## 2. Current state (as-found)

There are three moving parts today:

### 2.1 GenerateDocs → `oasislmf.github.io`
- Repo: [`OasisLMF/GenerateDocs`](https://github.com/OasisLMF/GenerateDocs)
- A Sphinx project (Furo theme) whose `src/sections/` holds ~60 hand-written
  `.rst` pages — the flagship conceptual docs: `financial-module.rst`,
  `modelling-methodology.rst`, `sampling-methodology.rst`, `correlation.rst`,
  `disaggregation.rst`, `vulnerability-adjustments.rst`, etc.
- `conf.py` also adds `../modules/OasisLMF` and `../modules/OasisPlatform` to
  `sys.path` and runs `autodoc` over cloned copies of those repos, plus `redoc`
  for four API/JSON schemas (analysis-settings, model-settings, Platform 1 & 2).
- `build.sh` clones the code repos (largely commented out) — **no version
  pinning**, so a build reflects whatever is checked out.
- CI (`build-deploy.yml`) publishes `build/html` to
  `OasisLMF/OasisLMF.github.io`.

### 2.2 `OasisLMF/docs/` (in-repo) → this repo's GitHub Pages
- A **second** Sphinx project living in this repository at `docs/`.
- `conf.py` runs **`sphinx-autoapi` over the entire `oasislmf` package**
  (`autoapi_dirs = ['../../oasislmf']`) — thousands of auto-generated pages — and
  imports the package at build time (`from oasislmf import __version__`, plus CLI
  and computation modules) to generate a CLI options reference.
- Hand-written content is thin: `installation`, `building-and-running-models`,
  `logging-configuration`, `options_config_file`, `environment-variables`.
- Several valuable files sit **loose in `docs/` and are never wired into the
  toctree**: `fm_architecture.md`, `calc_rules.xlsx`,
  `OED_financial_terms_supported.xlsx`, `OED_validation_guidelines.md`,
  `OED_currency_support.md`, `mdk-builtin-lookup.rst`.
- CI: `build-docs.yml` (PR artifact) and `build-and-deploy-docs.yml` (deploy to
  this repo's Pages). The deploy workflow contains a **commented-out block that
  previously pushed this build to `OasisLMF/OasisLMF.github.io`** — the exact
  target GenerateDocs owns. The two systems have collided here.

### 2.3 ktools / OasisPlatform
- ktools (C++) docs are hand-written Markdown under its own `docs/md/`
  (referenced from the financial-module page).
- OasisPlatform docs are pulled into GenerateDocs via autodoc.

---

## 3. Problems

1. **Two sources of truth, overlapping scope.** The same subsystems are
   documented (or half-documented) in both places, with no defined boundary.
2. **Conceptual docs drift.** The financial-module / methodology pages live in
   GenerateDocs, physically divorced from `oasislmf/pytools/…`. Nothing forces a
   docs update when behaviour changes (e.g. gulmc coverage-dependency work
   changes the *science* the correlation/disaggregation pages describe).
3. **No content model.** ~60 flat pages in one system; an unstructured autoapi
   dump in the other. No separation of tutorial / how-to / reference /
   explanation, and no per-audience entry points.
4. **Version smearing.** GenerateDocs does not pin the repos it documents; the
   in-repo build documents `HEAD`. Published docs correspond to no single
   release.
5. **Orphaned content.** In-repo files like `fm_architecture.md` never reach any
   site despite being good material.
6. **Contribution friction & rot.** Prose contributors must know GenerateDocs
   exists as a separate repo; raw `.rst` and no live-preview raise the bar.
7. **Slow, fragile in-repo build.** Full-package autoapi + build-time import of
   the package makes the build slow and coupled to the full runtime dependency
   set.

---

## 4. Target architecture — consolidate, then aggregate (Option B)

**Principle: each code repo owns the docs for its own code, in-repo, next to that
code. One thin orchestrator aggregates them into the published site, pinned to
released versions.**

```
OasisLMF/docs/          → MDK, pytools, CLI, financial module (explanation + reference),
                          modelling methodology, correlation, disaggregation, ...
                          + the ktools component docs (see ktools note below)
OasisPlatform/docs/     → platform, deployment, distributed execution, REST API

GenerateDocs (slimmed)  → orchestrator: pins each repo to a RELEASE TAG (manifest or
                          submodules), builds each, cross-links via intersphinx, and
                          publishes the combined versioned site to oasislmf.github.io
```

**ktools is being decommissioned — it does NOT own its own docs.** The ktools
components are **already reimplemented and present in `oasislmf/pytools`**
(gulpy/gulmc, fmpy, summarypy, etc.), so this is **not gated on any code
cutover** — it is a straightforward documentation task: **copy the ktools docs
into OasisLMF (mostly) and update them** to describe the pytools implementation,
with any platform-specific pieces going to OasisPlatform. Concretely:

- Copy the ktools Markdown under `docs/md/` (e.g. `CoreComponents.md`,
  `DataConversionComponents.md`, `fmprofiles.md`) into `OasisLMF/docs/` next to
  the Python implementation it describes, and update the content (command names,
  behaviour, file formats) to match pytools rather than the retired C++ binaries.
- Repoint the external links to `github.com/OasisLMF/ktools/...` in migrated
  pages (the financial-module page has several) to **in-repo cross-references**.
  Until then they are dead-link risks the `linkcheck` job should watch.
- ktools is a **content source to drain, not a docs home to maintain**; there is
  no `ktools/docs/` in the target architecture. Because the code already lives in
  OasisLMF, the drain can proceed now, independently of the wind-down timeline.

Why this shape:
- **Kills drift** — conceptual pages sit beside the code; a behaviour PR touches
  its docs in the same diff and the same review.
- **One source of truth per topic** — GenerateDocs stops holding prose; it only
  orchestrates.
- **Versioned** — the orchestrator pins tags, so `oasislmf.github.io` can offer a
  version selector and each build maps to real releases.
- **Lower contribution barrier** — docs live where contributors already are.
- **Survives the ktools decommission** — its docs are preserved in the repo that
  inherits the functionality instead of dying with the retired repo.

Trade-off: content must migrate out of GenerateDocs (and out of ktools), and
contributors must know which repo owns a page. Accepted.

---

## 5. Content model

### 5.1 Diátaxis (four modes, never mixed on a page)

| Mode | Purpose | OasisLMF examples | Maintained as |
|------|---------|-------------------|---------------|
| **Tutorials** | Learning by doing | "Run your first analysis (PiWind)", "Build a toy model with the MDK" | Hand-written, ideally executable |
| **How-to** | Task recipes | "Generate Oasis files", "Add reinsurance layers", "Configure distributed execution" | Hand-written |
| **Reference** | Dry facts | Python API, CLI options, `analysis_settings`/`model_settings` schemas, OED/ODS fields, ktools binaries | **Auto-generated** (autoapi / argparse / schema) |
| **Explanation** | Understanding | Financial module, sampling, correlation, disaggregation, vulnerability adjustments | Hand-written, **co-located with the algorithms** |

The reference layer is largely automatable (pytools already requires docstrings
by project style). That frees human effort for the explanation layer — where the
value and the drift risk both concentrate.

### 5.2 Audiences (each gets a landing page)

- **Analysts / end users** — run analyses, interpret ORD output
- **Model developers** — MDK, build/test models, keys/lookup
- **Platform operators** — deploy & scale OasisPlatform
- **Contributors** — architecture, dev setup, how to change the code

---

## 6. Tooling decisions

| Concern | Decision | Rationale |
|---------|----------|-----------|
| Engine | **Keep Sphinx** | Right tool; already in use in both systems |
| Theme | **Keep Furo** | Already branded (Oasis colours, Raleway) |
| Authoring | **Add MyST Markdown** alongside rst | Lower barrier; migrate rst incrementally, both compile |
| API reference | **`sphinx-autoapi`, but scoped** | AST-based (no import of heavy deps); scope per subsystem instead of whole-package dump |
| Landing pages | **`sphinx-design`** cards/grids | Per-audience entry points |
| UX | `sphinx-copybutton`, "Edit on GitHub", `sphinx-autobuild` for local live preview | Contributor experience |
| Executable docs | **`myst-nb`** for tutorials + methodology | Runs real examples at build time → **cannot silently rot** |
| Diagrams | **Mermaid** (`sphinxcontrib-mermaid`) | Diagrams as text, diff cleanly, no stale binary blobs |
| Publishing/versioning | Orchestrator builds per **release tag**; version selector | Fixes version smearing; consider Read the Docs if hosted search/versioning-as-a-service is wanted |

### Quality gates (keep code & docs honest)
- `sphinx.ext.linkcheck` in CI — **there is already link rot**: the
  financial-module page links to `github.com/simplitium/oed`, which moved to the
  OasisLMF org years ago.
- Docstring-coverage gate (`interrogate`) to keep the auto-generated reference
  complete.
- `literalinclude` / `myst-nb` for code samples — never paste code; include it
  from tested sources.
- PR template checkbox: "docs updated / not needed".

---

## 7. Phased plan

1. **Audit & decide boundaries.** Inventory GenerateDocs' ~60 pages + the in-repo
   pages; tag each with Diátaxis mode + audience + owning repo + freshness.
   Agree the GenerateDocs ↔ in-repo boundary. *(Deliverable: the mapping table.)*
2. **Fix the in-repo build first.** Scope autoapi per-subsystem; stop importing
   the whole package where avoidable; add `linkcheck`. Fast, low-risk wins.
3. **Restructure in place (Diátaxis + audience landing).** Reorganise the in-repo
   `docs/` toctree; wire in the orphaned files. *(This branch is step 2+3 for the
   Financial Module as a worked example.)*
4. **Migrate conceptual content into the owning repos.** Move financial-module /
   methodology / correlation / disaggregation from GenerateDocs into
   `OasisLMF/docs/explanation/`, next to the code.
5. **Drain ktools docs into OasisLMF (can start now).** The pytools components
   already exist in OasisLMF, so this is purely a docs task: copy the ktools
   component docs (`docs/md/CoreComponents.md`, `DataConversionComponents.md`,
   `fmprofiles.md`, …) into OasisLMF (mostly) beside the `pytools`
   implementations, **update the content to describe pytools** (command names,
   behaviour, formats), and convert the now-internal ktools links in migrated
   pages to in-repo cross-references. Any platform-specific pieces go to
   OasisPlatform. There is no `ktools/docs/` in the end state.
6. **Slim GenerateDocs to an orchestrator** with a version-pinning manifest
   (tags/submodules) + intersphinx cross-linking; add a version selector.
7. **Repeat migration** for OasisPlatform; retire duplicated pages.
8. **Add executable methodology** (`myst-nb`) for 2–3 flagship pages.

Each phase is independently shippable and reviewable.

---

## 8. What this branch changes

This proof-of-concept demonstrates **phase 2–3 on the in-repo docs, using the
Financial Module as the worked example** — chosen because it is the highest-value
and highest-drift subsystem, and its explanation content currently lives only in
GenerateDocs.

Concretely:
- Restructured the in-repo `docs/` around **Diátaxis + audience landing pages**
  (sphinx-design cards) instead of a flat toctree.
- Added an **Explanation** section and **co-located** the Financial Module
  content there: migrated the flagship `financial-module` page out of
  GenerateDocs, and wired in the previously-orphaned `fm_architecture.md`.
- Scoped **autoapi to the Financial Module** so the API reference sits directly
  beside its explanation (and the build is fast), demonstrating the
  "reference-next-to-explanation" pattern before rolling it wider.
- Added MyST Markdown, sphinx-design, sphinx-copybutton, and a `linkcheck` path.

Measured effect of the scoping change on the in-repo build:

| | Before (whole-package autoapi) | After (scoped to FM, this branch) |
|---|---|---|
| Build time | **~4m 6s** | **~20s** |
| Generated HTML pages | **519** (mostly an unstructured API dump) | 41 (structured) |
| Structure | flat toctree + auto dump | Diátaxis + audience landing |
| FM explanation on the site | only in GenerateDocs | **co-located in-repo** |
| `fm_architecture.md` | orphaned, unpublished | wired in |

The remaining build warnings are **pre-existing docstring / CLI-help formatting
issues in the `fm` source** (`compute_sparse.py`, `back_allocation.py`, and one
CLI help string) — not in the restructured pages. They are exactly what the
proposed docstring-quality gate would surface.

It is deliberately a *slice*, not the whole migration — enough for the team to
react to the shape before committing to the full consolidation.

---

## 9. Migration gotchas (found while building the POC)

Concrete, recurring issues to bake into the migration checklist and, where
possible, into CI so they are caught automatically rather than by eye.

1. **Furo rejects the `.. contents::` directive with a visible red ERROR box.**
   Furo already renders an "On this page" table of contents in the right sidebar,
   so any docutils `.. contents:: ... :local:` directive is turned into a hard,
   reader-visible error banner at the top of the page (found on
   `logging-configuration.rst`; fixed on this branch by removing the directive).
   *Action:* strip `.. contents::` directives during migration — this pattern is
   common in the GenerateDocs `.rst` pages, so expect multiple hits. A CI grep
   (`! grep -rn '.. contents::' source/`) or a `-W` "fail on warning" build would
   catch regressions.

2. **The `.. contents::` case is invisible to a plain build.** The page above
   built with **exit 0 and no Sphinx warning** — the error is injected by the
   theme into the HTML, not the docutils build. *Action:* the docs CI needs a
   check beyond "did Sphinx exit 0" — at minimum a grep of the built HTML for the
   Furo error string, ideally a headless render/screenshot smoke test of a few
   key pages. "Builds clean" ≠ "renders clean".

3. **Furo `.. contents::` finding generalises:** theme-specific rendering rules
   won't surface in the build log. Any theme upgrade or directive the theme
   dislikes can ship a broken page silently. Keep a small visual/HTML smoke-test
   step in CI.

4. **Dead external links build fine.** The migrated financial-module page
   contained a dead `simplitium/oed` link (repointed on this branch). Nothing in
   a normal build flags it — hence the `make linkcheck` job in the plan.

---

## 10. Open questions for the team

1. **Publishing target:** keep hand-rolled GitHub Actions + a version selector,
   or move to Read the Docs (hosted versioning/search/PR previews)?
2. **GenerateDocs' fate:** slim to an orchestrator (recommended) or retire it and
   let the in-repo sites cross-link directly via intersphinx?
3. **Boundary:** does any conceptual prose stay in GenerateDocs, or does *all*
   prose move to the owning repo?
4. **Executable docs appetite:** are we willing to run small model examples at
   docs-build time (`myst-nb`) to keep methodology pages self-verifying?
