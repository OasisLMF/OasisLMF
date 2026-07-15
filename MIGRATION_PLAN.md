# OasisLMF Documentation — Cross-Repo Migration Plan

**Status:** Draft for team review · **Companion to:** `DOCS_STRATEGY.md` (the *why*
and the target architecture). This document is the *what/where/how*: the concrete,
repo-by-repo migration — moving, synthesising, updating, and adding executable
(Jupyter/`myst-nb`) examples.

Grounded in a full inventory of the feeding repos cloned under `/home/sstruzik/`
(GenerateDocs, OasisPlatform, ktools, ODS_OpenExposureData) plus OasisLMF.

---

## 1. Repos in scope and their end-state role

| Repo | Default branch | End-state role |
|---|---|---|
| **OasisLMF** | main | **Owns** MDK/pytools/CLI docs + all GUL/FM methodology + results/ORD + ktools content (drained in). Diátaxis, in-repo `docs/`. |
| **OasisPlatform** | main | **Owns** platform/deployment/distributed/REST-API docs. Already has Sphinx + strong k8s docs; needs Diátaxis + persisted OpenAPI. |
| **ODS_OpenExposureData** | main | **Owns** the OED (exposure) standard reference (single source of truth — CSV data dictionary → `oed.json`). Publish as its own Sphinx site. |
| **ODS_OpenResultsData** | tbc | **Owns** the ORD (results) standard reference. Same single-source pattern as OED. Repo: `github.com/OasisLMF/ODS_OpenResultsData`. |
| **ODS_Tools** | tbc | **Owns** `ODS-tools.rst` / `ODTF.rst` (data-standards tooling). Inventory pending. |
| **ktools** | develop | **Drained, not owned.** Its `docs/md/*.md` move into OasisLMF; no `ktools/docs/` survives. |
| **GenerateDocs** | main | **Slimmed to an orchestrator**: home/entry-points + ecosystem pages + version-pinned aggregation (intersphinx + redoc) → publishes `oasislmf.github.io`. |

> This refines `DOCS_STRATEGY.md` §4 by adding the standards repos —
> **ODS_OpenExposureData** (OED), **ODS_OpenResultsData** (ORD), and **ODS_Tools** —
> as owning repos, on the single-source-of-truth principle. Owning-docs repos total
> five: OasisLMF, OasisPlatform, ODS_OpenExposureData, ODS_OpenResultsData, ODS_Tools;
> ktools is drained; GenerateDocs orchestrates.

---

## 2. Current-state summary (from the inventory)

- **GenerateDocs** — 48 `src/sections/*.rst` + `home/` (3) + `use_cases/` (3) +
  `releases/` (9, one empty) + `schema/` (4 redoc JSON) + **125 images / 21 MB**.
  Deploys to `OasisLMF/OasisLMF.github.io`. Redoc schema versions **hardcoded**
  (`PLAT_VER='2.5.4'`). **No notebooks.** Largest pages: `results.rst` (57 KB),
  `distributed_execution.rst` (40 KB), `financial-module.rst` (35 KB).
- **ktools** — 16 `docs/md/*.md` (all mapped to pytools below) + runnable
  `examples/*.py`. `ordleccalc` is the one component with no clean pytools match.
- **OasisPlatform** — has Sphinx (alabaster) + excellent in-repo k8s/Helm docs;
  **REST API OpenAPI schemas are CI-only artifacts with 3-day retention** (must be
  persisted/published to survive).
- **ODS_OpenExposureData** — owns the standard: CSV dictionary (`OEDInputFields.csv`,
  559 fields + 14 code lists) → generated `oed.json`; 8 narrative `Docs/*.rst`;
  `Examples/*.csv`. GenerateDocs currently **duplicates** this as hand-written prose.
- **OasisLMF** — in-repo `docs/` now restructured on `docs/restructure-poc`
  (Diátaxis, FM done as worked example); orphaned `fm_architecture.md` wired in.

---

## 3. The four content actions

Every source page gets one of: **MOVE** (relocate + co-locate), **SYNTHESISE**
(merge overlapping sources into one), **UPDATE** (rewrite stale/implementation-drifted
content), **DROP** (retire low-value stubs).

### 3.1 GenerateDocs → OasisLMF

| Source page(s) | Target section (OasisLMF `docs/source/`) | Action |
|---|---|---|
| `financial-module.rst` + `docs/fm_architecture.md` | `explanation/financial-module` (+ `fm-architecture`) | **SYNTHESISE** (already started in POC) |
| `modelling-methodology`, `sampling-methodology`, `correlation`, `disaggregation`, `post-loss-amplification`, `vulnerability-adjustments`, `absolute-damage`, `pre-analysis-adjustments`, `currency_conversion` | `explanation/` (co-located with `pytools/gulmc`, `pytools/pla`, etc.) | **MOVE** (UPDATE `vulnerability-adjustments` — flagged possibly stale) |
| `results.rst` (57 KB) — Oasis output *implementation* | `reference/outputs/` (co-located with `pytools/{elt,lec,plt,aal,summary}`) | **MOVE** (Oasis-specific bits); cross-link to ORD standard |
| `ORD.rst` — the ORD *standard* | → **ODS_OpenResultsData** (single source); pulled via intersphinx | **SYNTHESISE→ORD repo** |
| `keys-service`, `model-development-kit`, `OasisLMF-package`, `api-client`, `Oasis-models`, `Oasis-model-data-formats`, `Oasis-file-formats`, `geocoding` | `how-to/` + `reference/` | **MOVE**; **SYNTHESISE** MDK/install with existing `installation.rst` + `building-and-running-models.rst` |
| `pytools.rst`, `ktools.rst` | `reference/` (merge with drained ktools docs, §3.3) | **SYNTHESISE** |
| `model_settings`, `analysis_settings` (redoc stubs) | `reference/settings/` | **MOVE** (schemas from ods-tools; keep redoc/jsonschema render) |
| `Oasis-evaluation`, `camel`, `complex-model` | — | **UPDATE-or-DROP** (niche/stub; confirm ownership) |

### 3.2 GenerateDocs → OasisPlatform

| Source page(s) | Target | Action |
|---|---|---|
| `overview`, `platform_architecture`, `container_configuration`, `distributed_execution`, `distributed_configuration`, `deployment`, `API`, `Oasis-UI` | OasisPlatform `docs/` (Diátaxis) | **MOVE**; **SYNTHESISE** with existing in-repo k8s/Helm READMEs (avoid duplicating the already-strong `kubernetes/charts/README.md`) |
| `rest_api`, `platform_1`, `platform_2` (redoc) | OasisPlatform `docs/reference/api` | **MOVE** + **UPDATE**: persist the drf-spectacular OpenAPI JSON in-repo (fix 3-day-artifact problem), render with redoc there |

### 3.3 ktools → OasisLMF (drain; the pytools mapping)

| ktools doc | pytools target | Action |
|---|---|---|
| `FinancialModule.md`, `fmprofiles.md` | `explanation/financial-module`, `reference/fm-calcrules` (`pytools/fm`) | **SYNTHESISE + UPDATE** (verify all 38 calcrule_ids) |
| `CoreComponents.md`, `Specification.md`, `Overview.md`, `Introduction.md` | `explanation/` + `reference/stream-formats` (`pytools/{eve,getmodel,gul,gulmc,summary}`) | **MOVE + UPDATE** (stream/binary specs stay; C++ CLI → pytools CLI) |
| `OutputComponents.md`, `ORDOutputComponents.md` | `reference/outputs` (`pytools/{elt,lec,plt,aal}`) | **MOVE + UPDATE**; resolve **`ordleccalc`** (no clean pytools equivalent — confirm it lives in `pytools/lec`) |
| `DataConversionComponents.md`, `StreamConversionComponents.md` | `reference/converters` (`pytools/converters`) | **MOVE + UPDATE** (tool names → pytools) |
| `ValidationComponents.md` | `reference/validation` | **MOVE + UPDATE** |
| `MultiPeril.md`, `RandomNumbers.md` | `explanation/` (`pytools/gulmc`) | **MOVE** (conceptual; mostly stable) |
| `Workflows.md` | `how-to/pipelines` + notebooks (§4) | **UPDATE** (shell-pipe examples → pytools) |

**The "update" is the real work, not the copy:** ktools docs describe C++ single-letter
CLI flags, shell-pipe workflows, and binary behaviour. Binary/stream *format* specs are
largely stable; CLI/usage/workflows must be rewritten against the pytools implementation.

### 3.4 GenerateDocs / ODS → ODS_OpenExposureData (single source of truth)

| Source | Action |
|---|---|
| `OED.rst`, `ODS.rst` (GenerateDocs, hand-written, duplicate the spec) | **SYNTHESISE→ODS**: stand up ODS_OpenExposureData as its own Sphinx site (narrative `Docs/*.rst` already exist); **generate** field-reference tables from `oed.json`/CSV rather than hand-maintaining. GenerateDocs keeps only a stub + intersphinx link. |
| Oasis-specific OED bits (how FM implements OED terms; `OED_validation_guidelines.md`, `OED_currency_support.md` already in OasisLMF) | **Keep in OasisLMF**, cross-link to ODS |
| `ODTF.rst`, `ODS-tools.rst` | **MOVE to ODS_Tools** (pending its inventory) or keep as orchestrator ref — **decision** |

### 3.5 DROP

`complex-model.rst` (252 B stub), `errors.rst` (179 B stub), `Oasis-platform.rst` (bare
stub), `releases/oasis_platform.md` (0 B). Confirm no inbound links first (`linkcheck`).

### 3.6 Keep in GenerateDocs (orchestrator)

`home/` (introduction, FAQs, git-repo), `use_cases/` (model-developer, model-users,
installing-deploying), `index.rst`, `model-providers`, `SaaS-providers`, `versioning`,
`releases` (aggregated), `Oasis-workflow` (high-level, links out), the ecosystem images.

---

## 4. Executable examples — Jupyter / `myst-nb` plan

**None of the repos currently ship notebooks.** This is net-new, high-value work: runnable
examples that double as tests (a notebook that executes in CI cannot silently rot).

**Tooling decision:** use **`myst-nb`** (not `nbsphinx`) — it composes with the MyST
Markdown already chosen, runs notebooks at build time, and supports both `.ipynb` and
MyST-Markdown notebooks. Execute in CI with **pinned example data**; cache outputs to keep
builds fast.

**Example-data sources (already on disk):** `OasisPiWind` (reference model), ODS
`Examples/*.csv` (property/cyber/liability/marine), ktools `examples/input`+`static`,
`ktest` (FM worked examples).

| # | Notebook | Repo / section | Built from | Audience |
|---|---|---|---|---|
| 1 | Run your first analysis (PiWind end-to-end) | OasisLMF `tutorials/` | PiWind | analyst |
| 2 | Build a model with the MDK | OasisLMF `tutorials/` | `model-development-kit.rst` + PiWind | model-dev |
| 3 | Keys/lookup service walkthrough | OasisLMF `how-to/` | `keys-service.rst` | model-dev |
| 4 | Financial module worked example | OasisLMF `explanation/` | `ktest` FM cases | model-dev/analyst |
| 5 | Correlation & disaggregation in practice | OasisLMF `explanation/` | `correlation.rst` + gulmc | model-dev |
| 6 | Parse & analyse outputs (ELT/LEC/PLT/AAL, ORD) | OasisLMF `how-to/` | `results.rst` + `pytools/{elt,lec,plt,aal}` | analyst |
| 7 | ktools→pytools workflow equivalents | OasisLMF `how-to/pipelines` | convert ktools `examples/*.py` (aal/elt/plt/lec/gulandfm) | model-dev |
| 8 | Load & validate an OED exposure; CEDE→OED via ODTF | ODS_OpenExposureData `examples/` | ODS `Examples/*.csv` | analyst |
| 9 | Drive an analysis via the Platform API | OasisPlatform `tutorials/` | platform API client + PiWind | analyst/ops |
| 10 | Deploy on Kubernetes (Helm) walkthrough | OasisPlatform `how-to/` | `kubernetes/charts/README.md` | ops |

Per repo: add `myst-nb` + `jupyter` to docs deps, a `notebooks/` (or embedded) location,
pinned example-data fetch, and a CI job that **executes** the notebooks.

---

## 5. Decisions needed before/while executing

1. ✅ **RESOLVED — ORD ownership:** single-sourced in **`ODS_OpenResultsData`**
   (`github.com/OasisLMF/ODS_OpenResultsData`), pulled via intersphinx. The ORD *standard*
   (from `ORD.rst` + ktools `ORDOutputComponents.md`) lives there; Oasis output
   *implementation* (`results.rst`, `pytools/{elt,lec,plt,aal}`) stays in OasisLMF.
2. ✅ **RESOLVED — `ODS_Tools` owns its docs:** `ODTF.rst` / `ODS-tools.rst` move there.
   Inventory in progress (cloned + Explore agent running), same as ODS_OpenResultsData.
3. ⬜ **`ordleccalc`** — confirm its pytools home (likely folded into `pytools/lec`) so its
   docs have a target.
4. ⬜ **Publishing/versioning** — GitHub Actions + version selector vs Read the Docs (see
   the write-up handed to the team); affects how the orchestrator aggregates.
5. ⬜ **Redoc version pinning** — replace hardcoded `PLAT_VER='2.5.4'` with a per-release
   pin/manifest.

---

## 6. Orchestrator (GenerateDocs) target

**Keep:** home/entry-points, use-cases, ecosystem pages, `index.rst`, `conf.py`
(Furo + redoc + jsonschema), deploy workflow.
**Shed:** the 45 technical sections (→ owning repos), the `modules/` git-clone hack.
**Add:** a **version-pinning manifest** (each repo pinned to a release tag), **intersphinx**
to each owning repo's published site (OasisLMF, OasisPlatform, ODS), parameterised redoc
schema versions, a **version selector**.
**Fix:** release-notes automation (`oasis_platform.md` is empty; `build.sh` has the pull
code commented out).

---

## 7. Cross-cutting tooling & CI (apply per owning repo)

Replicate the OasisLMF POC baseline in OasisPlatform and ODS:
- Sphinx + Furo + MyST + sphinx-design + copybutton; **Diátaxis** structure.
- **Scoped autoapi** per subsystem (never whole-package; OasisLMF baseline proved 4m6s→20s).
- **`linkcheck`** job (catches the dead `simplitium/oed` class of link, and the temporary
  ktools→pytools link gaps during the drain).
- **Render smoke test** — grep built HTML for the Furo `.. contents::` error banner; a clean
  build ≠ a clean render (see `DOCS_STRATEGY.md` §9).
- **`myst-nb` execution** job for notebooks.
- OasisPlatform: **persist OpenAPI JSON in-repo** (end the 3-day-artifact problem).
- ODS: **generate** field reference from `oed.json`/CSV (don't hand-maintain).

---

## 8. Phasing

| Wave | Work | Depends on |
|---|---|---|
| **0. Land foundation** | Commit/PR `docs/restructure-poc` (structure + tooling + FM example) into OasisLMF | — |
| **1. OasisLMF content** | MOVE/SYNTHESISE methodology + FM + results + MDK + tools from GenerateDocs; **drain ktools** (can start now — code already in pytools) | Wave 0 |
| **2. OasisPlatform** | Diátaxis restructure; MOVE platform pages; persist+publish OpenAPI | Wave 0 (pattern) |
| **3. ODS single-source** | Stand up ODS Sphinx site; generate reference from CSV; replace GenerateDocs OED/ODS prose with intersphinx stubs | — (parallel) |
| **4. Notebooks** | Author the 10 notebooks; wire `myst-nb` + example data + CI execution | Waves 1–3 (target sections exist) |
| **5. Orchestrator** | Slim GenerateDocs; version-pin manifest; intersphinx wiring; version selector; deploy | Waves 1–3 published |
| **6. Cutover** | Point `oasislmf.github.io` at the aggregated build; retire duplicated pages; final `linkcheck` | Wave 5 |

Waves 1/2/3 are largely parallel across repos. Each is independently shippable.

---

## 9. Rough effort (relative)

| Bucket | Weight | Notes |
|---|---|---|
| MOVE (relocate + toctree + fix paths) | ~15% | Mechanical; scriptable |
| SYNTHESISE (merge overlaps) | ~20% | FM×3 sources, OED×2, MDK/install overlaps |
| UPDATE (ktools C++→pytools rewrites, stale pages, API v2) | ~30% | The real work; needs code verification |
| Notebooks (net-new, +CI execution) | ~20% | High value; example-data plumbing |
| Orchestrator + versioning + per-repo CI | ~15% | Intersphinx, redoc pinning, smoke tests |

---

## 10. Risks & gotchas (carry-over + new)

- **Clean build ≠ clean render** (Furo `.. contents::` red box; invisible to Sphinx). Add
  the HTML smoke test everywhere. (`DOCS_STRATEGY.md` §9)
- **Link rot during the drain** — ktools→pytools links dangle until targets land; guard with
  `linkcheck` and a link-redirect map.
- **Version smearing** — GenerateDocs pins nothing; redoc hardcodes 2.5.4. Fix in the
  orchestrator manifest.
- **OasisPlatform OpenAPI retention** — 3-day CI artifacts; must persist/publish or the API
  reference silently disappears.
- **OED duplication** — do NOT copy the spec into OasisLMF; single-source in ODS + intersphinx.
- **21 MB of images in GenerateDocs** — split to owning repos with the pages that use them;
  don't drag the whole `images/` tree along.
- **`ordleccalc` gap** — confirm the pytools home before migrating its docs.

---

## 11. Immediate next actions

1. **Commit & PR the OasisLMF foundation** (`docs/restructure-poc`) — unblocks everything.
2. **Resolve the 5 decisions in §5** (ORD ownership, ODS_Tools, ordleccalc, publishing, redoc pinning).
3. **Start Wave 1** on OasisLMF with **gulmc** (adjacent to FM; absorbs correlation/
   disaggregation + your just-finished coverage-dependency feature) and the **ktools drain**.
4. Optionally kick off **Wave 3 (ODS)** in parallel — it's independent.
