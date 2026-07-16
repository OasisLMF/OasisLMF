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
> six: OasisLMF, OasisPlatform, ODS_OpenExposureData, ODS_OpenResultsData, ODS_Tools,
> and **OasisModels** (example models — hosts cross-repo/model worked-example
> notebooks, kept fresh by its own model-run CI); ktools is drained; GenerateDocs
> orchestrates.
>
> **Placement rule for worked examples/notebooks:** an *executable* example that runs
> the engine on a model lives in the **model repo** (OasisModels) — co-located with the
> data and its model-run CI keeps outputs honest; the orchestrator has no engine CI to
> do that. Single-repo API/how-to notebooks live in their owning repo. GenerateDocs may
> hold *non-executable* cross-repo narrative that links to the executable examples.

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
| `ODTF.rst`, `ODS-tools.rst` (GenerateDocs) | **MOVE to ODS_Tools** — it owns ODTF (`ods_tools/odtf/`) + the OED loader/validator + CLI (convert/check/transform/combine/generate), and **bundles the `model_settings`/`analysis_settings` schemas** (`ods_tools/data/*.json`) that redoc renders. Good docstrings → autoapi. | **MOVE** |

**Standards repos need a Sphinx setup + generated reference.** None of ODS_OpenExposureData,
ODS_OpenResultsData, or ODS_Tools currently has a `conf.py`. For each: stand up the
OasisLMF-style Diátaxis Sphinx setup, publish independently, and pull into the aggregated
site via intersphinx.
- **ODS_OpenExposureData** already generates `oed.json` from its CSV dictionary — reuse it to
  generate the OED field reference (don't hand-maintain the prose).
- **ODS_OpenResultsData (ORD)** is dormant (last commit 2022); spec lives in `Schema/*.csv`
  + `Docs/ORD_Data_Spec.xlsx` + **9 Excel worked examples** (SELT→SPLT, EPT variants, TVAR…).
  Add an `ord.json` generator (mirroring `oed.json`) and convert the worked examples to
  notebooks.
- **ODS_Tools** owns the **settings-schema reference** (`model_settings`/`analysis_settings`)
  — so that reference is single-sourced here, not in OasisLMF.

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
4. ✅ **RESOLVED — Publishing:** **GitHub Actions + a version selector** (tag-triggered
   builds + version dropdown + PR previews). Chosen over Read the Docs because the target is
   a bespoke multi-repo orchestrated build (pinned tags + redoc + intersphinx), which fits a
   scriptable Actions build better than RTD's single-repo model.
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

**Working method (agreed):** create a `docs/migration` branch in each repo and do the work
**locally**; push and open PRs across **all** repos together once the whole migration is
ready, so cross-repo intersphinx links resolve at review time. OasisLMF's foundation is
already committed on `docs/restructure-poc`; `docs/migration` branches exist in the other
repos (GenerateDocs, OasisPlatform, ktools, ODS_OpenExposureData, ODS_OpenResultsData,
ODS_Tools).

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

---

## 12. Progress log

- **Foundation** (OasisLMF, branch `docs/migration`, commit `7b9643178`): Diátaxis
  structure, FM worked example, scoped autoapi, DOCS_STRATEGY + MIGRATION_PLAN + authoring
  prompt.
- **Wave 1 (gulmc) — increment 1:** migrated `modelling-methodology`,
  `sampling-methodology`, `correlation`, `disaggregation` from GenerateDocs into
  `explanation/` (images + paths fixed, dead `pytools`/`gulmc-pytools` links repointed to the
  reference); authored `coverage-dependency.md` for the new feature; widened autoapi to
  `pytools/gulmc`; wired reference + explanation nav. Build green (~25s), **renders clean**.
  - *Residual, deferred to the UPDATE pass* (~30 inherited docutils warnings in the
    bulk-copied rst; none affect rendering): 19 `----` transitions, 2 duplicate labels
    (`features_by_version`, `available_1.27`), 8 blank-line-before-list, 1 short title
    underline; plus pre-existing fm/gulmc docstring formatting.
  - *Note:* the coverage-dependency **API** will appear in the reference once
    `feature/gulmc-coverage-dependency` merges to main (this docs branch is off main).
- **Wave 1 — increment 2 (ktools drain → reference):** drained the whole ktools `docs/md`
  kernel set (15 pages + 21 figures) into `reference/kernel/` (already Markdown → builds
  under MyST). Skipped `FinancialModule.md` (duplicate — links repointed to
  `explanation/financial-module`); fixed the dead `InputConversionComponents.md` link;
  repointed the 3 external ktools links in `financial-module.rst` to the in-repo kernel
  pages; added a section banner (migrated-from-ktools, being updated to pytools) + toctree.
  Build green (~33s), **renders clean**, authored/edited content warning-free.
  - *Residual, deferred to the UPDATE pass* (all in drained content): `myst.header`
    non-consecutive-heading warnings **suppressed** in `conf.py` (ktools uses H1→H3 skips);
    ~86 `myst.xref_missing` from ktools' HTML `<a name>` in-page anchors (convert to MyST
    `(target)=` anchors or link to heading slugs); ~62 `----` transitions; plus the
    methodology-page residual from increment 1. None affect page rendering.
  - *Real UPDATE work still needed:* the kernel pages describe the **legacy C++ CLI**
    (`gulcalc -S100 …`, shell pipes) — rewrite for the pytools CLI; resolve `ordleccalc`'s
    pytools home; reconcile `ORDOutputComponents` against the ORD standard in
    `ODS_OpenResultsData`.
- **Wave 1 — increment 3 (gulmc how-to):** added `how-to/ground-up-losses.md` — task
  recipes for running gulmc (engine selection, sample count, RNG choice, correlation,
  disaggregation, coverage dependency, performance tuning, and the low-level kernel-pipeline
  invocation), wired into the How-to guides toctree. Build green, page + links warning-free.
  **The gulmc module now has the full Diátaxis set** (explanation + reference + how-to).
- **Wave 1 — increment 4 (results/ORD outputs):** migrated `results.rst` into
  `reference/outputs/` (repointed the dead ODS `:doc:` and the ktools link; added a banner
  clarifying the ORD-standard boundary); widened scoped autoapi to the output modules
  (`elt`, `lec`, `plt`, `aal`, `summary`, `pla`); added an "Outputs & results" reference
  section. Build green (~52s), renders clean, authored content + all `:doc:` repoints
  warning-free. `results.rst` inherited docutils warnings (transitions, 2 short underlines,
  blank-line-before-list) deferred to the UPDATE pass. The ORD **standard** single-sourcing
  is Wave 3 (`ODS_OpenResultsData`); the ORD-emitting kernel pages are already in
  `reference/kernel/`.
- **Wave 1 — increment 5 (keys-service):** migrated `keys-service.rst` → `explanation/`
  and the previously-orphaned in-repo `mdk-builtin-lookup.rst` → `how-to/` (co-located,
  images fixed); widened scoped autoapi to `oasislmf/lookup`. **Keys/lookup now has the full
  Diátaxis set** (explanation + how-to + reference). Build green, renders clean, no broken
  refs. `keys-service.rst` inherited docutils + code-highlighting warnings (its
  `<model_id>` placeholder snippets in json/python blocks) deferred to the UPDATE pass.
- **Wave 1 — increment 6 (MDK / model-dev):** migrated the model-developer set from
  GenerateDocs across Diátaxis — `Oasis-models` → explanation; `model-development-kit`,
  `api-client` → how-to; `OasisLMF-package`, `Oasis-model-data-formats`, `Oasis-file-formats`
  → reference (new "Model data & package" section). Cross-refs converted to **absolute**
  `:doc:` paths; 3 out-of-set refs (vulnerability-adjustments, rest_api, releases) softened;
  14 figures migrated. Build green, renders clean, no broken refs. Inherited docutils
  warnings deferred to UPDATE pass.
- **Wave 1 — increment 7 (methodology/adjustments):** migrated the remaining
  adjustment/methodology pages into `explanation/` (new "Adjustments & methodology"
  section): `vulnerability-adjustments`, `absolute-damage`, `post-loss-amplification`
  (4 figures), `pre-analysis-adjustments`, `currency_conversion`. Disaggregation `:doc:`
  repointed; geocoding ref softened (page not yet migrated). Build green, renders clean, no
  broken refs. Inherited docutils warnings deferred to the UPDATE pass.
- **Wave 1 — increment 8 (geocoding + evaluation):** migrated `geocoding` → explanation
  (Keys & lookup) and `Oasis-evaluation` → how-to. Build green, renders clean, no broken refs.
- **Wave 1 OasisLMF MOVE essentially complete.** The `model_settings`/`analysis_settings`
  redoc stubs are NOT an OasisLMF item — those schemas are owned by **ODS_Tools** (Wave 3).
  Remaining OasisLMF work: **notebooks** (convert ktools `examples/*.py` — first executable
  docs) and the **UPDATE pass** (clear inherited docutils/anchor warnings; rewrite ktools
  C++ CLI → pytools). Then Waves 2 (OasisPlatform) and 3 (ODS standards repos).
- **Wave 1 — increment 9 (executable notebooks):** stood up `myst-nb` (swapped
  `myst_parser`→`myst_nb`; `source_suffix` `.md`/`.ipynb`→`myst-nb`; `nb_execution_mode=cache`,
  **`nb_execution_raise_on_error=True`** so a notebook error fails the build). Authored the
  first executable notebook `tutorials/explore-model-data.md` (MyST-Markdown) — loads in-repo
  example model data (`test_model_1` CSVs under `tutorials/data/example_model/`) and explores
  footprint / vulnerability / damage-bins / exposure with pandas + a matplotlib plot;
  **executes at build**, outputs verified (16 tables + 1 plot rendered). First build ~1m13s
  (executes the notebook; cached thereafter).
  - **`myst-nb` execution = a smoke test** ("does the example still run against current
    code"), NOT output assertion. For CI output/regression testing add **`nbmake`**
    (`pytest --nbmake`) or `nbval`. The docs-build job is the smoke test.
- **Wave 1 — increment 10 (ORD outputs notebook):** added executable
  `tutorials/analyse-ord-results.md` — builds an ELT, simulates a year loss table, and
  derives + plots an OEP/AEP exceedance-probability curve (self-contained, synthetic data,
  seeded). Executes at build; EP-curve plot + tables verified. **Two executable notebooks
  now** (explore-model-data, analyse-ord-results).
- **Wave 1 — increment 11 (cross-repo e2e notebook, in OasisModels):** stood up an
  executable docs project in **OasisModels** (Furo + myst-nb) and added the high-level
  `tutorials/run-piwind-analysis.md` — shows `oasislmf model run -C config` (the current
  default is a **full pytools** pipeline: modelpy→gulmc→fmpy→summarypy→eltpy/pltpy/lecpy/aalpy,
  **no ktools binaries**) and analyses the ORD outputs (SELT, OEP/AEP EP curves for GUL & IL,
  losses at return periods) from a committed sample of a real PiWind run (`losses-20260715142725`).
  Engine shown as a command, not executed at build; analysis cells execute + verified by
  render. Committed on OasisModels `docs/migration` (`cb310bd`). Placement per the rule above.
  *(Corrected ORD semantics from ODS_OpenResultsData: EPType 1=OEP, 2=OEP TVaR, 3=AEP, 4=AEP TVaR;
  EPCalc 2=FullUncertainty.)*
- **Wave 1 — increment 12 (step-by-step pipeline notebook, in OasisModels):** added
  `tutorials/pipeline-step-by-step.md` decomposing the generated `run_kernel.sh` stage by
  stage — `evepy → gulmc → fmpy → summarypy → eltpy/pltpy/lecpy/aalpy` — with each tool's
  real command and the **intermediary stream data** (events; GUL item-level losses incl. the
  `sidx` statistics -1..-5; FM insured losses), produced by running each pytools tool once on
  a single event and converted with `bintocsv`. Engine shown as commands, not executed at
  build; sample-loading cells execute (render-verified). OasisModels `docs/migration` (`4ff029f`).
- **Wave 1 — increment 13 (OED load/validate notebook, in ODS_Tools):** stood up an
  executable docs project in **ODS_Tools** (Furo + myst-nb) and added
  `tutorials/load-validate-oed.md` — loads an OED location file with `ods_tools.oed`
  (typed DataFrame), runs the OED validation rules in `return` mode (catches a missing
  conditionally-required `LocPeril`), fixes it and re-validates (0 findings). Exercises the
  `ods_tools` **library** (light), so it **executes at docs-build** (render-verified).
  ODS_Tools `docs/migration` (`c0e626f`).
- **Notebooks still to author:** CEDE→OED via ODTF and currency conversion (ODS_Tools),
  keys/lookup and platform-API examples (their repos) — the rest of the §4 plan.
- **UPDATE pass — kernel pages (C++ → pytools), started (`da836e066`):** rewrote
  `reference/kernel/CoreComponents.md` for pytools — tool mapping (eve→evepy, getmodel→
  modelpy, gulcalc→gulpy/gulmc, fmcalc→fmpy, summarycalc→summarypy), pytools CLI/usage/
  examples grounded in the real `--help`, concepts kept; dropped the HTML `<a id>`/"Return
  to top" anchors (xref_missing 86→80, warnings 347→338); and `OutputComponents.md`
  (eltcalc→eltpy, pltcalc→pltpy, leccalc→lecpy, aalcalc→aalpy, kat→katpy; ORD-native;
  `7bae01d8d`; xref_missing 80→73, warnings 338→325). and the converter pages
  `StreamConversionComponents` (full rewrite → `bintocsv`/`csvtobin`) + `DataConversionComponents`
  (intro→`csvtobin`/`bintocsv` mapping table; format tables kept) (`e3de070ca`; xref_missing
  73→44, warnings 325→291). **Remaining kernel pages to update:** `ORDOutputComponents`,
  `ValidationComponents`, `Workflows` (pipelines → pytools), and light passes on
  `Specification`/`fmprofiles` (formats/rules stable) and the conceptual pages
  (`Introduction`/`Overview`/`ReferenceModelOverview`/`MultiPeril`/`RandomNumbers`).
  *(Note: DataConversion's per-section command examples still show legacy binary names,
  covered by its mapping-table banner — a deeper per-section pass is optional follow-up.)* Also drop remaining HTML anchors as each is rewritten (clears the rest
  of the ~80 xref_missing).
