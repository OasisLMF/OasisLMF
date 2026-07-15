Coverage dependency
===================

Coverage dependency lets one coverage's damage be **conditioned on another coverage's
damage** at the same location, so that correlated coverages no longer sample
independently. It is an opt-in ``gulmc`` feature; when it is not configured, results are
bit-identical to previous behaviour.

Overview
--------

In a standard ground-up loss calculation every coverage at a location (building structure,
contents, business interruption, ...) is sampled independently. That can produce
physically implausible realisations — for example the building structure 100% damaged in a
sample while contents are undamaged.

Coverage dependency ties a **dependent** coverage to a **source** coverage. In each sample,
the source's sampled **damage bin** is used to select the dependent's damage distribution,
through a purpose-built *conditional* (damage-transition) vulnerability. A typical use is
**contents conditioned on building**: how badly the contents are damaged depends on how
badly the structure was damaged in that same sample, at the same location and areaperil.

The dependence is expressed entirely in terms of damage **bins** — the source's sampled
damage bin indexes the dependent's conditional vulnerability. The source may use any damage
type (relative, absolute or duration); it is the sampled bin, not a damage ratio, that
drives the dependent.

How it works
------------

The source → dependent links form a **forest** (each dependent has exactly one direct
source; a source may drive several dependents; chains such as ``A → B → C`` are allowed and
validated to be acyclic). For each event, coverages are computed in a **depth-first order**
so that a source is always computed before its dependents. Each source records its sampled
damage bin per sample; a dependent then reads its source's bin and looks up the
corresponding column of its conditional vulnerability.

.. note::

   Under the default **full Monte Carlo** engine the linkage is per-sample (comonotonic):
   the dependent follows the source's realised bin sample by sample. Under
   ``--effective-damageability`` the dependent is supported too, but the linkage is
   **marginal only** — the dependent's damage distribution is the source's damage
   distribution pushed through the conditional vulnerability, without a per-sample tie.

Enabling and configuring
-------------------------

Three inputs work together. Only the model settings entry is required to *declare* the
dependency; the ``source_coverage_id`` column is populated automatically during file
generation, and the conditional vulnerability is model-provided static data.

1. Declare the dependency in model settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a ``coverage_dependency_settings`` block to ``model_settings.json``, listing
``{source_coverage_type, dependent_coverage_type}`` pairs (OED coverage type ids, e.g.
``1`` = buildings, ``3`` = contents):

.. code-block:: json

   {
     "model_settings": {
       "coverage_dependency_settings": [
         {"source_coverage_type": 1, "dependent_coverage_type": 3}
       ]
     }
   }

Each dependent coverage type may appear only once (it has exactly one source), and a
coverage type cannot depend on itself; violations are rejected when the settings are read.

2. The ``source_coverage_id`` column on the correlations input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The per-item link is carried on the **correlations** file as a new ``source_coverage_id``
column (``0`` = independent). You do not author this by hand: it is resolved automatically
during Oasis file generation from ``coverage_dependency_settings``, by matching a
dependent coverage to the source coverage of the configured type at the same location and
areaperil. A ``correlations.csv`` fragment then looks like:

.. code-block:: text

   item_id,peril_correlation_group,damage_correlation_value,hazard_group_id,hazard_correlation_value,source_coverage_id
   1,0,0.0,0,0.0,0
   2,0,0.0,0,0.0,0
   3,0,0.0,0,0.0,1

Here item ``3`` (a contents item) is driven by source coverage ``1`` (the building at the
same location); items ``1`` and ``2`` are independent (``source_coverage_id = 0``).

3. The conditional vulnerability (damage-transition matrix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A dependent coverage is driven by a new, optional static file
``conditional_vulnerability.csv`` (or ``.bin``): a damage-transition matrix
``P(dependent damage bin | source damage bin)``. Its columns are:

- ``vulnerability_id`` — the dependent's (conditional) vulnerability id;
- ``source_damage_bin`` — the source coverage's damage bin (``1 .. num_damage_bins``);
- ``damage_bin`` — the dependent coverage's resulting damage bin;
- ``probability`` — ``P(damage_bin | source_damage_bin)``.

For each source damage bin, the probabilities over the dependent's ``damage_bin`` values
form the conditional distribution. For example, a matrix where contents track the building
one bin lower with some spread:

.. code-block:: text

   vulnerability_id,source_damage_bin,damage_bin,probability
   100,1,1,1.0
   100,2,1,0.4
   100,2,2,0.6
   100,3,2,0.4
   100,3,3,0.6

The matrix is sized ``num_damage_bins x num_damage_bins`` — it is indexed by damage bins,
independent of the footprint's hazard-intensity resolution, so ``num_damage_bins`` may
differ from the number of intensity bins. A source damage bin with no rows is treated as an
all-zero column, i.e. "that source damage produces no dependent damage" — leaving a source
bin undefined is a valid modelling choice, not an error.

Rules and constraints
---------------------

The engine validates the configuration up front and fails loudly rather than silently
producing wrong losses:

- **Dependents must use a conditional vulnerability.** A coverage linked to a source
  (``source_coverage_id > 0``) must use a ``vulnerability_id`` present in
  ``conditional_vulnerability``; otherwise the run is aborted.
- **Independents must not use a conditional vulnerability.** A coverage with no source
  cannot use a conditional vulnerability, because a damage-transition matrix has no meaning
  without a source damage bin to index it.
- **No aggregate dependents.** A dependent coverage may not use an aggregate vulnerability.
- **Acyclic links only.** Source links may not form a cycle, point outside the coverage
  set, or be self-referential.

Per-location activation
~~~~~~~~~~~~~~~~~~~~~~~~~

The dependency is activated **per location**, only where the source and dependent share the
same areaperil with **item counts that line up** (the same multiset of areaperils, so each
dependent item pairs with the corresponding source item). Where the keys server returns the
dependent at a different areaperil from the source, or the item counts do not align, the
dependent is **demoted to independent** and a warning is logged — the run continues, it is
simply no longer driven by the source at that location.

Zero-TIV (uninsured) sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A source coverage with zero TIV (for example an uninsured building) is **retained** so that
it can still drive its dependent — a source's damage is physical and does not depend on
whether the source itself is insured. Such a source is not special-cased: it flows as an
ordinary zero-TIV, zero-loss coverage, and appears with zero loss in the outputs.

.. note::

   The behaviours described here are exercised end-to-end by
   ``tests/pytools/gulmc/test_coverage_dependency.py``, which is the source of truth for the
   configuration formats and examples on this page.
