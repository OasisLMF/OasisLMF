Dynamic footprint
=================

A dynamic footprint is built per event at run time, from two tables, instead of being read from a
pre-computed footprint file. It is a ``gulmc`` feature (``dynamic_footprint`` option); ``modelpy``
cannot run a dynamic footprint and refuses a model that has one.

Why two tables
--------------

A static footprint is one table over **event x areaperil**. For a building-resolution model that is
not storable: tens of thousands of events against millions of areaperils. A dynamic footprint
factorises it into two smaller tables, recombined for each event:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - File
     - Keyed on
     - Says
   * - ``event_definition.parquet``
     - event x **section**
     - how severe the event was in that region, as a return period
   * - ``hazard_case.parquet``
     - **areaperil** x return period
     - the intensity at that place for an event of that return period

A **section** is the unit at which an event's severity is defined (a river catchment, a rainfall
cell); an **areaperil** is the unit at which the hazard response is defined. Sections can be coarse
and areaperils fine. A new event set, a climate scenario say, needs only new ``event_definition``
rows; the hazard case is unchanged.

The model data
--------------

``event_definition`` holds, per event and section, the two return periods bracketing the event's
severity, ``rp_from`` and ``rp_to``, and an ``interpolation`` weight between them.

``hazard_case`` holds ``section_id``, ``areaperil_id``, ``return_period`` and ``intensity``. Several
rows for one (section, areaperil, return period) form a discrete distribution of intensity: the
hazard uncertainty at that place. An optional ``probability`` column weights them; without it the
rows weigh equally, so a value repeated twice has twice the weight.

Either file may be partitioned by section (a directory of ``section_id=N`` partitions) or be a
single file, and either may be sparse: a section absent from the event definition has no events, and
a section absent from the hazard case is not affected by the modelled perils.

Building an event's footprint
-----------------------------

For each section the event affects, and each areaperil of that section, the intensity distribution
at ``rp_from`` is interpolated towards the one at ``rp_to``. An areaperil lying in several of the
event's sections then has one distribution per section, and those are merged into one. Intensities
are floored to integers once, after the merge.

Interpolating between return periods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two brackets' distributions are paired **by cumulative probability**: the j-th quantile at
``rp_from`` is interpolated with the j-th quantile at ``rp_to``. The brackets' cumulative
probability breakpoints are merged into one set of levels, and each interval between consecutive
levels becomes one realisation, whose probability is the interval's width. The brackets may hold
different numbers of realisations, with different probabilities.

For example, interpolating half-way from a single value at ``rp_from`` to three values at ``rp_to``:

.. list-table::
   :header-rows: 1

   * - Level interval
     - ``rp_from``
     - ``rp_to``
     - Interpolated
   * - 0 - 0.25
     - 10
     - 16
     - 13 at 0.25
   * - 0.25 - 0.75
     - 10
     - 20
     - 15 at 0.5
   * - 0.75 - 1
     - 10
     - 24
     - 17 at 0.25

With the same number of realisations at the same probabilities in both brackets this is pairing by
rank. A bracket the hazard case does not cover for an areaperil reads as intensity 0.

Drawn as curves, the pairing is easiest to see on an example where both brackets are uncertain:

.. figure:: /_static/images/dynamic_footprint/interpolation_light.svg
   :figclass: only-light
   :alt: Step curves of intensity against cumulative probability for the rp_from bracket, the rp_to bracket and the interpolated result
   :width: 100%

   Each bracket drawn as its intensity at every cumulative probability level (its quantile curve). ``rp_from`` has two values at 0.5 each, ``rp_to`` three at 0.25 / 0.5 / 0.25. Their breakpoints, 0.25, 0.5 and 0.75 (dotted), cut the levels into four intervals; in each, the two curves' values are interpolated, here half-way, giving 13, 15, 20 and 22 at 0.25 each.

.. figure:: /_static/images/dynamic_footprint/interpolation_dark.svg
   :figclass: only-dark
   :alt: Step curves of intensity against cumulative probability for the rp_from bracket, the rp_to bracket and the interpolated result
   :width: 100%

   Each bracket drawn as its intensity at every cumulative probability level (its quantile curve). ``rp_from`` has two values at 0.5 each, ``rp_to`` three at 0.25 / 0.5 / 0.25. Their breakpoints, 0.25, 0.5 and 0.75 (dotted), cut the levels into four intervals; in each, the two curves' values are interpolated, here half-way, giving 13, 15, 20 and 22 at 0.25 each.

Areaperils in several sections (confluences)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An areaperil can carry hazard under several sections: a property at a river confluence can be
flooded by either catchment, and each catchment floods on its own schedule. When one event drives
more than one of an areaperil's sections, their distributions are merged by the
``dynamic_footprint_section_merge`` option:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Option
     - Merged distribution
   * - ``max`` (default)
     - The largest intensity, the sections' uncertainties moving together: the sections are paired
       by cumulative probability, as the return period brackets are, and the larger intensity is
       kept at each level.
   * - ``max_independent``
     - The largest intensity, the sections' uncertainties independent: ``P(max <= x)`` is the
       product of the sections' ``P(intensity <= x)``.
   * - ``sum``
     - The sum of the intensities, paired by cumulative probability. For intensities that
       accumulate rather than compete; not for water depth.

With no hazard uncertainty, one realisation per section, ``max`` and ``max_independent`` both keep
the largest intensity. They differ only for uncertain hazard. With one section at 140 / 160 / 180
and another at 120 / 150 / 190, each at probabilities 0.25 / 0.5 / 0.25:

.. list-table::
   :header-rows: 1

   * - Option
     - Merged
     - Mean
     - ``P(intensity >= 180)``
   * - ``max``
     - 140 at 0.25, 160 at 0.5, 190 at 0.25
     - 162.50
     - 0.25
   * - ``max_independent``
     - 140 at 0.0625, 150 at 0.125, 160 at 0.375, 180 at 0.1875, 190 at 0.25
     - 168.75
     - 0.4375

.. figure:: /_static/images/dynamic_footprint/max_together_light.svg
   :figclass: only-light
   :alt: Step curves of intensity against cumulative probability for two sections and their maximum, which follows the higher curve
   :width: 100%

   ``max`` pairs the sections by cumulative probability, as the brackets above, and keeps the higher intensity at each level: the merged curve (dashed) is the upper envelope of the sections' curves. Section 1 governs the lower levels, section 2 the top one.

.. figure:: /_static/images/dynamic_footprint/max_together_dark.svg
   :figclass: only-dark
   :alt: Step curves of intensity against cumulative probability for two sections and their maximum, which follows the higher curve
   :width: 100%

   ``max`` pairs the sections by cumulative probability, as the brackets above, and keeps the higher intensity at each level: the merged curve (dashed) is the upper envelope of the sections' curves. Section 1 governs the lower levels, section 2 the top one.

.. figure:: /_static/images/dynamic_footprint/max_options_light.svg
   :figclass: only-light
   :alt: Cumulative distribution curves of two sections, of their maximum when their uncertainties move together, and of their independent maximum
   :width: 100%

   The same two sections as cumulative distributions, ``P(intensity <= x)``. The maximum is at most x only if both sections are: moving together, that is the lower of the two curves (``max``); independent, it is their product (``max_independent``), which is lower again. Lower curves put more probability on high intensities: just below 180, ``max`` has reached 0.75 but ``max_independent`` only 0.5625, so the chance of at least 180 is 0.25 against 0.4375.

.. figure:: /_static/images/dynamic_footprint/max_options_dark.svg
   :figclass: only-dark
   :alt: Cumulative distribution curves of two sections, of their maximum when their uncertainties move together, and of their independent maximum
   :width: 100%

   The same two sections as cumulative distributions, ``P(intensity <= x)``. The maximum is at most x only if both sections are: moving together, that is the lower of the two curves (``max``); independent, it is their product (``max_independent``), which is lower again. Lower curves put more probability on high intensities: just below 180, ``max`` has reached 0.75 but ``max_independent`` only 0.5625, so the chance of at least 180 is 0.25 against 0.4375.

The independent maximum is heavier in the tail, because either section being high is enough. Which
one fits is a question about the model: whether the sections' residual uncertainties at the same
place, once the event definition has fixed each section's return period, move together or not.

Each merged realisation keeps the return period of the section giving its intensity, the larger
one on a tie; return period protection is therefore decided on the section that dominates. An
areaperil in a single section is unchanged whatever the option.

.. note::

   Taking the largest intensity treats the sections as alternatives competing at the same place. At
   a real confluence the water from both rivers can interact, backwater raising the level above
   either river alone; that has to be represented in the hazard case itself, and ``max`` is then a
   lower bound.

Declaring an areaperil's sections
---------------------------------

The keys carry a location's sections in the ``section_id`` column, several of them separated by
``;`` (``1;2``). Emitting one keys row per section does **not** work: the rows share an item, and
all but the first are dropped (`issue 2203 <https://github.com/OasisLMF/OasisLMF/issues/2203>`_).

The built-in lookup produces that form with a ``join_rows`` step after any step that matches a
location several times. Placed after an ``rtree`` step on overlapping section polygons, with
``"columns": ["section_id"]`` as its parameter, it gives a location falling in several sections one
row, its section ids joined in numeric order (see :doc:`keys-service`).

Choosing the merge
------------------

``dynamic_footprint_section_merge`` is a run option of the loss generation, and can be set as any
other: on the command line (``--dynamic-footprint-section-merge``), in the MDK configuration file,
with the ``OASIS_DYNAMIC_FOOTPRINT_SECTION_MERGE`` environment variable (under
``OASIS_ENV_OVERRIDE``), or in a ``computation_settings`` block. A model sets its own default in
the ``computation_settings`` of its model settings, as a parameter (``dropdown_parameters`` entry
named ``dynamic_footprint_section_merge``); an analysis can override it in the
``computation_settings`` of its analysis settings. ``gulmc`` and the footprint server
(``servedata``) take it as ``--section-merge``.

Example
-------

The `DynamicFootprint <https://github.com/OasisLMF/OasisModels/tree/main/DynamicFootprint>`_
example model includes a confluence example: three overlapping sections over 18 areaperils, in
every combination of one, two and three sections, each with no hazard uncertainty, the same number
of realisations everywhere, and different numbers between sections and return periods. Its
``tests/test_2`` runs them, one loss line per location.

.. note::

   OasisLMF 2.5.4 to 2.5.8 merged an areaperil's sections into an equal-weight mixture, and paired
   return period brackets holding different numbers of realisations by rank, interpolating the
   unmatched ones from intensity 0. Models with either feature give different results from those
   versions.
