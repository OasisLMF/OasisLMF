Pre Analysis Adjustments
========================

On this page
------------

* :ref:`introduction_paa`
* :ref:`how_it_works_paa`
* :ref:`multiprocessing_paa`
* :ref:`example_models_paa`

|

.. _introduction_paa:

Introduction
************

The Oasis modelling platform is designed to model individual buildings with known locations and vulnerability attributes. However, 
this exposure data can sometimes be aggregated, low resolution or missing key attributes, such as location data – a situation 
which is particularly true in the developing world. A pre-analysis adjustment step allows the user to overcome the issues that could 
arise from this by performing data cleansing of any errors or inconsistencies in their OED exposure data, before it is used in a 
model run. The code and cofig for the pre-analysis step are completely customisable; the user can change these to modify input 
files in any way they desire to achieve a particular output, and automating this kind of preparation improves the quality of 
analyses.

|

.. _how_it_works_paa:

How it works
************

Currently in the Oasis platform, as of August 2022, exposure must be converted into detailed data before being imported into the 
platform for analysis. The format for this data is one building per row in the location file. This can be done outside of the 
system, or alternatively the model developer, as part of the Oasis model assets, can provide a pre-analysis routine to generate a 
modified OED location file from an input OED location file.

The purpose of a pre-analysis routines is to provide flexibility to manipulate the OED input files before the model is run, for 
augmentation as required by the model. An example pre-analysis ‘hook’ for the PiWind model can be found `here 
<https://github.com/OasisLMF/OasisPiWind/blob/main/src/exposure_modification/exposure_pre_analysis_example.py>`_.

|

.. _multiprocessing_paa:

Multiprocessing
****************

For large portfolios the pre-analysis hook can be the slowest step in the workflow, so, like the
keys/lookup service, it can be run across multiple processes. It is controlled by the same
``lookup_multiprocessing``/``lookup_num_processes``/``lookup_num_chunks`` parameters as the
keys/lookup service, and chunked the same way - the hook is instantiated once per chunk and
called with a subset of ``exposure_data``, and the resulting location/account dataframes are
merged back together afterwards, in chunk order, so the output doesn't depend on which process
finishes first. Unlike the keys/lookup service, chunks are formed from unique
``(PortNumber, AccNumber)`` combinations rather than individual locations, so a single account's
location rows are never split across two chunks.

Because the framework has no visibility into what a pre-analysis hook actually does, a hook must
opt in to multiprocessing, in the same way a lookup class does, by setting a class attribute:

.. code-block:: python

    class ExposurePreAnalysis:
        multiproc_enabled = True

        def __init__(self, exposure_data, exposure_pre_analysis_setting, **kwargs):
            ...

Hooks without ``multiproc_enabled = True`` always run in a single process. Only opt in if the
hook gives the same result whether it sees the whole portfolio or one group of accounts at a
time - the intended use cases described above (per-location geocoding, disaggregation, exposure
enhancement) usually do. A hook is **not** safe to opt in if it:

* needs to see locations/accounts outside of a single account (e.g. whole-portfolio
  aggregation or optimisation), or
* numbers or counts rows across the portfolio - e.g. ``df['LocNumber'] = df.index + 1``, or a
  module-level counter used to make ``LocNumber`` values unique. Each chunk's dataframe starts
  from its own index, and module-level state is shared between the chunks that the same worker
  process happens to pick up, so the result would change from run to run, or
* reads or modifies ``exposure_data.ri_info``/``exposure_data.ri_scope`` (these are not
  chunked or merged back - only the main process's copy is kept), or
* has side effects on shared files under ``input_dir`` that aren't safe for multiple
  processes to write concurrently.

Setting ``lookup_multiprocessing=False`` runs every hook in a single process, whether or not it
has opted in.

|

.. _example_models_paa:

Example models
**************

Oasis currently offers two toy models that demonstrate the possible options for pre-analysis adjustment: 
:doc:`Disaggregation </explanation/disaggregation>` via `PiWind Postcode 
<https://github.com/OasisLMF/OasisModels/tree/main/PiWindPostcode>`_, and Geocoding via 
`PiWind Pre Analysis <https://github.com/OasisLMF/OasisModels/tree/main/PiWindPreAnalysis>`_.

For more information on these model:

* :doc:`Disaggregation </explanation/disaggregation>`

* :doc:`Geocoding </explanation/geocoding>`
