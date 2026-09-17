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
keys/lookup service, it can be run across multiple processes. By default it is chunked the same
way the keys/lookup service is - the hook is instantiated once per chunk and called with a subset
of ``exposure_data``, and the resulting location/account dataframes are merged back together
afterwards. Unlike the keys/lookup service, chunks are formed from unique ``(PortNumber,
AccNumber)`` combinations rather than individual locations, so a single account's location rows
are never split across two chunks.

Three parameters control this behaviour, each mirroring an equivalent keys/lookup parameter and
defaulting to that parameter's resolved value if not set explicitly:

* ``exposure_pre_analysis_multiprocessing`` (defaults to ``lookup_multiprocessing``, i.e. ``True``)
* ``exposure_pre_analysis_num_processes`` (defaults to ``lookup_num_processes``, i.e. auto-sized)
* ``exposure_pre_analysis_num_chunks`` (defaults to ``lookup_num_chunks``, i.e. auto-sized)

Because the framework has no visibility into what a pre-analysis hook actually does, chunking is
only safe for hooks that operate independently per location/account - the intended use cases
described above (geocoding, disaggregation, exposure enhancement). A hook is **not** compatible
with the default chunked behaviour if it:

* needs to see locations/accounts outside of a single account (e.g. whole-portfolio
  aggregation or optimisation), or
* reads or modifies ``exposure_data.ri_info``/``exposure_data.ri_scope`` (these are not
  chunked or merged back - only the main process's copy is kept), or
* has side effects on shared files under ``input_dir`` that aren't safe for multiple
  processes to write concurrently.

Set ``exposure_pre_analysis_multiprocessing=False`` to disable chunking for such a model.

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
