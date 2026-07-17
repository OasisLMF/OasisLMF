Pre Analysis Adjustments
========================

On this page
------------

* :ref:`introduction_paa`
* :ref:`how_it_works_paa`
* :ref:`example_models_paa`

|

.. _introduction_paa:

Introduction
************

----

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

----

Currently in the Oasis platform, as of August 2022, exposure must be converted into detailed data before being imported into the 
platform for analysis. The format for this data is one building per row in the location file. This can be done outside of the 
system, or alternatively the model developer, as part of the Oasis model assets, can provide a pre-analysis routine to generate a 
modified OED location file from an input OED location file.

The purpose of a pre-analysis routines is to provide flexibility to manipulate the OED input files before the model is run, for 
augmentation as required by the model. An example pre-analysis ‘hook’ for the PiWind model can be found `here 
<https://github.com/OasisLMF/OasisPiWind/blob/main/src/exposure_modification/exposure_pre_analysis_example.py>`_.

|

.. _example_models_paa:

Example models
**************

----

Oasis currently offers two toy models that demonstrate the possible options for pre-analysis adjustment: 
:doc:`Disaggregation </explanation/disaggregation>` via `PiWind Postcode 
<https://github.com/OasisLMF/OasisModels/tree/develop/PiWindPostcode>`_, and Geocoding via 
`PiWind Pre Analysis <https://github.com/OasisLMF/OasisModels/tree/feature/geocode/PiWindPreAnalysis>`_.

For more information on these model:

* :doc:`Disaggregation </explanation/disaggregation>`

* Geocoding