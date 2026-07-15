Oasis Models
============

On this page
------------

* :ref:`intro_models`
* :ref:`piwind_models`



|

.. _intro_models:

Oasis Models
------------

Oasis makes available a number of example models to demonstrate how different functionality options are implemented in 
Oasis. The available models are:

----

Deterministic Model
*******************

This is a single event model which allows users to apply deterministic losses to a portfolio, defining the damage factors 
in the OED location file. It is similar to the ``exposure`` feature in the oasislmf package, but can be deployed as a model in 
it's own right to model deterministic losses which can then be passed through the Oasis financial module.

This model is availible to use `here <https://github.com/OasisLMF/OasisModels/tree/develop/DeterministicModel>`_.

----

Paris Windstorm
****************

This is very small, single peril model used for demonstration of how to build a simple model in Oasis.

This model is availible to use `here <https://github.com/OasisLMF/OasisModels/tree/develop/ParisWindstorm>`_.

----

PiWind
******

This is the original test model in Oasis and is an example of a multi-peril model implementation representing ficticious 
events with wind and flood affecting the Town of Melton Mowbray in England.

This model is availible to use `here <https://github.com/OasisLMF/OasisModels/tree/develop/PiWind>`_.

More information on this model can be found here: :ref:`piwind_models`

----

PiWind Absolute Damage
**********************

This model expands upon the PiWind model with the absolute damage option. This option allows model providers to include 
absolute damage amounts rather than damage factors in the damage bin dictionary. If the damage factors are less than or 
equal to 1 in the damage bin dictionary, the factor will be applied as normal during the loss calculation, by applying the 
sampled damage factor to the TIV to give a simulated loss; but with absolute damage factors, where the factor is greater 
than 1, the TIV is not used in the calculation at all, but rather the absolute damage is applied as the loss.

This model is availible to use from `here <https://github.com/OasisLMF/OasisModels/tree/develop/PiWindAbsoluteDamage>`_.

----

PiWind Complex Model
********************

This is a version of the PiWind model which uses the complex model integreation approach to generate ground up losses in a 
custoim module, which then sits in the workflow and replaces the standard ground up loss calculation from Oasis.

This model is availible to use from `here <https://github.com/OasisLMF/OasisModels/tree/develop/PiWindComplexModel>`_.

----

PiWind Postcode
***************

This is a variant of the original PiWind model designed for running exposures whose locations are known at postcode level 
rather than by latitude and longitude. This model demonstrates the disaggregation features of Oasis.

This model is availible to use `here <https://github.com/OasisLMF/OasisModels/tree/develop/PiWindPostcode>`_.

----

PiWind Post Loss Amplification
******************************

This is a version of the PiWind model with post loss amplification factors applied. Major catastrophic events can 
give rise to inflated and/or deflated costs depending on that specific situation. To account for this, the ground up 
losses produced by the GUL calculation component are multiplied by post loss amplification factors, by the component 
plapy.

This model is availible to use `here <https://github.com/OasisLMF/OasisModels/tree/develop/PiWindPostLossAmplification>`_.

----

PiWind Post Pre Analysis
************************

This model builds upon the original PiWind model with a pre-analysis adjustment hook. This step allows the user to modify input 
files before they are processed in the analysis. This functionality is utilised by this model by implementing an external geocoder: 
this checks the location data before it is analysed for any addresses that are missing OED location data. If an address is found t
o be incomplete, it is geocoded to fill these gaps.

This model is availible to use `here <https://github.com/OasisLMF/OasisModels/tree/develop/PiWindPreAnalysis>`_.

----

PiWind Single Peril
*******************

This is a simplified variant of the original PiWind model which has single peril (wind only) and would be a good basis for
a single peril model in Oasis.

This model is availible to use `here <https://github.com/OasisLMF/OasisModels/tree/develop/PiWindSinglePeril>`_.

----

PiWind Vulnerability Adjustments
*********************************

This model showcases how specific adjustments to the vulnerabilities can be introduced in the ``analysis_settings.json``
file. Three adjustment methods are demonstrated: scaling vulnerability curves by a factor, replacing an entire
vulnerability curve, and applying adjustments at aggregate level. See vulnerability adjustments for full details.

This model is availible to use `here <https://github.com/OasisLMF/OasisModels/tree/develop/PiWindVulnerability>`_.

----

Dynamic Footprint
*****************

This is a version of the PiWind model which demonstrates the dynamic (stochastic) footprint feature. Instead of using
pre-computed static hazard footprint files, the model generates footprint data dynamically per event at runtime using a
custom footprint module. This approach is suited to models where the hazard field is generated programmatically or
where stochastic hazard selection is required.

This model is availible to use `here <https://github.com/OasisLMF/OasisModels/tree/develop/DynamicFootprint>`_.

----

PiWind S3
*********

This variant of PiWind demonstrates running an Oasis model with model data stored on AWS S3. It contains the minimal
set of configuration files and options required to use S3 as the model data backend via AWS credentials.

This model is availible to use `here <https://github.com/OasisLMF/OasisModels/tree/develop/PiWindS3>`_.

----

PiWind Azure
************

This variant of PiWind demonstrates running an Oasis model with model data stored on Azure Blob Storage. It contains
the minimal configuration required to use Azure as the model data backend.

This model is availible to use `here <https://github.com/OasisLMF/OasisModels/tree/develop/PiWindAzure>`_.

----

.. note::
    More information about these models can be found `here <https://github.com/OasisLMF/OasisModels/tree/develop>`_.

|

.. _piwind_models:

PiWind - toy model
------------------

----

Oasis has developed a toy model, PiWind, available `here <https://github.com/OasisLMF/OasisPiWind>`_. PiWind is a wind storm 
model for a small area of the UK. The data is mocked up to illustrate the Oasis data formats and functionality, and is not 
meant to be a usable risk model. The PiWind toy model is availible to use from `here <https://github.com/OasisLMF/
OasisModels/tree/develop/PiWind>`_.

There are three main components to a catastrophe risk model deployed in Oasis. A fuller discussion of the components of a 
hazard model can be found in :doc:`/explanation/modelling-methodology`.

**Hazard footprint data:**
    This holds the hazard intensity data for each event in the stochastic event set. The hazard intensity footprint is
    defined on a model specific geospatial grid, and each grid cell is assigned a unique identifier.Note that a model may 
    cover multiple perils, each with a different overlaid area peril grid. For example, a hurricane model will usually 
    cover both wind and storm surge perils. Each peril has a defined hazard intensity measure, such as wind speed in metres 
    per second.The Oasis Platform allows uncertainty to be specified in the hazard intensity measure in a particular grid 
    cell for each event.

**Vulnerability data:**
    This holds curves that define the distribution of damage as a proportion of replacement value given the level of hazard 
    intensity.Different curves as specified for structures with different building characteristics.For example, a 
    wood-framed building will have a different vulnerability to wind damage as compared to a building of concrete 
    construction.The curves also define the uncertainty in damage at different hazard levels.The Oasis Platform does not 
    make any assumptions about the form of the damage distributions and represents them all as discrete distributions.

**Keys lookup logic:**
    This is model specific logic that maps a set of exposure attributes into the model specific grid and vulnerability type.
    A unique mapping is made for each location, coverage and peril combination. The lookup also provides informative 
    messages about any exposures that will not be modelled.For example, an exposure may not be modelled if there is 
    insufficiently detailed address information or if the exposure is not within the geographic scope of the model.
    
The PiWind model is a very small example model, so it's files can be saved to a GitHub repository and easily queried.For 
real models the data sets can get much larger, in some cases more than 1 TB for a single model.The following link is to a 
Jupyter notebook that illustrates the setup of the PiWind model and how it can be ran using the Oasis MDK: 
`Running PiWind <https://mybinder.org/v2/gh/OasisLMF/OasisPiWind/master>`_.