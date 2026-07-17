Geocoding
=========

On this page
------------

* :ref:`introduction_geocoding`
* :ref:`how_it_works_geocoding`
* :ref:`example_geocoding`
* :ref:`links_geocoding`

|

.. _introduction_geocoding:

Introduction
************

----

The Oasis modelling platform is designed to model individual buildings with known locations and vulnerability attributes. However, 
this exposure data can sometimes be missing vital location data – a situation which is particularly true in the developing world.

Incomplete exposure data can negatively affect the performance when a model run, and the consequent uncertainty from this is not 
always captured in loss output. To overcome this issue, models can be integrated with geocoding in the pre-analysis step. This 
feature fills in incomplete OED fields for addresses in the exposure location data, based on the available information about an 
address provided.

An example of the geocoding step can be seen in our toy model `PiWind Pre Analysis 
<https://github.com/OasisLMF/OasisModels/tree/develop/PiWindPreAnalysis>`_, which is available for use from `here 
<https://github.com/OasisLMF/OasisModels/tree/develop/PiWindPreAnalysis>`_.

.. note::
    Oasis does not do any of the geocoding for this model. The geocoding aspect is performed by `Precisely's Geocode API 
    <https://docs.precisely.com/docs/sftw/precisely-apis/main/en-us/webhelp/apis/Geocode/geocode_desc.html>`_ and is integrated 
    into the model using the pre-analysis adjustment functionality.

|

.. _how_it_works_geocoding:

How it works
************

----

Currently in the Oasis platform, as of August 2022, exposure must be converted into detailed data before being imported into the 
platform for analysis. The format for this data is one building per row in the location file. This can be done outside of the 
system, or alternatively the model developer, as part of the Oasis model assets, can provide a pre-analysis routine to generate a 
modified OED location file from an input OED location file.

Pre-analysis routines provide flexibility to manipulate the OED input files before the model is run, for 
augmentation as required by the model. They are completely customisable ofr changing input files in whatever way a user requires. 
An example pre-analysis ‘hook’ for the PiWind model can be found `here 
<https://github.com/OasisLMF/OasisPiWind/blob/main/src/exposure_modification/exposure_pre_analysis_example.py>`_.

The Oasis toy model `PiWind Pre Analysis <https://github.com/OasisLMF/OasisModels/tree/develop/PiWindPreAnalysis>`_ uses 
the pre-analysis feature to integrate an external geocoder. The purpose of the geocoding is to ‘complete’ the location data in 
the OED input by calculating the values for any empty fields for addresses in the location file, that would hinder the performance 
of the model if left incomplete.

|

Geocoding for latitude and longitude
####################################

|

In the OED input, the typical location fields that define an address are: CountryCode, PostalCode, City, StreetAddress, Latitude, 
and longitude. The fields are not limited to this, but these listed describe the physical location of an address. If one or more 
of these fields are missing, the ability of the model to correctly assign these addresses can be affect; there could be multiple 
streets with the same name in a country, or multiple addresses that are the same in different countries, etc. This is typically 
not an issue when only one field empty, as latitude and longitude, or another field, will dispel any ambiguity in the accuracy of 
the address. However, if more are missing, especially latitude and longitude, issues can arise. 

The geocoding pre-analysis step overcomes this by calculating any incomplete OED fields in preparation for the model run. It uses 
Precisely’s Geocode API to achieve this, which is built into the script for the pre-analysis step. More information on this 
service offered by Precisely can be found `here 
<https://docs.precisely.com/docs/sftw/precisely-apis/main/en-us/webhelp/apis/Geocode/geocode_desc.html>`_.

This script takes the OED location file and runs through it line-by-line, checking the fields in each address. If an address has 
empty values for its latitude and longitude fields, the remaining location data (what is available from CountryCode, PostalCode, 
City, StreetAddress) is sent off for geocoding.

This geocoding step takes in the incomplete address data, checks it against its extensive database of locations, and returns a 
detailed response of information about that address – this includes its latitude and longitude. These two values are then inserted 
into their corresponding empty fields to make that address complete. In addition, two new OED fields are added that indicate the 
presence of geocoding: Geocoder and GeocodeQuality. Geocoder is set to ‘Precisely’ by default, as this is what the pre-analysis 
step uses. GeocodeQuality is a value between 0 and 1 that indicates the precision of the geocoded values (e.g. 80% is entered as 
0.8). More information on how quality is quantified can be found `here 
<https://docs.precisely.com/docs/sftw/precisely-apis/main/en-us/webhelp/apis/Geocode/Geocode/LI_GGM_Geo_ReturnValuesDefaults.html>`_. 

Once this has ran through the entire location file, all addresses should be complete with every field accounted for with 
corresponding values. This exposure data is then written over the old, incomplete file and is then ready for model run.

|

.. _example_geocoding:

Example of geocoding
********************

----

Below is example of the geocode pre-analysis step that demonstrates latitude and longitude fields being completed when they have 
not been provided in the original location file. The table below shows a location file with empty entries for latitude and 
longitude.

.. csv-table::
   :header: PortNumber,AccNumber,LocNumber,IsTenant,BuildingID,CountryCode,Latitude,Longitude,StreetAddress,PostalCode,OccupancyCode,ConstructionCode,LocPerilsCovered,BuildingTIV,OtherTIV,ContentsTIV,BITIV,LocCurrency,OEDVersion

   1,A11111,100030535219,1,1,GB,,,1 BENTLEY STREET,LE13 1LY,1120,5204,WSS,150000,0,37500,15000,GBP,2.0.0
   1,A11111,100030535220,1,1,GB,,,2 BENTLEY STREET,LE13 1LY,1120,5204,WW1,150000,0,37500,15000,GBP,2.0.0
   1,A11111,100030535221,1,1,GB,52.7658503,-0.8832562,3 BENTLEY STREET,LE13 1LY,1120,5204,WW1,150000,0,37500,15000,GBP,2.0.0
   1,A11111,100030535222,1,1,GB,52.7659084,-0.882736,4 BENTLEY STREET,LE13 1LY,1120,5204,WW1,150000,0,37500,15000,GBP,2.0.0

|

The geocode pre-analysis step identifies that the address in this row are incomplete and sends it for geocoding. The geocoder 
returns the values for the latitude and longitude, and these are inserted to this row to complete the address data, along with the 
geocode fields(the addresses that aren't geocoded are blank for these two fields).

.. csv-table::
   :header: PortNumber,AccNumber,LocNumber,IsTenant,BuildingID,CountryCode,Latitude,Longitude,StreetAddress,PostalCode,OccupancyCode,ConstructionCode,LocPerilsCovered,BuildingTIV,OtherTIV,ContentsTIV,BITIV,LocCurrency,OEDVersion,Geocoder,GeocodeQuality

   1,A11111,100030535219,1,1,GB,52.7657126,-0.8831089,1 BENTLEY STREET,LE13 1LY,1120,5204,WSS,150000.0,0.0,37500.0,15000.0,GBP,2.0.0,Precisely,0.05
   1,A11111,100030535220,1,1,GB,52.7657510,-0.8829107,2 BENTLEY STREET,LE13 1LY,1120,5204,WW1,150000.0,0.0,37500.0,15000.0,GBP,2.0.0,Precisely,0.05
   1,A11111,100030535221,1,1,GB,52.7658503,-0.8832562,3 BENTLEY STREET,LE13 1LY,1120,5204,WW1,150000.0,0.0,37500.0,15000.0,GBP,2.0.0,,
   1,A11111,100030535222,1,1,GB,52.7659084,-0.882736,4 BENTLEY STREET,LE13 1LY,1120,5204,WW1,150000.0,0.0,37500.0,15000.0,GBP,2.0.0,,

|

This data is then written over the old location file to be processes by the model.

|

.. _links_geocoding:

Links for further information
*****************************

----

* The example model PiWind Pre Analysis, with geocoding, can be found `here 
  <https://github.com/OasisLMF/OasisModels/tree/develop/PiWindPreAnalysis>`_.

* More information on Precisely’s geocoding API can be found `here 
  <https://docs.precisely.com/docs/sftw/precisely-apis/main/en-us/webhelp/apis/Geocode/geocode_desc.html>`_.
