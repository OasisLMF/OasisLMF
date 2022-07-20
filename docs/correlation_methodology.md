Correlation methodology
=======================

## Overview 
This document explains the approach to modelling correlation within the Oasis model execution framework. This follows consultation with a steering group formed of model developers within reinsurance companies and brokers, and third-party commercial model providers.

## Introduction
In large catastrophes, there is a tendency for losses across multiple locations to be correlated, meaning relatively high losses across locations or low losses across locations tend to occur together. The correlation is stronger the closer together the exposures are located. 

Two main reasons why this would be the case for buildings situated close together are;
* They experience similar a hazard intensity in an event; flood depth, windspeed etc.
* They have similar vulnerability characteristics (such being built by the same developer) and similar modes of failure/types of damage given the hazard intensity.

Correlation increases the range of potential claims at a portfolio level and particularly for large, rare events, a model can significantly underestimate uncertainty and extreme losses if this correlation is not captured. It is therefore desirable to allow modellers and users the ability to express views on the degree of spatial correlation in Oasis so that the effect on portfolio risk can be modelled.

## Ground up loss methodology overview
The Oasis kernel performs Monte Carlo sampling of ground loss from the effective damage probability distribution using uniform random numbers. The method is a linear interpolation of each random number against the cumulative distribution function to compute a damage ratio (value between 0 and 1) which is multiplied by the total insured value of the exposed asset to produce ground up loss samples. This is done at the individual building coverage level, for each modelled peril.

The effective damage distribution is a probabilistic summation of all possible levels of hazard intensity (including the chance of zero impact) and all resulting possible levels of damage for a given event and exposure. It therefore represents all sources of ‘secondary uncertainty’ including the overall chance of damage, to a particular exposure given the event.

The approach to sampling correlated losses described here involves generating correlated random numbers that are used to sample a damage factor from each building’s effective damage distribution, i.e., from the secondary uncertainty distribution around a mean value, per peril.  The correlation factor represents a mixture of the two sources of correlation described above (which may therefore be harder to specify than if the two sources of correlation were modelled separately). 

#### Random sampling of two effective damage distributions with similar (left) and different (right) random numbers
![alt text](sampling-charts.jpg "Random sampling of two effective damage distributions with similar (left) and different (right) random numbers.")

How correlated the resulting damage samples will be depends on the following factors;

* How high the correlation factor is (how close together the uniform random numbers tend to be)
* How similar the effective damage distributions are. This is driven both by the hazard correlation (e.g., similar probability of hazard/chance of loss) and by vulnerability correlation (similar shaped conditional damage distributions).

## Scope
Only correlation within the context of a single event is considered, there is no attempt to correlate losses between events over time or take physical conditions into account prior to the event.

We don’t use correlation factors to sample correlated coverage losses within the same site, and the default assumption of full correlation will remain (unless overridden). However, the general approach may be extended to coverage correlation at a later stage.

There can be multiple hazards in an event which can give rise to loss. There may be the same peril type, for example flooding from different sources such as river flood / heavy rainfall, or there may be completely different perils and types of damage (e.g., high wind speeds causing roof damage, and flooding causing ground floor damage). 

Limited modelling of peril correlation will be supported as follows: model developers can group the same peril types together to fully correlate them at a location or treat different peril types as independent from each other. Each model defines their own hazard intensity measure(s) for each peril and therefore blanket correlation rules for general peril types are not proposed.

The correlation functionality described here is available use for any standard Oasis model. Complex models that use bespoke correlation methodologies can continue to be used as before, or the new functionality could be incorporated within the complex model wrapper by the model provider. 

## Correlation functionality prior to v1.27

Items are the smallest calculation unit in Oasis, and they represent the loss to an individual coverage of a single physical risk from a particular peril. Items are grouped together into item groups by assigning a common group_id.

The group_id is generated using a hashing function based on information from the OED location file. It ensures that the group_id will be the same regardless of what set of exposure files the risk appears in, leading to repeatable losses for the risk across different analyses for the same model. 

An item group by default represents a single physical risk with one or more coverages and perils at a location, unless;

* specified differently in model settings json (e.g., group by areaperil, group by postcode, etc.) 
* overridden by the user through the CorrelationGroup field in the OED location file
* the risk is disaggregated and multiple subrisks are assigned the same group_id (using the exposure pre-analysis options)

All items that share the same group_id are fully correlated, meaning that the same random number is used to sample loss from the respective effective damage cdfs. This includes multiple coverages and perils of a single physical risk, and multiple coverages and perils of multiple subrisks (in the case of disaggregated risks for example, or generally whenever the group_id is set to represent multiple physical risks).

#### Example item groups for two physical risks

| Item_id | Peril code   |  Coverage   | Location     | Group_id     |
|:--------|--------------|-------------| -------------|-------------:|
| 1       | WTC          |  Building   | 1            |     567      |
| 2       | WSS          |  Building   | 1            |     567      |
| 3       | WTC          |  Contents   | 1            |     567      |
| 4       | WSS          |  Contents   | 1            |     567      |
| 5       | WTC          |  Building   | 2            |     123      |
| 6       | WSS          |  Building   | 2            |     123      |
| 7       | WTC          |  Contents   | 2            |     123      |
| 8       | WSS          |  Contents   | 2            |     123      |

Note that the group_id values are arbitrary because they are generated from a hashing function.

In this example;
* Item_ids 1-4 are fully correlated
* Item_ids 5-8 are fully correlated
* Group_id 567 items are fully independent from group_id 123 items.

#### Example item group for a disaggregated risk with two sub-risks

| Item_id | Peril code   |  Coverage   | Location     | Group_id     | Subrisk |
|:--------|--------------|-------------| -------------|--------------|--------:|
| 1       | WTC          |  Building   | 1            |     567      | 1       |
| 2       | WSS          |  Building   | 1            |     567      | 1       |
| 3       | WTC          |  Contents   | 1            |     567      | 1       |
| 4       | WSS          |  Contents   | 1            |     567      | 1       |
| 5       | WTC          |  Building   | 2            |     567      | 2       |
| 6       | WSS          |  Building   | 2            |     567      | 2       |
| 7       | WTC          |  Contents   | 2            |     567      | 2       |
| 8       | WSS          |  Contents   | 2            |     567      | 2       |

All item_ids are fully correlated in this example.

Note that disaggregation may be performed using the exposure_pre_analysis options.

## Correlation functionality v1.27 and later

### Methodology overview

This functionality builds on the existing functionality and enables model developers to;

• Specify **peril correlation groups** for the perils covered in their model to either fully correlate damage from different perils, or make damage from different perils independent, at each location, per event.
• For each peril correlation group, specify a default **global correlation factor** to correlate damage across locations per event.

It enables users to vary the global correlation factors for each peril correlation group in the model as a runtime option.

A peril is represented by an OED single peril code and a group of perils is one or more perils associated with a single event. The intensity measures used to represent hazard under each peril code may vary from model to model.

### Peril correlation groups and factors

Peril correlation group and factors are specified in model settings and used to; 

* seed a set of random numbers for each peril correlation group which are independent of each other
* set a correlation factor among item groups within each peril group. 

If specified, the peril correlation group_id should be an integer starting from 1 and it will be used as part of the hashing function to generate the item group_ids. A group_id will therefore be distinct for each item group and a peril correlation group.

In model settings, the covered perils for a model are currently specified within the ‘lookup_settings’ attribute, which could be extended to include the peril correlation group as follows;

#### Example 1 - Tropical Cyclone with storm surge

##### Model_settings.json

```
“lookup_settings”:[
  {“id”:”WSS”,”desc”:”Single Peril: Storm Surge”,”peril_correlation_group”:1},
  {“id”:”WTC”,”desc”:”Single Peril: Tropical Cyclone”,”peril_correlation_group”:2 },
],
“correlation_ settings”:[
  {”peril_correlation_group”:1, “correlation_value”:0.5},
  {”peril_correlation_group”:2, “correlation_value”:0.2},
]
```
The WTC peril code here represents tropical cyclone wind hazard measured by wind speed, whereas WSS represents inundation depth from waves on coastal areas. These are different peril types and damage arising from each within the same event may be assumed to be independent by the modeller, and the perils can be assigned to different peril correlation groups. For a given location, Wind and Storm surge items will have independent random numbers drawn.

Using the single physical risk example from above, the peril correlation group ids 1 and 2 are used in the hashing function to generate different item group ids.

#### Example 1 item group data

| Item_id | Peril code   |  Coverage   | Location     | Group_id     |
|:--------|--------------|-------------| -------------|-------------:|
| 1       | WTC          |  Building   | 1            |     131      |
| 2       | WSS          |  Building   | 1            |     76       |
| 3       | WTC          |  Contents   | 1            |     131      |
| 4       | WSS          |  Contents   | 1            |     76       |
| 5       | WTC          |  Building   | 2            |     237      |
| 6       | WSS          |  Building   | 2            |     98       |
| 7       | WTC          |  Contents   | 2            |     237      |
| 8       | WSS          |  Contents   | 2            |     98       |

* Item_ids 1 and 3 are fully correlated
* Item_ids 2 and 4 are fully correlated
* Item_ids 1 and 3 are independent from 2 and 4.

* Item_ids 5 and 7 are fully correlated
* Item_ids 6 and 8 are fully correlated
* Item_ids 5 and 7 are independent from 6 and 8.

* Group_id 131 items are 20% correlated with group_id 237 items
* Group_id 76 items are 50% correlated with group_id 98 items

#### Example 2 - Multiple flood hazards

##### Model_settings.json

```
“lookup_settings”:[
  {“id”:”WSS”,”desc”:”Single Peril: Storm Surge”,”peril_correlation_group”:1}
  {“id”:”ORF”,”desc”:”Single Peril: River / Fluvial Flood”,”peril_correlation_group”:1},
  {“id”:”OSF”,”desc”:”Single Peril: Flash / Surface / Pluvial Flood”,”peril_correlation_group”:1},
],
“correlation_ settings”:[
  {”peril_correlation_group”:1, “correlation_value”:0.5}
]
```
River flood (ORF) and Surface/Pluvial flooding (OSF), and storm surge (WSS) hazard are all the same peril type and use the same intensity measure, inundation depth, so they are assigned to the same peril group.

This means that;

1)  river flood, flash flood and storm surge losses for a particular physical risk are fully correlated (because they belong to the same peril correlation group). 
2)  the correlation across all locations (item groups) is 0.5

#### Example 2 item group data

| Item_id | Peril code   |  Coverage   | Location     | Group_id     |
|:--------|--------------|-------------| -------------|-------------:|
| 1       | ORF          |  Building   | 1            |     65       |
| 2       | OSF          |  Building   | 1            |     65       |
| 3       | WSS          |  Building   | 1            |     65       |
| 4       | ORF          |  Contents   | 1            |     65       |
| 5       | OSF          |  Contents   | 1            |     65       |
| 6       | WSS          |  Contents   | 1            |     65       |
| 7       | ORF          |  Building   | 2            |     12       |
| 8       | OSF          |  Building   | 2            |     12       |
| 9       | WSS          |  Building   | 2            |     12       |
| 10      | ORF          |  Contents   | 2            |     12       |
| 11      | OSF          |  Contents   | 2            |     12       |
| 12      | WSS          |  Contents   | 2            |     12       |


* Item_ids 1,2,3,4,5,6 are fully correlated
* Item_ids 7,8,9,10,11,12 are fully correlated
* Group_id 65 items are 50% correlated with group_id 12 items

### Correlation options for the user (**not implemented**)

The user can override correlation factors for each peril correlation group by mirroring the correlation settings in the model settings and changing the factors either directly in the analysis settings json in the MDK or through UI options.

analysis_settings.json

```
“correlation_ settings”:[
{”peril_correlation_group”:1, “correlation_value”:0.7},
{”peril_correlation_group”:2, “correlation_value”:0.4},
]
```

### Defaults
If peril correlation groups and correlation factors are not specified in model settings, the generated exposure input files and the model results will remain exactly the same. Results will be backwards compatible with versions prior to 1.27.

The implicit assumptions in this case are that 1) all peril codes belong to the same peril correlation group and 2) a correlation value of 0 between item groups. This means losses will be fully correlated across all coverages and perils within an item group, and there will be zero correlation between item groups. 

### Method of generating correlated random numbers

A one-factor Gaussian copula generates correlated random numbers across group_ids for each peril group and event. 

For an event, for each peril correlation group k and sample j, a random number Y_jk  ~ N(0,1) is generated as the correlated random variable across groups. The set of random numbers is seeded from the event and peril correlation group k so that it is repeatable.

For each event, sample j and group_id ik (ik = i locations times k peril groups), one random number, X_ijk  ~ N(0,1) is generated as the noise/uncorrelated variable. The group_id is hashed from the location details and the peril correlation group id so that the random numbers are repeatable for the same item group and peril correlation group across analyses.

The dependent variable Z_ijk  ~ N(0,1) for peril correlation group k, sample j and group_id ik is

![alt text](eqn1.jpg "One factor Gaussian copula")

Where ρ_k is the input correlation factor for peril correlation group k.

The normal inverse function is used to transform independent uniform random numbers generated from the chosen RNG function (Mersenne Twister / Latin Hypercube) into the normally distributed random variables, X_ijk and Y_jk. The cumulative normal distribution function is used to transform the dependent normally distributed Z_ijk values to the uniform distribution, which are the correlated uniform random numbers to use for damage interpolation of the cdf.

