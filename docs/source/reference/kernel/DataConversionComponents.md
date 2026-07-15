![alt text](/_static/images/kernel/banner.jpg "banner")
# 4.4 Data conversion components <a id="dataconversioncomponents"></a>

The following components convert input data in csv format to the binary format required by the calculation components in the reference model;

**Static data**
* **[aggregatevulnerabilitytobin](#aggregatevulnerability)** converts the aggregate vulnerability data. 
* **[damagebintobin](#damagebins)** converts the damage bin dictionary. 
* **[footprinttobin](#footprint)** converts the event footprint.
* **[lossfactorstobin](#lossfactors)** converts the lossfactors data. 
* **[randtobin](#rand)** converts a list of random numbers. 
* **[vulnerabilitytobin](#vulnerability)** converts the vulnerability data.
* **[weightstobin](#weights)** converts the weights data.

A reference [intensity bin dictionary](#intensitybins) csv should also exist, although there is no conversion component for this file because it is not needed for calculation purposes. 

**Input data**
* **[amplificationtobin](#amplifications)** converts the amplifications data.
* **[coveragetobin](#coverages)** converts the coverages data.
* **[ensembletobin](#ensemble)** converts the ensemble data.
* **[evetobin](#events)** converts a list of event_ids.
* **[itemtobin](#items)** converts the items data.
* **[gulsummaryxreftobin](#gulsummaryxref)** converts the gul summary xref data.
* **[fmpolicytctobin](#fmpolicytc)** converts the fm policytc data.
* **[fmprogrammetobin](#fmprogramme)** converts the fm programme data.
* **[fmprofiletobin](#fmprofile)** converts the fm profile data.
* **[fmsummaryxreftobin](#fmsummaryxref)** converts the fm summary xref data.
* **[fmxreftobin](#fmxref)** converts the fm xref data.
* **[occurrencetobin](#occurrence)** converts the event occurrence data.
* **[returnperiodtobin](#returnperiod)** converts a list of return periods.
* **[periodstobin](#periods)** converts a list of weighted periods (optional).
* **[quantiletobin](#quantile)** converts a list of quantiles (optional).

These components are intended to allow users to generate the required input binaries from csv independently of the original data store and technical environment. All that needs to be done is first generate the csv files from the data store (SQL Server database, etc).

The following components convert the binary input data required by the calculation components in the reference model into csv format;

**Static data**
* **[aggregatevulnerabilitytocsv](#aggregatevulnerability)** converts the aggregate vulnerability data. 
* **[damagebintocsv](#damagebins)** converts the damage bin dictionary. 
* **[footprinttocsv](#footprint)** converts the event footprint.
* **[lossfactorstocsv](#lossfactors)** converts the lossfactors data. 
* **[randtocsv](#rand)** converts a list of random numbers. 
* **[vulnerabilitytocsv](#vulnerability)** converts the vulnerability data.
* **[weightstocsv](#weights)** converts the weights data.

**Input data**
* **[amplificationtocsv](#amplifications)** converts the amplifications data.
* **[coveragetocsv](#coverages)** converts the coverages data.
* **[ensembletocsv](#ensemble)** converts the ensemble data.
* **[evetocsv](#events)** converts a list of event_ids.
* **[itemtocsv](#items)** converts the items data.
* **[gulsummaryxreftocsv](#gulsummaryxref)** converts the gul summary xref data.
* **[fmpolicytctocsv](#fmpolicytc)** converts the fm policytc data.
* **[fmprogrammetocsv](#fmprogramme)** converts the fm programme data.
* **[fmprofiletocsv](#fmprofile)** converts the fm profile data.
* **[fmsummaryxreftocsv](#fmsummaryxref)** converts the fm summary xref data.
* **[fmxreftocsv](#fmxref)** converts the fm xref data.
* **[occurrencetocsv](#occurrence)** converts the event occurrence data.
* **[returnperiodtocsv](#returnperiod)** converts a list of return periods.
* **[periodstocsv](#returnperiod)** converts a list of weighted periods (optional).
* **[quantiletocsv](#quantile)** converts a list of quantiles (optional).

These components are provided for the convenience of viewing the data and debugging.

## Static data

 <a id="aggregatevulnerability"></a>
### aggregate vulnerability
***
The aggregate vulnerability file is  required for the gulmc component. It contains the conditional distributions of damage for each intensity bin and for each vulnerability_id. This file must have the following location and filename;

* static/aggregate_vulnerability.bin

##### File format

The csv file should contain the following fields and include a header row.


| Name                           | Type   |  Bytes | Description                                   | Example     |
|:-------------------------------|--------|--------| :---------------------------------------------|------------:|
| aggregate_vulnerability_id     | int    |    4   | Oasis vulnerability_id                        |     45      |
| vulnerability_id               | int    |    4   | Oasis vulnerability_id                        |     45      |

If this file is present, the weights.bin or weights.csv file must also be present. The data should not contain nulls.

##### aggregatevulnerabilitytobin
```
$ aggregatevulnerabilitytobin < aggregate_vulnerability.csv > aggregate_vulnerability.bin
```

##### aggregatevulnerabilitytocsv
```
$ aggregatevulnerabilitytocsv < aggregate_vulnerability.bin > aggregate_vulnerability.csv
```

[Return to top](#dataconversioncomponents)

 <a id="damagebins"></a>
### damage bin dictionary
***
The damage bin dictionary is a reference table in Oasis which defines how the effective damageability cdfs are discretized on a relative damage scale (normally between 0 and 1). It is required by getmodel and gulcalc and must have the following location and filename;

* static/damage_bin_dict.bin

##### File format

The csv file should contain the following fields and include a header row.


| Name              | Type   |  Bytes | Description                                                   | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------|------------:|
| bin_index         | int    |    4   | Identifier of the damage bin                                  |     1       |
| bin_from          | float  |    4   | Lower damage threshold for the bin                            |   0.01      |
| bin_to            | float  |    4   | Upper damage threshold for the bin                            |   0.02      |
| interpolation     | float  |    4   | Interpolation damage value for the bin (usually the mid-point)|   0.015     |

The interval_type field has been deprecated and will be filled with zeros in the binary file. It does not need to be included as the final column in the csv file:

| Name              | Type   |  Bytes | Description                                                    | Example     |
|:------------------|--------|--------| :--------------------------------------------------------------|------------:|
| interval_type     | int    |    4   | Identifier of the interval type, e.g. closed, open (deprecated)|    0        | 

The data should be ordered by bin_index ascending and not contain nulls. The bin_index should be a contiguous sequence of integers starting from 1.

##### damagebintobin
```
$ damagebintobin < damage_bin_dict.csv > damage_bin_dict.bin
```
Validation checks on the damage bin dictionary csv file are conducted by default during conversion to binary format. These can be suppressed with the -N argument:
```
$ damagebintobin -N < damage_bin_dict.csv > damage_bin_dict.bin
```

##### damagebintocsv
```
$ damagebintocsv < damage_bin_dict.bin > damage_bin_dict.csv
```
The deprecated interval_type field can be sent to the output using the -i argument:
```
$ damagebintocsv -i < damage_bin_dict.bin > damage_bin_dict.csv
```

[Return to top](#dataconversioncomponents)

 <a id="intensitybins"></a>
#### intensity bin dictionary

The intensity bin dictionary defines the meaning of the bins of the hazard intensity measure. The hazard intensity measure could be flood depth, windspeed, peak ground acceleration etc, depending on the type of peril. The range of hazard intensity values in the model is discretized into bins, each with a unique and contiguous bin_index listed in the intensity bin dictionary. The bin_index is used as a reference in the footprint file (field intensity_bin_index) to specify the hazard intensity for each event and areaperil.

This file is for reference only as it is not used in the calculation so there is no component to convert it to binary format.

The csv file should contain the following fields and include a header row.

| Name              | Type   |  Bytes | Description                                                   | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------|------------:|
| bin_index         | int    |    4   | Identifier of the intensity bin                               |     1       |
| bin_from          | float  |    4   | Lower intensity threshold for the bin                         |   56        |
| bin_to            | float  |    4   | Upper intensity threshold for the bin                         |   57        |
| interpolation     | float  |    4   | Mid-point intensity value for the bin                         |   0.015     |
| interval_type     | int    |    4   | Identifier of the interval type, e.g. closed, open            |   1         | 

The data should be ordered by bin_index ascending and not contain nulls. The bin_index should be a contiguous sequence of integers starting from 1.

[Return to top](#dataconversioncomponents)

<a id="footprint"></a>
### footprint
***
The event footprint is required for the getmodel component, as well as an index file containing the starting positions of each event block. These must have the following location and filenames;

* static/footprint.bin
* static/footprint.idx

##### File format
The csv file should contain the following fields and include a header row.

| Name               | Type   |  Bytes | Description                                                   | Example     |
|:-------------------|--------|--------| :-------------------------------------------------------------|------------:|
| event_id           | int    |    4   | Oasis event_id                                                |     1       |
| areaperil_id       | int    |    4   | Oasis areaperil_id                                            |   4545      |
| intensity_bin_index| int    |    4   | Identifier of the intensity bin                               |     10      |
| prob               | float  |    4   | The probability mass for the intensity bin between 0 and 1    |    0.765    | 
The data should be ordered by event_id, areaperil_id and not contain nulls. 

##### footprinttobin
```
$ footprinttobin -i {number of intensity bins} < footprint.csv
```
This command will create a binary file footprint.bin and an index file footprint.idx in the working directory. The number of intensity bins is the maximum value of intensity_bin_index.

Validation checks on the footprint csv file are conducted by default during conversion to binary format. These can be suppressed with the -N argument:
```
$ footprinttobin -i {number of intensity bins} -N < footprint.csv > footprint.bin
```

There is an additional parameter -n, which should be used when there is only one record per event_id and areaperil_id, with a single intensity_bin_index value and prob = 1. This is the special case 'no hazard intensity uncertainty'. In this case, the usage is as follows.

```
$ footprinttobin -i {number of intensity bins} -n < footprint.csv
```
Both parameters -i and -n are held in the header of the footprint.bin and used in getmodel.

The output binary and index file names can be explicitly set using the -b and
 -x flags respectively:

```
$ footprinttobin -i {number of intensity bins} -b {output footprint binary file name} -x {output footprint index file name} < footprint.csv
```

Both output binary and index file names must be given to use this option.

In the case of very large footprint files, it may be preferrable to compress the data as it is written to the binary file. Compression is performed using [zlib](https://zlib.net/) by issuing the -z flag. If the -u flag is used in addition, the index file will include the uncompressed data size. It is recommended to use the -u flag to prevent any memory issues during decompression with getmodel or footprinttocsv:

```
$ footprinttobin -i {number of intensity bins} -z < footprint.csv
$ footprinttobin -i {number of intensity bins} -z -u < footprint.csv
```
The value of the -u parameter is held in the same location as -n in the header of the footprint.bin file, left-shifted by 1.

##### footprinttocsv
```
$ footprinttocsv > footprint.csv
```
footprinttocsv requires a binary file footprint.bin and an index file footprint.idx to be present in the working directory.

Input binary and index file names can be explicitly set using the -b and -x flags respectively:

```
$ footprinttocsv -b {input footprint binary file name} -x {input footprint index file name} > footprint.csv
```

Both input binary and index file name must be given to use this option.

Footprint binary files that contain compressed data require the -z argument to be issued:

```
$ footprinttocsv -z > footprint.csv
```

[Return to top](#dataconversioncomponents)

<a id="lossfactors"></a>
### Loss Factors
***
The lossfactors binary maps the event_id/amplification_id pairs with post loss amplification factors, and is supplied by the model providers. The first 4 bytes are preserved for future use and the data format is as follows. It is required by Post Loss Amplification (PLA) workflow must have the following location and filename;

* static/lossfactors.bin

#### File format
The csv file should contain the following fields and include a header row.

| Name              | Type   |  Bytes | Description                                               | Example     |
|:------------------|--------|--------| :---------------------------------------------------------|------------:|
| event_id          | int    |    4   | Event ID                                                  |     1       |
| count             | int    |    4   | Number of amplification IDs associated with the event ID  |     1       |
| amplification_id  | int    |    4   | Amplification ID                                          |     1       |
| factor            | float  |    4   | The uplift factor                                         |     1.01    |

All fields must not have null values. The csv file will not contain the count, and the conversion tools will add/remove this count.

##### lossfactorstobin
```
$ lossfactorstobin < lossfactors.csv > lossfactors.bin
```

##### lossfactorstocsv
```
$ lossfactorstocsv < lossfactors.bin > lossfactors.csv
```

[Return to top](#dataconversioncomponents)

<a id="rand"></a>
### Random numbers 
***
A random number file may be provided for the gulcalc component as an option (using gulcalc -r parameter) The random number binary contains a list of random numbers used for ground up loss sampling in the kernel calculation. It must have the following location and filename;

* static/random.bin

If the gulcalc -r parameter is not used, the random number binary is not required and random numbers are instead generated dynamically during the calculation, using the -R parameter to specify how many should be generated. 

The random numbers can be imported from a csv file using the component randtobin.

##### File format
The csv file should contain a simple list of random numbers and include a header row.

| Name              | Type   |  Bytes | Description                    | Example     |
|:------------------|--------|--------| :------------------------------|------------:|
| rand              | float  |    4   | Number between 0 and 1         |  0.75875    |  


##### randtobin
```
$ randtobin < random.csv > random.bin
```

##### randtocsv

There are a few parameters available which allow the generation of a random number csv file as follows;

* -r convert binary float input to csv
* -g generate random numbers {number of random numbers}
* -S seed value {seed value}

```
$ randtocsv -r < random.bin > random.csv
$ randtocsv -g 1000000 > random.csv
$ randtocsv -g 1000000 -S 1234 > random.csv
```
The -S {seed value} option produces repeatable random numbers, whereas usage of -g alone will generate a different set every time.

[Return to top](#dataconversioncomponents)

<a id="vulnerability"></a>
### vulnerability
***
The vulnerability file is  required for the getmodel component. It contains the conditional distributions of damage for each intensity bin and for each vulnerability_id. This file must have the following location and filename;

* static/vulnerability.bin

##### File format
The csv file should contain the following fields and include a header row.

| Name                 | Type   |  Bytes | Description                                   | Example   |
|:---------------------|--------|--------| :---------------------------------------------|----------:|
| vulnerability_id     | int    |    4   | Oasis vulnerability_id                        |     45    |
| intensity_bin_index  | int    |    4   | Identifier of the hazard intensity bin        |     10    |
| damage_bin_index     | int    |    4   | Identifier of the damage bin                  |     20    |
| prob                 | float  |    4   | The probability mass for the damage bin       |    0.186  | 

The data should be ordered by vulnerability_id, intensity_bin_index and not contain nulls. 

##### vulnerabilitytobin

```
$ vulnerabilitytobin -d {number of damage bins} < vulnerability.csv > vulnerability.bin
```
The parameter -d number of damage bins is the maximum value of damage_bin_index. This is held in the header of vulnerability.bin and used by getmodel.

Validation checks on the vulnerability csv file are conducted by default during conversion to binary format. These can be suppressed with the -N argument:
```
$ vulnerabilitytobin -d {number of damage bins} -N < vulnerability.csv > vulnerability.bin
```

In the case of very large vulnerability files, it may be preferrable to create an index file to improve performance. Issuing the -i flag creates vulnerability.bin and vulnerability.idx in the current working directory:

```
$ vulnerabilitytobin -d {number of damage bins} -i < vulnerability.csv
```

Additionally, the data can be compressed as it is written to the binary file. Compression is performed with [zlib](https://zlib.net/) by issuing the -z flag. This creates vulnerability.bin.z and vulnerability.idx.z in the current working directory:

```
$ vulnerabilitytobin -d {number of damage bins} -i < vulnerability.csv
```

The getmodel component will look for the presence of index files in the following order to determine which algorithm to use to extract data from vulnerability.bin:

1. static/vulnerability.idx.z
2. static/vulnerability.idx

##### vulnerabilitytocsv
```
$ vulnerabilitytocsv < vulnerability.bin > vulnerability.csv
$ vulnerabilitytocsv -i > vulnerability.csv
$ vulnerabilitytocsv -z > vulnerability.csv
```
[Return to top](#dataconversioncomponents)

<a id="weights"></a>
### Weights
***
The vulnerability weights binary contains the the weighting of each vulnerability function in all areaperil IDs. The data format is as follows. It is required by gulmc with the aggregate_vulnerability file and must have the following location and filename;

* static/weights.bin

#### File format
The csv file should contain the following fields and include a header row.

| Name              | Type   |  Bytes | Description                                               | Example     |
|:------------------|--------|--------| :---------------------------------------------------------|------------:|
| areaperil_id      | int    |    4   | Areaperil ID                                              |     1       |
| vulnerability_id  | int    |    4   | Vulnerability ID                                          |     1       |
| weight            | float  |    4   | The weighting factor                                      |     1.0     |

All fields must not have null values.

##### weightstobin
```
$ weightstobin < weights.csv > weights.bin
```

##### weightstocsv
```
$ weightstocsv < weights.bin > weights.csv
```

[Return to top](#dataconversioncomponents)

## Input data

<a id="amplifications"></a>
### Amplifications
***
The amplifications binary contains the list of item IDs mapped to amplification IDs. The data format is as follows. It is required by Post Loss Amplification (PLA) workflow must have the following location and filename;

* input/amplifications.bin

#### File format
The csv file should contain the following fields and include a header row.

| Name              | Type   |  Bytes | Description                                   | Example     |
|:------------------|--------|--------| :---------------------------------------------|------------:|
| item_id           | int    |    4   | Item ID                                       |     1       |
| amplification_id  | int    |    4   | Amplification ID                              |     1       |

The item_id must start from 1 and must be contiguous and not have null values. The binary file only contains the amplification IDs and assumes the item_ids would start from 1 and are contiguous.

##### amplificationtobin
```
$ amplificationtobin < amplifications.csv > amplifications.bin
```

##### amplificationtocsv
```
$ amplificationtocsv < amplifications.bin > amplifications.csv
```

[Return to top](#dataconversioncomponents)

<a id="coverages"></a>
### Coverages
***
The coverages binary contains the list of coverages and the coverage TIVs. The data format is as follows. It is required by gulcalc and fmcalc and must have the following location and filename;

* input/coverages.bin

#### File format
The csv file should contain the following fields and include a header row.

| Name              | Type   |  Bytes | Description                                   | Example     |
|:------------------|--------|--------| :---------------------------------------------|------------:|
| coverage_id       | int    |    4   | Identifier of the coverage                    |     1       |
| tiv               | float  |    4   | The total insured value of the coverage       |   200000    |

Coverage_id must be an ordered contiguous sequence of numbers starting at 1. 

##### coveragetobin
```
$ coveragetobin < coverages.csv > coverages.bin
```

##### coveragetocsv
```
$ coveragetocsv < coverages.bin > coverages.csv
```

[Return to top](#dataconversioncomponents)

<a id="ensemble"></a>
### ensemble
***
The ensemble file is used for ensemble modelling (multiple views) which maps sample IDs to particular ensemble ID groups. It is an optional file for use with AAL and LEC. It must have the following location and filename;
* input/ensemble.bin

##### File format
The csv file should contain a list of event_ids (integers) and include a header.

| Name              | Type   |  Bytes | Description         | Example     |
|:------------------|--------|--------| :-------------------|------------:|
| sidx              | int    |    4   | Sample ID           |   1         |
| ensemble_id       | int    |    4   | Ensemble ID         |   1         |

##### ensembletobin
```
$ ensembletobin < ensemble.csv > ensemble.bin
```

##### ensembletocsv
```
$ ensembletocsv < ensemble.bin > ensemble.csv
```
[Return to top](#dataconversioncomponents)

<a id="events"></a>
### events
***
One or more event binaries are required by eve. It must have the following location and filename;
* input/events.bin

##### File format
The csv file should contain a list of event_ids (integers) and include a header.

| Name              | Type   |  Bytes | Description         | Example     |
|:------------------|--------|--------| :-------------------|------------:|
| event_id          | int    |    4   | Oasis event_id      |   4545      |

##### evetobin
```
$ evetobin < events.csv > events.bin
```

##### evetocsv
```
$ evetocsv < events.bin > events.csv
```
[Return to top](#dataconversioncomponents)

<a id="items"></a>
### items
***
The items binary contains the list of exposure items for which ground up loss will be sampled in the kernel calculations. The data format is as follows. It is required by gulcalc and outputcalc and must have the following location and filename;

* input/items.bin

##### File format
The csv file should contain the following fields and include a header row.

| Name              | Type   | Bytes | Description                                    | Example   |
|:------------------|--------|-------|:-----------------------------------------------|----------:|
| item_id           | int    |   4   | Identifier of the exposure item                |    1      |
| coverage_id       | int    |   4   | Identifier of the coverage                     |    3      |
| areaperil_id      | int    |   4   | Identifier of the locator and peril            |   4545    |
| vulnerability_id  | int    |   4   | Identifier of the vulnerability distribution   |   645     |
| group_id			| int    |   4   | Identifier of the correlaton group             |    3      |


The data should be ordered by areaperil_id, vulnerability_id ascending and not contain nulls. item_id must be a contiguous sequence of numbers starting from 1.

##### itemtobin
```
$ itemtobin < items.csv > items.bin
```

##### itemtocsv
```
$ itemtocsv < items.bin > items.csv
```

[Return to top](#dataconversioncomponents)

<a id="gulsummaryxref"></a>
### gul summary xref
***
The gulsummaryxref binary is a cross reference file which determines how item or coverage losses from gulcalc output are summed together into at various summary levels in summarycalc. It is required by summarycalc and must have the following location and filename;

* input/gulsummaryxref.bin

##### File format
The csv file should contain the following fields and include a header row.

| Name                  | Type   |  Bytes | Description                                   | Example  |
|:----------------------|--------|--------|:----------------------------------------------|---------:|
| item_id / coverage_id | int    |    4   | Identifier of the item or coverage            |   3      |
| summary_id            | int    |    4   | Identifier of the summary level grouping      |   3      |
| summaryset_id         | int    |    4   | Identifier of the summary set                 |   1      |

* The first field must be consistent with the corresponding index in the ground up loss stream. If it is loss stream (stream_id 2/1) then item_id must be used. If it is gulcalc coverage stream (stream_id 1/2) then coverage_id must be used. 
* The data should not contain nulls and there should be at least one summary set in the file.  
* Valid entries for summaryset_id is all integers between 0 and 9 inclusive. 
* Within each summary set, all coverages_id's from the coverages file must be present.

One summary set consists of a common summaryset_id and each item_id being assigned a summary_id. An example is as follows.

| item_id       | summary_id     | summaryset_id    |
|:--------------|----------------|-----------------:|
| 1             | 1              |    1             | 
| 2             | 1              |    1             | 
| 3             | 1              |    1             |
| 4             | 2              |    1             |
| 5             | 2              |    1             |
| 6             | 2              |    1             |

This shows, for summaryset_id=1, items 1-3 being grouped into summary_id = 1 and items 4-6 being grouped into summary_id = 2.  This could be an example of a 'site' level grouping, for example. The summary_ids should be held in a dictionary which contains the description of the ids to make meaning of the output results.  For instance;

| summary_id   | summaryset_id    | summary_desc    |
|:-------------|------------------|----------------:|
| 1            |    1             |  site_435       |
| 2            |    1             |  site_958       |

This cross reference information is not required in ktools.

Up to 10 summary sets may be provided in gulsummaryxref, depending on the required summary reporting levels for the analysis. Here is an example of the 'site' summary level with summaryset_id=1, plus an 'account' summary level with summaryset_id = 2. In summary set 2, the account summary level includes both sites because all items are assigned a summary_id of 1.

| item_id       | summary_id   | summaryset_id     |
|:--------------|--------------|------------------:|
| 1             | 1            |    1              | 
| 2             | 1            |    1              | 
| 3             | 1            |    1              |
| 4             | 2            |    1              |
| 5             | 2            |    1              |
| 6             | 2            |    1              |
| 1             | 1            |    2              | 
| 2             | 1            |    2              | 
| 3             | 1            |    2              |
| 4             | 1            |    2              |
| 5             | 1            |    2              |
| 6             | 1            |    2              |


##### gulsummaryxreftobin
```
$ gulsummaryxreftobin < gulsummaryxref.csv > gulsummaryxref.bin
```

##### gulsummaryxreftocsv
```
$ gulsummaryxreftocsv < gulsummaryxref.bin > gulsummaryxref.csv
```

[Return to top](#dataconversioncomponents)

<a id="fmprogramme"></a>
### fm programme 
***
The fm programme binary file contains the level heirarchy and defines aggregations of losses required to perform a loss calculation, and is required for fmcalc only. 

This must have the following location and filename;
* input/fm_programme.bin

##### File format
The csv file should contain the following fields and include a header row.

| Name                     | Type   |  Bytes | Description                                    | Example     |
|:-------------------------|--------|--------| :----------------------------------------------|------------:|
| from_agg_id              | int    |    4   | Oasis Financial Module from_agg_id             |    1        |
| level_id                 | int    |    4   | Oasis Financial Module level_id                |     1       |
| to_agg_id                | int    |    4   | Oasis Financial Module to_agg_id               |     1       |

* All fields must have integer values and no nulls
* Must have at least one level, starting from level_id = 1, 2 3 ...
* For level_id = 1, the set of values in from_agg_id must be equal to the set of item_ids in the input ground up loss stream (which has fields event_id, item_id, idx, gul).  Therefore level 1 always defines a group of items.
* For subsequent levels, the from_agg_id must be the distinct values from the previous level to_agg_id field.
* The from_agg_id and to_agg_id values, for each level, should be a contiguous block of integers (a sequence with no gaps).  This is not a strict rule in this version and it will work with non-contiguous integers, but it is recommended as best practice.

##### fmprogrammetobin
```
$ fmprogrammetobin < fm_programme.csv > fm_programme.bin
``` 

##### fmprogrammetocsv
```
$ fmprogrammetocsv < fm_programme.bin > fm_programme.csv
```

[Return to top](#dataconversioncomponents)

<a id="fmprofile"></a>
### fm profile
***
The fmprofile binary file contains the list of calculation rules with profile values (policytc_ids) that appear in the policytc file. This is required for fmcalc only. 

There are two versions of this file and either one or the other can be used at a time.

* fm_profile - original set of fields
* fm_profile_step - extended set of fields to include step policy fields

They must be in the following location with filename formats;
* input/fm_profile.bin
* input/fm_profile_step.bin

##### File format
The csv file should contain the following fields and include a header row.

**fm_profile**

| Name                     | Type   |  Bytes | Description                                    | Example     |
|:-------------------------|--------|--------| :----------------------------------------------|------------:|
| policytc_id              | int    |    4   | Primary key 						              |     34      |
| calcrule_id              | int    |    4   | The calculation rule that applies to the terms |     12      |
| deductible_1             | int    |    4   | First deductible                               |    0.03     |
| deductible_2             | float  |    4   | Second deductible                              |   50000     |
| deductible_3             | float  |    4   | Third deductible                               |   100000    |
| attachment_1             | float  |    4   | Attachment point, or excess					  |   1000000   |
| limit_1				   | float  |    4   | Limit                                          |   5000000   |
| share_1                  | float  |    4   | First proportional share                       |   0.8       |
| share_2                  | float  |    4   | Second proportional share                      |   0.25      |
| share_3                  | float  |    4   | Third proportional share                       |   1         |

**fm_profile_step**

| Name                     | Type   |  Bytes | Description                                    | Example     |
|:-------------------------|--------|--------| :----------------------------------------------|------------:|
| policytc_id              | int    |    4   | Primary key 						              |     34      |
| calcrule_id              | int    |    4   | The calculation rule that applies to the terms |     12      |
| deductible_1             | int    |    4   | First deductible                               |    0.03     |
| deductible_2             | float  |    4   | Second deductible                              |   50000     |
| deductible_3             | float  |    4   | Third deductible                               |   100000    |
| attachment_1             | float  |    4   | Attachment point, or excess					  |   1000000   |
| limit_1				   | float  |    4   | First limit                                    |   5000000   |
| share_1                  | float  |    4   | First proportional share                       |   0.8       |
| share_2                  | float  |    4   | Second proportional share                      |   0.25      |
| share_3                  | float  |    4   | Third proportional share                       |   1         |
| step_id                  | int    |    4   | Step number                                    |   1         |
| trigger_start            | float  |    4   | Start trigger for payout                       |   0.05      |
| trigger_end              | float  |    4   | End trigger for payout      		              |   0.15      |
| payout_start             | float  |    4   | Start payout                                   |   100       |
| payout_end               | float  |    4   | End payout                                     |   200       |
| limit_2                  | float  |    4   | Second limit                                   |   3000000   |
| scale_1                  | float  |    4   | Scaling (inflation) factor 1                   |   0.03      |
| scale_2                  | float  |    4   | Scaling (inflation) factor 2                   |   0.2       |

* A reference table listing the valid values for calcrule_id and which of the fields are required is available in Appendix [B FM Profiles](fmprofiles.md).
* All distinct policytc_id values that appear in the policytc table must appear once in the policytc_id field of the profile table. We suggest that policytc_id=1 is included by default using calcrule_id = 100 and all fields = 0 as a default 'null' calculation rule whenever no terms and conditions apply to a particular level_id / agg_id in the policytc table.
* Any fields that are not required for the profile should be set to zero.

##### fmprofiletobin
```
$ fmprofiletobin < fm_profile.csv > fm_profile.bin
$ fmprofiletobin -S < fm_profile_step.csv > fm_profile_step.bin
``` 

##### fmprofiletocsv
```
$ fmprofiletocsv < fm_profile.bin > fm_profile.csv
$ fmprofiletocsv -S < fm_profile_step.bin > fm_profile_step.csv
```
[Return to top](#dataconversioncomponents)

<a id="fmpolicytc"></a>
### fm policytc
***
The fm policytc binary file contains the cross reference between the aggregations of losses defined in the fm programme file at a particular level and the calculation rule that should be applied as defined in the fm profile file. This file is required for fmcalc only. 

This  must have the following location and filename;
* input/fm_policytc.bin

##### File format
The csv file should contain the following fields and include a header row.


| Name                     | Type   |  Bytes | Description                                    | Example     |
|:-------------------------|--------|--------| :----------------------------------------------|------------:|
| layer_id                 | int    |    4   | Oasis Financial Module layer_id                |    1        |
| level_id                 | int    |    4   | Oasis Financial Module level_id                |     1       |
| agg_id                   | int    |    4   | Oasis Financial Module agg_id                  |     1       |
| policytc_id              | int    |    4   | Oasis Financial Module policytc_id             |     1       |

* All fields must have integer values and no nulls
* Must contain the same levels as the fm programme where level_id = 1, 2, 3 ...
* For every distinct combination of to_agg_id and level_id in the programme table, there must be a corresponding record matching level_id and agg_id values in the policytc table with a valid value in the policytc_id field.  
* layer_id = 1 at all levels except the last where there may be multiple layers, with layer_id = 1, 2, 3 ... This allows for the specification of several policy contracts applied to the same aggregation of losses defined in the programme table.
 
##### fmpolicytctobin
```
$ fmpolicytctobin < fm_policytc.csv > fm_policytc.bin
``` 

##### fmpolicytctocsv
```
$ fmpolicytctocsv < fm_policytc.bin > fm_policytc.csv
```

[Return to top](#dataconversioncomponents)

<a id="fmsummaryxref"></a>
### fm summary xref
***
The fm summary xref binary is a cross reference file which determines how losses from fmcalc output are summed together at various summary levels by summarycalc. It is required by summarycalc and must have the following location and filename;

* input/fmsummaryxref.bin

##### File format
The csv file should contain the following fields and include a header row.

| Name              | Type   |  Bytes | Description                                                        | Example     |
|:------------------|--------|--------| :------------------------------------------------------------------|------------:|
| output_id         | int    |    4   | Identifier of the coverage                                         |   3         |
| summary_id        | int    |    4   | Identifier of the summary level group for one or more output losses|   1         |
| summaryset_id     | int    |    4   | Identifier of the summary set (0 to 9 inclusive)                   |   1         |

* The data should not contain nulls and there should be at least one summary set in the file.  
* Valid entries for summaryset_id is all integers between 0 and 9 inclusive. 
* Within each summary set, all output_id's from the fm xref file must be present.

One summary set consists of a common summaryset_id and each output_id being assigned a summary_id. An example is as follows.

| output_id   | summary_id   | summaryset_id    |
|:------------|--------------|-----------------:|
| 1           | 1            |    1             | 
| 2           | 2            |    1             | 

This shows, for summaryset_id=1, output_id=1 being assigned summary_id = 1 and output_id=2 being assigned summary_id = 2.  

If the output_id represents a policy level loss output from fmcalc (the meaning of output_id is defined in the fm xref file) then no further grouping is performed by summarycalc and this is an example of a 'policy' summary level grouping.

Up to 10 summary sets may be provided in this file, depending on the required summary reporting levels for the analysis. Here is an example of the 'policy' summary level with summaryset_id=1, plus an 'account' summary level with summaryset_id = 2. In summary set 2, the 'account' summary level includes both policy's because both output_id's are assigned a summary_id of 1.

| output_id   | summary_id   | summaryset_id    |
|:------------|--------------|-----------------:|
| 1           | 1            |    1             | 
| 2           | 2            |    1             | 
| 1           | 1            |    2             |
| 2           | 1            |    2             |

If a more detailed summary level than policy is required for insured losses, then the user should specify in the fm profile file to back-allocate fmcalc losses to items. Then the output_id represents back-allocated policy losses to item, and in the fmsummaryxref file these can be grouped into any summary level, such as site, zipcode, line of business or region, for example. The user needs to define output_id in the fm xref file, and group them together into meaningful summary levels in the fm summary xref file, hence these two files must be consistent with respect to the meaning of output_id.

##### fmsummaryxreftobin
```
$ fmsummaryxreftobin < fmsummaryxref.csv > fmsummaryxref.bin
```

##### fmsummaryxreftocsv
```
$ fmsummaryxreftocsv < fmsummaryxref.bin > fmsummaryxref.csv
```

[Return to top](#dataconversioncomponents)

<a id="fmxref"></a>
### fm xref 
***
The fmxref binary file contains cross reference data specifying the output_id in the fmcalc as a combination of agg_id and layer_id, and is required by fmcalc. 

This must be in the following location with filename format;
* input/fm_xref.bin

##### File format
The csv file should contain the following fields and include a header row.

| Name                        | Type   |  Bytes | Description                                    | Example     |
|:----------------------------|--------|--------| :----------------------------------------------|------------:|
| output_id                   | int    |    4   | Identifier of the output group of losses       |     1       |
| agg_id                      | int    |    4   | Identifier of the agg_id to output             |     1       |
| layer_id                    | int    |    4   | Identifier of the layer_id to output           |     1       |

The data should not contain any nulls.

The output_id represents the summary level at which losses are output from fmcalc, as specified by the user.

There are two cases;
* losses are output at the final level of aggregation (represented by the final level to_agg_id's from the fm programme file ) for each contract or layer (represented by the final level layer_id's in the fm policytc file)
* losses are back-allocated to the item level and output by item (represented by the from_agg_id of level 1 from the fm programme file) for each policy contract / layer (represented by the final level layer_id's in the fm policytc file)

For example, say there are two policy layers (with layer_ids=1 and 2) which applies to the sum of losses from 4 items (the summary level represented by agg_id=1). Without back-allocation, the policy summary level of losses can be represented as two output_id's as follows;

| output_id | agg_id   | layer_id    |
|:----------|----------|------------:|
| 1         | 1        |    1        | 
| 2         | 1        |    2        | 

If the user wants to back-allocate policy losses to the items and output the losses by item and policy, then the item-policy summary level of losses would be represented by 8 output_id's, as follows;

| output_id | agg_id   | layer_id    |
|:----------|----------|------------:|
| 1         | 1        |    1        | 
| 2         | 2        |    1        | 
| 3         | 3        |    1        | 
| 4         | 4        |    1        |
| 5         | 1        |    2        | 
| 6         | 2        |    2        |
| 7         | 3        |    2        | 
| 8         | 4        |    2        |

The fm summary xref file must be consistent with respect to the meaning of output_id in the fmxref file.

##### fmxreftobin
```
$ fmxreftobin < fm_xref.csv > fm_xref.bin
``` 

##### fmxreftocsv
```
$ fmxreftocsv < fm_xref.bin > fm_xref.csv
``` 

[Return to top](#dataconversioncomponents)

<a id="occurrence"></a>
### occurrence
***
The occurrence file is required for certain output components which, in the reference model, are leccalc, pltcalc and aalcalc.  In general, some form of event occurence file is required for any output which involves the calculation of loss metrics over a period of time.  The occurrence file assigns occurrences of the event_ids to numbered periods. A period can represent any length of time, such as a year, or 2 years for instance. The output metrics such as mean, standard deviation or loss exceedance probabilities are with respect to the chosen period length.  Most commonly in catastrophe modelling, the period of interest is a year.

The occurrence file also includes date fields.  
* occ_year, occ_month, occ_day. These are all integers representing occurrence year, month and day.
* occ_hour, occ_monute. These are optional and are all integers representing occurrence hour and minute.


##### File format
The csv file should contain the following fields and include a header row.

| Name              | Type   |  Bytes | Description                                                         | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| event_id          | int    |    4   | The occurrence event_id                                             |  45567      |
| period_no         | int    |    4   | A numbered period in which the event occurs                         |  56876      |
| occ_year          | int    |    4   | the year number of the event occurrence                             |   56876     |
| occ_month         | int    |    4   | the month of the event occurrence                                   |   5         |
| occ_day           | int    |    4   | the day of the event occurrence                                     |   16        |

The occurrence year in this example is a scenario numbered year, which cannot be expressed as a real date in a standard calendar.

In addition, the following fields are optional and should comprise the sixth and seventh column respectively:

| Name              | Type   |  Bytes | Description                                                         | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| occ_hour          | int    |    4   | The hour of the event occurrence                                    |   13        |
| occ_minute        | int    |    4   | The minute of the event occurrence                                  |   52        |

The date fields are converted to a single number through an algorithm for efficient storage in the binary file. The data type for this field is either an integer when the optional date fields are not included or a long long integer when the these date fields are included. This should not be confused with the deprecated occ_date_id field.

##### occurrencetobin
A required parameter is -P, the total number of periods of event occurrences. The total number of periods is held in the header of the binary file and used in output calculations.

```
$ occurrencetobin -P10000 < occurrence.csv > occurrence.bin
```
If it is desirable to include the occ_hour and occ_minute fields in the binary file, the -H argument should be given. A flag to signify the presence of these fields is set in the header of the binary file, which is read by other kiools components. If these fields do not exist in the csv file, both are assigned the value of 0 when written to the binary file.
```
$ occurrencetobin -P10000 -H < occurrence.csv > occurrence.bin
```

##### occurrencetocsv
```
$ occurrencetocsv < occurrence.bin > occurrence.csv
```
[Return to top](#dataconversioncomponents)

<a id="returnperiod"></a>
### return period 
***
The returnperiods binary file is a list of return periods that the user requires to be included in loss exceedance curve (leccalc) results.

This must be in the following location with filename format;
* input/returnperiods.bin


##### File format
The csv file should contain the following field and include a header.

| Name                        | Type   |  Bytes | Description          | Example     |
|:----------------------------|--------|--------| :--------------------|------------:|
| return_period               | int    |    4   | Return period        |     250     |


##### returnperiodtobin
```
$ returnperiodtobin < returnperiods.csv > returnperiods.bin
``` 

##### returnperiodtocsv
```
$ returnperiodtocsv < returnperiods.bin > returnperiods.csv
``` 

[Return to top](#dataconversioncomponents)

<a id="periods"></a>
### periods
***
The periods binary file is a list of all the periods that are in the model and is optional for weighting the periods in the calculation. The file is used in the calculation of the loss exceedance curve (leccalc) and aalcalc results.

This must be in the following location with filename format;
* input/periods.bin


##### File format
The csv file should contain the following field and include a header.

| Name               | Type   |  Bytes | Description                                  | Example     |
|:----------------------------|--------|--------| :-----------------------------------|------------:|
| period_no          | int    |    4   | A numbered period in which the event occurs  |   4545      |
| weight             | int    |    4   | relative weight to P, the maximum period_no  |   0.0003    |

All periods must be present in this file (no gaps in period_no from 1 to P).

##### periodstobin
```
$ periodstobin < periods.csv > periods.bin
``` 

##### periodstocsv
```
$ periodstocsv < periods.bin > periods.csv
``` 

[Return to top](#dataconversioncomponents)

<a id="quantile"></a>
### Quantile
***
The quantile binary file contains a list of user specified quantile floats. The data format is as follows. It is optionally used by the Quantile Event/Period Loss tables and must have the following location and filename;

* input/quantile.bin

#### File format
The csv file should contain the following fields and include a header row.

| Name              | Type   |  Bytes | Description                                               | Example     |
|:------------------|--------|--------| :---------------------------------------------------------|------------:|
| quantile          | float  |    4   | Quantile float                                            |     0.1     |

All fields must not have null values.

##### quantiletobin
```
$ quantiletobin < quantile.csv > quantile.bin
```

##### quantiletocsv
```
$ quantiletocsv < quantile.bin > quantile.csv
```

[Return to top](#dataconversioncomponents)

[Go to 4.5 Stream conversion components section](StreamConversionComponents.md)

[Back to Contents](Contents.md)
