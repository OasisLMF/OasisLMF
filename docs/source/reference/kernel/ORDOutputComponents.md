# 4.3 ORD Output Components <a id="ordoutputcomponents"></a>

As well as the set of legacy outputs described in OutputComponents.md, ktools also supports Open Results Data "ORD" output calculations and reports. 

Open Results Data is a data standard for catastrophe loss model results developed as part of Open Data Standards "ODS". ODS is curated by OasisLMF and governed by the Open Data Standards Steering Committee (SC), comprised of industry experts representing (re)insurers, brokers, service providers and catastrophe model vendors. More information about ODS can be found [here](https://github.com/OasisLMF/OpenDataStandards).

ktools supports a subset of the fields in each of the ORD reports, which are given in more detail below.  In most cases, the existing components for legacy outputs are used to generate ORD format outputs when called with extra command line switches, although there is a dedicated component call ordleccalc to generate all of the EPT reports.  In overview, here are the mappings from component to ORD report:

* **summarycalctocsv** generates SELT
* **eltcalc** generates MELT, QELT
* **pltcalc** generates SPLT, MPLT, QPLT
* **ordleccalc** generates EPT and PSEPT
* **aalcalc** generates ALT

<a id="summarycalc"></a>
### summarycalctocsv 
***
Summarycalctocsv takes the summarycalc loss stream, which contains the individual loss samples by event and summary_id, and outputs them in ORD format. Summarycalc is a core component that aggregates the individual building or coverage loss samples into groups that are of interest from a reporting perspective. This is covered in [Core Components](DataConversionComponents.md)

##### Parameters

* -o The ORD output flag
* -p {filename.parquet} outputs the SELT in parquet format

##### Usage
```
$ [stdin component] | summarycalctocsv [parameters] > selt.csv
$ summarycalctocsv [parameters] > selt.csv < [stdin].bin
```

##### Example

```
$ eve 1 1 | getmodel | gulcalc -r -S100 -a1 -i - | summarycalc -i -1 - | summarycalctocsv -o > selt.csv
$ eve 1 1 | getmodel | gulcalc -r -S100 -a1 -i - | summarycalc -i -1 - | summarycalctocsv -p selt.parquet
$ eve 1 1 | getmodel | gulcalc -r -S100 -a1 -i - | summarycalc -i -1 - | summarycalctocsv -p selt.parquet -o > selt.csv
$ summarycalctocsv -o > selt.csv < summarycalc.bin
$ summarycalctocsv -p selt.parquet < summarycalc.bin
$ summarycalctocsv -p selt.parquet -o > selt.csv < summarycalc.bin
```

##### Internal data
None.

##### Output

The Sample ELT output is a csv file with the following fields;

| Name              	| Type   |  Bytes | Description                                                 				| Example     |
|:----------------------|--------|--------| :---------------------------------------------------------------------------|------------:|
| EventId           	| int    |    4   | Model event_id                                             					|  45567      |
| SummaryId         	| int    |    4   | SummaryId  representing a grouping of losses                           	 	|   10        |
| SampleId 	        	| int    |    4   | The sample number						                               	 	|  2          |
| Loss          		| float  |    4   | The loss sample                                                           	|   13645.78  |
| ImpactedExposure  	| float  |    4   | Exposure value impacted by the event for the sample 						|   70000     |

### eltcalc <a id="eltcalc"></a>
***
The program calculates loss by SummaryId and EventId. There are two variants (in addition to the sample variant SELT output by summarycalc, above);

* Moment ELT (MELT) outputs Mean and Standard deviation of loss, as well as EventRate, ChanceOfLoss, MaxLoss, FootprintExposure, MeanImpactedExposure and MaxImpactedExposure
* Quantile ELT (QELT) outputs loss quantiles for the provided set of probabilites. 

##### Parameters

* -M {filename.csv} outputs the MELT in csv format
* -Q {filename.csv} outputs the QELT in csv format
* -m {filename.parquet} outputs the MELT in parquet format
* -q {filename.parquet} outputs the QELT in parquet format

##### Usage
```
$ [stdin component] | eltcalc -M [filename.csv] -Q [filename.csv] -m [filename.parquet] -q [filename.parquet]
$ eltcalc  -M [filename.csv] -Q [filename.csv] -m [filename.parquet] -q [filename.parquet] < [stdin].bin
```

##### Example
```
$ eve 1 1 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - | eltcalc -M MELT.csv -Q QELT.csv
$ eve 1 1 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - | eltcalc -m MELT.parquet -q QELT.parquet
$ eve 1 1 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - | eltcalc -M MELT.csv -Q QELT.csv -m MELT.parquet -q QELT.parquet
$ eltcalc  -M MELT.csv -Q QELT.csv < summarycalc.bin
$ eltcalc  -m MELT.parquet -Q QELT.parquet < summarycalc.bin
$ eltcalc  -M MELT.csv -Q QELT.csv -m MELT.parquet -q QELT.parquet < summarycalc.bin
```

##### Internal data

The Quantile report requires the quantile.bin file

* input/quantile.bin

##### Calculation

###### MELT
For each SummaryId and EventId, the sample mean and standard deviation is calculated from the sampled losses in the summarycalc stream and output to file.  The analytical mean is also output as a seperate record, differentiated by a 'SampleType' field. Variations of the exposure value are also output (see below for details).

###### QELT
For each SummaryId and EventId, this report provides the probability and the corresponding loss quantile computed from the samples.  The list of probabilities is provided as input in the quantile.bin file.

Quantiles are cut points dividing the range of a probability distribution into continuous intervals with equal probabilities, or dividing the observations in a sample set in the same way. In this case we are computing the quantiles of loss from the sampled losses by event and summary for a user-provided list of probabilities. For each provided probability p, the loss quantile is the sampled loss which is bigger than the proportion p of the observed samples. 

In practice this is calculated by sorting the samples in ascending order of loss and using linear interpolation between the ordered observations to compute the precise loss quantile for the required probability.

The algorithm used for the quantile estimate type and interpolation scheme from a finite sample set is R-7 referred to in Wikipedia https://en.wikipedia.org/wiki/Quantile

If p is the probability, and the sample size is N, then the position of the ordered samples required for the quantile is computed by;

(N-1)p + 1

In general, this value will be a fraction rather than an integer, representing a value in between two ordered samples. Therefore for an integer value of k between 1 and N-1 with k < (N-1)p + 1 < k+1 , the loss quantile Q(p) is calculated by a linear interpolation of the kth ordered sample X(k) and the k+1 th ordered sample X(k+1) as follows;

Q(p) = X(k) * (1-h) + X(k+1) * h

where h = (N-1)p + 1 - k

##### Output

The Moment ELT output is a csv file with the following fields;

| Name              	| Type   |  Bytes | Description                                                 				| Example     |
|:----------------------|--------|--------| :---------------------------------------------------------------------------|------------:|
| EventId           	| int    |    4   | Model event_id                                             					|  45567      |
| SummaryId         	| int    |    4   | SummaryId  representing a grouping of losses                           	 	|   10        |
| SampleType        	| int    |    4   | 1 for analytical mean, 2 for sample mean                               	 	|  2          |
| EventRate			 	| float  |    4   | Annual frequency of event computed by relative frequency of occurrence 		|   0.01      | 
| ChanceOfLoss		 	| float  |    4   | Probability of a loss calculated from the effective damage distributions	|   0.95      | 
| MeanLoss          	| float  |    4   | Mean                                                                    	|   1345.678  |
| SDLoss            	| float  |    4   | Sample standard deviation for SampleType=2                              	|    945.89   |
| MaxLoss     	     	| float  |    4   | Maximum possible loss calculated from the effective damage distribution     |   75000     |
| FootprintExposure 	| float  |    4   | Exposure value impacted by the model's event footprint                 		|   80000     |
| MeanImpactedExposure  | float  |    4   | Mean exposure impacted by the event across the samples (where loss > 0 )    |   65000     |
| MaxImpactedExposure  	| float  |    4   | Maximum exposure impacted by the event across the samples (where loss > 0)  |   70000     |

The Quantile ELT output is a csv file with the following fields;

| Name              	| Type   |  Bytes | Description                                                 				| Example     |
|:----------------------|--------|--------| :---------------------------------------------------------------------------|------------:|
| EventId           	| int    |    4   | Model event_id                                             					|  45567      |
| SummaryId         	| int    |    4   | SummaryId  representing a grouping of losses                           	 	|   10        |
| Quantile 	        	| float  |    4   | The probability associated with the loss quantile                     	 	|    0.9      |
| Loss 		          	| float  |    4   | The loss quantile                                                          	|   1345.678  |

[Return to top](#ordoutputcomponents)

### pltcalc <a id="pltcalc"></a>
***
The program calculates loss by Period, EventId and SummaryId and outputs the results in ORD format. There are three variants;

* Sample PLT (SPLT) outputs individual loss samples by SampleId, as well as PeriodWeight, Year, Month, Day, Hour, Minute and ImpactedExposure
* Moment PLT (MPLT) outputs Mean and Standard deviation of loss, as well as PeriodWeight, Year, Month, Day, Hour, Minute, ChanceOfLoss, MaxLoss, FootprintExposure, MeanImpactedExposure and MaxImpactedExposure
* Quantile PLT (QPLT) outputs loss quantiles for the provided set of probabilites as well as PeriodWeight, Year, Month, Day, Hour, Minute 

##### Parameters

* -S {filename.csv} outputs the SPLT in csv format
* -M {filename.csv} outputs the MPLT in csv format
* -Q {filename.csv} outputs the QPLT in csv format
* -s {filename.parquet} outputs the SPLT in parquet format
* -m {filename.parquet} outputs the MPLT in parquet format
* -q {filename.parquet} outputs the QPLT in parquet format

##### Usage
```
$ [stdin component] | pltcalc -S [filename.csv] -M [filename.csv] -Q [filename.csv] -s [filename.parquet] -m [filename.parquet] -q [filename.parquet]
$ pltcalc -S [filename.csv] -M [filename.csv] -Q [filename.csv] -s [filename.parquet] -m [filename.parquet] -q [filename.parquet] < [stdin].bin
```

##### Example
```
$ eve 1 1 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - | pltcalc -S SPLT.csv -M MPLT.csv -Q QPLT.csv
$ eve 1 1 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - | pltcalc -s SPLT.parquet -m MPLT.parquet -q QPLT.parquet
$ eve 1 1 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - | pltcalc -S SPLT.csv -M MPLT.csv -Q QPLT.csv -s SPLT.parquet -m MPLT.parquet -q QPLT.parquet
$ pltcalc -S SPLT.csv -M MPLT.csv -Q QPLT.csv < summarycalc.bin
$ pltcalc -s SPLT.parquet -m MPLT.parquet -q QPLT.parquet < summarycalc.bin
$ pltcalc -S SPLT.csv -M MPLT.csv -Q QPLT.csv -s SPLT.parquet -m MPLT.parquet -q QPLT.parquet < summarycalc.bin
```

##### Internal data

pltcalc requires the occurrence.bin file

* input/occurrence.bin

The Quantile report additionally requires the quantile.bin file

* input/quantile.bin

pltcalc will optionally use the following file if present

* input/periods.bin

##### Calculation

###### SPLT
For each Period, EventId and SummaryId, the individual loss samples are output by SampleId. The sampled event losses from the summarycalc stream are assigned to a Period for each occurrence of the EventId in the occurrence file.

###### MPLT
For each Period, EventId and SummaryId, the sample mean and standard deviation is calculated from the sampled event losses in the summarycalc stream and output to file.  The analytical mean is also output as a seperate record, differentiated by a 'SampleType' field. Variations of the exposure value are also output (see below for more details).

###### QPLT
For each Period, EventId and SummaryId, this report provides the probability and the corresponding loss quantile computed from the samples.  The list of probabilities is provided in the quantile.bin file.

See QELT for the method of computing the loss quantiles.

##### Output

The Sample PLT output is a csv with the folling fields

| Name              	| Type   |  Bytes | Description                                                 				| Example     |
|:----------------------|--------|--------| :---------------------------------------------------------------------------|------------:|
| Period            	| int    |    4   | The period in which the event occurs                      					|  500        |
| PeriodWeight        	| int    |    4   | The weight of the period (frequency relative to the total number of periods)|  0.001      |
| EventId           	| int    |    4   | Model event_id                                             					|  45567      |
| Year 		        	| int    |    4   | The year in which the event occurs			                           	 	|   1970      |
| Month		         	| int    |    4   | The month number in which the event occurs                           	 	|    5        |
| Day 		         	| int    |    4   | The day number in which the event occurs	                           	 	|   22        |
| Hour		         	| int    |    4   | The hour in which the event occurs			                           	 	|   11        |
| Minute	         	| int    |    4   | The minute in which the event occurs		                           	 	|   45        |
| SummaryId         	| int    |    4   | SummaryId  representing a grouping of losses                           	 	|   10        |
| SampleId 	        	| int    |    4   | The sample number						                               	 	|  2          |
| Loss          		| float  |    4   | The loss sample                                                           	|   13645.78  |
| ImpactedExposure  	| float  |    4   | Exposure impacted by the event for the sample 								|   70000     |

The Moment PLT output is a csv file with the following fields;

| Name              	| Type   |  Bytes | Description                                                 				| Example     |
|:----------------------|--------|--------| :---------------------------------------------------------------------------|------------:|
| Period            	| int    |    4   | The period in which the event occurs                      					|  500        |
| PeriodWeight        	| int    |    4   | The weight of the period (frequency relative to the total number of periods)|  0.001      |
| EventId           	| int    |    4   | Model event_id                                             					|  45567      |
| Year 		        	| int    |    4   | The year in which the event occurs			                           	 	|   1970      |
| Month		         	| int    |    4   | The month number in which the event occurs                           	 	|    5        |
| Day 		         	| int    |    4   | The day number in which the event occurs	                           	 	|   22        |
| Hour		         	| int    |    4   | The hour in which the event occurs			                           	 	|   11        |
| Minute	         	| int    |    4   | The minute in which the event occurs		                           	 	|   45        |
| SummaryId         	| int    |    4   | SummaryId  representing a grouping of losses                           	 	|   10        |
| SampleType        	| int    |    4   | 1 for analytical mean, 2 for sample mean                               	 	|  2          |
| ChanceOfLoss		 	| float  |    4   | Probability of a loss calculated from the effective damage distributions	|   0.95      |
| MeanLoss          	| float  |    4   | Mean                                                                    	|   1345.678  |
| SDLoss            	| float  |    4   | Sample standard deviation for SampleType=2                              	|    945.89   |
| MaxLoss     	     	| float  |    4   | Maximum possible loss calculated from the effective damage distribution     |   75000     |
| FootprintExposure 	| float  |    4   | Exposure value impacted by the model's event footprint                 		|   80000     |
| MeanImpactedExposure  | float  |    4   | Mean exposure impacted by the event across the samples (where loss > 0 )    |   65000     |
| MaxImpactedExposure  	| float  |    4   | Maximum exposure impacted by the event across the samples (where loss > 0)  |   70000     |

The Quantile PLT output is a csv file with the following fields;

| Name              	| Type   |  Bytes | Description                                                 				| Example     |
|:----------------------|--------|--------| :---------------------------------------------------------------------------|------------:|
| Period            	| int    |    4   | The period in which the event occurs                      					|  500        |
| PeriodWeight        	| int    |    4   | The weight of the period (frequency relative to the total number of periods)|  0.001      |
| EventId           	| int    |    4   | Model event_id                                             					|  45567      |
| Year 		        	| int    |    4   | The year in which the event occurs			                           	 	|   1970      |
| Month		         	| int    |    4   | The month number in which the event occurs                           	 	|    5        |
| Day 		         	| int    |    4   | The day number in which the event occurs	                           	 	|   22        |
| Hour		         	| int    |    4   | The hour in which the event occurs			                           	 	|   11        |
| Minute	         	| int    |    4   | The minute in which the event occurs		                           	 	|   45        |
| SummaryId         	| int    |    4   | SummaryId representing a grouping of losses                           	 	|   10        |
| Quantile 	        	| float  |    4   | The probability associated with the loss quantile                     	 	|    0.9      |
| Loss 		          	| float  |    4   | The loss quantile                                                          	|   1345.678  |

[Return to top](#ordoutputcomponents)

### ordleccalc <a id="ordleccalc"></a>
***
This component produces several variants of loss exceedance curves, known as Exceedance Probability Tables "EPT" under ORD. 

An Exceedance Probability Table is a set of user-specified percentiles of (typically) annual loss on one of two bases – AEP (sum of losses from all events in a year) or OEP (maximum of any one event’s losses in a year).  In ORD the percentiles are expressed as Return Periods, which is the reciprocal of the percentile. 

How EPTs are derived in general depends on the mathematical methodology of calculating the underlying ground up and insured losses. 

In the Oasis kernel the methodology is Monte Carlo sampling from damage distributions, which results in several samples (realisations) of an event loss for every event in the model's catalogue. The event losses are assigned to a year timeline and the years are rank ordered by loss. The method of computing the percentiles is by taking the ratio of the frequency of years with a loss exceeding a given threshold over the total number of years.

The OasisLMF approach gives rise to five variations of calculation of these statistics:

*	EP Table from Mean Damage Losses  – this means do the loss calculation for a year using the event mean damage loss computed by numerical integration of the effective damageability distributions. 
*	EP Table of Sample Mean Losses – this means do the loss calculation for a year using the statistical sample event mean.
*	Full Uncertainty EP Table – this means do the calculation across all samples (treating the samples effectively as repeat years) - this is the most accurate of all the single EP Curves.
*	Per Sample EPT (PSEPT) – this means calculate the EP Curve for each sample and leave it at the sample level of detail, resulting in multiple "curves".
*	Per Sample mean EPT – this means average the loss at each return period of the Per Sample EPT. 

Exceedance Probability Tables are further generalised in Oasis to represent not only annual loss percentiles but loss percentiles over any period of time. Thus the typical use of 'Year' label in outputs is replaced by the more general term 'Period', which can be any period of time as defined in the model data 'occurrence' file (although the normal period of interest is a year).


##### Parameters

* -K{sub-directory}. The subdirectory of /work containing the input summarycalc binary files.
Then the following parameters must be specified for at least one analysis type;
* Analysis type. Use -F for Full Uncertainty Aggregate, -f for Full Uncertainty Occurrence, -W for Per Sample Aggregate,  -w for Per Sample Occurrence, -S for Sample Mean Aggregate, -s for Sample Mean Occurrence, -M for Per Sample Mean Aggregate, -m for Per Sample Mean Occurrence
* -O {ept.csv} is the output flag for the EPT csv (for analysis types -F, -f, -S, -s, -M, -m)
* -o {psept.csv} is the output flag for the PSEPT csv (for analysis types -W or -w)
* -P {ept.parquet} is the output flag for the EPT parquet file (for analysis types -F, -f, -S, -s, -M, -m)
* -p {psept.parquet} is the output flag for the PSEPT parquet file (for analysis types -W or -w)

An optional parameter is; 
* -r. Use return period file - use this parameter if you are providing a file with a specific list of return periods. If this file is not present then all calculated return periods will be returned, for losses greater than zero.

##### Usage

```
$ ordleccalc [parameters] 

```

##### Examples
```
'First generate summarycalc binaries by running the core workflow, for the required summary set
$ eve 1 2 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - > work/summary1/summarycalc1.bin
$ eve 2 2 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - > work/summary1/summarycalc2.bin

'Then run ordleccalc, pointing to the specified sub-directory of work containing summarycalc binaries.
'Write aggregate and occurrence full uncertainty
$ ordleccalc -Ksummary1 -F -f -O ept.csv
$ ordleccalc -Ksummary1 -F -f -P ept.parquet
$ ordleccalc -Ksummary1 -F -f -O ept.csv -P ept.parquet

'Write occurrence per sample (PSEPT)
$ ordleccalc -Ksummary1 -w -o psept.csv
$ ordleccalc -Ksummary1 -w -p psept.parquet
$ ordleccalc -Ksummary1 -w -o psept.csv -p psept.parquet

'Write aggregate and occurrence per sample (written to PSEPT) and per sample mean (written to EPT file)
$ ordleccalc -Ksummary1 -W -w -M -m -O ept.csv -o psept.csv
$ ordleccalc -Ksummary1 -W -w -M -m -P ept.parquet -p psept.parquet
$ ordleccalc -Ksummary1 -W -w -M -m -O ept.csv -o psept.csv -P ept.parquet -p psept.parquet

'Write full output
$ ordleccalc -Ksummary1 -F -f -W -w -S -s -M -m -O ept.csv -o psept.csv
$ ordleccalc -Ksummary1 -F -f -W -w -S -s -M -m -P ept.parquet -p psept.parquet
$ ordleccalc -Ksummary1 -F -f -W -w -S -s -M -m -O ept.csv -o pseept.csv -P ept.parquet -p psept.parquet
```

##### Internal data

ordleccalc requires the occurrence.bin file

* input/occurrence.bin

and will optionally use the following additional files if present

* input/returnperiods.bin
* input/periods.bin

ordleccalc does not have a standard input that can be streamed in. Instead, it reads in summarycalc binary data from a file in a fixed location.  The format of the binaries must match summarycalc standard output. The location is in the 'work' subdirectory of the present working directory. For example;

* work/summarycalc1.bin
* work/summarycalc2.bin
* work/summarycalc3.bin

The user must ensure the work subdirectory exists.  The user may also specify a subdirectory of /work to store these files. e.g.

* work/summaryset1/summarycalc1.bin
* work/summaryset1/summarycalc2.bin
* work/summaryset1/summarycalc3.bin

The reason for ordleccalc not having an input stream is that the calculation is not valid on a subset of events, i.e. within a single process when the calculation has been distributed across multiple processes.  It must bring together all event losses before assigning event losses to periods and ranking losses by period.  The summarycalc losses for all events (all processes) must be written to the /work folder before running leccalc.

##### Calculation

All files with extension .bin from the specified subdirectory are read into memory, as well as the occurrence.bin. The summarycalc losses are grouped together and sampled losses are assigned to period according to which period the events are assigned to in the occurrence file.

If multiple events occur within a period;
* For **aggregate** loss exceedance curves, the sum of losses is calculated.
* For **occurrence** loss exceedance curves, the maximum loss is calculated.

The 'EPType' field in the output identifies the basis of loss exceedance curve.

The 'EPTypes' are;

1. OEP
2. OEP TVAR
3. AEP
4. AEP TVAR

TVAR results are generated automatically if the OEP or AEP report is selected in the analysis options. TVAR, or Tail Conditional Expectation (TCE), is computed by averaging the rank ordered losses exceeding a given return period loss from the respective OEP or AEP result.

Then the calculation differs by EPCalc type, as follows;

1. The mean damage loss (sidx = -1) is output as a standard exceedance probability table.  If the calculation is run with 0 samples, then leccalc will still return the mean damage loss exceedance curve.  

2. Full uncertainty - all losses by period are rank ordered to produce a single loss exceedance curve. 

3. Per Sample mean - the return period losses from the Per Sample EPT are averaged, which produces a single loss exceedance curve.

4. Sample mean - the losses by period are first averaged across the samples, and then a single loss exceedance table is created from the period sample mean losses.

All four of the above variants are output into the same file when selected.

Finally, the fifth variant, the Per Sample EPT is output to a separate file. In this case, for each sample, losses by period are rank ordered to produce a loss exceedance curve for each sample.


##### Output

Exceedance Probability Tables (EPT)

csv files with the following fields;

**Exceedance Probability Table (EPT)**

| Name              | Type   |  Bytes | Description                                                         | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| SummaryId         | int    |    4   | identifier representing a summary level grouping of losses          |   10        |
| EPCalc            | int    |    4   | 1, 2, 3 or 4 with meanings as given above	                        |    2        |
| EPType 			| int    |    4   | 1, 2, 3 or 4 with meanings as given above                           |    1        |
| ReturnPeriod		| float  |    4   | return period interval                                              |    250      |
| loss              | float  |    4   | loss exceedance threshold or TVAR for return period                 |    546577.8 |

**Per Sample Exceedance Probability Tables (PSEPT)**

| Name              | Type   |  Bytes | Description                                                         | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| SummaryId         | int    |    4   | identifier representing a summary level grouping of losses          |   10        |
| SampleID          | int    |    4   | Sample number								                        |    20       |
| EPType 			| int    |    4   | 1, 2, 3 or 4			                                            |    3        |
| ReturnPeriod		| float  |    4   | return period interval                                              |    250      |
| loss              | float  |    4   | loss exceedance threshold or TVAR for return period                 |    546577.8 |

##### Period weightings

An additional feature of ordleccalc is available to vary the relative importance of the period losses by providing a period weightings file to the calculation. In this file, a weight can be assigned to each period make it more or less important than neutral weighting (1 divided by the total number of periods). For example, if the neutral weight for period 1 is 1 in 10000 years, or 0.0001, then doubling the weighting to 0.0002 will mean that period's loss reoccurrence rate would double.  Assuming no other period losses, the return period of the loss of period 1 in this example would be halved.

All period_nos must appear in the file from 1 to P (no gaps). There is no constraint on the sum of weights. Periods with zero weight will not contribute any losses to the loss exceedance curve.

This feature will be invoked automatically if the periods.bin file is present in the input directory.

[Return to top](#ordoutputcomponents)

### aalcalc <a id="aalcalc"></a>
***
aalcalc outputs the Average Loss Table (ALT) which contains the average annual loss and standard deviation of annual loss by SummaryId.

Two types of average and standard deviation of loss are calculated; analytical (SampleType 1) and sample (SampleType 2).  If the analysis is run with zero samples, then only SampleType 1 statistics are returned.

##### Internal data

aalcalc requires the occurrence.bin file 

* input/occurrence.bin

aalcalc does not have a standard input that can be streamed in. Instead, it reads in summarycalc binary data from a file in a fixed location.  The format of the binaries must match summarycalc standard output. The location is in the 'work' subdirectory of the present working directory. For example;

* work/summarycalc1.bin
* work/summarycalc2.bin
* work/summarycalc3.bin

The user must ensure the work subdirectory exists.  The user may also specify a subdirectory of /work to store these files. e.g.

* work/summaryset1/summarycalc1.bin
* work/summaryset1/summarycalc2.bin
* work/summaryset1/summarycalc3.bin

The reason for aalcalc not having an input stream is that the calculation is not valid on a subset of events, i.e. within a single process when the calculation has been distributed across multiple processes.  It must bring together all event losses before assigning event losses to periods and finally computing the final statistics.  

##### Parameters

* -K{sub-directory}. The sub-directory of /work containing the input aalcalc binary files.
* -o. The ORD format flag
* -p {filename}. The ORD parquet format flag

##### Usage

```
$ aalcalc [parameters] > alt.csv
```

##### Examples

```
'First generate summarycalc binaries by running the core workflow, for the required summary set
$ eve 1 2 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - > work/summary1/summarycalc1.bin
$ eve 2 2 | getmodel | gulcalc -r -S100 -c - | summarycalc -g -1 - > work/summary1/summarycalc2.bin

'Then run aalcalc, pointing to the specified sub-directory of work containing summarycalc binaries.
$ aalcalc -o -Ksummary1 > alt.csv
$ aalcalc -p alt.parquet -Ksummary1
$ allcalc -o -p alt.parquet -Ksummary1 > alt.csv
```

##### Output

csv file containing the following fields;

| Name                | Type   |  Bytes | Description                                                         | Example     |
|:--------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| SummaryId           | int    |    4   | SummaryId representing a grouping of losses                         |   10        |
| SampleType          | int    |    4   | 1 for analytical statistics, 2 for sample statistics                |    1        |
| MeanLoss            | float  |    8   | average annual loss                                                 |    6785.9   |
| SDLoss			  | float  |    8   | standard deviation of loss                                          |    54657.8  |


##### Calculation

The occurrence file and summarycalc files from the specified subdirectory are read into memory. Event losses are assigned to period according to which period the events occur in and summed by period and by sample.

For type 1, the mean and standard deviation of numerically integrated mean period losses are calculated across the periods. For type 2 the mean and standard deviation of the sampled period losses are calculated across all samples (sidx > 1) and periods. 


##### Period weightings

An additional feature of aalcalc is available to vary the relative importance of the period losses by providing a period weightings file to the calculation. In this file, a weight can be assigned to each period make it more or less important than neutral weighting (1 divided by the total number of periods). For example, if the neutral weight for period 1 is 1 in 10000 years, or 0.0001, then doubling the weighting to 0.0002 will mean that period's loss reoccurrence rate would double and the loss contribution to the average annual loss would double.  

All period_nos must appear in the file from 1 to P (no gaps). There is no constraint on the sum of weights. Periods with zero weight will not contribute any losses to the AAL.

This feature will be invoked automatically if the periods.bin file is present in the input directory.

[Return to top](#ordoutputcomponents)

[Go to 4.4 Data conversion components section](DataConversionComponents.md)

[Back to Contents](Contents.md)
