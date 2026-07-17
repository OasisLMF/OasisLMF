# 3. Specification

### Introduction

This section specifies the data stream structures and core components in the in-memory kernel.

The data stream structures are;
* **cdf stream**
* gulmc stream(deprecated)
* **loss stream**
* **summary stream**

The stream data structures have been designed to minimise the volume flowing through the pipeline, using data packet 'headers' to remove redundant data. For example, indexes which are common to a block of data are defined as a header record and then only the variable data records that are relevant to the header key are part of the data stream. The names of the data fields given below are unimportant, only their position in the data stream is important in order to perform the calculations defined in the program.

The components are;

* **evepy**
* **modelpy**
* **gulmc**
* **fmpy**
* **summarypy**
* **outputcalc**


The components have a standard input (stdin) and/or output (stdout) data stream structure. evepy is the stream-initiating component which only has a standard output stream, whereas "outputcalc" (a generic name representing an extendible family of output calculation components) is a stream-terminating component with only a standard input stream.

An implementation of each of the above components is provided in the [Reference Model](ReferenceModelOverview.md), where usage instructions and command line parameters are provided. A functional overview is given below.

#### Stream types

The architecture supports multiple stream types. Therefore a developer can define a new type of data stream within the framework by specifying a unique stream_id of the stdout of one or more of the components, or even write a new component which performs an intermediate calculation between the existing components.

The stream_id is the first 4 byte header of the stdout streams. The higher byte is reserved to identify the type of stream, and the 2nd to 4th bytes hold the identifier of the stream. This is used for validation of pipeline commands to report errors if the components are not being used in the correct order.

The current reserved values are as follows;

Higher byte;

| Byte 1 |  Stream name  		|
|:-------|:---------------------|
|    0   | cdf          		|
|    1   | gulmc (deprecated) |
|    2   | loss          		|
|    3   | summary       		|

Reserved stream_ids;

| Byte 1   | Bytes 2-4    |  Description                                                             		         
|:---------|--------------|:---------------------------------------------------------------------------------|
|    0     |     1        |  cdf - Oasis format effective damageability CDF output                           |
|	 1     |     1        |  gulmc - Oasis format item level ground up loss sample output (deprecated)     |
|    1     |     2        |  gulmc - Oasis format coverage level ground up loss sample output (deprecated) |
|    2     |     1        |  loss -  Oasis format loss sample output (any loss perspective)                  |
|    3     |     1        |  summary - Oasis format summary level loss sample output                         |

The supported standard input and output streams of the reference model components are summarized here;

| Component    | Standard input                        |  Standard output                      | Stream option parameters          			|
|:-------------|:--------------------------------------|:--------------------------------------|:-------------------------------------------|
| modelpy     | none                                  | 0/1 cdf                               | none                              			|
| gulmc      | 0/1 cdf                               | 2/1 loss                              | -i -a{}                           			|
| fmpy       | 2/1 loss                              | 2/1 loss                              | none                              			|
| summarypy  | 2/1 loss                              | 3/1 summary                           | -i input from gulmc, -f input from fmpy| 
| outputcalc   | 3/1 summary                           | none                                  | none                              			| 


## Stream structure


### cdf stream

Stream header packet structure

| Name              | Type   |  Bytes | Description                                                         | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| stream_id         | int    |   1/3  | Identifier of the data stream type.                                 |    0/1      |

Data header packet structure

| Name              | Type   |  Bytes | Description                                                         | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| event_id          | int    |    4   | Oasis event_id                                                      |   4545      |
| areaperil_id      | int    |    4   | Oasis areaperil_id                                                  |  345456     |
| vulnerability_id  | int    |    4   | Oasis vulnerability_id                                              |   345       |
| no_of_bins        | int    |    4   | Number of records (bins) in the data package                        |    20       |        

Data packet structure (record repeated no_of_bin times)

| Name              | Type   |  Bytes | Description                                                         | Example     |
|:------------------|--------|--------| :-------------------------------------------------------------------|------------:|
| prob_to           | float  |    4   | The cumulative probability at the upper damage bin threshold        |     0.765   |
| bin_mean          | float  |    4   | The conditional mean of the damage bin                              |     0.45    |



### loss stream

Stream header packet structure

| Name              | Type   |  Bytes | Description                          | Example     |
|:------------------|--------|--------| :------------------------------------|------------:|
| stream_id         | int    |   1/3  | Identifier of the data stream type.  |    2/1      |
| no_of_samples     | int    |   4    | Number of samples                    |    100      |

Data header packet structure

| Name               | Type   |  Bytes | Description                                              | Example     |
|:-------------------|--------|--------| :--------------------------------------------------------|------------:|
| event_id           | int    |    4   | Oasis event_id                                           |   4545      |
| item_id /output_id | int    |    4   | Oasis item_id (gulmc) or output_id (fmpy)            |    300      |

Data packet structure

| Name              | Type   |  Bytes | Description                          | Example     |
|:------------------|--------|--------| :------------------------------------|------------:|
| sidx              | int    |    4   | Sample index                         |   10        |
| loss              | float  |    4   | The  loss for the sample             | 5625.675    |

The data packet may be a variable length and so a sidx of 0 identifies the end of the data packet.

There are five values of sidx with special meaning as follows;

| sidx   |  Meaning                                      | Required / optional|
|:-------|:----------------------------------------------|--------------------|
|   -5   | maximum loss                                  |   optional         |
|	-4	 | chance of loss                                |   optional         |
|   -3   | impacted exposure                             |   required         |
|   -2   | numerical integration standard deviation loss |   optional         |
|   -1   | numerical integration mean loss               |   required         |

sidx -5 to -1 must come at the beginning of the data packet before the other samples in ascending order (-5 to -1).  


### summary stream

Stream header packet structure

| Name              | Type   |  Bytes | Description                          | Example     |
|:------------------|--------|--------| :------------------------------------|------------:|
| stream_id         | int    |   1/3  | Identifier of the data stream type.  |    3/1      |
| no_of_samples     | int    |   4    | Number of samples                    |    100      |
| summary_set       | int    |   4    | Identifier of the summary set        |    2        |

Data header packet structure

| Name              | Type   |  Bytes | Description                                             | Example     |
|:------------------|--------|--------| :-------------------------------------------------------|------------:|
| event_id          | int    |    4   | Oasis event_id                                          |   4545      |
| summary_id        | int    |    4   | Oasis summary_id                                        |    300      |
| exposure_value    | float  |    4   | Impacted exposure (sum of sidx -3 losses for summary_id)|    987878   |

Data packet structure

| Name              | Type   |  Bytes | Description                          | Example     |
|:------------------|--------|--------| :------------------------------------|------------:|
| sidx              | int    |    4   | Sample index                         |   10        |
| loss              | float  |    4   | The loss for the sample              | 5625.675    |

The data packet may be a variable length and so a sidx of 0 identifies the end of the data packet.

The sidx -1 mean loss may be present (if non-zero)

| sidx   |  Meaning                                      | Required / optional|
|:-------|:----------------------------------------------|--------------------|
|   -1   | numerical integration mean loss               |   optional         |



## Components 


### evepy 

evepy is an 'event emitter' and its job is to read a list of events from file and send out a subset of events as a binary data stream. It has no standard input and emits a list of event_ids, which are 4 byte integers.

evepy is used to partition lists of events such that a workflow can be distributed across multiple processes.


### modelpy 

modelpy is the component which generates a stream of effective damageability cdfs for a given set of event_ids and the impacted exposed items on the basis of their areaperil_ids (location) and vulnerability_ids (damage function). 


### gulmc 

gulmc is the component which calculates ground up loss. It takes the modelpy output as standard input and based on the sampling parameters specified, performs Monte Carlo sampling and numerical integration. The output is a stream of ground up loss samples in Oasis kernel format with random samples identified by positive sample indexes (sidx 1 and greater), and special meaning samples assigned to negative sample indexes.

gulmc also supports the combining and back-allocation of losses arising from multiple subperils impacting the same coverage with some options.


### fmpy 

fmpy is the component which takes the loss stream as standard input and output and applies the policy terms and conditions to produce insured loss samples. fmpy can be called recursively to perform multiple sequential applications of financial terms (e.g for inuring reinsurance following direct insurance). The output is a table of loss samples in Oasis kernel format, including the (re)insured loss for the numerical integration mean (sidx=-1), and the impacted exposure (sidx=-3). 


### summarypy
summarypy is a component which sums the sampled losses from either gulmc or fmpy to the users required level(s) for reporting results.  This is a simple sum of the loss value by event_id, sidx and summary_id, where summary_id is a grouping of coverage_id or item_id for gulmc or output_id for fmpy defined in the user's input files.  



### outputcalc 

Outputcalc is a general term for an end-of-pipeline component which represents one of a potentially unlimited set of output components. Some examples are provided in the Reference Model. These are; 

* eltpy
* lecpy
* aalpy
* pltpy
* ordleccalc

The output components generate results such as an event loss table or loss exceedance curve from the sampled output from summarypy.  The output is a results table in csv format or parquet format. 


[Go to 4. Reference model](ReferenceModelOverview.md)

[Back to Contents](Contents.md)
