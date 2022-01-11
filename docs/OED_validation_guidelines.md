OED validation guidelines
=======================

# Overview 
This document contains some guidelines on what is a valid set of OED files to be imported into the OasisLMF platform. This is different to what is theoretically possible to import under the full OED schema because OasisLMF supports only a subset of the fields defined in the schema. Therefore the validation rules for importing exposures in Oasis are different and will also sometimes vary between versions of the OasisLMF software.

This document should be read alongside [OED financial terms supported](OED_financial_terms_supported.xlsx) which contains a detailed list of the OED fields supported by OasisLMF.

## Minimum file requirements
Only the OED location file is required for running a ground up loss analysis.  The OED account file may be provided if there are direct insurance terms, and the OED Reinsurance info file and the OED Reinsurance scope file may be provided together if there are reinsurance terms. 

## OED Versions supported
* OasisLMF 1.15.21-LTS supports OED v1 format files
* OasisLMF 1.23-LTS and later releases support OED v2 format files

## Minimum required fields
The OED schema has a fixed set of required fields, but the required fields for loading exposures into the Oasis platform will vary depending on what model is being run.  This is because the main function of the Oasis platform is to run exposures against a catastrophe model and each model has a particular set of fields needed for exposure geo-location and vulnerability identification.

The fields that are always required in OasisLMF are;
* OED location - PortNumber, AccNumber, LocNumber, and at least one of BuildingTIV, OtherTIV, ContentsTIV, BITIV
* OED account - PortNumber, AccNumber, PolNumber

Examples of additional location fields required for the OasisLMF PiWind model are;

* BuildingTIV, ContentsTIV, LocPerilsCovered, OccupancyCode, Latitude, Longitude.

## Definition of location, account and policy
A location is identified by each unique combination of the values in PortNumber, AccNumber and LocNumber. Generally each record in the OED location file should represent a location with some exceptions as detailed in the 'Uniqueness requirements' section below.

An account is identified by each unique combination of the values in PortNumber, AccNumber. Each account in the OED account file, if provided, must have at least one PolNumber. 

A policy is identified by each unique combination of the values in PortNumber, AccNumber, PolNumber, and LayerNumber, although LayerNumber is an optional field.  Generally each record in the OED account file should represent a policy with some exceptions as detailed in the 'Uniqueness requirements' section below.

## Uniqueness requirements 

The OED account and location file can contain duplicate records of location, account, and policies in order to accommodate additional policy data.  This includes multiple 'special conditions' that apply to a policy, and step policies with multiple steps per policy

The following fields are required to contain a unique combination of values per record.

### OasisLMF 1.15.21-LTS
* OED location
  * PortNumber, AccNumber, LocNumber
* OED account
  * PortNumber, AccNumber, PolNumber, LayerNumber, CondNumber, StepNumber

### OasisLMF 1.23-LTS and later
* OED location
  * PortNumber, AccNumber, LocNumber, CondTag
* OED account
  * PortNumber, AccNumber, PolNumber, LayerNumber, CondNumber, CondTag, StepNumber

## Special Conditions
Special conditions may be used to represent extra policy conditions applying to one or more subsets of locations under a policy.

OasisLMF does not yet support special conditions that apply to particular perils as it does not read the financial terms peril codes (LocPeril, CondPeril, PolPeril, AccPeril) and assumes all financial terms apply to all locations regardless of peril. Therefore any locations that are not subject to a particular peril sublimit being modelled should be removed from the input files.

### OasisLMF 1.15.21-LTS
The CondNumber field in the location file identifies the subset of locations to which a policy condition applies. There may not be more than one condition on a location, i.e. no duplicates of PortNumber, AccNumber, LocNumber in the location file with different CondNumbers.  

The account file contains the CondNumber field which links to the CondNumber in the location file. Every CondNumber must apply to all policies under an account if there is more than one policy. More than one CondNumber per policy gives rise to valid duplicates of the policy record. The financial terms for a given CondNumber must match if repeated on multiple rows. Only one priority of condition is supported, i.e. CondPriority = 1 for all CondNumbers.


### OasisLMF 1.23-LTS and later
The CondTag field in the location file identifies the subset of locations to which a policy condition applies. 

The account file contains the CondTag field, which links to the CongTag in the location file, and a CondNumber field which represents a particular set of financial terms for the condition. For each policy in the account file, one or more CondTags may be specified along with a CondNumber. More than one pair of CondTag, CondNumber values per policy gives rise to valid duplicates of the policy record. The financial terms for a given CondNumber must match if repeated on multiple rows. Each CondTag is assigned a CondPriority which is the order in which the fiancial terms apply. CondPriority must be the same for a given CondTag if it occurs on multiple records within an account.

#### More than one condition per location

If a location is subject to more than one condition, they must be heirarchal. This means that if the location is assigned more than one CondTag in the location file, giving rise to valid duplicates in the location file, then the associated CondTags in the account file must have different CondPriorities. The terms are applied in order of CondPriority in this case. There may not be multiple CondTags for a location in the location file which link to the CondTags in the account file with the same CondPriority.

## Cross validation
If both OED location and account file are provided, the list of unique values of PortNumber, AccNumber must match between the files.

### OasisLMF 1.15.21-LTS
If CondNumber is used in the location file for an account, each CondNumber value appearing in the location file must be appear for every PolNumber,LayerNumber under the same account in the account file.

### OasisLMF 1.23-LTS and later
If CondTag is used in the location file for an account, each CondTag value appearing in the location file must be specified for at least one PolNumber,LayerNumber under the same account in the account file.

## Step Policies
A policy record in the account file is a step policy if data is populated in the StepNumber field and some of the other step fields (see 'OED Input Fields' within the [Open Exposure Data spec](https://github.com/OasisLMF/OpenDataStandards/blob/master/OpenExposureData/Docs/OpenExposureData_Spec.xlsx) with BackEndTableName 'StepFunctions' and 'Steps' for a full list). It is common to have multiple records in the account file for a step policy, with each record corresponding to each defined step in the policy.

Step policies are incompatible with any other financial terms in OED account and location files in OasisLMF within the same account. Loc, Cond, Pol and Acc financial fields should not be populated for accounts containing policies which have step terms. In particular, the CondNumber field must not be populated when StepNumber is populated, and vice versa StepNumber should not be populated where there are CondNumbers populated. Step policies and normal policies may co-exist in the same set of OED files as long as they belong to different accounts.

Step policy financial terms operate on modelled losses generated for BuildingTIV and ContentsTIV only. OtherTIV and BITIV may be populated and used to generate ground up losses, but losses for these coverages will not be used as inputs to the step policy terms.