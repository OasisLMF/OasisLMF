
------------------------------------------
--Pi Wind
------------------------------------------
Declare	@SupplierId int = (Select ISNULL(Max(SupplierId),0) + 1 From Supplier)
Declare	@ModelFamilyId int = (Select ISNULL(Max(ModelFamilyId),0) + 1 From ModelFamily)
Declare	@ModelId int = (Select ISNULL(Max(ModelId),0) + 1 From Model)
Declare	@ServiceId int = (Select ISNULL(Max(ServiceId),0) + 1 From Service)
Declare	@OasisSystemId int = (Select ISNULL(Max(OasisSystemId),0) + 1 From OasisSystem)
Declare	@OasisSystemServiceId int = (Select ISNULL(Max(OasisSystemServiceId),0) + 1 From OasisSystemService)
Declare	@ModelResourceId int = (Select ISNULL(Max(ModelResourceId),0) From ModelResource)
Declare	@ModelPerilId int = (Select ISNULL(Max(ModelPerilId),0) From ModelPeril)
Declare	@ModelCoverageTypeId int = (Select ISNULL(Max(ModelCoverageTypeId),0) From ModelCoverageType)

Declare	@ModelLicenseId int = (Select ISNULL(Max(ModelLicenseId),0) + 1 From ModelLicense)
Declare	@OasisUserId int = (Select ISNULL(Max(OasisUserId),0) + 1 From OasisUser) --to be fixed
Declare	@UserLicenseId int = (Select ISNULL(Max(UserLicenseId),0) + 1 From UserLicense)

Declare	@ResourceId int = (Select ISNULL(Max(ResourceId),0) + 1 From [Resource])
Declare	@FileId int = (Select ISNULL(Max(FileId),0) + 1 From [File])
Declare	@TransformId int = (Select ISNULL(Max(TransformId),0) + 1 From [Transform])
Declare	@FileResourceId int = (Select ISNULL(Max(FileResourceId),0) + 1 From [FileResource])
Declare @ProfileResourceId int = (Select ISNULL(Max(ProfileResourceId),0) + 1 From [ProfileResource])
Declare @ProfileId int = (Select ISNULL(Max(ProfileId),0) + 1 From [Profile])
Declare @ProfileElementId int = (Select ISNULL(Max(ProfileElementId),0) From [ProfileElement])
Declare @ProfileValueDetailID int = (Select ISNULL(Max(ProfileValueDetailID),0) + 1 From [ProfileValueDetail])

--Supplier
INSERT [dbo].[Supplier] ([SupplierID], [SupplierName], [SupplierDesc], [SupplierLegalName], [SupplierAddress], [SupplierPostcode], [SupplierTelNo], [Deleted]) 
		VALUES (@SupplierId,N'Oasis',N'Oasis',N'Oasis',N'',N'',N'',0)

--ModelFamily
INSERT [dbo].[ModelFamily] ([ModelFamilyID], [ModelFamilyName], [SupplierID]) 
		VALUES (@ModelFamilyId, N'PiWind', @SupplierId)

--Model
INSERT [dbo].[Model] ([ModelID], [ModelName], [ModelFamilyID], [ModelDescription], [VersionRef], [ReleaseDate], [Contact], [ModelTypeId], [Deleted]) 
		VALUES (@ModelId, N'PiWind', @ModelFamilyId, N'PiWind', N'1.0', N'Jun 2017', N'', 1, 0)

--Service
INSERT [dbo].[Service] ([ServiceID], [ServiceName], [ServiceDesc], [ServiceTypeId], [ModelId]) 
		VALUES (@ServiceId, N'PiWind Oasis', N'PiWind Oasis Mid Tier', 1, @ModelId)
INSERT [dbo].[Service] ([ServiceID], [ServiceName], [ServiceDesc], [ServiceTypeId], [ModelId]) 
		VALUES (@ServiceId+1, N'PiWind API', N'PiWind Oasis Key Lookup Service', 2, @ModelId)

--OasisSystem
INSERT [dbo].[OasisSystem] ([OasisSystemID], [OasisSystemName], [OasisSystemDescription], [url], [Port], [SysConfigID]) 
		VALUES (@OasisSystemId, N'PiWind Oasis Mid Tier', N'PiWind Oasis Mid Tier', N'%OASIS_API_IP%', %OASIS_API_PORT%, 4)  
INSERT [dbo].[OasisSystem] ([OasisSystemID], [OasisSystemName], [OasisSystemDescription], [url], [Port], [SysConfigID])  
		VALUES (@OasisSystemId+1, N'PiWind API', N'PiWind Lookup Service', N'http://' + N'%KEYS_SERVICE_IP%' + N':' + N'%KEYS_SERVICE_PORT%' + N'/OasisLMF/PiWind/0.0.0.1/get_keys', %KEYS_SERVICE_PORT%, NULL)

--OasisSystemService
INSERT [dbo].[OasisSystemService] ([OasisSystemServiceID], [OasisSystemID], [ServiceID]) 
		VALUES (@OasisSystemServiceId, @OasisSystemId, @ServiceId)
INSERT [dbo].[OasisSystemService] ([OasisSystemServiceID], [OasisSystemID], [ServiceID]) 
		VALUES (@OasisSystemServiceId+1, @OasisSystemId+1, @ServiceId+1)

--ModelResource
Create Table #ModelResource
		(
		ModelResourceId int identity,
		ModelResourceName nvarchar(255),
		ResourceTypeID int,
		OasisSystemID int,
		ModelID int,
		ModelResourceValue nvarchar(255),
		)

INSERT #ModelResource (ModelResourceName,ResourceTypeID,OasisSystemID,ModelID,ModelResourceValue) VALUES (N'model_file_extension', 305, @OasisSystemID+1, @ModelID, N'csv')
INSERT #ModelResource (ModelResourceName,ResourceTypeID,OasisSystemID,ModelID,ModelResourceValue) VALUES (N'PiWind API', 1000, @OasisSystemID+1, @ModelID, N'')
INSERT #ModelResource (ModelResourceName,ResourceTypeID,OasisSystemID,ModelID,ModelResourceValue) VALUES (N'ModelGroupField', 300, @OasisSystemID, @ModelID, N'LocID')
INSERT #ModelResource (ModelResourceName,ResourceTypeID,OasisSystemID,ModelID,ModelResourceValue) VALUES (N'module_supplier_id', 301, @OasisSystemID, @ModelID, N'OasisLMF') 
INSERT #ModelResource (ModelResourceName,ResourceTypeID,OasisSystemID,ModelID,ModelResourceValue) VALUES (N'model_version_id', 302, @OasisSystemID, @ModelID, N'PiWind')   
INSERT #ModelResource (ModelResourceName,ResourceTypeID,OasisSystemID,ModelID,ModelResourceValue) VALUES (N'peril_wind', 1001, @OasisSystemId, @ModelID, N'checkbox')
INSERT #ModelResource (ModelResourceName,ResourceTypeID,OasisSystemID,ModelID,ModelResourceValue) VALUES (N'event_set', 1001, @OasisSystemId, @ModelID, N'dropdown')
INSERT #ModelResource (ModelResourceName,ResourceTypeID,OasisSystemID,ModelID,ModelResourceValue) VALUES (N'event_occurance_id', 1001, @OasisSystemId, @ModelID, N'dropdown')
INSERT #ModelResource (ModelResourceName,ResourceTypeID,OasisSystemID,ModelID,ModelResourceValue) VALUES (N'event_set', 303, @OasisSystemId, @ModelID, N'P')
INSERT #ModelResource (ModelResourceName,ResourceTypeID,OasisSystemID,ModelID,ModelResourceValue) VALUES (N'event_occurance_id', 304, @OasisSystemId, @ModelID, N'1')


INSERT	[dbo].[ModelResource] ([ModelResourceID], [ModelResourceName], [ResourceTypeID], [OasisSystemID], [ModelID], [ModelResourceValue])
SELECT	[ModelResourceID] + @ModelResourceId as [ModelResourceID], 
		[ModelResourceName], 
		[ResourceTypeID], 
		[OasisSystemID], 
		[ModelID], 
		[ModelResourceValue]
FROM	#ModelResource

DROP TABLE #ModelResource

--ModelLicense
INSERT [dbo].[ModelLicense] ([ModelLicenseID], [ModelID], [CompanyID], [ModelLicenseName], [ModelVersionDescription], [LicenseStartDate], [LicenseEndDate], [LicenseType], [LicenseContractRef]) 
		VALUES (@ModelLicenseId, @ModelId, 1, N'Model License', N'Model License', '01/01/1900', '12/31/9999', 'Dummy License', '')

--OasisUser
INSERT [dbo].[OasisUser] ([OasisUserID], [OasisUserName], [ModelLicenseID], [OasisSystemID], [SystemLogin], [SystemPassword], [Django Login], [Django Password]) 
		VALUES (@OasisUserId, N'OasisUserPiWind', @ModelLicenseId, @OasisSystemId, N'Root', N'Password', N'', N'')

--UserLicense
INSERT [dbo].[UserLicense] ([UserLicenseId], [BFEUserID], [OasisUserID]) VALUES (@UserLicenseId, 1, @OasisUserId)

--ModelPeril
INSERT ModelPeril Values (@ModelPerilId+1,@ModelId,1,1,'Wind')

--ModelPeril
INSERT ModelCoverageType Values (@ModelCoverageTypeId+1,@ModelId,1,'1','Buildings')
INSERT ModelCoverageType Values (@ModelCoverageTypeId+2,@ModelId,2,'2','Other Structures')
INSERT ModelCoverageType Values (@ModelCoverageTypeId+3,@ModelId,3,'3','Contents')
INSERT ModelCoverageType Values (@ModelCoverageTypeId+4,@ModelId,4,'4','Business Interuption')

---------------------------------------------------------
--Transforms
---------------------------------------------------------
--Params
Set	@ResourceId = (Select ISNULL(Max(ResourceId),0) + 1 From [Resource])
Set	@FileId = (Select ISNULL(Max(FileId),0) + 1 From [File])
Set	@FileResourceId = (Select ISNULL(Max(FileResourceId),0) + 1 From [FileResource])
Set	@TransformId = (Select ISNULL(Max(TransformId),0) + 1 From [Transform])
Set	@ProfileId = (Select ISNULL(Max(ProfileId),0) + 1 From [Profile])

--Loc
--
-- |XSD1|-->|XSLT1|-->|XSD2a|
--                       |
--                       V
--                    |XSD2B|-->|XSLT2|-->|XSD3|
--                       |
--                       V
--                   |Profile1|
--Acc
--
-- |XSD4|-->|XSLT3|-->|XSD5a|
--                       |
--                       V
--                    |XSD5b|
--                       |
--                       V
--                   |Profile2|

--Declare Resource IDs
Declare @FromLocXSDResourceID		int = @ResourceId + 0  -- XSD1
Declare @XSLTLocResourceID			int = @ResourceId + 1  -- XSLT1
Declare @ToLocAXSDResourceID		int = @ResourceId + 2  -- XSD2a
Declare @ToLocBXSDResourceID		int = @ResourceId + 3  -- XSD2b
Declare @FromModelXSDResourceID		int = @ResourceId + 4  -- XSLT2
Declare @XSLTModelResourceID		int = @ResourceId + 5  -- XSLT2
Declare @ToModelXSDResourceID		int = @ResourceId + 6  -- XSD3
Declare @LocProfileResourceID		int = @ResourceId + 7  -- Profile1
Declare @FromAccXSDResourceID		int = @ResourceId + 8  -- XSD4
Declare @XSLTAccResourceID			int = @ResourceId + 9  -- XSLT3
Declare @ToAccAXSDResourceID		int = @ResourceId + 10 -- XSD5a
Declare @ToAccBXSDResourceID		int = @ResourceId + 11 -- XSD5b
Declare @AccProfileResourceID		int = @ResourceId + 12 -- Profile2

--Declare File IDs
Declare @FromLocXSDFileID			int = @FileId + 0  -- XSD1
Declare @XSLTLocFileID				int = @FileId + 1  -- XSLT1
Declare @ToLocAXSDFileID			int = @FileId + 2  -- XSD2a
Declare @ToLocBXSDFileID			int = @FileId + 3  -- XSD2b
Declare @XSLTModelFileID			int = @FileId + 4  -- XSLT2
Declare @ToModelXSDFileID			int = @FileId + 5  -- XSD3
Declare @FromAccXSDFileID			int = @FileId + 6  -- XSD4
Declare @XSLTAccFileID				int = @FileId + 7  -- XSLT3
Declare @ToAccAXSDFileID			int = @FileId + 8  -- XSD5a
Declare @ToAccBXSDFileID			int = @FileId + 9  -- XSD5b

--Declare File Resource IDs
Declare @FromLocXSDFileResourceID	int = @FileResourceId + 0  -- XSD1
Declare @XSLTLocFileResourceID		int = @FileResourceId + 1  -- XSLT1
Declare @ToLocAXSDFileResourceID	int = @FileResourceId + 2  -- XSD2a
Declare @ToLocBXSDFileResourceID	int = @FileResourceId + 3  -- XSD2b
Declare @FromModelXSDFileResourceID int = @FileResourceId + 4  -- XSD2b
Declare @XSLTModelFileResourceID	int = @FileResourceId + 5  -- XSLT2
Declare @ToModelXSDFileResourceID	int = @FileResourceId + 6  -- XSD3
Declare @FromAccXSDFileResourceID	int = @FileResourceId + 7  -- XSD4
Declare @XSLTAccFileResourceID		int = @FileResourceId + 8  -- XSLT3
Declare @ToAccAXSDFileResourceID	int = @FileResourceId + 9  -- XSD5a
Declare @ToAccBXSDFileResourceID	int = @FileResourceId + 10 -- XSD5b

--Declare Profile IDs
Declare @LocProfileId int = @ProfileID     -- Profile1
Declare @AccProfileId int = @ProfileID + 1 -- Profile2

-------------insert data------------------------
--Transform
Insert Into Transform Values (@TransformId,'Wind Source to Canonical','Source AccLoc to Canonical for PiWind',1)
Insert Into Transform Values (@TransformId+1,'PiWind Canonical to Model','Canonical to Model for PiWind',2)

--Resource
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@FromLocXSDResourceID,	'Transform',@TransformId,NULL,120)
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@XSLTLocResourceID,		'Transform',@TransformId,NULL,124)
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@ToLocAXSDResourceID,		'Transform',@TransformId,NULL,121)
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@ToLocBXSDResourceID,		'Transform',@TransformId,NULL,129)
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@FromModelXSDResourceID,	'Transform',@TransformId+1,NULL,120)
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@XSLTModelResourceID,		'Transform',@TransformId+1,NULL,124)
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@ToModelXSDResourceID,	'Transform',@TransformId+1,NULL,121)
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@LocProfileResourceID,	'Transform',@TransformId,NULL,118)
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@FromAccXSDResourceID,	'Transform',@TransformId,NULL,122)
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@XSLTAccResourceID,		'Transform',@TransformId,NULL,125)
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@ToAccAXSDResourceID,		'Transform',@TransformId,NULL,123)
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@ToAccBXSDResourceID,		'Transform',@TransformId,NULL,130)
INSERT [dbo].[Resource] ([ResourceId], [ResourceTable], [ResourceKey], [ResourceQualifier], [ResourceTypeID]) VALUES (@AccProfileResourceID,	'Transform',@TransformId,NULL,119)

--File
INSERT [dbo].[File] ([FileId], [FileName], [FileDesc], [SourceID], [OwnerID], [LocationID], [DateTimeCreated], [DateTimeUpdated], [DateTimeDeleted], [OwnerNameCreated], [OwnerNameUpdated], [OwnerNameDeleted], [FileTypeId]) VALUES (@FromLocXSDFileID,		N'Generic_Windstorm_SourceLoc.xsd', N'Source Loc Validation File', 1, 1, 106, getdate(), getdate(), NULL, N'Sys', N'Sys', NULL, 109) --XSD1
INSERT [dbo].[File] ([FileId], [FileName], [FileDesc], [SourceID], [OwnerID], [LocationID], [DateTimeCreated], [DateTimeUpdated], [DateTimeDeleted], [OwnerNameCreated], [OwnerNameUpdated], [OwnerNameDeleted], [FileTypeId]) VALUES (@XSLTLocFileID,			N'MappingMapToGeneric_Windstorm_CanLoc_A.xslt', N'Source to Canonical Loc Tranformation File', 1, 1, 105, getdate(), getdate(), NULL, N'Sys', N'Sys', NULL, 110) --XSLT1
INSERT [dbo].[File] ([FileId], [FileName], [FileDesc], [SourceID], [OwnerID], [LocationID], [DateTimeCreated], [DateTimeUpdated], [DateTimeDeleted], [OwnerNameCreated], [OwnerNameUpdated], [OwnerNameDeleted], [FileTypeId]) VALUES (@ToLocAXSDFileID,		N'Generic_Windstorm_CanLoc_A.xsd', N'Canonical Loc Validation File', 1, 1, 106, getdate(), getdate(), NULL, N'Sys', N'Sys', NULL, 109) --XSD2a
INSERT [dbo].[File] ([FileId], [FileName], [FileDesc], [SourceID], [OwnerID], [LocationID], [DateTimeCreated], [DateTimeUpdated], [DateTimeDeleted], [OwnerNameCreated], [OwnerNameUpdated], [OwnerNameDeleted], [FileTypeId]) VALUES (@ToLocBXSDFileID,		N'Generic_Windstorm_CanLoc_B.xsd', N'Canonical Loc Profile File', 1, 1, 106, getdate(), getdate(), NULL, N'Sys', N'Sys', NULL, 109) --XSD2b
INSERT [dbo].[File] ([FileId], [FileName], [FileDesc], [SourceID], [OwnerID], [LocationID], [DateTimeCreated], [DateTimeUpdated], [DateTimeDeleted], [OwnerNameCreated], [OwnerNameUpdated], [OwnerNameDeleted], [FileTypeId]) VALUES (@XSLTModelFileID,		N'MappingMapToPiWind_ModelLoc.xslt', N'Canonical to Model Loc Tranformation File', 1, 1, 105, getdate(), getdate(), NULL, N'Sys', N'Sys', NULL, 110) --XSLT2
INSERT [dbo].[File] ([FileId], [FileName], [FileDesc], [SourceID], [OwnerID], [LocationID], [DateTimeCreated], [DateTimeUpdated], [DateTimeDeleted], [OwnerNameCreated], [OwnerNameUpdated], [OwnerNameDeleted], [FileTypeId]) VALUES (@ToModelXSDFileID,		N'PiWind_ModelLoc.xsd', N'Canonical Loc Validation File', 1, 1, 106, getdate(), getdate(), NULL, N'Sys', N'Sys', NULL, 109) --XSD3
INSERT [dbo].[File] ([FileId], [FileName], [FileDesc], [SourceID], [OwnerID], [LocationID], [DateTimeCreated], [DateTimeUpdated], [DateTimeDeleted], [OwnerNameCreated], [OwnerNameUpdated], [OwnerNameDeleted], [FileTypeId]) VALUES (@FromAccXSDFileID,		N'Generic_SourceAcc.xsd', N'Source Acc Validation File', 1, 1, 106, getdate(), getdate(), NULL, N'Sys', N'Sys', NULL, 109) --XSD4
INSERT [dbo].[File] ([FileId], [FileName], [FileDesc], [SourceID], [OwnerID], [LocationID], [DateTimeCreated], [DateTimeUpdated], [DateTimeDeleted], [OwnerNameCreated], [OwnerNameUpdated], [OwnerNameDeleted], [FileTypeId]) VALUES (@XSLTAccFileID,			N'MappingMapToGeneric_CanAcc_A.xslt', N'Source to Canonical Acc Tranformation File', 1, 1, 105, getdate(), getdate(), NULL, N'Sys', N'Sys', NULL, 110) --XSLT3
INSERT [dbo].[File] ([FileId], [FileName], [FileDesc], [SourceID], [OwnerID], [LocationID], [DateTimeCreated], [DateTimeUpdated], [DateTimeDeleted], [OwnerNameCreated], [OwnerNameUpdated], [OwnerNameDeleted], [FileTypeId]) VALUES (@ToAccAXSDFileID,		N'Generic_CanAcc_A.xsd', N'Canonical Acc Validation File', 1, 1, 106, getdate(), getdate(), NULL, N'Sys', N'Sys', NULL, 109) --XSD5a
INSERT [dbo].[File] ([FileId], [FileName], [FileDesc], [SourceID], [OwnerID], [LocationID], [DateTimeCreated], [DateTimeUpdated], [DateTimeDeleted], [OwnerNameCreated], [OwnerNameUpdated], [OwnerNameDeleted], [FileTypeId]) VALUES (@ToAccBXSDFileID,		N'Generic_CanAcc_B.xsd', N'Canonical Acc Profile File', 1, 1, 106, getdate(), getdate(), NULL, N'Sys', N'Sys', NULL, 109) --XSD5b

--FileResource
INSERT [dbo].[FileResource] ([FileResourceId], [FileId], [ResourceId]) VALUES (@FromLocXSDFileResourceID,	@FromLocXSDFileID,	@FromLocXSDResourceID)
INSERT [dbo].[FileResource] ([FileResourceId], [FileId], [ResourceId]) VALUES (@XSLTLocFileResourceID,		@XSLTLocFileID,		@XSLTLocResourceID)
INSERT [dbo].[FileResource] ([FileResourceId], [FileId], [ResourceId]) VALUES (@ToLocAXSDFileResourceID,	@ToLocAXSDFileID,	@ToLocAXSDResourceID)
INSERT [dbo].[FileResource] ([FileResourceId], [FileId], [ResourceId]) VALUES (@ToLocBXSDFileResourceID,	@ToLocBXSDFileID,	@ToLocBXSDResourceID)

INSERT [dbo].[FileResource] ([FileResourceId], [FileId], [ResourceId]) VALUES (@FromModelXSDFileResourceID,	@ToLocBXSDFileID,	@FromModelXSDResourceID)
INSERT [dbo].[FileResource] ([FileResourceId], [FileId], [ResourceId]) VALUES (@XSLTModelFileResourceID,	@XSLTModelFileID,	@XSLTModelResourceID)
INSERT [dbo].[FileResource] ([FileResourceId], [FileId], [ResourceId]) VALUES (@ToModelXSDFileResourceID,	@ToModelXSDFileID,	@ToModelXSDResourceID)

INSERT [dbo].[FileResource] ([FileResourceId], [FileId], [ResourceId]) VALUES (@FromAccXSDFileResourceID,	@FromAccXSDFileID,	@FromAccXSDResourceID)
INSERT [dbo].[FileResource] ([FileResourceId], [FileId], [ResourceId]) VALUES (@XSLTAccFileResourceID,		@XSLTAccFileID,		@XSLTAccResourceID)
INSERT [dbo].[FileResource] ([FileResourceId], [FileId], [ResourceId]) VALUES (@ToAccAXSDFileResourceID,	@ToAccAXSDFileID,	@ToAccAXSDResourceID)
INSERT [dbo].[FileResource] ([FileResourceId], [FileId], [ResourceId]) VALUES (@ToAccBXSDFileResourceID,	@ToAccBXSDFileID,	@ToAccBXSDResourceID)

--Profile
INSERT [dbo].[Profile] ([ProfileID], [ProfileName], [ProfileTypeID]) VALUES (@LocProfileId, 'Canonical Loc Windstorm', 1)
INSERT [dbo].[Profile] ([ProfileID], [ProfileName], [ProfileTypeID]) VALUES (@AccProfileId, 'Canonical Acc Windstorm', 2)

--ProfileResource
INSERT [dbo].[ProfileResource] ([ProfileResourceID], [ProfileID], [ResourceId]) VALUES (@ProfileResourceId, @LocProfileId, @LocProfileResourceID)
INSERT [dbo].[ProfileResource] ([ProfileResourceID], [ProfileID], [ResourceId]) VALUES (@ProfileResourceId+1, @AccProfileId, @AccProfileResourceID)

--ProfileElements
insert into [ProfileElement] values (1+@ProfileElementId,'ROW_ID',@LocProfileId,5,3)
insert into [ProfileElement] values (2+@ProfileElementId,'ACCNTNUM',@LocProfileId,16,6)
insert into [ProfileElement] values (3+@ProfileElementId,'LOCNUM',@LocProfileId,5,10)
insert into [ProfileElement] values (4+@ProfileElementId,'POSTALCODE',@LocProfileId,5,7)
insert into [ProfileElement] values (5+@ProfileElementId,'COUNTY',@LocProfileId,5,50)
insert into [ProfileElement] values (6+@ProfileElementId,'COUNTYCODE',@LocProfileId,5,30)
insert into [ProfileElement] values (7+@ProfileElementId,'CRESTA',@LocProfileId,5,43)
insert into [ProfileElement] values (8+@ProfileElementId,'CITY',@LocProfileId,5,59)
insert into [ProfileElement] values (9+@ProfileElementId,'CITYCODE',@LocProfileId,5,60)
insert into [ProfileElement] values (10+@ProfileElementId,'STATE',@LocProfileId,5,51)
insert into [ProfileElement] values (11+@ProfileElementId,'STATECODE',@LocProfileId,5,29)
insert into [ProfileElement] values (12+@ProfileElementId,'ADDRMATCH',@LocProfileId,5,48)
insert into [ProfileElement] values (13+@ProfileElementId,'COUNTRY',@LocProfileId,5,52)
insert into [ProfileElement] values (14+@ProfileElementId,'COUNTRYGEOID',@LocProfileId,5,44)
insert into [ProfileElement] values (15+@ProfileElementId,'CNTRYSCHEME',@LocProfileId,5,45)
insert into [ProfileElement] values (16+@ProfileElementId,'CNTRYCODE',@LocProfileId,5,46)
insert into [ProfileElement] values (17+@ProfileElementId,'LATITUDE',@LocProfileId,5,8)
insert into [ProfileElement] values (18+@ProfileElementId,'LONGITUDE',@LocProfileId,5,9)
insert into [ProfileElement] values (19+@ProfileElementId,'BLDGSCHEME',@LocProfileId,5,11)
insert into [ProfileElement] values (20+@ProfileElementId,'BLDGCLASS',@LocProfileId,5,12)
insert into [ProfileElement] values (21+@ProfileElementId,'OCCSCHEME',@LocProfileId,5,13)
insert into [ProfileElement] values (22+@ProfileElementId,'OCCTYPE',@LocProfileId,5,14)
insert into [ProfileElement] values (23+@ProfileElementId,'YEARBUILT',@LocProfileId,5,31)
insert into [ProfileElement] values (24+@ProfileElementId,'YEARUPGRAD',@LocProfileId,5,32)
insert into [ProfileElement] values (25+@ProfileElementId,'NUMSTORIES',@LocProfileId,5,33)
insert into [ProfileElement] values (26+@ProfileElementId,'NUMBLDGS',@LocProfileId,5,34)
insert into [ProfileElement] values (27+@ProfileElementId,'WSCV1VAL',@LocProfileId,5,2)
insert into [ProfileElement] values (28+@ProfileElementId,'WSCV2VAL',@LocProfileId,5,2)
insert into [ProfileElement] values (29+@ProfileElementId,'WSCV3VAL',@LocProfileId,5,2)
insert into [ProfileElement] values (30+@ProfileElementId,'WSCV4VAL',@LocProfileId,5,2)
insert into [ProfileElement] values (31+@ProfileElementId,'WSCV5VAL',@LocProfileId,5,2)
insert into [ProfileElement] values (32+@ProfileElementId,'WSCV6VAL',@LocProfileId,5,2)
insert into [ProfileElement] values (33+@ProfileElementId,'WSCV7VAL',@LocProfileId,5,2)
insert into [ProfileElement] values (34+@ProfileElementId,'WSCV8VAL',@LocProfileId,5,2)
insert into [ProfileElement] values (35+@ProfileElementId,'WSCV9VAL',@LocProfileId,5,2)
insert into [ProfileElement] values (36+@ProfileElementId,'WSCV10VAL',@LocProfileId,5,2)
insert into [ProfileElement] values (37+@ProfileElementId,'WSCV1LIMIT',@LocProfileId,14,15)
insert into [ProfileElement] values (38+@ProfileElementId,'WSCV2LIMIT',@LocProfileId,14,15)
insert into [ProfileElement] values (39+@ProfileElementId,'WSCV3LIMIT',@LocProfileId,14,15)
insert into [ProfileElement] values (40+@ProfileElementId,'WSCV4LIMIT',@LocProfileId,14,15)
insert into [ProfileElement] values (41+@ProfileElementId,'WSCV5LIMIT',@LocProfileId,14,15)
insert into [ProfileElement] values (42+@ProfileElementId,'WSCV6LIMIT',@LocProfileId,14,15)
insert into [ProfileElement] values (43+@ProfileElementId,'WSCV7LIMIT',@LocProfileId,14,15)
insert into [ProfileElement] values (44+@ProfileElementId,'WSCV8LIMIT',@LocProfileId,14,15)
insert into [ProfileElement] values (45+@ProfileElementId,'WSCV9LIMIT',@LocProfileId,14,15)
insert into [ProfileElement] values (46+@ProfileElementId,'WSCV10LMT',@LocProfileId,14,15)
insert into [ProfileElement] values (47+@ProfileElementId,'WSCV1DED',@LocProfileId,14,16)
insert into [ProfileElement] values (48+@ProfileElementId,'WSCV2DED',@LocProfileId,14,16)
insert into [ProfileElement] values (49+@ProfileElementId,'WSCV3DED',@LocProfileId,14,16)
insert into [ProfileElement] values (50+@ProfileElementId,'WSCV4DED',@LocProfileId,14,16)
insert into [ProfileElement] values (51+@ProfileElementId,'WSCV5DED',@LocProfileId,14,16)
insert into [ProfileElement] values (52+@ProfileElementId,'WSCV6DED',@LocProfileId,14,16)
insert into [ProfileElement] values (53+@ProfileElementId,'WSCV7DED',@LocProfileId,14,16)
insert into [ProfileElement] values (54+@ProfileElementId,'WSCV8DED',@LocProfileId,14,16)
insert into [ProfileElement] values (55+@ProfileElementId,'WSCV9DED',@LocProfileId,14,16)
insert into [ProfileElement] values (56+@ProfileElementId,'WSCV10DED',@LocProfileId,14,16)
insert into [ProfileElement] values (57+@ProfileElementId,'WSSITELIM',@LocProfileId,14,17)
insert into [ProfileElement] values (58+@ProfileElementId,'WSSITEDED',@LocProfileId,14,18)
insert into [ProfileElement] values (59+@ProfileElementId,'WSCOMBINEDLIM',@LocProfileId,14,19)
insert into [ProfileElement] values (60+@ProfileElementId,'WSCOMBINEDDED',@LocProfileId,14,20)
insert into [ProfileElement] values (61+@ProfileElementId,'COND1TYPE',@LocProfileId,14,53)
insert into [ProfileElement] values (62+@ProfileElementId,'COND1NAME',@LocProfileId,14,54)
insert into [ProfileElement] values (63+@ProfileElementId,'COND1DEDUCTIBLE',@LocProfileId,14,56)
insert into [ProfileElement] values (64+@ProfileElementId,'COND1LIMIT',@LocProfileId,14,55)
insert into [ProfileElement] values (65+@ProfileElementId,'ROOFGEOM',@LocProfileId,5,37)
insert into [ProfileElement] values (66+@ProfileElementId,'ROW_ID',@AccProfileId,16,1)
insert into [ProfileElement] values (67+@ProfileElementId,'ACCNTNUM',@AccProfileId,16,6)
insert into [ProfileElement] values (68+@ProfileElementId,'POLICYNUM',@AccProfileId,16,21)
insert into [ProfileElement] values (69+@ProfileElementId,'POLICYTYPE',@AccProfileId,16,22)
insert into [ProfileElement] values (70+@ProfileElementId,'UNDCOVAMT',@AccProfileId,20,23)
insert into [ProfileElement] values (71+@ProfileElementId,'PARTOF',@AccProfileId,20,24)
insert into [ProfileElement] values (72+@ProfileElementId,'MINDEDAMT',@AccProfileId,20,25)
insert into [ProfileElement] values (73+@ProfileElementId,'MAXDEDAMT',@AccProfileId,20,26)
insert into [ProfileElement] values (74+@ProfileElementId,'BLANDEDAMT',@AccProfileId,20,27)
insert into [ProfileElement] values (75+@ProfileElementId,'BLANLIMAMT',@AccProfileId,20,28)



--ProfileValueDetail
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (1+@ProfileValueDetailID ,27+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+1,1)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (2+@ProfileValueDetailID ,28+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+3,2)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (3+@ProfileValueDetailID ,29+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+4,3)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (4+@ProfileValueDetailID ,30+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+1,4)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (5+@ProfileValueDetailID ,31+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+2,5)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (6+@ProfileValueDetailID ,32+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+3,6)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (7+@ProfileValueDetailID ,33+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+4,7)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (8+@ProfileValueDetailID ,34+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+2,8)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (9+@ProfileValueDetailID ,35+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+2,9)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (10+@ProfileValueDetailID ,36+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+2,10)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (11+@ProfileValueDetailID ,37+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+1,1)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (12+@ProfileValueDetailID ,38+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+3,2)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (13+@ProfileValueDetailID ,39+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+4,3)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (14+@ProfileValueDetailID ,40+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+1,4)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (15+@ProfileValueDetailID ,41+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+2,5)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (16+@ProfileValueDetailID ,42+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+3,6)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (17+@ProfileValueDetailID ,43+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+4,7)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (18+@ProfileValueDetailID ,44+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+2,8)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (19+@ProfileValueDetailID ,45+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+2,9)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (20+@ProfileValueDetailID ,46+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+2,10)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (21+@ProfileValueDetailID ,47+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+1,1)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (22+@ProfileValueDetailID ,48+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+3,2)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (23+@ProfileValueDetailID ,49+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+4,3)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (24+@ProfileValueDetailID ,50+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+1,4)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (25+@ProfileValueDetailID ,51+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+2,5)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (26+@ProfileValueDetailID ,52+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+3,6)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (27+@ProfileValueDetailID ,53+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+4,7)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (28+@ProfileValueDetailID ,54+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+2,8)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (29+@ProfileValueDetailID ,55+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+2,9)
INSERT [dbo].[ProfileValueDetail] ([ProfileValueDetailID], [ProfileElementID],[PerilID],[CoverageTypeID],[ElementDimensionID]) VALUES (30+@ProfileValueDetailID ,56+@ProfileElementId,@ModelPerilId+1,@ModelCoverageTypeId+2,10)


GO
