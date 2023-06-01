__all__ = [
    'MIN_RUN_SETTINGS',
    'MIN_MODEL_SETTINGS',
    'MIN_KEYS',
    'MIN_LOC',
    'MIN_ACC',
    'MIN_INF',
    'MIN_SCP',
    'FAKE_PRE_ANALYSIS_MODULE',
]

from os import path

MIN_RUN_SETTINGS = {
    "model_supplier_id": "M-sup",
    "model_name_id": 'M-name',
    "model_settings": {},
    "number_of_samples": 1,
    "gul_output": True,
    "gul_summaries": [
        {
            "id": 1,
            "eltcalc": True,
        }
    ]
}

MIN_MODEL_SETTINGS = {
    "version": "3",
    "model_settings": {},
    "lookup_settings": {},
    "model_default_samples": 10
}


MIN_KEYS = """LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID
1,WSS,1,154,8
1,WTC,1,54,2
1,WSS,3,154,11
1,WTC,3,54,5
"""

MIN_LOC = """PortNumber,AccNumber,LocNumber,IsTenant,BuildingID,CountryCode,Latitude,Longitude,StreetAddress,PostalCode,OccupancyCode,ConstructionCode,LocPerilsCovered,BuildingTIV,OtherTIV,ContentsTIV,BITIV,LocCurrency,OEDVersion
1,A11111,10002082046,1,1,GB,52.76698052,-0.895469856,1 ABINGDON ROAD,LE13 0HL,1050,5000,WW1,220000,0,0,0,GBP,2.0.0
"""
MIN_ACC = """PortNumber,AccNumber,AccCurrency,PolNumber,PolPerilsCovered,PolInceptionDate,PolExpiryDate,LayerNumber,LayerParticipation,LayerLimit,LayerAttachment,OEDVersion
1,A11111,GBP,Layer1,WW1,2018-01-01,2018-12-31,1,0.3,5000000,500000,2.0.0
"""
MIN_INF = """ReinsNumber,ReinsLayerNumber,ReinsName,ReinsPeril,ReinsInceptionDate,ReinsExpiryDate,CededPercent,RiskLimit,RiskAttachment,OccLimit,OccAttachment,PlacedPercent,ReinsCurrency,InuringPriority,ReinsType,RiskLevel,UseReinsDates,OEDVersion
1,1,ABC QS,WW1,2018-01-01,2018-12-31,1,0,0,0,0,1,GBP,1,SS,LOC,N,2.0.0
"""
MIN_SCP = """ReinsNumber,PortNumber,AccNumber,PolNumber,LocGroup,LocNumber,CedantName,ProducerName,LOB,CountryCode,ReinsTag,CededPercent,OEDVersion
1,1,A11111,,,10002082047,,,,,,0.1,2.0.0
"""

FAKE_PRE_ANALYSIS_MODULE = path.join(path.dirname(__file__), 'fake_pre_analysis.py')
