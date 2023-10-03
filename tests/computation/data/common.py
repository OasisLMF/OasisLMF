__all__ = [
    'MIN_RUN_SETTINGS',
    'IL_RUN_SETTINGS',
    'RI_RUN_SETTINGS',
    'MIN_MODEL_SETTINGS',
    'INVALID_RUN_SETTINGS',
    'RI_AAL_SETTINGS',
    'PARQUET_GUL_SETTINGS',
    'MIN_KEYS',
    'MIN_KEYS_ERR',
    'MIN_LOC',
    'MIN_ACC',
    'MIN_INF',
    'MIN_SCP',
    'FAKE_PRE_ANALYSIS_MODULE',
    'FAKE_IL_ITEMS_RETURN',
    'FAKE_MODEL_RUNNER',
    'FAKE_MODEL_RUNNER__OLD',
    'EXPECTED_KEYS',
    'EXPECTED_ERROR',
    'GROUP_FIELDS_MODEL_SETTINGS',
    'OLD_GROUP_FIELDS_MODEL_SETTINGS',
    'merge_dirs',
]

import os
import shutil
from pathlib import Path
import pandas as pd


def merge_dirs(src_root, dst_root):
    for root, dirs, files in os.walk(src_root):
        for f in files:
            src = os.path.join(root, f)
            rel_dst = os.path.relpath(src, src_root)
            abs_dst = os.path.join(dst_root, rel_dst)
            Path(abs_dst).parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(os.path.join(root, f), abs_dst)


INVALID_RUN_SETTINGS = {
    "model_supplier_id": "M-sup",
    "model_name_id": 'M-name',
    "model_settings": {
        "event_set": "NOT-FOUND",
        "event_occurrence_id": "NOT-FOUND"
    },
    "number_of_samples": 1,
    "gul_output": True,
    "gul_summaries": [
        {
            "id": 1,
            "eltcalc": True,
        }
    ]
}


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

IL_RUN_SETTINGS = {
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
    ],
    "il_output": True,
    "il_summaries": [
        {
            "id": 1,
            "eltcalc": True,
        }
    ]
}

RI_RUN_SETTINGS = {
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
    ],
    "il_output": True,
    "il_summaries": [
        {
            "id": 1,
            "eltcalc": True,
        }
    ],
    "ri_output": True,
    "ri_summaries": [
        {
            "id": 1,
            "eltcalc": True,
        }
    ]
}

RI_AAL_SETTINGS = {
    "model_supplier_id": "M-sup",
    "model_name_id": 'M-name',
    "model_settings": {},
    "number_of_samples": 1,
    "gul_output": True,
    "model_settings": {
    },
    "gul_summaries": [
        {
            "aalcalc": True,
            "eltcalc": True,
            "id": 1,
            "lec_output": False
        }
    ],
    "il_output": True,
    "il_summaries": [
        {
            "aalcalc": True,
            "eltcalc": True,
            "id": 1,
            "lec_output": False
        }
    ],
    "ri_output": True,
    "ri_summaries": [
        {
            "aalcalc": True,
            "eltcalc": True,
            "id": 1,
            "lec_output": False
        }
    ]
}


PARQUET_GUL_SETTINGS = {
    "model_supplier_id": "M-sup",
    "model_name_id": 'M-name',
    "model_settings": {},
    "number_of_samples": 1,
    "gul_output": True,
    "gul_summaries": [
        {
            "id": 1,
            "ord_output": {
                "elt_sample": True,
                "parquet_format": True,
            }
        }
    ]
}


MIN_MODEL_SETTINGS = {
    "version": "3",
    "model_settings": {},
    "lookup_settings": {},
    "model_default_samples": 10
}

GROUP_FIELDS_MODEL_SETTINGS = {
    "version": "3",
    "model_settings": {},
    "lookup_settings": {},
    "model_default_samples": 10,
    "data_settings": {
        "damage_group_fields": ["PortNumber", "AccNumber", "LocNumber"],
        "hazard_group_fields": ["PortNumber", "AccNumber", "LocNumber"]
    }
}

OLD_GROUP_FIELDS_MODEL_SETTINGS = {
    "version": "3",
    "model_settings": {},
    "lookup_settings": {},
    "model_default_samples": 10,
    "data_settings": {
        "group_fields": ["PortNumber", "AccNumber", "LocNumber"],
    }
}

MIN_KEYS = """LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID
1,WSS,1,154,8
1,WTC,1,54,2
1,WSS,3,154,11
1,WTC,3,54,5
"""

MIN_KEYS_ERR = """LocID,PerilID,CoverageTypeID,Status,Message
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

EXPECTED_KEYS = b'LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID,AmplificationID\n1,WSS,1,1000,8,2\n1,WTC,1,500,2,1\n1,WSS,3,1000,11,2\n1,WTC,3,500,5,1\n'
EXPECTED_ERROR = b'LocID,PerilID,CoverageTypeID,Status,Message\n1,WEC,1,noreturn,unsuported peril_id\n1,WEC,3,noreturn,unsuported peril_id\n'


FAKE_IL_ITEMS_RETURN = pd.read_csv(os.path.join(os.path.dirname(__file__), 'il_inputs_df_return.csv'))
FAKE_PRE_ANALYSIS_MODULE = os.path.join(os.path.dirname(__file__), 'fake_pre_analysis.py')

FAKE_MODEL_RUNNER = os.path.join(os.path.dirname(__file__), 'fake_model_runner')
FAKE_MODEL_RUNNER__OLD = os.path.join(os.path.dirname(__file__), 'fake_model_runner__old')
