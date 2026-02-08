__all__ = [
    'MIN_RUN_SETTINGS',
    'IL_RUN_SETTINGS',
    'RI_RUN_SETTINGS',
    'MIN_MODEL_SETTINGS',
    'INVALID_RUN_SETTINGS',
    'RI_AAL_SETTINGS',
    'RI_ALL_OUTPUT_SETTINGS',
    'PARQUET_GUL_SETTINGS',
    'MIN_KEYS',
    'MIN_KEYS_ERR',
    'MIN_LOC',
    'N2_LOC',
    'MIN_ACC',
    'MIN_INF',
    'MIN_SCP',
    'FAKE_PRE_ANALYSIS_MODULE',
    'FAKE_COMPLEX_LOOKUP_MODULE',
    'FAKE_MODEL_SETTINGS_JSON',
    'FAKE_IL_ITEMS_RETURN',
    'FAKE_MODEL_RUNNER',
    'FAKE_MODEL_RUNNER__OLD',
    'EXPECTED_KEYS',
    'EXPECTED_ERROR',
    'EXPECTED_KEYS_COMPLEX',
    'EXPECTED_ERROR_COMPLEX',
    'GROUP_FIELDS_MODEL_SETTINGS',
    'OLD_GROUP_FIELDS_MODEL_SETTINGS',
    'merge_dirs',
    'ALL_EXPECTED_SCRIPT',
    'MIN_RUN_CORRELATIONS_SETTINGS',
    'CORRELATIONS_MODEL_SETTINGS',
    'EXPECTED_CORRELATION_CSV'

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
            "ord_output": {
                "elt_sample": True,
            }
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
            "ord_output": {
                "elt_sample": True,
            }
        }
    ]
}

MIN_RUN_CORRELATIONS_SETTINGS = {
    "model_supplier_id": "M-sup",
    "model_name_id": 'M-name',
    "model_settings": {
        'correlation_settings': [
            {"peril_correlation_group": 1, "damage_correlation_value": "0.7", "hazard_correlation_value": "0.4"},
            {"peril_correlation_group": 2, "damage_correlation_value": "0.5", "hazard_correlation_value": "0.2"}
        ]
    },
    "number_of_samples": 1,
    "gul_output": True,
    "gul_summaries": [
        {
            "id": 1,
            "ord_output": {
                "elt_sample": True,
            }
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
            "ord_output": {
                "elt_sample": True,
            }
        }
    ],
    "il_output": True,
    "il_summaries": [
        {
            "id": 1,
            "ord_output": {
                "elt_sample": True,
            }
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
            "ord_output": {
                "elt_sample": True,
            }
        }
    ],
    "il_output": True,
    "il_summaries": [
        {
            "id": 1,
            "ord_output": {
                "elt_sample": True,
            }
        }
    ],
    "ri_output": True,
    "ri_summaries": [
        {
            "id": 1,
            "ord_output": {
                "elt_sample": True,
            }
        }
    ],
    "rl_output": True,
    "rl_summaries": [
        {
            "id": 1,
            "ord_output": {
                "elt_sample": True,
            }
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
            "id": 1,
            "ord_output": {
                "elt_sample": True,
                "alt_period": True,
                "ept_mean_sample_oep": False,
            }
        }
    ],
    "il_output": True,
    "il_summaries": [
        {
            "id": 1,
            "ord_output": {
                "elt_sample": True,
                "alt_period": True,
                "ept_mean_sample_oep": False,
            }
        }
    ],
    "ri_output": True,
    "ri_summaries": [
        {
            "id": 1,
            "ord_output": {
                "elt_sample": True,
                "alt_period": True,
                "ept_mean_sample_oep": False,
            }
        }
    ]
}


RI_ALL_OUTPUT_SETTINGS = {
    "model_supplier_id": "M-sup",
    "model_name_id": 'M-name',
    "model_settings": {},
    "number_of_samples": 1,
    "gul_output": True,
    "return_periods": [1, 10],
    "event_ids": [1, 2],
    "model_settings": {
    },
    "gul_summaries": [
        {
            "id": 1,
            "ord_output": {
                "return_period_file": False,
                "plt_sample": True,
                "plt_quantile": True,
                "plt_moment": True,
                "elt_sample": True,
                "elt_quantile": True,
                "elt_moment": True,
                "alt_period": True,
                "alt_meanonly": True,
                "ept_full_uncertainty_aep": True,
                "ept_full_uncertainty_oep": True,
                "ept_mean_sample_aep": True,
                "ept_mean_sample_oep": True,
                "ept_per_sample_mean_aep": True,
                "ept_per_sample_mean_oep": True,
                "psept_aep": True,
                "psept_oep": True,
                "parquet_format": True,
            }
        }
    ],
    "il_output": True,
    "il_summaries": [
        {
            "id": 1,
            "ord_output": {
                "plt_sample": True,
                "plt_quantile": True,
                "plt_moment": True,
                "elt_sample": True,
                "elt_quantile": True,
                "elt_moment": True,
                "alt_period": True,
                "alt_meanonly": True,
                "ept_full_uncertainty_aep": True,
                "ept_full_uncertainty_oep": True,
                "ept_mean_sample_aep": True,
                "ept_mean_sample_oep": True,
                "ept_per_sample_mean_aep": True,
                "ept_per_sample_mean_oep": True,
                "psept_aep": True,
                "psept_oep": True,
                "parquet_format": True,
            }
        }
    ],
    "ri_output": True,
    "ri_summaries": [
        {
            "id": 1,
            "ord_output": {
                "plt_sample": True,
                "plt_quantile": True,
                "plt_moment": True,
                "elt_sample": True,
                "elt_quantile": True,
                "elt_moment": True,
                "alt_period": True,
                "alt_meanonly": True,
                "ept_full_uncertainty_aep": True,
                "ept_full_uncertainty_oep": True,
                "ept_mean_sample_aep": True,
                "ept_mean_sample_oep": True,
                "ept_per_sample_mean_aep": True,
                "ept_per_sample_mean_oep": True,
                "psept_aep": True,
                "psept_oep": True,
                "parquet_format": True,
            }
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

CORRELATIONS_MODEL_SETTINGS = {
    "version": "3",
    "model_settings": {},
    "lookup_settings": {
        "supported_perils": [
            {"id": "WSS", "desc": "Single Peril: Storm Surge", "peril_correlation_group": 1},
            {"id": "WTC", "desc": "Single Peril: Tropical Cyclone", "peril_correlation_group": 2},
            {"id": "WW1", "desc": "Group Peril: Windstorm with storm surge"},
            {"id": "WW2", "desc": "Group Peril: Windstorm w/o storm surge"}
        ]
    },
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
1,A11111,10002082046,1,1,GB,52.76698052,-0.895469856,1 ABINGDON ROAD,LE13 0HL,1050,5000,WW1,220000,0,0,0,GBP,latest version
"""
MIN_ACC = """PortNumber,AccNumber,AccCurrency,PolNumber,PolPerilsCovered,PolPeril,PolInceptionDate,PolExpiryDate,LayerNumber,LayerParticipation,LayerLimit,LayerAttachment,OEDVersion
1,A11111,GBP,Layer1,WW1,WW1,2018-01-01,2018-12-31,1,0.3,5000000,500000,latest version
"""
MIN_INF = """ReinsNumber,ReinsLayerNumber,ReinsName,ReinsPeril,ReinsInceptionDate,ReinsExpiryDate,CededPercent,RiskLimit,RiskAttachment,OccLimit,OccAttachment,PlacedPercent,ReinsCurrency,InuringPriority,ReinsType,RiskLevel,UseReinsDates,OEDVersion
1,1,ABC QS,WW1,2018-01-01,2018-12-31,1,0,0,0,0,1,GBP,1,SS,LOC,N,latest version
"""
MIN_SCP = """ReinsNumber,PortNumber,AccNumber,PolNumber,LocGroup,LocNumber,CedantName,ProducerName,LOB,CountryCode,ReinsTag,CededPercent,OEDVersion
1,1,A11111,,,10002082047,,,,,,0.1,latest version
"""

N2_LOC = """PortNumber,AccNumber,LocNumber,IsTenant,BuildingID,CountryCode,Latitude,Longitude,StreetAddress,PostalCode,OccupancyCode,ConstructionCode,LocPerilsCovered,BuildingTIV,OtherTIV,ContentsTIV,BITIV,LocCurrency,OEDVersion
1,A11111,10002082046,1,1,GB,52.76698052,-0.895469856,1 ABINGDON ROAD,LE13 0HL,1050,5000,WW1,220000,0,0,0,GBP,latest version
1,A11111,10002082047,1,1,GB,52.76697956,-0.89536613,2 ABINGDON ROAD,LE13 0HL,1050,5000,WW1,790000,0,0,0,GBP,latest version
"""

EXPECTED_KEYS = b'LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID,AmplificationID\n1,WSS,1,1000,8,2\n1,WTC,1,500,2,1\n1,WSS,3,1000,11,2\n1,WTC,3,500,5,1\n'
EXPECTED_ERROR = b'LocID,PerilID,CoverageTypeID,Status,Message\n'

EXPECTED_KEYS_COMPLEX = b'LocID,PerilID,CoverageTypeID,ModelData\n1,WTC,1,"{""area_peril_id"": 54, ""vulnerability_id"": 2}"\n1,WTC,3,"{""area_peril_id"": 54, ""vulnerability_id"": 5}"\n1,WSS,1,"{""area_peril_id"": 154, ""vulnerability_id"": 8}"\n1,WSS,3,"{""area_peril_id"": 154, ""vulnerability_id"": 11}"\n'
EXPECTED_ERROR_COMPLEX = b'LocID,PerilID,CoverageTypeID,Status,Message\n'

FAKE_IL_ITEMS_RETURN = pd.read_csv(os.path.join(os.path.dirname(__file__), 'il_inputs_df_return.csv'))
FAKE_PRE_ANALYSIS_MODULE = os.path.join(os.path.dirname(__file__), 'fake_pre_analysis.py')
FAKE_COMPLEX_LOOKUP_MODULE = os.path.join(os.path.dirname(__file__), 'fake_complex_lookup.py')

FAKE_MODEL_SETTINGS_JSON = os.path.join(os.path.dirname(__file__), 'fake_model_settings.json')

FAKE_MODEL_RUNNER = os.path.join(os.path.dirname(__file__), 'fake_model_runner')
FAKE_MODEL_RUNNER__OLD = os.path.join(os.path.dirname(__file__), 'fake_model_runner__old')

ALL_EXPECTED_SCRIPT = os.path.join(os.path.dirname(__file__), 'ord_bash_script_{0}.sh')

EXPECTED_CORRELATION_CSV = b'item_id,peril_correlation_group,damage_correlation_value,hazard_group_id,hazard_correlation_value\n1,1,0.7,833720067,0.4\n2,2,0.5,741910550,0.2\n'

EXPECTED_SUMMARY_INFO_CSV = b'summary_id,LocNumber,AccNumber,PolNumber,AccCurrency,tiv\n1,10002082046,A11111,Layer1,GBP,220000.0\n'
