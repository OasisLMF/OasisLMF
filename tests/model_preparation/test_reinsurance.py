import os
import subprocess
import time
import unittest
import hypothesis

from backports.tempfile import TemporaryDirectory
from collections import OrderedDict

import pandas as pd
import numpy as np

from pandas.util.testing import assert_frame_equal

from parameterized import parameterized

from oasislmf.model_preparation import (
    oed,
    reinsurance_layer,
)

from oasislmf.model_execution import bin
from oasislmf.utils.data import get_dataframe, set_dataframe_column_dtypes

import shutil

class TestReinsurance(unittest.TestCase):

    def setUp(self):
        self.exposure_1_items_df = pd.DataFrame.from_dict({
            'item_id':          [   1,      2,      3,      4,      5,      6,      7,      8,      9,      10,     11,     12  ],
            'coverage_id':      [   1,      2,      3,      4,      5,      6,      7,      8,      9,      10,     11,     12  ],
            'area_peril_id':    [   -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1  ],
            'vulnerability_id': [   -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1  ],
            'group_id':         [   1,      1,      1,      2,      2,      2,      3,      3,      3,      4,      4,      4   ]
            })
        self.exposure_1_coverages_df = pd.DataFrame.from_dict({
            'coverage_id':  [   1,      2,      3,      4,      5,      6,      7,      8,      9,      10,     11,     12  ],
            'tiv':          [   1000,   500,    500,    1000,   500,    500,    1000,   500,    500,    1000,   500,    500 ]
            })
        self.exposure_1_xref_descriptions_df = pd.DataFrame.from_dict({
            'loc_idx':          [   1,      1,      1,      2,      2,      2,      3,      3,      3,      4,      4,      4       ],
            'output_id':           [   1,      2,      3,      4,      5,      6,      7,      8,      9,      10,     11,     12      ],
            'portnumber':       [   '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1'     ],
            'polnumber':        [   '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1'     ],
            'accnumber':        [   '1',    '1',    '1',    '1',    '1',    '1',    '2',    '2',    '2',    '2',    '2',    '2'     ],
            'locnumber':        [   '1',    '1',    '1',    '2',    '2',    '2',    '1',    '1',    '1',    '2',    '2',    '2'     ],
            'locgroup':         [   'ABC',  'ABC',  'ABC',  '',     '',     '',     'ABC',  'ABC',  'ABC',  '',     '',     ''      ],
            'cedantname':       [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'producername':     [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'lob':              [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'countrycode':      [   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US'    ],
            'reinstag':         [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'coverage_type_id': [   1,      3,      4,      1,      3,      4,      1,      3,      4,      1,      3,      4       ],
            'peril_id':         [   1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1       ],
            'tiv':              [   1000.0, 500.0,  500.0,  1000.0, 500.0,  500.0,  1000.0, 500.0,  500.0,  1000.0, 500.0,  500.0   ]
        })
        self.exposure_2_xref_descriptions_df = pd.DataFrame.from_dict({
            'loc_idx':          [   1,      1,      1,      2,      2,      2,      3,      3,      3       ],
            'output_id':           [   1,      2,      3,      4,      5,      6,      7,      8,      9       ],
            'portnumber':       [   '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1'     ],
            'polnumber':        [   'P1',   'P1',   'P1',   'P1',   'P1',   'P1',   'P1',   'P1',   'P1'    ],
            'accnumber':        [   'A1',   'A1',   'A1',   'A2',   'A2',   'A2',   'A2',   'A2',   'A2'    ],
            'locnumber':        [   'L1',   'L1',   'L1',   'L2',   'L2',   'L2',   'L3',   'L3',   'L3'    ],
            'locgroup':         [   'ABC',  'ABC',  'ABC',  '',     '',     '',     '',     '',     ''      ],
            'cedantname':       [   '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'producername':     [   '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'lob':              [   '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'countrycode':      [   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US'    ],
            'reinstag':         [   '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'coverage_type_id': [   1,      3,      4,      1,      3,      4,      1,      2,      3       ],
            'peril_id':         [   1,      1,      1,      1,      1,      1,      1,      1,      1       ],
            'tiv':              [   1000.0, 500.0,  500.0,  1000.0, 500.0,  500.0,  1000.0, 500.0,  500.0   ]
        })
        self.exposure_3_xref_descriptions_df = pd.DataFrame.from_dict({
            'loc_idx':          [   1,      1,      1,      2,      2,      2,      3,      3,      3,      4,      4,      4,      5,      5,      5       ],
            'output_id':          [   1,      2,      3,      4,      5,      6,      7,      8,      9,      10,     11,     12,     13,     14,     15      ],           
            'portnumber':       [   '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '2',    '2',    '2'     ],
            'polnumber':        [   'P1',   'P1',   'P1',   'P2',   'P2',   'P2',   'P2',   'P2',   'P2',   'P1',   'P1',   'P1',   'P1',   'P1',   'P1'    ],
            'accnumber':        [   'A1',   'A1',   'A1',   'A1',   'A1',   'A1',   'A2',   'A2',   'A2',   'A2',   'A2',   'A2',   'A1',   'A1',   'A1'    ],
            'locnumber':        [   'L1',   'L1',   'L1',   'L1',   'L1',   'L1',   'L2',   'L2',   'L2',   'L2',   'L2',   'L2',   'L1',   'L1',   'L1'    ],
            'locgroup':         [   'ABC',  'ABC',  'ABC',  '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'cedantname':       [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'producername':     [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'lob':              [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'countrycode':      [   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US'    ],
            'reinstag':         [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'coverage_type_id': [   1,      3,      4,      1,      3,      4,      1,      3,      4,      1,      3,      4,      1,      2,      3       ],
            'peril_id':         [   1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1       ],
            'tiv':              [   1000.0, 500.0,  500.0,  1000.0, 500.0,  500.0,  1000.0, 500.0,  500.0,  1000.0, 500.0,  500.0,  1000.0, 500.0,  500.0   ]
        })


    def test_single_loc_level_fac(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber': ['1'],
            'CededPercent': [1.0],
            'RiskLimit': [10.0],
            'RiskAttachment': [0.0],
            'OccLimit': [0.0],
            'OccAttachment': [0.0],
            'InuringPriority': [1],
            'ReinsType': ['FAC'],
            'PlacedPercent': [1.0],
            'TreatyShare': [1.0]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber': ['1'],
            'PortNumber': ['1'],
            'AccNumber': ['1'],
            'PolNumber': ['1'],
            'LocGroup': [''],
            'LocNumber': ['1'],
            'CedantName': [''],
            'ProducerName': [''],
            'LOB': [''],
            'CountryCode': [''],
            'ReinsTag': [''],
            'RiskLevel': ['LOC'],
            'CededPercent': [1.0]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='LOC',
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  3,  4,  1,  2,  3,  4   ],
                    'profile_id':   [   2,  3,  1,  1,  1,  2,  2,  2,  2   ] 
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3       ],
                    'calcrule_id':  [   14,     12,     24      ],
                    'deductible1':  [   0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    0.0     ],
                    'limit':        [   0.0,    0.0,    10.0    ],
                    'share1':       [   0.0,    0.0,    1.0     ],
                    'share2':       [   0.0,    0.0,    1.0     ],
                    'share3':       [   0.0,    0.0,    1.0     ]
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  2,  3,  4,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)

    def test_single_pol_level_fac(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber': ['1'],
            'CededPercent': [1.0],
            'RiskLimit': [10.0],
            'RiskAttachment': [0.0],
            'OccLimit': [0.0],
            'OccAttachment': [0.0],
            'InuringPriority': [1],
            'ReinsType': ['FAC'],
            'PlacedPercent': [1.0],
            'TreatyShare': [1.0]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber': ['1'],
            'PortNumber': ['1'],
            'AccNumber': ['1'],
            'PolNumber': ['1'],
            'LocGroup': [''],
            'LocNumber': [''],
            'CedantName': [''],
            'ProducerName': [''],
            'LOB': [''],
            'CountryCode': [''],
            'ReinsTag': [''],
            'RiskLevel': ['POL'],
            'CededPercent': [1.0]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='POL',
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1, 1,  1,  1,  1,  1,  1    ],
                    'level_id':     [   3, 2,  2,  1,  1,  1,  1    ],
                    'agg_id':       [   1, 1,  2,  1,  2,  3,  4    ],
                    'profile_id':   [   2, 3,  1,  2,  2,  2,  2    ] 
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,     2,      3    ],
                    'calcrule_id':  [   14,    12,     24   ],
                    'deductible1':  [   0.0,   0.0,    0.0  ],
                    'deductible2':  [   0.0,   0.0,    0.0  ],
                    'deductible3':  [   0.0,   0.0,    0.0  ],
                    'attachment':   [   0.0,   0.0,    0.0  ],
                    'limit':        [   0.0,   0.0,    10.0 ],
                    'share1':       [   0.0,   0.0,    1.0  ],
                    'share2':       [   0.0,   0.0,    1.0  ],
                    'share3':       [   0.0,   0.0,    1.0  ]
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1, 2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3, 3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1, 1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)        

    def test_single_acc_level_fac(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber': ['1'],
            'CededPercent': [1.0],
            'RiskLimit': [10.0],
            'RiskAttachment': [0.0],
            'OccLimit': [0.0],
            'OccAttachment': [0.0],
            'InuringPriority': [1],
            'ReinsType': ['FAC'],
            'PlacedPercent': [1.0],
            'TreatyShare': [1.0]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber': ['1'],
            'PortNumber': ['1'],
            'AccNumber': ['1'],
            'PolNumber': [''],
            'LocGroup': [''],
            'LocNumber': [''],
            'CedantName': [''],
            'ProducerName': [''],
            'LOB': [''],
            'CountryCode': [''],
            'ReinsTag': [''],
            'RiskLevel': ['ACC'],
            'CededPercent': [1.0]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='ACC',
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   2,  3,  1,  2,  2,  2,  2   ] 
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,     2,      3       ],
                    'calcrule_id':  [   14,    12,     24      ],
                    'deductible1':  [   0.0,   0.0,    0.0     ],
                    'deductible2':  [   0.0,   0.0,    0.0     ],
                    'deductible3':  [   0.0,   0.0,    0.0     ],
                    'attachment':   [   0.0,   0.0,    0.0     ],
                    'limit':        [   0.0,   0.0,    10.0    ],
                    'share1':       [   0.0,   0.0,    1.0     ],
                    'share2':       [   0.0,   0.0,    1.0     ],
                    'share3':       [   0.0,   0.0,    1.0     ]
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)        


    def test_single_lgr_level_fac(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber': ['1'],
            'CededPercent': [1.0],
            'RiskLimit': [10.0],
            'RiskAttachment': [0.0],
            'OccLimit': [0.0],
            'OccAttachment': [0.0],
            'InuringPriority': [1],
            'ReinsType': ['FAC'],
            'PlacedPercent': [1.0],
            'TreatyShare': [1.0]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber': ['1'],
            'PortNumber': ['1'],
            'AccNumber': [''],
            'PolNumber': [''],
            'LocGroup': ['ABC'],
            'LocNumber': [''],
            'CedantName': [''],
            'ProducerName': [''],
            'LOB': [''],
            'CountryCode': [''],
            'ReinsTag': [''],
            'RiskLevel': ['LGR'],
            'CededPercent': [1.0]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='LGR',
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1, 1,  1,  1,  1,  1,  1    ],
                    'level_id':     [   3, 2,  2,  1,  1,  1,  1    ],
                    'agg_id':       [   1, 1,  2,  1,  2,  3,  4    ],
                    'profile_id':   [   2, 1,  3,  2,  2,  2,  2    ] 
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':  [    1,      2,      3       ],
                    'calcrule_id': [    14,     12,     24      ],
                    'deductible1': [    0.0,    0.0,    0.0     ],
                    'deductible2': [    0.0,    0.0,    0.0     ],
                    'deductible3': [    0.0,    0.0,    0.0     ],
                    'attachment':  [    0.0,    0.0,    0.0     ],
                    'limit':       [    0.0,    0.0,    10.0    ],
                    'share1':      [    0.0,    0.0,    1.0     ],
                    'share2':      [    0.0,    0.0,    1.0     ],
                    'share3':      [    0.0,    0.0,    1.0     ]
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id': [    1,  2,  1,  2,  3,  4,  4,  5,  6,  10, 11, 12, 1,  2,  3,  7,  8,  9   ],
                    'level_id':    [    3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':   [    1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)        

    def test_multiple_facs_same_inuring_level(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   '1',    '2',    '3',    '4'     ],
            'CededPercent':     [   1.0,    1.0,    1.0,    1.0     ],
            'RiskLimit':        [   10.0,   10.0,   10.0,   10.0    ],
            'RiskAttachment':   [   0.0,    0.0,    0.0,    0.0     ],
            'OccLimit':         [   0.0,    0.0,    0.0,    0.0     ],
            'OccAttachment':    [   0.0,    0.0,    0.0,    0.0     ],
            'InuringPriority':  [   1,      1,      1,      1       ],
            'ReinsType':        [   'FAC',  'FAC',  'FAC',  'FAC'   ],
            'PlacedPercent':    [   1.0,    1.0,    1.0,    1.0     ],
            'TreatyShare':      [   1.0,    1.0,    1.0,    1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   '1',    '2',    '3',    '4'     ],
            'PortNumber':   [   '1',    '1',    '1',    '1'     ],
            'AccNumber':    [   '1',    '1',    '1',    '1'     ],
            'PolNumber':    [   '1',    '1',    '1',    ''      ],
            'LocGroup':     [   '',     '',     '',     ''      ],
            'LocNumber':    [   '1',    '2',    '',     ''      ],
            'CedantName':   [   '',     '',     '',     ''      ],
            'ProducerName': [   '',     '',     '',     ''      ],
            'LOB':          [   '',     '',     '',     ''      ],
            'CountryCode':  [   '',     '',     '',     ''      ],
            'ReinsTag':     [   '',     '',     '',     ''      ],
            'RiskLevel':    [   'LOC',  'LOC',  'POL',  'ACC'   ],
            'CededPercent': [   1.0,    1.0,    1.0,    1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [
            reinsurance_layer.RiInputs(
                risk_level='LOC',
                inuring_priority=1,
                ri_inputs=reinsurance_layer.RiLayerInputs(
                    fm_policytc=pd.DataFrame.from_dict({
                        'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2   ],
                        'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1,  3,  2,  2,  2,  2,  1,  1,  1,  1   ],
                        'agg_id':       [   1,  1,  2,  3,  4,  1,  2,  3,  4,  1,  1,  2,  3,  4,  1,  2,  3,  4   ],
                        'profile_id':   [   2,  3,  1,  1,  1,  2,  2,  2,  2,  2,  1,  4,  1,  1,  2,  2,  2,  2   ]
                    }),
                    fm_profile=pd.DataFrame.from_dict({
                        'profile_id':  [    1,      2,      3,      4       ],
                        'calcrule_id': [    14,     12,     24,     24      ],
                        'deductible1': [    0.0,    0.0,    0.0,    0.0     ],
                        'deductible2': [    0.0,    0.0,    0.0,    0.0     ],
                        'deductible3': [    0.0,    0.0,    0.0,    0.0     ],
                        'attachment':  [    0.0,    0.0,    0.0,    0.0     ],
                        'limit':       [    0.0,    0.0,    10.0,   10.0    ],
                        'share1':      [    0.0,    0.0,    1.0,    1.0     ],
                        'share2':      [    0.0,    0.0,    1.0,    1.0     ],
                        'share3':      [    0.0,    0.0,    1.0,    1.0     ]
                    }),
                    fm_programme=pd.DataFrame.from_dict({
                        'from_agg_id': [    1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                        'level_id':    [    3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                        'to_agg_id':   [    1,  1,  1,  1,  1,  2,  3,  4,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]
                    })
                )
            ),
            reinsurance_layer.RiInputs(
                risk_level='POL',
                inuring_priority=1,
                ri_inputs=reinsurance_layer.RiLayerInputs(
                    fm_policytc=pd.DataFrame.from_dict({
                        'layer_id':     [   1,  1,   1,  1,  1,  1,  1   ],
                        'level_id':     [   3,  2,   2,  1,  1,  1,  1   ],
                        'agg_id':       [   1,  1,   2,  1,  2,  3,  4   ],
                        'profile_id':   [   2,  3,   1,  2,  2,  2,  2   ] 
                    }),
                    fm_profile=pd.DataFrame.from_dict({
                        'profile_id':  [    1,      2,      3       ],
                        'calcrule_id': [    14,     12,     24      ],
                        'deductible1': [    0.0,    0.0,    0.0     ],
                        'deductible2': [    0.0,    0.0,    0.0     ],
                        'deductible3': [    0.0,    0.0,    0.0     ],
                        'attachment':  [    0.0,    0.0,    0.0     ],
                        'limit':       [    0.0,    0.0,    10.0    ],
                        'share1':      [    0.0,    0.0,    1.0     ],
                        'share2':      [    0.0,    0.0,    1.0     ],
                        'share3':      [    0.0,    0.0,    1.0     ]
                    }),
                    fm_programme=pd.DataFrame.from_dict({
                        'from_agg_id': [    1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                        'level_id':    [    3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                        'to_agg_id':   [    1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]
                    })
                )
            ),
            reinsurance_layer.RiInputs(
                risk_level='ACC',
                inuring_priority=1,
                ri_inputs=reinsurance_layer.RiLayerInputs(
                    fm_policytc=pd.DataFrame.from_dict({
                        'layer_id':     [   1, 1,  1,  1,  1,  1,  1    ],
                        'level_id':     [   3, 2,  2,  1,  1,  1,  1    ],
                        'agg_id':       [   1, 1,  2,  1,  2,  3,  4    ],
                        'profile_id':   [   2, 3,  1,  2,  2,  2,  2    ] 
                    }),
                    fm_profile=pd.DataFrame.from_dict({
                        'profile_id':  [    1,      2,      3       ],
                        'calcrule_id': [    14,     12,     24      ],
                        'deductible1': [    0.0,    0.0,    0.0     ],
                        'deductible2': [    0.0,    0.0,    0.0     ],
                        'deductible3': [    0.0,    0.0,    0.0     ],
                        'attachment':  [    0.0,    0.0,    0.0     ],
                        'limit':       [    0.0,    0.0,    10.0    ],
                        'share1':      [    0.0,    0.0,    1.0     ],
                        'share2':      [    0.0,    0.0,    1.0     ],
                        'share3':      [    0.0,    0.0,    1.0     ]
                    }),
                    fm_programme=pd.DataFrame.from_dict({
                        'from_agg_id': [    1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                        'level_id':    [    3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                        'to_agg_id':   [    1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]
                    })
                )
            )        
        ]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)        

    def test_single_loc_level_PR_all_risks(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber': [1],
            'CededPercent': [1.0],
            'RiskLimit': [10.0],
            'RiskAttachment': [5.0],
            'OccLimit': [37.5],
            'OccAttachment': [0.0],
            'InuringPriority': [1],
            'ReinsType': ['PR'],
            'PlacedPercent': [1.0],
            'TreatyShare': [1.0]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber': [1],
            'PortNumber': ['1'],
            'AccNumber': [''],
            'PolNumber': [''],
            'LocGroup': [''],
            'LocNumber': [''],
            'CedantName': [''],
            'ProducerName': [''],
            'LOB': [''],
            'CountryCode': [''],
            'ReinsTag': [''],
            'RiskLevel': ['LOC'],
            'CededPercent': [1.0]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='LOC',
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  3,  4,  1,  2,  3,  4   ],
                    'profile_id':   [   4,  3,  3,  3,  3,  2,  2,  2,  2   ] 
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,      4       ],
                    'calcrule_id':  [   14,     12,     24,     23      ],
                    'deductible1':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    5.0,    0.0     ],
                    'limit':        [   0.0,    0.0,    10.0,   37.5    ],
                    'share1':       [   0.0,    0.0,    1.0,    0.0     ],
                    'share2':       [   0.0,    0.0,    1.0,    1.0     ],
                    'share3':       [   0.0,    0.0,    1.0,    1.0     ]
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  2,  3,  4,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)

    def test_single_lgr_level_PR_all_risks(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   10.0    ],
            'RiskAttachment':   [   5.0     ],
            'OccLimit':         [   37.5    ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'PR'    ],
            'PlacedPercent':    [   1.0     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   ''     ],
            'PolNumber':    [   ''     ],
            'LocGroup':     [   'ABC'      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'LGR'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='LGR',
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   4,  1,  3,  1,  1,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,      4       ],
                    'calcrule_id':  [   14,     12,     24,     23      ],
                    'deductible1':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    5.0,    0.0     ],
                    'limit':        [   0.0,    0.0,    10.0,   37.5    ],
                    'share1':       [   0.0,    0.0,    1.0,    0.0     ],
                    'share2':       [   0.0,    0.0,    1.0,    1.0     ],
                    'share3':       [   0.0,    0.0,    1.0,    1.0     ]
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  4,  5,  6,  10, 11, 12, 1,  2,  3,  7,  8,  9   ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)

    def test_single_pol_level_PR_all_risks(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber': [1],
            'CededPercent': [1.0],
            'RiskLimit': [10.0],
            'RiskAttachment': [5.0],
            'OccLimit': [37.5],
            'OccAttachment': [0.0],
            'InuringPriority': [1],
            'ReinsType': ['PR'],
            'PlacedPercent': [1.0],
            'TreatyShare': [1.0]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber': [1],
            'PortNumber': ['1'],
            'AccNumber': ['1'],
            'PolNumber': ['1'],
            'LocGroup': [''],
            'LocNumber': [''],
            'CedantName': [''],
            'ProducerName': [''],
            'LOB': [''],
            'CountryCode': [''],
            'ReinsTag': [''],
            'RiskLevel': ['POL'],
            'CededPercent': [1.0]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='POL',
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   4,  3,  1,  2,  2,  1,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,      4       ],
                    'calcrule_id':  [   14,     12,     24,     23      ],
                    'deductible1':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    5.0,    0.0     ],
                    'limit':        [   0.0,    0.0,    10.0,   37.5    ],
                    'share1':       [   0.0,    0.0,    1.0,    0.0     ],
                    'share2':       [   0.0,    0.0,    1.0,    1.0     ],
                    'share3':       [   0.0,    0.0,    1.0,    1.0     ]
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12   ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)

    def test_single_acc_level_PR_all_risks(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber': [1],
            'CededPercent': [1.0],
            'RiskLimit': [10.0],
            'RiskAttachment': [5.0],
            'OccLimit': [37.5],
            'OccAttachment': [0.0],
            'InuringPriority': [1],
            'ReinsType': ['PR'],
            'PlacedPercent': [1.0],
            'TreatyShare': [1.0]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber': [1],
            'PortNumber': ['1'],
            'AccNumber': ['1'],
            'PolNumber': [''],
            'LocGroup': [''],
            'LocNumber': [''],
            'CedantName': [''],
            'ProducerName': [''],
            'LOB': [''],
            'CountryCode': [''],
            'ReinsTag': [''],
            'RiskLevel': ['ACC'],
            'CededPercent': [1.0]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='ACC',
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   4,  3,  1,  2,  2,  1,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,      4       ],
                    'calcrule_id':  [   14,     12,     24,     23      ],
                    'deductible1':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    5.0,    0.0     ],
                    'limit':        [   0.0,    0.0,    10.0,   37.5    ],
                    'share1':       [   0.0,    0.0,    1.0,    0.0     ],
                    'share2':       [   0.0,    0.0,    1.0,    1.0     ],
                    'share3':       [   0.0,    0.0,    1.0,    1.0     ]
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12   ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)

    def test_single_loc_level_SS_all_risks_loc(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'SS'    ],
            'PlacedPercent':    [   1.0     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      1,      1,      1       ],
            'PortNumber':   [   '1',    '1',    '1',    '1'     ],
            'AccNumber':    [   '1',    '1',     '1',   '1'     ],
            'PolNumber':    [   '',     '',     '',     ''      ],
            'LocGroup':     [   '',     '',     '',     ''      ],
            'LocNumber':    [   '1',    '2',    '3',    '4'      ],
            'CedantName':   [   '',     '',     '',     ''      ],
            'ProducerName': [   '',     '',     '',     ''      ],
            'LOB':          [   '',     '',     '',     ''      ],
            'CountryCode':  [   '',     '',     '',     ''      ],
            'ReinsTag':     [   '',     '',     '',     ''      ],
            'RiskLevel':    [   'LOC',  'LOC',  'LOC',  'LOC'   ],
            'CededPercent': [   0.1,    0.1,    0.1,    0.1     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='LOC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  3,  4,  1,  2,  3,  4   ],
                    'profile_id':   [   7,  3,  4,  1,  1,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,                  4,                  5,                  6,                  7               ],
                    'calcrule_id':  [   14,     12,     24,                 24,                 24,                 24,                 23              ],
                    'deductible1':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0             ],
                    'deductible2':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0             ],
                    'deductible3':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0             ],
                    'attachment':   [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0             ],
                    'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0,    9999999999999.0,    9999999999999.0,    9999999999999.0 ],  
                    'share1':       [   0.0,    0.0,    0.1,                0.1,                0.1,                0.1,                0.0             ],  
                    'share2':       [   0.0,    0.0,    1.0,                1.0,                1.0,                1.0,                1.0             ],  
                    'share3':       [   0.0,    0.0,    1.0,                1.0,                1.0,                1.0,                1.0             ]   
                }), 
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  2,  3,  4,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_loc_level_SS_all_risks_loc_pol(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'SS'    ],
            'PlacedPercent':    [   1.0     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      1,      1,      1       ],
            'PortNumber':   [   '1',    '1',    '1',    '1'     ],
            'AccNumber':    [   '1',    '1',    '1',    '1'     ],
            'PolNumber':    [   '1',    '1',    '1',    '1'      ],
            'LocGroup':     [   '',     '',     '',     ''      ],
            'LocNumber':    [   '1',    '2',    '3',    '4'      ],
            'CedantName':   [   '',     '',     '',     ''      ],
            'ProducerName': [   '',     '',     '',     ''      ],
            'LOB':          [   '',     '',     '',     ''      ],
            'CountryCode':  [   '',     '',     '',     ''      ],
            'ReinsTag':     [   '',     '',     '',     ''      ],
            'RiskLevel':    [   'LOC',  'LOC',  'LOC',  'LOC'   ],
            'CededPercent': [   0.1,    0.1,    0.1,    0.1     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='LOC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  3,  4,  1,  2,  3,  4   ],
                    'profile_id':   [   7,  3,  4,  1,  1,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,                  4,                  5,                  6,                  7               ],
                    'calcrule_id':  [   14,     12,     24,                 24,                 24,                 24,                 23              ],
                    'deductible1':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0             ],
                    'deductible2':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0             ],
                    'deductible3':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0             ],
                    'attachment':   [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0             ],
                    'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0,    9999999999999.0,    9999999999999.0,    9999999999999.0 ],  
                    'share1':       [   0.0,    0.0,    0.1,                0.1,                0.1,                0.1,                0.0             ],  
                    'share2':       [   0.0,    0.0,    1.0,                1.0,                1.0,                1.0,                1.0             ],  
                    'share3':       [   0.0,    0.0,    1.0,                1.0,                1.0,                1.0,                1.0             ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  2,  3,  4,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_loc_level_SS_all_risks(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'SS'    ],
            'PlacedPercent':    [   1.0     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      1       ],
            'PortNumber':   [   '1',    '1'     ],
            'AccNumber':    [   '1',    '2'     ],
            'PolNumber':    [   '1',    '2'     ],
            'LocGroup':     [   '',     ''      ],
            'LocNumber':    [   '',    ''     ],
            'CedantName':   [   '',     ''      ],
            'ProducerName': [   '',     ''      ],
            'LOB':          [   '',     ''      ],
            'CountryCode':  [   '',     ''      ],
            'ReinsTag':     [   '',     ''      ],
            'RiskLevel':    [   'POL',  'POL'   ],
            'CededPercent': [   0.1,    0.1     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='POL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   5,  3,  1,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,                  4,                  5                   ],
                    'calcrule_id':  [   14,     12,     24,                 24,                 23                  ],
                    'deductible1':  [   0.0,    0.0,    0.0,                0.0,                0.0                 ],
                    'deductible2':  [   0.0,    0.0,    0.0,                0.0,                0.0                 ],
                    'deductible3':  [   0.0,    0.0,    0.0,                0.0,                0.0                 ],
                    'attachment':   [   0.0,    0.0,    0.0,                0.0,                0.0                 ],
                    'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0,    9999999999999.0     ],  
                    'share1':       [   0.0,    0.0,    0.1,                0.1,                0.0                 ],  
                    'share2':       [   0.0,    0.0,    1.0,                1.0,                1.0                 ],  
                    'share3':       [   0.0,    0.0,    1.0,                1.0,                1.0                 ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_acc_level_SS_all_risks(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'SS'    ],
            'PlacedPercent':    [   1.0     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      1       ],
            'PortNumber':   [   '1',    '1'     ],
            'AccNumber':    [   '1',    '2'     ],
            'PolNumber':    [   '',     ''      ],
            'LocGroup':     [   '',     ''      ],
            'LocNumber':    [   '',     ''      ],
            'CedantName':   [   '',     ''      ],
            'ProducerName': [   '',     ''      ],
            'LOB':          [   '',     ''      ],
            'CountryCode':  [   '',     ''      ],
            'ReinsTag':     [   '',     ''      ],
            'RiskLevel':    [   'ACC',  'ACC'   ],
            'CededPercent': [   0.1,    0.1     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='ACC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   5,  3,  4,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,                  4,                  5                   ],
                    'calcrule_id':  [   14,     12,     24,                 24,                 23                  ],
                    'deductible1':  [   0.0,    0.0,    0.0,                0.0,                0.0                 ],
                    'deductible2':  [   0.0,    0.0,    0.0,                0.0,                0.0                 ],
                    'deductible3':  [   0.0,    0.0,    0.0,                0.0,                0.0                 ],
                    'attachment':   [   0.0,    0.0,    0.0,                0.0,                0.0                 ],
                    'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0,    9999999999999.0     ],  
                    'share1':       [   0.0,    0.0,    0.1,                0.1,                0.0                 ],  
                    'share2':       [   0.0,    0.0,    1.0,                1.0,                1.0                 ],  
                    'share3':       [   0.0,    0.0,    1.0,                1.0,                1.0                 ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_QS_one_risk_with_pct_ceded_and_pct_placed_and_risk_limit_and_occ_limit(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   0.5     ],
            'RiskLimit':        [   50.0    ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   40.0    ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'QS'    ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   ''      ],
            'PolNumber':    [   ''      ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'ACC'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='ACC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   4,  3,  3,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,       4      ],
                    'calcrule_id':  [   14,     12,     24,      23     ],
                    'deductible1':  [   0.0,    0.0,    0.0,     0.0    ],
                    'deductible2':  [   0.0,    0.0,    0.0,     0.0    ],
                    'deductible3':  [   0.0,    0.0,    0.0,     0.0    ],
                    'attachment':   [   0.0,    0.0,    0.0,     0.0    ],
                    'limit':        [   0.0,    0.0,    50.0,    40.0   ],  
                    'share1':       [   0.0,    0.0,    0.5,     0.0    ],  
                    'share2':       [   0.0,    0.0,    1.0,     0.8    ],  
                    'share3':       [   0.0,    0.0,    1.0,     1.0    ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_QS_all_acc_with_pct_ceded_and_pct_placed_and_risk_limit_and_occ_limit(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   0.5     ],
            'RiskLimit':        [   50.0    ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   40.0    ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'QS'    ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   '1'     ],
            'PolNumber':    [   ''      ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'ACC'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='ACC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   4,  3,  1,  2,  2,  1,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,       4      ],
                    'calcrule_id':  [   14,     12,     24,      23     ],
                    'deductible1':  [   0.0,    0.0,    0.0,     0.0    ],
                    'deductible2':  [   0.0,    0.0,    0.0,     0.0    ],
                    'deductible3':  [   0.0,    0.0,    0.0,     0.0    ],
                    'attachment':   [   0.0,    0.0,    0.0,     0.0    ],
                    'limit':        [   0.0,    0.0,    50.0,    40.0   ],  
                    'share1':       [   0.0,    0.0,    0.5,     0.0    ],
                    'share2':       [   0.0,    0.0,    1.0,     0.8    ],  
                    'share3':       [   0.0,    0.0,    1.0,     1.0    ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_loc_level_SS_with_pct_ceded_and_pct_placed(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'SS'    ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      1,      1,      1       ],
            'PortNumber':   [   '1',    '1',    '1',    '1'     ],
            'AccNumber':    [   '1',    '1',    '2',    '2'     ],
            'PolNumber':    [   '1',    '1',    '1',    '1'     ],
            'LocGroup':     [   '',     '',     '',     ''      ],
            'LocNumber':    [   '1',    '2',    '1',    '2'     ],
            'CedantName':   [   '',     '',     '',     ''      ],
            'ProducerName': [   '',     '',     '',     ''      ],
            'LOB':          [   '',     '',     '',     ''      ],
            'CountryCode':  [   '',     '',     '',     ''      ],
            'ReinsTag':     [   '',     '',     '',     ''      ],
            'RiskLevel':    [   'LOC',  'LOC',  'LOC',  'LOC'   ],
            'CededPercent': [   0.1,    0.1,    0.1,    0.1     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='LOC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  3,  4,  1,  2,  3,  4   ],
                    'profile_id':   [   7,  3,  4,  5,  6,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,                  4,                  5,                  6,                  7                   ],
                    'calcrule_id':  [   14,     12,     24,                 24,                 24,                 24,                 23                  ],
                    'deductible1':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0                 ],
                    'deductible2':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0                 ],
                    'deductible3':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0                 ],
                    'attachment':   [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0                 ],
                    'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0,    9999999999999.0,    9999999999999.0,    9999999999999.0     ],  
                    'share1':       [   0.0,    0.0,    0.1,                0.1,                0.1,                0.1,                0.0                 ],  
                    'share2':       [   0.0,    0.0,    1.0,                1.0,                1.0,                1.0,                0.8                 ],  
                    'share3':       [   0.0,    0.0,    1.0,                1.0,                1.0,                1.0,                1.0                 ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  2,  3,  4,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_pol_level_SS_with_pct_ceded_and_pct_placed(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'SS'    ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      1       ],
            'PortNumber':   [   '1',    '1'     ],
            'AccNumber':    [   '1',    '1'     ],
            'PolNumber':    [   '1',    '2'     ],
            'LocGroup':     [   '',     ''      ],
            'LocNumber':    [   '',     ''      ],
            'CedantName':   [   '',     ''      ],
            'ProducerName': [   '',     ''      ],
            'LOB':          [   '',     ''      ],
            'CountryCode':  [   '',     ''      ],
            'ReinsTag':     [   '',     ''      ],
            'RiskLevel':    [   'POL',  'POL'   ],
            'CededPercent': [   0.1,    0.1     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='POL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   5,  3,  1,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,                  4,                  5                   ],
                    'calcrule_id':  [   14,     12,     24,                 24,                 23                  ],
                    'deductible1':  [   0.0,    0.0,    0.0,                0.0,                0.0                 ],
                    'deductible2':  [   0.0,    0.0,    0.0,                0.0,                0.0                 ],
                    'deductible3':  [   0.0,    0.0,    0.0,                0.0,                0.0                 ],
                    'attachment':   [   0.0,    0.0,    0.0,                0.0,                0.0                 ],
                    'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0,    9999999999999.0     ],  
                    'share1':       [   0.0,    0.0,    0.1,                0.1,                0.0                 ],  
                    'share2':       [   0.0,    0.0,    1.0,                1.0,                0.8                 ],  
                    'share3':       [   0.0,    0.0,    1.0,                1.0,                1.0                 ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_QS_all_risks_with_pct_ceded_and_pct_placed(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   0.5     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'QS'    ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      1       ],
            'PortNumber':   [   '1',    '1'     ],
            'AccNumber':    [   '1',    '1'     ],
            'PolNumber':    [   '1',    '2'     ],
            'LocGroup':     [   '',     ''      ],
            'LocNumber':    [   '',     ''      ],
            'CedantName':   [   '',     ''      ],
            'ProducerName': [   '',     ''      ],
            'LOB':          [   '',     ''      ],
            'CountryCode':  [   '',     ''      ],
            'ReinsTag':     [   '',     ''      ],
            'RiskLevel':    [   'POL',  'POL'   ],
            'CededPercent': [   0.1,    0.1     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='POL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   4,  3,  1,  2,  2,  1,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,                  4                  ],
                    'calcrule_id':  [   14,     12,     24,                 23                 ],
                    'deductible1':  [   0.0,    0.0,    0.0,                0.0                ],
                    'deductible2':  [   0.0,    0.0,    0.0,                0.0                ],
                    'deductible3':  [   0.0,    0.0,    0.0,                0.0                ],
                    'attachment':   [   0.0,    0.0,    0.0,                0.0                ],
                    'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0    ],  
                    'share1':       [   0.0,    0.0,    0.5,                0.0                ],  
                    'share2':       [   0.0,    0.0,    1.0,                0.8                ],  
                    'share3':       [   0.0,    0.0,    1.0,                1.0                ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_QS_with_account_level_risk_limits(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   0.5     ],
            'RiskLimit':        [   10.0    ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'QS'    ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   ''      ],
            'PolNumber':    [   ''      ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'ACC'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='ACC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   4,  3,  3,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,                  4                   ],
                    'calcrule_id':  [   14,     12,     24,                 23                  ],
                    'deductible1':  [   0.0,    0.0,    0.0,                0.0                 ],
                    'deductible2':  [   0.0,    0.0,    0.0,                0.0                 ],
                    'deductible3':  [   0.0,    0.0,    0.0,                0.0                 ],
                    'attachment':   [   0.0,    0.0,    0.0,                0.0                 ],
                    'limit':        [   0.0,    0.0,    10.0,               9999999999999.0     ],  
                    'share1':       [   0.0,    0.0,    0.5,                0.0                 ],  
                    'share2':       [   0.0,    0.0,    1.0,                0.8                 ],  
                    'share3':       [   0.0,    0.0,    1.0,                1.0                 ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_cxl(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   10.0    ],
            'OccAttachment':    [   50.0    ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'CXL'   ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   ''      ],
            'PolNumber':    [   ''      ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'SEL'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='SEL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  1,  2,  3,  4   ],
                    'profile_id':   [   3,  2,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3                  ],
                    'calcrule_id':  [   14,     12,     24                 ],
                    'deductible1':  [   0.0,    0.0,    0.0                ],
                    'deductible2':  [   0.0,    0.0,    0.0                ],
                    'deductible3':  [   0.0,    0.0,    0.0                ],
                    'attachment':   [   0.0,    0.0,    50.0               ],
                    'limit':        [   0.0,    0.0,    10.0               ],  
                    'share1':       [   0.0,    0.0,    1.0                ],  
                    'share2':       [   0.0,    0.0,    0.8                ],  
                    'share3':       [   0.0,    0.0,    1.0                ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_cxl_0_occ_limit_treat_as_unlimited(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0    ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   50.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'CXL'   ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   ''      ],
            'PolNumber':    [   ''      ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'SEL'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='SEL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  1,  2,  3,  4   ],
                    'profile_id':   [   3,  2,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3                  ],
                    'calcrule_id':  [   14,     12,     24                 ],
                    'deductible1':  [   0.0,    0.0,    0.0                ],
                    'deductible2':  [   0.0,    0.0,    0.0                ],
                    'deductible3':  [   0.0,    0.0,    0.0                ],
                    'attachment':   [   0.0,    0.0,    50.0               ],
                    'limit':        [   0.0,    0.0,    9999999999999.0    ],  
                    'share1':       [   0.0,    0.0,    1.0                ],  
                    'share2':       [   0.0,    0.0,    0.8                ],  
                    'share3':       [   0.0,    0.0,    1.0                ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_QS_with_policy_level_risk_limits(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   0.5     ],
            'RiskLimit':        [   10.0    ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'QS'    ],
            'PlacedPercent':    [   1.0     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   ''      ],
            'PolNumber':    [   ''      ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'POL'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='POL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   4,  3,  3,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,      4                   ],
                    'calcrule_id':  [   14,     12,     24,     23                  ],
                    'deductible1':  [   0.0,    0.0,    0.0,    0.0                 ],
                    'deductible2':  [   0.0,    0.0,    0.0,    0.0                 ],
                    'deductible3':  [   0.0,    0.0,    0.0,    0.0                 ],
                    'attachment':   [   0.0,    0.0,    0.0,    0.0                 ],
                    'limit':        [   0.0,    0.0,    10,     9999999999999.0     ],  
                    'share1':       [   0.0,    0.0,    0.5,    0.0                 ],  
                    'share2':       [   0.0,    0.0,    1.0,    1.0                 ],  
                    'share3':       [   0.0,    0.0,    1.0,    1.0                 ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_QS_with_location_level_risk_limits(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   0.5     ],
            'RiskLimit':        [   10.0    ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'QS'    ],
            'PlacedPercent':    [   1.0     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   ''      ],
            'PolNumber':    [   ''      ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'LOC'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='LOC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  3,  4,  1,  2,  3,  4   ],
                    'profile_id':   [   4,  3,  3,  3,  3,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,      4                   ],
                    'calcrule_id':  [   14,     12,     24,     23                  ],
                    'deductible1':  [   0.0,    0.0,    0.0,    0.0                 ],
                    'deductible2':  [   0.0,    0.0,    0.0,    0.0                 ],
                    'deductible3':  [   0.0,    0.0,    0.0,    0.0                 ],
                    'attachment':   [   0.0,    0.0,    0.0,    0.0                 ],
                    'limit':        [   0.0,    0.0,    10,     9999999999999.0     ],  
                    'share1':       [   0.0,    0.0,    0.5,    0.0                 ],  
                    'share2':       [   0.0,    0.0,    1.0,    1.0                 ],  
                    'share3':       [   0.0,    0.0,    1.0,    1.0                 ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  2,  3,  4,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_QS(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   0.5     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'QS'    ],
            'PlacedPercent':    [   1.0     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   ''      ],
            'PolNumber':    [   ''      ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'ACC'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='ACC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4   ],
                    'profile_id':   [   4,  3,  3,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,                  4                   ],
                    'calcrule_id':  [   14,     12,     24,                 23                  ],
                    'deductible1':  [   0.0,    0.0,    0.0,                0.0                 ],
                    'deductible2':  [   0.0,    0.0,    0.0,                0.0                 ],
                    'deductible3':  [   0.0,    0.0,    0.0,                0.0                 ],
                    'attachment':   [   0.0,    0.0,    0.0,                0.0                 ],
                    'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0     ],  
                    'share1':       [   0.0,    0.0,    0.5,                0.0                 ],  
                    'share2':       [   0.0,    0.0,    1.0,                1.0                 ],  
                    'share3':       [   0.0,    0.0,    1.0,                1.0                 ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_loc_level_PR_loc_filter_1(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   0.5     ],
            'RiskLimit':        [   10.0    ],
            'RiskAttachment':   [   5.0     ],
            'OccLimit':         [   50.0    ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'PR'    ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      1       ],
            'PortNumber':   [   '1',    '1'     ],
            'AccNumber':    [   'A1',    'A2'   ],
            'PolNumber':    [   '',     ''      ],
            'LocGroup':     [   '',     ''      ],
            'LocNumber':    [   'L1',   'L2'    ],
            'CedantName':   [   '',     ''      ],
            'ProducerName': [   '',     ''      ],
            'LOB':          [   '',     ''      ],
            'CountryCode':  [   '',     ''      ],
            'ReinsTag':     [   '',     ''      ],
            'RiskLevel':    [   'LOC',  'LOC'   ],
            'CededPercent': [   1.0,    1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_2_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='LOC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  2,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  3,  1,  2,  3   ],
                    'profile_id':   [   4,  3,  3,  1,  2,  2,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,      4       ],
                    'calcrule_id':  [   14,     12,     24,     23      ],
                    'deductible1':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    5.0,    0.0     ],
                    'limit':        [   0.0,    0.0,    10.0,   50.0    ],  
                    'share1':       [   0.0,    0.0,    0.5,    0.0     ],  
                    'share2':       [   0.0,    0.0,    1.0,    0.8     ],  
                    'share3':       [   0.0,    0.0,    1.0,    1.0     ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  3,  1,  2,  3,  1,  2,  3,  4,  5,  6,  7,  8,  9   ],
                    'level_id':     [   3,  3,  3,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  3,  1,  1,  1,  2,  2,  2,  3,  3,  3   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_loc_level_PR_loc_filter_2(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   0.5     ],
            'RiskLimit':        [   10.0    ],
            'RiskAttachment':   [   5.0     ],
            'OccLimit':         [   50.0    ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'PR'    ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      1       ],
            'PortNumber':   [   '1',    '1'     ],
            'AccNumber':    [   'A1',    'A2'     ],
            'PolNumber':    [   '',     ''      ],
            'LocGroup':     [   '',     ''      ],
            'LocNumber':    [   'L1',   'XX'    ],
            'CedantName':   [   '',     ''      ],
            'ProducerName': [   '',     ''      ],
            'LOB':          [   '',     ''      ],
            'CountryCode':  [   '',     ''      ],
            'ReinsTag':     [   '',     ''      ],
            'RiskLevel':    [   'LOC',  'LOC'   ],
            'CededPercent': [   1.0,    1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_2_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='LOC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  2,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  3,  1,  2,  3   ],
                    'profile_id':   [   4,  3,  1,  1,  2,  1,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,      4       ],
                    'calcrule_id':  [   14,     12,     24,     23      ],
                    'deductible1':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    5.0,    0.0     ],
                    'limit':        [   0.0,    0.0,    10.0,   50.0    ],  
                    'share1':       [   0.0,    0.0,    0.5,    0.0     ],  
                    'share2':       [   0.0,    0.0,    1.0,    0.8     ],  
                    'share3':       [   0.0,    0.0,    1.0,    1.0     ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  3,  1,  2,  3,  1,  2,  3,  4,  5,  6,  7,  8,  9  ],
                    'level_id':     [   3,  3,  3,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1  ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  3,  1,  1,  1,  2,  2,  2,  3,  3,  3  ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)        


    def test_single_loc_level_PR_pol_and_loc_filter_2(self):

        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   0.5     ],
            'RiskLimit':        [   10.0    ],
            'RiskAttachment':   [   5.0     ],
            'OccLimit':         [   50.0    ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'PR'    ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      1       ],
            'PortNumber':   [   '1',    '1'     ],
            'AccNumber':    [   'A1',    'A2'   ],
            'PolNumber':    [   'P1',     'P1'  ],
            'LocGroup':     [   '',     ''      ],
            'LocNumber':    [   'L1',   'XX'    ],
            'CedantName':   [   '',     ''      ],
            'ProducerName': [   '',     ''      ],
            'LOB':          [   '',     ''      ],
            'CountryCode':  [   '',     ''      ],
            'ReinsTag':     [   '',     ''      ],
            'RiskLevel':    [   'LOC',  'LOC'   ],
            'CededPercent': [   1.0,    1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_2_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='LOC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  2,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  3,  1,  2,  3   ],
                    'profile_id':   [   4,  3,  1,  1,  2,  1,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,      4       ],
                    'calcrule_id':  [   14,     12,     24,     23      ],
                    'deductible1':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    5.0,    0.0     ],
                    'limit':        [   0.0,    0.0,    10.0,   50.0    ],  
                    'share1':       [   0.0,    0.0,    0.5,    0.0     ],  
                    'share2':       [   0.0,    0.0,    1.0,    0.8     ],  
                    'share3':       [   0.0,    0.0,    1.0,    1.0     ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  3,  1,  2,  3,  1,  2,  3,  4,  5,  6,  7,  8,  9  ],
                    'level_id':     [   3,  3,  3,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1  ],
                    'to_agg_id':    [   1,  1,  1,  1,  2,  3,  1,  1,  1,  2,  2,  2,  3,  3,  3  ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)        


    def test_single_CXL_no_filter(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   10.0    ],
            'OccAttachment':    [   50.0    ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'CXL'   ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   ''     ],
            'AccNumber':    [   ''      ],
            'PolNumber':    [   ''      ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'SEL'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_3_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='SEL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4,  5   ],
                    'profile_id':   [   3,  2,  2,  2,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3       ],
                    'calcrule_id':  [   14,     12,     24      ],
                    'deductible1':  [   0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    50.0    ],
                    'limit':        [   0.0,    0.0,    10.0    ],  
                    'share1':       [   0.0,    0.0,    1.0     ],  
                    'share2':       [   0.0,    0.0,    0.8     ],  
                    'share3':       [   0.0,    0.0,    1.0     ]
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  5,  1,  2,  3,  4,  5,  6,  10, 11, 12, 7,  8,  9,  13, 14, 15  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  1,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  5   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_CXL_port_filter(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   10.0    ],
            'OccAttachment':    [   50.0    ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'CXL'   ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   ''      ],
            'PolNumber':    [   ''      ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'SEL'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_3_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='SEL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4,  5   ],
                    'profile_id':   [   3,  2,  1,  2,  2,  2,  2,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3       ],
                    'calcrule_id':  [   14,     12,     24      ],
                    'deductible1':  [   0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    50.0    ],
                    'limit':        [   0.0,    0.0,    10.0    ],  
                    'share1':       [   0.0,    0.0,    1.0     ],  
                    'share2':       [   0.0,    0.0,    0.8     ],  
                    'share3':       [   0.0,    0.0,    1.0     ]
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  5,  1,  2,  3,  4,  5,  6,  10, 11, 12, 7,  8,  9,  13, 14, 15  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  1,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  5   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_CXL_port_acc_filter(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   10.0    ],
            'OccAttachment':    [   50.0    ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'CXL'   ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   'A1'    ],
            'PolNumber':    [   ''      ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'SEL'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_3_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='SEL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4,  5   ],
                    'profile_id':   [   3,  2,  1,  2,  2,  1,  1,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3       ],
                    'calcrule_id':  [   14,     12,     24      ],
                    'deductible1':  [   0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    50.0    ],
                    'limit':        [   0.0,    0.0,    10.0    ],  
                    'share1':       [   0.0,    0.0,    1.0     ],  
                    'share2':       [   0.0,    0.0,    0.8     ],  
                    'share3':       [   0.0,    0.0,    1.0     ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  5,  1,  2,  3,  4,  5,  6,  10, 11, 12, 7,  8,  9,  13, 14, 15  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  1,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  5   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)        


    def test_single_CXL_port_acc_pol_filter(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   10.0    ],
            'OccAttachment':    [   50.0    ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'CXL'   ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   'A2'    ],
            'PolNumber':    [   'P1'    ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   ''      ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'SEL'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_3_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='SEL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4,  5   ],
                    'profile_id':   [   3,  2,  1,  1,  1,  2,  1,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3       ],
                    'calcrule_id':  [   14,     12,     24      ],
                    'deductible1':  [   0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    50.0    ],
                    'limit':        [   0.0,    0.0,    10.0    ],  
                    'share1':       [   0.0,    0.0,    1.0     ],  
                    'share2':       [   0.0,    0.0,    0.8     ],  
                    'share3':       [   0.0,    0.0,    1.0     ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  5,  1,  2,  3,  4,  5,  6,  10, 11, 12, 7,  8,  9,  13, 14, 15  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  1,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  5   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_CXL_port_pol_loc_filter(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   10.0    ],
            'OccAttachment':    [   50.0    ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'CXL'   ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   ''      ],
            'PolNumber':    [   'P1'    ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   'L2'    ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'SEL'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_3_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='SEL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4,  5   ],
                    'profile_id':   [   3,  2,  1,  1,  1,  2,  1,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3       ],
                    'calcrule_id':  [   14,     12,     24      ],
                    'deductible1':  [   0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    50.0    ],
                    'limit':        [   0.0,    0.0,    10.0    ],  
                    'share1':       [   0.0,    0.0,    1.0     ],  
                    'share2':       [   0.0,    0.0,    0.8     ],  
                    'share3':       [   0.0,    0.0,    1.0     ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  5,  1,  2,  3,  4,  5,  6,  10, 11, 12, 7,  8,  9,  13, 14, 15  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  1,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  5   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_CXL_port_acc_loc_filter(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   10.0    ],
            'OccAttachment':    [   50.0    ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'CXL'   ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   'A1'      ],
            'PolNumber':    [   ''    ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   'L1'    ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'SEL'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_3_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='SEL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4,  5   ],
                    'profile_id':   [   3,  2,  1,  2,  2,  1,  1,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3       ],
                    'calcrule_id':  [   14,     12,     24      ],
                    'deductible1':  [   0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    50.0    ],
                    'limit':        [   0.0,    0.0,    10.0    ],  
                    'share1':       [   0.0,    0.0,    1.0     ],  
                    'share2':       [   0.0,    0.0,    0.8     ],  
                    'share3':       [   0.0,    0.0,    1.0     ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  5,  1,  2,  3,  4,  5,  6,  10, 11, 12, 7,  8,  9,  13, 14, 15  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  1,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  5   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_CXL_port_acc_pol_loc_filter(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   0.0     ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   10.0    ],
            'OccAttachment':    [   50.0    ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'CXL'   ],
            'PlacedPercent':    [   0.8     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1       ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   'A1'    ],
            'PolNumber':    [   'P1'    ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   'L1'    ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'SEL'   ],
            'CededPercent': [   1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_3_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='SEL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1   ],
                    'level_id':     [   3,  2,  2,  1,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  1,  2,  3,  4,  5   ],
                    'profile_id':   [   3,  2,  1,  2,  1,  1,  1,  1   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3       ],
                    'calcrule_id':  [   14,     12,     24      ],
                    'deductible1':  [   0.0,    0.0,    0.0     ],
                    'deductible2':  [   0.0,    0.0,    0.0     ],
                    'deductible3':  [   0.0,    0.0,    0.0     ],
                    'attachment':   [   0.0,    0.0,    50.0    ],
                    'limit':        [   0.0,    0.0,    10.0    ],  
                    'share1':       [   0.0,    0.0,    1.0     ],  
                    'share2':       [   0.0,    0.0,    0.8     ],  
                    'share3':       [   0.0,    0.0,    1.0     ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  1,  2,  3,  4,  5,  1,  2,  3,  4,  5,  6,  10, 11, 12, 7,  8,  9,  13, 14, 15  ],
                    'level_id':     [   3,  3,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  1,  2,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  5   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_multiple_SS_same_inuring_level(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1,      2       ],
            'CededPercent':     [   1.0,    1.0     ],
            'RiskLimit':        [   0.0,    0.0     ],
            'RiskAttachment':   [   0.0,    0.0     ],
            'OccLimit':         [   0.0,    0.0     ],
            'OccAttachment':    [   0.0,    0.0     ],
            'InuringPriority':  [   1,      1       ],
            'ReinsType':        [   'SS',   'SS'    ],
            'PlacedPercent':    [   1.0,    1.0     ],
            'TreatyShare':      [   1.0,    1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      1,      2,      2       ],
            'PortNumber':   [   '1',    '1',    '1',    '1'     ],
            'AccNumber':    [   '1',    '1',    '2',    '2'     ],
            'PolNumber':    [   '1',    '1',    '1',    '1'     ],
            'LocGroup':     [   '',     '',     '',     ''      ],
            'LocNumber':    [   '1',    '2',    '1',    '2'     ],
            'CedantName':   [   '',     '',     '',     ''      ],
            'ProducerName': [   '',     '',     '',     ''      ],
            'LOB':          [   '',     '',     '',     ''      ],
            'CountryCode':  [   '',     '',     '',     ''      ],
            'ReinsTag':     [   '',     '',     '',     ''      ],
            'RiskLevel':    [   'LOC',  'LOC',  'LOC',  'LOC'   ],
            'CededPercent': [   0.1,    0.1,    0.1,    0.1     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='LOC',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                    'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2   ],
                    'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1,  3,  2,  2,  2,  2,  1,  1,  1,  1   ],
                    'agg_id':       [   1,  1,  2,  3,  4,  1,  2,  3,  4,  1,  1,  2,  3,  4,  1,  2,  3,  4   ],
                    'profile_id':   [   5,  3,  4,  1,  1,  2,  2,  2,  2,  8,  1,  1,  6,  7,  2,  2,  2,  2   ]
                }),
                fm_profile=pd.DataFrame.from_dict({
                    'profile_id':   [   1,      2,      3,                  4,                  5,                  6,                  7,                  8                   ],
                    'calcrule_id':  [   14,     12,     24,                 24,                 23,                 24,                 24,                 23                  ],
                    'deductible1':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0,                0.0                 ],
                    'deductible2':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0,                0.0                 ],
                    'deductible3':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0,                0.0                 ],
                    'attachment':   [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0,                0.0,                0.0                 ],
                    'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0,    9999999999999.0,    9999999999999.0,    9999999999999.0,    9999999999999.0     ],  
                    'share1':       [   0.0,    0.0,    0.1,                0.1,                0.0,                0.1,                0.1,                0.0                 ],  
                    'share2':       [   0.0,    0.0,    1.0,                1.0,                1.0,                1.0,                1.0,                1.0                 ],  
                    'share3':       [   0.0,    0.0,    1.0,                1.0,                1.0,                1.0,                1.0,                1.0                 ]   
                }),
                fm_programme=pd.DataFrame.from_dict({
                    'from_agg_id':  [   1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                    'level_id':     [   3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                    'to_agg_id':    [   1,  1,  1,  1,  1,  2,  3,  4,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)                        


    def test_multiple_SS_different_inuring_levels(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1,      2       ],
            'CededPercent':     [   1.0,    1.0     ],
            'RiskLimit':        [   0.0,    0.0     ],
            'RiskAttachment':   [   0.0,    0.0     ],
            'OccLimit':         [   0.0,    0.0     ],
            'OccAttachment':    [   0.0,    0.0     ],
            'InuringPriority':  [   1,      2       ],
            'ReinsType':        [   'SS',   'SS'    ],
            'PlacedPercent':    [   1.0,    1.0     ],
            'TreatyShare':      [   1.0,    1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      1,      2,      2       ],
            'PortNumber':   [   '1',    '1',    '1',    '1'     ],
            'AccNumber':    [   '1',    '1',    '1',    '2'     ],
            'PolNumber':    [   '1',    '1',    '1',    '1'     ],
            'LocGroup':     [   '',     '',     '',     ''      ],
            'LocNumber':    [   '1',    '2',    '1',    '2'     ],
            'CedantName':   [   '',     '',     '',     ''      ],
            'ProducerName': [   '',     '',     '',     ''      ],
            'LOB':          [   '',     '',     '',     ''      ],
            'CountryCode':  [   '',     '',     '',     ''      ],
            'ReinsTag':     [   '',     '',     '',     ''      ],
            'RiskLevel':    [   'LOC',  'LOC',  'LOC',  'LOC'   ],
            'CededPercent': [   0.1,    0.1,    0.1,    0.1     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [
            reinsurance_layer.RiInputs(
                risk_level='LOC',   
                inuring_priority=1,
                ri_inputs=reinsurance_layer.RiLayerInputs(
                    fm_policytc=pd.DataFrame.from_dict({
                        'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                        'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1   ],
                        'agg_id':       [   1,  1,  2,  3,  4,  1,  2,  3,  4   ],
                        'profile_id':   [   5,  3,  4,  1,  1,  2,  2,  2,  2   ]
                    }),
                    fm_profile=pd.DataFrame.from_dict({
                        'profile_id':   [   1,      2,      3,                  4,                  5                  ],
                        'calcrule_id':  [   14,     12,     24,                 24,                 23                 ],
                        'deductible1':  [   0.0,    0.0,    0.0,                0.0,                0.0                ],
                        'deductible2':  [   0.0,    0.0,    0.0,                0.0,                0.0                ],
                        'deductible3':  [   0.0,    0.0,    0.0,                0.0,                0.0                ],
                        'attachment':   [   0.0,    0.0,    0.0,                0.0,                0.0                ],
                        'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0,    9999999999999.0    ],  
                        'share1':       [   0.0,    0.0,    0.1,                0.1,                0.0                ],  
                        'share2':       [   0.0,    0.0,    1.0,                1.0,                1.0                ],  
                        'share3':       [   0.0,    0.0,    1.0,                1.0,                1.0                ]   
                    }),
                    fm_programme=pd.DataFrame.from_dict({
                        'from_agg_id':  [   1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                        'level_id':     [   3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                        'to_agg_id':    [   1,  1,  1,  1,  1,  2,  3,  4,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                    })
                )),
            reinsurance_layer.RiInputs(
                risk_level='LOC',   
                inuring_priority=2,
                ri_inputs=reinsurance_layer.RiLayerInputs(
                    fm_policytc=pd.DataFrame.from_dict({
                        'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                        'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1   ],
                        'agg_id':       [   1,  1,  2,  3,  4,  1,  2,  3,  4   ],
                        'profile_id':   [   5,  3,  1,  1,  4,  2,  2,  2,  2   ]
                    }),
                    fm_profile=pd.DataFrame.from_dict({
                        'profile_id':   [   1,      2,      3,                  4,                  5                  ],
                        'calcrule_id':  [   14,     12,     24,                 24,                 23                 ],
                        'deductible1':  [   0.0,    0.0,    0.0,                0.0,                0.0                ],
                        'deductible2':  [   0.0,    0.0,    0.0,                0.0,                0.0                ],
                        'deductible3':  [   0.0,    0.0,    0.0,                0.0,                0.0                ],
                        'attachment':   [   0.0,    0.0,    0.0,                0.0,                0.0                ],
                        'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0,    9999999999999.0    ],  
                        'share1':       [   0.0,    0.0,    0.1,                0.1,                0.0                ],  
                        'share2':       [   0.0,    0.0,    1.0,                1.0,                1.0                ],  
                        'share3':       [   0.0,    0.0,    1.0,                1.0,                1.0                ]   
                    }),
                    fm_programme=pd.DataFrame.from_dict({
                        'from_agg_id':  [   1,  2,  3,  4,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                        'level_id':     [   3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                        'to_agg_id':    [   1,  1,  1,  1,  1,  2,  3,  4,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                    })
                ))
                ]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)                        


    def test_multiple_QS_same_inuring_level(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1,      2       ],
            'CededPercent':     [   0.2,    0.3     ],
            'RiskLimit':        [   0.0,    0.0     ],
            'RiskAttachment':   [   0.0,    0.0     ],
            'OccLimit':         [   0.0,    0.0     ],
            'OccAttachment':    [   0.0,    0.0     ],
            'InuringPriority':  [   1,      1       ],
            'ReinsType':        [   'QS',   'QS'    ],
            'PlacedPercent':    [   1.0,    1.0     ],
            'TreatyShare':      [   1.0,    1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      2       ],
            'PortNumber':   [   '1',    '1'     ],
            'AccNumber':    [   '',     ''     ],
            'PolNumber':    [   '',     ''     ],
            'LocGroup':     [   '',     ''      ],
            'LocNumber':    [   '',     ''     ],
            'CedantName':   [   '',     ''      ],
            'ProducerName': [   '',     ''      ],
            'LOB':          [   '',     ''      ],
            'CountryCode':  [   '',     ''      ],
            'ReinsTag':     [   '',     ''      ],
            'RiskLevel':    [   'SEL',  'SEL'   ],
            'CededPercent': [   1.0,    1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='SEL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                        'layer_id':     [   1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2   ],
                        'level_id':     [   3,  2,  1,  1,  1 , 1,  3,  2,  1,  1,  1 , 1   ],
                        'agg_id':       [   1,  1,  1,  2,  3,  4,  1,  1,  1,  2,  3,  4   ],
                        'profile_id':   [   4,  3,  2,  2,  2,  2,  6,  5,  2,  2,  2,  2   ]
                    }),
                    fm_profile=pd.DataFrame.from_dict({
                        'profile_id':   [   1,      2,      3,                  4,                  5,                  6                   ],
                        'calcrule_id':  [   14,     12,     24,                 23,                 24,                 23                  ],
                        'deductible1':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0                 ],
                        'deductible2':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0                 ],
                        'deductible3':  [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0                 ],
                        'attachment':   [   0.0,    0.0,    0.0,                0.0,                0.0,                0.0                 ],
                        'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0,    9999999999999.0,    9999999999999.0     ],  
                        'share1':       [   0.0,    0.0,    0.2,                0.0,                0.3,                0.0                 ],  
                        'share2':       [   0.0,    0.0,    1.0,                1.0,                1.0,                1.0                 ],  
                        'share3':       [   0.0,    0.0,    1.0,                1.0,                1.0,                1.0                 ]   
                    }),
                    fm_programme=pd.DataFrame.from_dict({
                        'from_agg_id':  [   1,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                        'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                        'to_agg_id':    [   1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                    })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_multiple_QS_different_inuring_level(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1,      2       ],
            'CededPercent':     [   0.2,    0.3     ],
            'RiskLimit':        [   0.0,    0.0     ],
            'RiskAttachment':   [   0.0,    0.0     ],
            'OccLimit':         [   0.0,    0.0     ],
            'OccAttachment':    [   0.0,    0.0     ],
            'InuringPriority':  [   1,      2       ],
            'ReinsType':        [   'QS',   'QS'    ],
            'PlacedPercent':    [   1.0,    1.0     ],
            'TreatyShare':      [   1.0,    1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      2       ],
            'PortNumber':   [   '1',    '1'     ],
            'AccNumber':    [   '',     ''     ],
            'PolNumber':    [   '',     ''     ],
            'LocGroup':     [   '',     ''      ],
            'LocNumber':    [   '',     ''     ],
            'CedantName':   [   '',     ''      ],
            'ProducerName': [   '',     ''      ],
            'LOB':          [   '',     ''      ],
            'CountryCode':  [   '',     ''      ],
            'ReinsTag':     [   '',     ''      ],
            'RiskLevel':    [   'SEL',  'SEL'   ],
            'CededPercent': [   1.0,    1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [
            reinsurance_layer.RiInputs(
                risk_level='SEL',   
                inuring_priority=1,
                ri_inputs=reinsurance_layer.RiLayerInputs(
                    fm_policytc=pd.DataFrame.from_dict({
                            'layer_id':     [   1,  1,  1,  1,  1,  1   ],
                            'level_id':     [   3,  2,  1,  1,  1 , 1   ],
                            'agg_id':       [   1,  1,  1,  2,  3,  4   ],
                            'profile_id':   [   4,  3,  2,  2,  2,  2   ]
                        }),
                        fm_profile=pd.DataFrame.from_dict({
                            'profile_id':   [   1,      2,      3,                  4                  ],
                            'calcrule_id':  [   14,     12,     24,                 23                 ],
                            'deductible1':  [   0.0,    0.0,    0.0,                0.0                ],
                            'deductible2':  [   0.0,    0.0,    0.0,                0.0                ],
                            'deductible3':  [   0.0,    0.0,    0.0,                0.0                ],
                            'attachment':   [   0.0,    0.0,    0.0,                0.0                ],
                            'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0    ],  
                            'share1':       [   0.0,    0.0,    0.2,                0.0                ],  
                            'share2':       [   0.0,    0.0,    1.0,                1.0                ],  
                            'share3':       [   0.0,    0.0,    1.0,                1.0                ]   
                        }),
                        fm_programme=pd.DataFrame.from_dict({
                            'from_agg_id':  [   1,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                            'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                            'to_agg_id':    [   1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                        })
                )),
            reinsurance_layer.RiInputs(
                risk_level='SEL',   
                inuring_priority=2,
                ri_inputs=reinsurance_layer.RiLayerInputs(
                    fm_policytc=pd.DataFrame.from_dict({
                            'layer_id':     [   1,  1,  1,  1,  1,  1   ],
                            'level_id':     [   3,  2,  1,  1,  1 , 1   ],
                            'agg_id':       [   1,  1,  1,  2,  3,  4   ],
                            'profile_id':   [   4,  3,  2,  2,  2,  2   ]
                        }),
                        fm_profile=pd.DataFrame.from_dict({
                            'profile_id':   [   1,      2,      3,                  4                  ],
                            'calcrule_id':  [   14,     12,     24,                 23                 ],
                            'deductible1':  [   0.0,    0.0,    0.0,                0.0                ],
                            'deductible2':  [   0.0,    0.0,    0.0,                0.0                ],
                            'deductible3':  [   0.0,    0.0,    0.0,                0.0                ],
                            'attachment':   [   0.0,    0.0,    0.0,                0.0                ],
                            'limit':        [   0.0,    0.0,    9999999999999.0,    9999999999999.0    ],  
                            'share1':       [   0.0,    0.0,    0.3,                0.0                ],  
                            'share2':       [   0.0,    0.0,    1.0,                1.0                ],  
                            'share3':       [   0.0,    0.0,    1.0,                1.0                ]   
                        }),
                        fm_programme=pd.DataFrame.from_dict({
                            'from_agg_id':  [   1,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                            'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                            'to_agg_id':    [   1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                        })
                ))
                ]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)        


    def test_multiple_CXL_same_inuring_level(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1,      2       ],
            'CededPercent':     [   0.2,    0.3     ],
            'RiskLimit':        [   0.0,    0.0     ],
            'RiskAttachment':   [   0.0,    0.0     ],
            'OccLimit':         [   10.0,   30.0    ],
            'OccAttachment':    [   20.0,   40.0    ],
            'InuringPriority':  [   1,      1       ],
            'ReinsType':        [   'CXL',  'CXL'   ],
            'PlacedPercent':    [   1.0,    1.0     ],
            'TreatyShare':      [   1.0,    1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      2       ],
            'PortNumber':   [   '1',    '1'     ],
            'AccNumber':    [   '',     ''     ],
            'PolNumber':    [   '',     ''     ],
            'LocGroup':     [   '',     ''      ],
            'LocNumber':    [   '',     ''     ],
            'CedantName':   [   '',     ''      ],
            'ProducerName': [   '',     ''      ],
            'LOB':          [   '',     ''      ],
            'CountryCode':  [   '',     ''      ],
            'ReinsTag':     [   '',     ''      ],
            'RiskLevel':    [   'SEL',  'SEL'   ],
            'CededPercent': [   1.0,    1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [reinsurance_layer.RiInputs(
            risk_level='SEL',   
            inuring_priority=1,
            ri_inputs=reinsurance_layer.RiLayerInputs(
                fm_policytc=pd.DataFrame.from_dict({
                        'layer_id':     [   1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2   ],
                        'level_id':     [   3,  2,  1,  1,  1 , 1,  3,  2,  1,  1,  1 , 1   ],
                        'agg_id':       [   1,  1,  1,  2,  3,  4,  1,  1,  1,  2,  3,  4   ],
                        'profile_id':   [   3,  2,  2,  2,  2,  2,  4,  2,  2,  2,  2,  2   ]
                    }),
                    fm_profile=pd.DataFrame.from_dict({
                        'profile_id':   [   1,      2,      3,      4       ],
                        'calcrule_id':  [   14,     12,     24,     24      ],
                        'deductible1':  [   0.0,    0.0,    0.0,    0.0     ],
                        'deductible2':  [   0.0,    0.0,    0.0,    0.0     ],
                        'deductible3':  [   0.0,    0.0,    0.0,    0.0     ],
                        'attachment':   [   0.0,    0.0,    20.0,   40.0    ],
                        'limit':        [   0.0,    0.0,    10.0,   30.0    ],  
                        'share1':       [   0.0,    0.0,    0.2,    0.3     ],  
                        'share2':       [   0.0,    0.0,    1.0,    1.0     ],  
                        'share3':       [   0.0,    0.0,    1.0,    1.0     ]   
                    }),
                    fm_programme=pd.DataFrame.from_dict({
                        'from_agg_id':  [   1,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                        'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                        'to_agg_id':    [   1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                    })
            )
        )]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_multiple_CXL_different_inuring_level(self):
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1,      2       ],
            'CededPercent':     [   0.2,    0.3     ],
            'RiskLimit':        [   0.0,    0.0     ],
            'RiskAttachment':   [   0.0,    0.0     ],
            'OccLimit':         [   10.0,   30.0    ],
            'OccAttachment':    [   20.0,   40.0    ],
            'InuringPriority':  [   1,      2       ],
            'ReinsType':        [   'CXL',  'CXL'   ],
            'PlacedPercent':    [   1.0,    1.0     ],
            'TreatyShare':      [   1.0,    1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      2       ],
            'PortNumber':   [   '1',    '1'     ],
            'AccNumber':    [   '',     ''      ],
            'PolNumber':    [   '',     ''      ],
            'LocGroup':     [   '',     ''      ],
            'LocNumber':    [   '',     ''      ],
            'CedantName':   [   '',     ''      ],
            'ProducerName': [   '',     ''      ],
            'LOB':          [   '',     ''      ],
            'CountryCode':  [   '',     ''      ],
            'ReinsTag':     [   '',     ''      ],
            'RiskLevel':    [   'SEL',  'SEL'   ],
            'CededPercent': [   1.0,    1.0     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            self.exposure_1_items_df,
            self.exposure_1_coverages_df,
            self.exposure_1_xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        expected_ri_inputs = [
            reinsurance_layer.RiInputs(
                risk_level='SEL',   
                inuring_priority=1,
                ri_inputs=reinsurance_layer.RiLayerInputs(
                    fm_policytc=pd.DataFrame.from_dict({
                            'layer_id':     [   1,  1,  1,  1,  1,  1   ],
                            'level_id':     [   3,  2,  1,  1,  1 , 1   ],
                            'agg_id':       [   1,  1,  1,  2,  3,  4   ],
                            'profile_id':   [   3,  2,  2,  2,  2,  2   ]
                        }),
                        fm_profile=pd.DataFrame.from_dict({
                            'profile_id':   [   1,      2,      3      ],
                            'calcrule_id':  [   14,     12,     24     ],
                            'deductible1':  [   0.0,    0.0,    0.0    ],
                            'deductible2':  [   0.0,    0.0,    0.0    ],
                            'deductible3':  [   0.0,    0.0,    0.0    ],
                            'attachment':   [   0.0,    0.0,    20.0   ],
                            'limit':        [   0.0,    0.0,    10.0   ],  
                            'share1':       [   0.0,    0.0,    0.2    ],  
                            'share2':       [   0.0,    0.0,    1.0    ],  
                            'share3':       [   0.0,    0.0,    1.0    ]   
                        }),
                        fm_programme=pd.DataFrame.from_dict({
                            'from_agg_id':  [   1,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                            'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                            'to_agg_id':    [   1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                        })
                )
            ),
            reinsurance_layer.RiInputs(
                risk_level='SEL',   
                inuring_priority=2,
                ri_inputs=reinsurance_layer.RiLayerInputs(
                    fm_policytc=pd.DataFrame.from_dict({
                            'layer_id':     [   1,  1,  1,  1,  1,  1   ],
                            'level_id':     [   3,  2,  1,  1,  1 , 1   ],
                            'agg_id':       [   1,  1,  1,  2,  3,  4   ],
                            'profile_id':   [   3,  2,  2,  2,  2,  2   ]
                        }),
                        fm_profile=pd.DataFrame.from_dict({
                            'profile_id':   [   1,      2,      3      ],
                            'calcrule_id':  [   14,     12,     24     ],
                            'deductible1':  [   0.0,    0.0,    0.0    ],
                            'deductible2':  [   0.0,    0.0,    0.0    ],
                            'deductible3':  [   0.0,    0.0,    0.0    ],
                            'attachment':   [   0.0,    0.0,    40.0   ],
                            'limit':        [   0.0,    0.0,    30.0   ],  
                            'share1':       [   0.0,    0.0,    0.3    ],  
                            'share2':       [   0.0,    0.0,    1.0    ],  
                            'share3':       [   0.0,    0.0,    1.0    ]   
                        }),
                        fm_programme=pd.DataFrame.from_dict({
                            'from_agg_id':  [   1,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12  ],
                            'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                            'to_agg_id':    [   1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4   ]  
                        })
                )
            )
        ]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def test_single_pol_level_fac_with_direct_layers(self):
        
        items_df = pd.DataFrame.from_dict({
            'item_id':          [   1,      2,      3,      4,      5,      6,      7,      8,      9,      10,     11,     12  ],
            'coverage_id':      [   4,      10,     5,      11,     6,      12,     7,      1,      8,      2,      9,      3   ],
            'area_peril_id':    [   -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1  ],
            'vulnerability_id': [   -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1,     -1  ],
            'group_id':         [   1,      1,      1,      2,      2,      2,      3,      3,      3,      4,      4,      4   ]
            })
        
        coverages_df = pd.DataFrame.from_dict({
            'coverage_id':   [   1,     2,      3,      4,      5,      6,      7,      8,      9,      10,     11,     12  ],
            'tiv':          [   1000,   500,    500,    1000,   500,    500,    1000,   500,    500,    1000,   500,    500 ]
            })
        
        xref_descriptions_df = pd.DataFrame.from_dict({
            'loc_idx':          [   1,      1,      2,      1,      1,      2,      1,      1,      2,      3,     11,      12,     3,      14,     15,     3,      17,     18      ],
            'output_id':           [   1,      2,      3,      4,      5,      6,      7,      8,      9,      10,     11,     12,     13,     14,     15,     16,     17,     18      ],
            'portnumber':       [   '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1'     ],
            'polnumber':        [   '1',    '2',    '1',    '1',    '2',    '1',    '1',    '2',    '1',    '1',    '1',    '2',    '1',    '1',    '2',    '1',    '1',    '2'     ],
            'accnumber':        [   '1',    '1',    '2',    '1',    '1',    '2',    '1',    '1',    '2',    '2',    '1',    '1',    '2',    '1',    '1',    '2',    '1',    '1'     ],
            'locnumber':        [   '2',    '2',    '2',    '2',    '2',    '2',    '2',    '2',    '2',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1',    '1'     ],
            'locgroup':         [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'cedantname':       [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'producername':     [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'lob':              [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'countrycode':      [   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US',   'US'    ],
            'reinstag':         [   '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     '',     ''      ],
            'coverage_type_id': [   1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1       ],
            'peril_id':         [   1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1       ],
            'tiv':              [   1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1       ]
        })
        
        ri_info_df = pd.DataFrame.from_dict({
            'ReinsNumber':      [   1       ],
            'CededPercent':     [   1.0     ],
            'RiskLimit':        [   10.0    ],
            'RiskAttachment':   [   0.0     ],
            'OccLimit':         [   0.0     ],
            'OccAttachment':    [   0.0     ],
            'InuringPriority':  [   1       ],
            'ReinsType':        [   'FAC'    ],
            'PlacedPercent':    [   1.0     ],
            'TreatyShare':      [   1.0     ]
        })        
        ri_scope_df = pd.DataFrame.from_dict({
            'ReinsNumber':  [   1,      ],
            'PortNumber':   [   '1'     ],
            'AccNumber':    [   '1'     ],
            'PolNumber':    [   ''      ],
            'LocGroup':     [   ''      ],
            'LocNumber':    [   '1'     ],
            'CedantName':   [   ''      ],
            'ProducerName': [   ''      ],
            'LOB':          [   ''      ],
            'CountryCode':  [   ''      ],
            'ReinsTag':     [   ''      ],
            'RiskLevel':    [   'LOC'   ],
            'CededPercent': [   0.1     ]
        })
        
        ri_inputs = reinsurance_layer._get_ri_inputs(
            items_df,
            coverages_df,
            xref_descriptions_df,
            ri_info_df,
            ri_scope_df)

        ri_inputs[0].ri_inputs.fm_policytc.to_csv('/tmp/fm_policytc.csv', index=False)
        ri_inputs[0].ri_inputs.fm_programme.to_csv('/tmp/fm_programme.csv', index=False)
        ri_inputs[0].ri_inputs.fm_profile.to_csv('/tmp/fm_profile.csv', index=False)

        expected_ri_inputs = [
            reinsurance_layer.RiInputs(
                risk_level='LOC',   
                inuring_priority=1,
                ri_inputs=reinsurance_layer.RiLayerInputs(
                    fm_policytc=pd.DataFrame.from_dict({
                            'layer_id':     [   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                            'level_id':     [   3,  2,  2,  2,  2,  1,  1,  1 , 1,  1,  1   ],
                            'agg_id':       [   1,  1,  2,  3,  4,  1,  2,  3,  4,  5,  6   ],
                            'profile_id':   [   2,  3,  1,  1,  1,  2,  2,  2,  2,  2,  2   ]
                        }),
                        fm_profile=pd.DataFrame.from_dict({
                            'profile_id':   [   1,      2,      3       ],
                            'calcrule_id':  [   14,     12,     24      ],
                            'deductible1':  [   0.0,    0.0,    0.0     ],
                            'deductible2':  [   0.0,    0.0,    0.0     ],
                            'deductible3':  [   0.0,    0.0,    0.0     ],
                            'attachment':   [   0.0,    0.0,    0.0     ],
                            'limit':        [   0.0,    0.0,    10.0    ],  
                            'share1':       [   0.0,    0.0,    1.0     ],  
                            'share2':       [   0.0,    0.0,    1.0     ],  
                            'share3':       [   0.0,    0.0,    1.0     ]
                        }),
                        fm_programme=pd.DataFrame.from_dict({
                            'from_agg_id':  [   1,  2,  3,  4,  1,  2,  3,  4,  5,  6,  11, 14, 17, 12, 15, 18, 1,  4,  7,  2,  5,  8,  10, 13, 16, 3,  6,  9   ],
                            'level_id':     [   3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1   ],
                            'to_agg_id':    [   1,  1,  1,  1,  1,  1,  2,  2,  3,  4,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  5,  6,  6,  6   ]  
                        })
                )
            )
        ]

        self.assert_ri_inputs_equal(expected_ri_inputs, ri_inputs)


    def assert_ri_inputs_equal(self, ri_inputs_1, ri_inputs_2):
        self.assertEqual(len(ri_inputs_1), len(ri_inputs_2))
        for i in range(len(ri_inputs_1)):
            self.assertEqual(ri_inputs_1[i].risk_level, ri_inputs_2[i].risk_level)
            self.assertEqual(ri_inputs_1[i].inuring_priority, ri_inputs_2[i].inuring_priority)
            assert_frame_equal(
                ri_inputs_1[i].ri_inputs.fm_policytc,
                ri_inputs_2[i].ri_inputs.fm_policytc,
                check_index_type=False)
            assert_frame_equal(
                ri_inputs_1[i].ri_inputs.fm_profile, 
                ri_inputs_2[i].ri_inputs.fm_profile,
                check_index_type=False)
            assert_frame_equal(
                ri_inputs_1[i].ri_inputs.fm_programme, 
                ri_inputs_2[i].ri_inputs.fm_programme,
                check_index_type=False)
