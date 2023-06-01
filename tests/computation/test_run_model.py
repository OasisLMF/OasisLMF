import json
import io
import os

import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from tempfile import NamedTemporaryFile, TemporaryDirectory

import oasislmf
from oasislmf.manager import OasisManager

from collections import ChainMap

MIN_SETTINGS = {
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

class TestRunModel(unittest.TestCase):

    @staticmethod
    def create_tmp_files(file_list):
        return {f: NamedTemporaryFile() for f in file_list}

    @staticmethod
    def create_tmp_dirs(dirs_list):
        return {d: TemporaryDirectory() for d in dirs_list}

    @staticmethod
    def write_json(tmpfile, data):
        with open(tmpfile.name, mode='w') as f:
            f.write(json.dumps(data))

    @staticmethod
    def write_str(tmpfile, data):
        with open(tmpfile.name, mode='w') as f:
            f.write(data)

    @staticmethod
    def combine_args(dict_list):
        return dict(ChainMap(*dict_list))

    @staticmethod
    def called_args(mock_obj):
        return {k:v for k,v in mock_obj.call_args.kwargs.items()  if isinstance(v, (str, int))}

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        # Args
        cls.default_args = cls.manager._params_run_model()
        #cls.blank_args = {k:None for k,v in cls.default_args.items()}

        cls.pre_hook_args = cls.manager._params_exposure_pre_analysis()
        cls.gen_files_args = cls.manager._params_generate_files()
        cls.gen_loss_args = cls.manager._params_generate_losses()
        # Tempfiles
        cls.tmp_dirs = cls.create_tmp_dirs([a for a in cls.default_args.keys() if 'dir' in a])
        cls.tmp_files = cls.create_tmp_files(
            [a for a in cls.default_args.keys() if 'csv' in a] + 
            [a for a in cls.default_args.keys() if 'json' in a]
        )

    def setUp(self):
        self.min_args = {
            'oed_location_csv': self.tmp_files['oed_location_csv'].name,
            'analysis_settings_json': self.tmp_files['analysis_settings_json'].name,
            'keys_data_csv': self.tmp_files['keys_data_csv'].name,
            'model_data_dir':  self.tmp_dirs['model_data_dir'].name
        }
        self.write_json(self.tmp_files.get('analysis_settings_json'), MIN_SETTINGS)
        self.write_str(self.tmp_files.get('oed_location_csv'), MIN_LOC)
        self.write_str(self.tmp_files.get('keys_data_csv'), MIN_KEYS)


    def test_args__default_combine(self):
        expt_combined_args = self.combine_args([
            self.pre_hook_args,
            self.gen_files_args,
            self.gen_loss_args,
        ])
        self.assertEqual(expt_combined_args, self.default_args)


    def test_args__min_required(self):
        files_mock = MagicMock()
        losses_mock = MagicMock()
        run_dir = self.tmp_dirs.get('model_run_dir').name
        losses_mock._get_output_dir.return_value = run_dir

        with patch.object(oasislmf.computation.run.model, 'GenerateFiles', files_mock), \
             patch.object(oasislmf.computation.run.model, 'GenerateLosses', losses_mock):
            self.manager.run_model(**self.min_args)

        files_called_kwargs = self.called_args(files_mock)
        losses_called_kwargs = self.called_args(losses_mock)
        expected_called_kwargs = self.combine_args([self.min_args,
            {
                'model_run_dir': run_dir,
                'oasis_files_dir': os.path.join(run_dir, 'input')
            }    
        ])    
            
        files_mock.assert_called_once()
        losses_mock.assert_called_once()
        self.assertEqual(files_called_kwargs, expected_called_kwargs)
        self.assertEqual(losses_called_kwargs, expected_called_kwargs)
        
        

#    def test_ktools_args(self):
#        """
#        "ktools_legacy_stream": false,
#        "ktools_num_processes": -1,
#        "ktools_event_shuffle": 1,
#        "ktools_alloc_rule_gul": 0,
#        "ktools_num_gul_per_lb": 0,
#        "ktools_num_fm_per_lb": 0,
#        "ktools_disable_guard": false,
#        "ktools_fifo_relative": false,
#        "ktools_alloc_rule_il": 2,
#        "ktools_alloc_rule_ri": 3,
#        """
#
#
#
#    def test_pytools_args(self):
#        """
#        "fmpy": true,
#        "fmpy_low_memory": false,
#        "fmpy_sort_output": false,
#        "modelpy": true,
#        "model_py_server": false,
#        "gulpy": true,
#        "gulpy_random_generator": 1,
#        "gulmc": false,
#        "gulmc_random_generator": 1,
#        "gulmc_effective_damageability": false,
#        "gulmc_vuln_cache_size": 200,
#        """
#
#
#    def test_settings_files(self):
#        """
#        """
#
#
#
#
#    #def test_get_exposure_data_config(self):
#    #    import ipdb; ipdb.set_trace()
#    #    self.run_model = self.manager.run_model
