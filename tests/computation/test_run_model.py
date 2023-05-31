import unittest
from unittest.mock import MagicMock, patch


from oasislmf.manager import OasisManager

class TestRunModel(unittest.TestCase):
    def setUp(self):
        self.manager = OasisManager()
        self.run_model = self.manager.run_model
        self.default_args = self.manager._params_run_model()


    def test_default_args(self):
        import ipdb; ipdb.set_trace()
        

