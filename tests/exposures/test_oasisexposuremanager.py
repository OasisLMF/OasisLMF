from unittest import TestCase

from oasislmf.exposures.manager import OasisExposuresManager
from oasislmf.models import OasisModel


class OasisExposureManagerAddModel(TestCase):
    def test_models_is_empty___model_is_added_to_model_dict(self):
        model = OasisModel('supplier', 'model', 'version')

        manager = OasisExposuresManager()
        manager.add_model(model)

        self.assertEqual({model.key: model}, manager.models)

    def test_manager_already_contains_a_model_with_the_given_key___model_is_replaced_in_models_dict(self):
        first = OasisModel('supplier', 'model', 'version')
        second = OasisModel('supplier', 'model', 'version')

        manager = OasisExposuresManager(oasis_models=[first])
        manager.add_model(second)

        self.assertIs(second, manager.models[second.key])

    def test_manager_already_contains_a_diferent_model___model_is_added_to_dict(self):
        first = OasisModel('first', 'model', 'version')
        second = OasisModel('second', 'model', 'version')

        manager = OasisExposuresManager(oasis_models=[first])
        manager.add_model(second)

        self.assertEqual({
            first.key: first,
            second.key: second,
        }, manager.models)
