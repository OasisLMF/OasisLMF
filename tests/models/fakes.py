from hypothesis.strategies import text

from oasislmf.models import OasisModel


def fake_model(supplier='supplier', model='model', version='version', resources=None):
    return OasisModel(supplier, model, version, resources=resources)
