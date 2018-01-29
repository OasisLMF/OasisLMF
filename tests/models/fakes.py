from hypothesis.strategies import text

from oasislmf.models import OasisModel


def fake_model(supplier=None, model=None, version=None, resources=None):
    supplier = supplier if supplier is not None else text().example()
    model = model if model is not None else text().example()
    version = version if version is not None else text().example()

    return OasisModel(supplier, model, version, resources=resources)
