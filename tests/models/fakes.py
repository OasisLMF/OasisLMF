from oasislmf.exposures.manager import OasisExposuresManager


def fake_model(supplier='supplier', model='model', version='version', resources=None):
    return OasisExposuresManager().create_model(supplier, model, version, resources=resources)
