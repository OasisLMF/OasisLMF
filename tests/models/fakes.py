from oasislmf.model_preparation.manager import OasisManager as om


def fake_model(supplier='supplier', model='model', version='version', resources=None):
    return om().create_model(supplier, model, version, resources=resources)
