from .mictresnet import *
from .resnet3d import *


def get_classification_model(name, **kwargs):
    models = {
        'mictresnet': get_mictresnet,
        'resnet3d': get_resnet3d,
    }
    return models[name.lower()](**kwargs)
