from .ucf101 import UCF101Classification

datasets = {
    'ucf101': UCF101Classification,
}


def get_classification_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
