import pandas as pd
import toml


def create_default_config():
    config = {
        "data_parameters": {
            "num_classes": 5
        },
        "model_parameters": {
            "model_name": "facebook/detr-resnet-101",
            "batch_size": 32,
            "epochs": 30,
            "learning_rate": 6e-4,
        },
        "optimizer_parameters": {
            "eta_min": 1e-6
        },
        "meta_parameters": {
            "path_to_data": "",
            "path_to_metadata": ""
        }
    }
    return config


def save_config(config, filename):
    with open(filename, 'w') as f:
        toml.dump(config, f)


def load_config(filename):
    with open(filename, 'r') as f:
        config = toml.load(f)
    return config