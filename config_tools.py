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
            "grad_clipping": 0.1
        },
        "optimizer_parameters": {
            "eta_min": 1e-6,
            "weight_decay": 1e-5,
            "learning_rate": 6e-4,
        },
        "metadata_parameters": {
            "path_to_data": "/workspace/datasets/hack/data/weld_data",
            "path_to_unsplitted_metadata": "/workspace/datasets/hack/data/weld_data/weld_data.csv",
            "path_to_train_metadata": "/workspace/atomic-hack-2.0-solution/metadata/train.json",
            "path_to_test_metadata": "/workspace/atomic-hack-2.0-solution/metadata/test.json",
            "split_ratio": 0.7
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
