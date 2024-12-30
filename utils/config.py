# Copyright (c) CUBOX, Inc. and its affiliates.

import yaml

def load_config(config_path):
    """Loads the configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def update_config(config_path, updates):
    """Updates specific keys in the configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config.update(updates)
    with open(config_path, 'w') as file:
        yaml.dump(config, file)