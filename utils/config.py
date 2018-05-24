import os

import yaml
from dotmap import DotMap

CONFIG_FILE = '{}/config.yml'.format(os.getcwd())


def get_config():
    """
    This function creates a DotMap configuration object from the config.yml file.
    :return: DotMap object of the configuration
    """
    with open(CONFIG_FILE) as config_file:
        config_yaml = yaml.load(config_file)
    return DotMap(config_yaml, _dynamic=False)


config = get_config()
