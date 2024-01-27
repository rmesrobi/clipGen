import os
import configparser

def parse_config():
    config = configparser.ConfigParser()
    config.read('config.conf')
    config_keys = config.sections()
    content = {k: dict(config[k]) for k in config_keys}
    return content