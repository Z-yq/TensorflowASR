
import yaml
from collections import UserDict

from utils.tools import preprocess_paths, append_default_keys_dict, check_key_in_dict


def load_yaml(path):
    with open(preprocess_paths(path), "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


class UserConfig(UserDict):
    """ User config class for training, testing or infering """

    def __init__(self, common,model):
        # assert default, "Default dict for config must be set"
        custom = load_yaml(common)
        model_config=load_yaml(model)
        custom.update(model_config)
        super(UserConfig, self).__init__(custom)


    def __missing__(self, key):
        return None
