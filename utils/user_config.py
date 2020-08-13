# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
