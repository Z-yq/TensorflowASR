
import codecs

import os
from utils.normalize import NSWNormalizer
import pypinyin as ppy
import logging

def preprocess_paths(paths):
    if isinstance(paths, list):
        return [os.path.abspath(os.path.expanduser(path)) for path in paths]
    return os.path.abspath(os.path.expanduser(paths)) if paths else None

class TextFeaturizer:

    def __init__(self, config: dict,show=False):
        self.config = config
        self.normlizer=NSWNormalizer
        self.config["vocabulary"] = preprocess_paths(self.config["vocabulary"])
        self.config["spker"] = preprocess_paths(self.config["spker"])
        self.config["maplist"] = preprocess_paths(self.config["maplist"])
        with open(self.config['spker']) as f:
            spks=f.readlines()
        self.spker_map={}
        for idx,spk in enumerate(spks):
            self.spker_map[spk.strip()]=idx
        with open(self.config["maplist"], encoding='utf-8') as f:
            data = f.readlines()

        map_dict={}
        for line in data:
            line = line.strip()
            line = line.replace('[', '').replace(']', '')
            content = line.split('\t')
            key = content[0]
            value = content[1].split(" ")

            for i in range(1, 6):
                key_ = key[:-1] + str(i)
                value_ = [value[0], value[-1][:-1] + str(i)]
                map_dict[key_] = value_
                map_dict[key_[:-1] + 'r' + key_[-1]] = value_ + ["er"]
        self.map_dict = map_dict

        self.num_classes = 0
        lines = []
        with codecs.open(self.config["vocabulary"], "r", "utf-8") as fin:
            lines.extend(fin.readlines())
        if show:
            logging.info('load token at {}'.format(self.config['vocabulary']))
        self.token_to_index = {}
        self.index_to_token = {}
        self.vocab_array = []

        index = 0
        if self.config["blank_at_zero"]:
            self.blank = 0
            index = 1

        for line in lines:
            line = line.strip()  # Strip the '\n' char
            if line.startswith("#") or not line or line == "\n":
                continue
            self.token_to_index[line] = index
            self.index_to_token[index] = line
            self.vocab_array.append(line)

            index += 1
        self.num_classes = index
        if not self.config["blank_at_zero"]:
            self.blank = index
            self.num_classes += 1
        self.stop=self.endid()
        self.pad=self.blank
        self.stop=-1


    def endid(self):
        return self.token_to_index['<END>']


    def extract(self, text):
        """
        Convert string to a list of integers
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints
        """
        text=self.normlizer(text).normalize()
        pinyins=ppy.pinyin(text,8,neutral_tone_with_five=True)
        pinyins=[i[0] for i in pinyins]

        tokens = []
        for py in pinyins:
            if py[-1]=="5":
                py=py[:-1]+'1'
            if py in self.map_dict:
                tokens += self.map_dict[py]
            else:
                if len(py) > 1 and py !='sil':
                    py = list(py)
                    tokens+=py
                else:
                    tokens += [py]

        feats=[self.token_to_index[token] for token in tokens]
        return feats

    def iextract(self,indx):
        texts=[self.index_to_token[idx] for idx in indx ]
        return texts

