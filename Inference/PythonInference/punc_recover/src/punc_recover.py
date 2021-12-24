import logging
import os

import numpy as np
import onnxruntime
from utils.text_featurizers import TextFeaturizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Punc():
    def __init__(self, config,):
        self.running_config = config['running_config']
        self.model_config = config['model_config']
        self.vocab_featurizer = TextFeaturizer(config['punc_vocab'])
        self.bd_featurizer = TextFeaturizer(config['punc_biaodian'])
        self.compile()

    def get_angles(self,pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    def get_pos_encoding(self):

        angle_rads = self.get_angles(np.arange(self.model_config['pe_input'])[:, np.newaxis],
                                np.arange(self.model_config['d_model'])[np.newaxis, :],
                                self.model_config['d_model'])

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        self.pos_encode_inputs= np.array(pos_encoding, 'float32')

    def creat_mask(self,seq):
        seq_pad = np.array(seq == 0, 'float32')
        return seq_pad[:, np.newaxis, np.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def compile(self):
        self.model =onnxruntime.InferenceSession('./punc_recover/models/punc.onnx')
        self.get_pos_encoding()

    def punc_recover(self, txt):
        x = [self.vocab_featurizer.startid()] + self.vocab_featurizer.extract(txt) + [self.vocab_featurizer.endid()]
        x = np.array([x], 'int32')
        mask = self.creat_mask(x)
        model_inputs=self.model.get_inputs()
        data = {model_inputs[0].name: x,
                model_inputs[1].name: mask,
                model_inputs[2].name: self.pos_encode_inputs, }

        pred = self.model.run([self.model.get_outputs()[0].name], input_feed=data)[0]
        pred = pred[0,1:-1]
        new_txt = []
        for t, b in zip(txt, pred):
            new_txt.append(t)
            if b.argmax() > 1 and b.max() >= 0.65:
                new_txt.append(self.bd_featurizer.vocab_array[b.argmax()])
        return new_txt



