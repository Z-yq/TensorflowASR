import logging
import os

import numpy as np

from LMmodel import punc_transformer
from LMmodel.tf2_trm import Transformer, tf
from utils.text_featurizers import TextFeaturizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class LM():
    def __init__(self, config, punc_config=None):
        self.config = config
        self.am_featurizer = TextFeaturizer(config['am_token'])
        self.lm_featurizer = TextFeaturizer(config['lm_token'])
        self.model_config = self.config['model_config']
        self.model_config.update(
            {'input_vocab_size': self.am_featurizer.num_classes, 'target_vocab_size': self.lm_featurizer.num_classes})
        self.punc_config = punc_config
        if punc_config:
            self.punc_vocab_featurizer = TextFeaturizer(punc_config['punc_vocab'])
            self.punc_bd_featurizer = TextFeaturizer(punc_config['punc_biaodian'])
            self.punc_model_config = self.punc_config['model_config']
            self.punc_model_config.update({'input_vocab_size': self.punc_vocab_featurizer.num_classes,
                                           'bd_vocab_size': self.punc_bd_featurizer.num_classes})

    def load_model(self, training=True):
        self.model = Transformer(**self.model_config)
        if self.punc_config is not None:
            self.punc_model = punc_transformer.Transformer(**self.punc_model_config)
        if not training:
            self.model._build()

            if self.punc_config is not None:
                self.punc_model._build()
            self.load_checkpoint()

        self.model.start_id = self.lm_featurizer.start
        self.model.end_id = self.lm_featurizer.stop

    def convert_to_pb(self, export_path):
        import tensorflow as tf
        self.model.inference(np.ones([1, 10], 'int32'))

        concrete_func = self.model.inference.get_concrete_function()
        tf.saved_model.save(self.model, export_path, signatures=concrete_func)

    def load_checkpoint(self, ):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(self.config['running_config']["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        if self.punc_config is not None:
            self.checkpoint_dir = os.path.join(self.punc_config['running_config']["outdir"], "checkpoints")
            files = os.listdir(self.checkpoint_dir)
            files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
            self.punc_model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))

    def encode(self, word, token):
        x = [token.start]
        for i in word:
            x.append(token.token_to_index[i])
        x.append(token.stop)
        return np.array(x)[np.newaxis, :]

    def decode(self, out, token):
        de = []
        for i in out[1:]:
            de.append(token.index_to_token[i])
        return de

    def predict(self, pins):
        x = self.encode(pins, self.am_featurizer)
        result = self.model.inference(x)
        return result

    def creat_mask(self, seq):
        seq_pad = tf.cast(tf.equal(seq, 0), tf.float32)
        return seq_pad[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


    def only_chinese(self, word):
        n = ''
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                n += ch
        return n

    def punc_predict(self, txt):
        chinese = self.only_chinese(txt)
        x=np.array(self.encode(chinese, self.punc_vocab_featurizer),'int32')
        mask = self.creat_mask(x)
        result = self.punc_model.inference(x, mask)
        decoded = self.punc_decoded(chinese,result[0].numpy())
        value = self.iextract(decoded, txt)
        return value

    def iextract(self, decoded, input_strs):


        idx = 0
        inp = list(input_strs)
        for n in decoded:
            idx_ = inp.index(n[0], idx)
            inp[idx_] = ''.join(n)
            idx = idx_ + 1

        return inp


    def punc_decoded(self,chinese,bd_out):
        de = []
        for i in range(1, len(chinese) + 1):
            now = [chinese[i - 1]]

            if bd_out[i].argmax(-1) > 1 and bd_out[i].max() >= 0.8:
                result = bd_out[i].argmax(-1)
                now.append(self.punc_bd_featurizer.vocab_array[result])
            de.append(now)
        return de

