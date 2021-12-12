import logging
import os

import numpy as np
import pypinyin

from punc_recover.models.punc_transformer import PuncTransformer,tf

from utils.text_featurizers import TextFeaturizer
from utils.user_config import UserConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Punc():
    def __init__(self, config,):
        self.running_config = config['running_config']
        self.model_config = config['model_config']
        self.vocab_featurizer = TextFeaturizer(config['punc_vocab'])
        self.bd_featurizer = TextFeaturizer(config['punc_biaodian'])
        self.compile()

    def creat_mask(self, seq):
        seq_pad = tf.cast(tf.equal(seq, 0), tf.float32)
        return seq_pad[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    def compile(self):
        self.model = PuncTransformer(num_layers=self.model_config['num_layers'],
                                     d_model=self.model_config['d_model'],
                                     enc_embedding_dim=self.model_config['enc_embedding_dim'],
                                     num_heads=self.model_config['num_heads'],
                                     dff=self.model_config['dff'],
                                     input_vocab_size=self.vocab_featurizer.num_classes,
                                     bd_vocab_size=self.bd_featurizer.num_classes,
                                     pe_input=self.model_config['pe_input'],
                                     rate=self.model_config['rate'])
        self.model._build()
        self.load_checkpoint()
        self.model.summary(line_length=100)

    def load_checkpoint(self, ):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))

    def punc_recover(self,txt):
        x=[self.vocab_featurizer.startid()]+self.vocab_featurizer.extract(txt)+[self.vocab_featurizer.endid()]
        x=tf.constant([x],tf.int32)
        mask=self.creat_mask(x)
        pred=self.model.inference(x,mask)[0]
        pred=pred.numpy()
        pred=pred[1:]
        new_txt=[]
        for t,b in zip(txt,pred):
            new_txt.append(t)
            if b.argmax()>1 and b.max()>=0.8:
                new_txt.append(self.bd_featurizer.vocab_array[b.argmax()])
        return new_txt






if __name__ == '__main__':

    # USE CPU:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # USE one GPU:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # limit cpu to 1 core:
    # import tensorflow as tf
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)

    config = UserConfig(r'./punc_recover/configs/data.yml', r'./punc_recover/configs/punc_settings.yml')
    punc = Punc(config)

    # first inference will be slow,it is normal
    print(punc.punc_recover('谢谢你的爱'))
