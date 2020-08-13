from LMmodel.tf2_trm import Transformer,create_masks
import tensorflow as tf
import os
import logging
import numpy as np
from utils.text_featurizers import TextFeaturizer
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
class LM():
    def __init__(self,config):
        self.config=config
        self.vocab_featurizer = TextFeaturizer(config['lm_vocab'])
        self.word_featurizer = TextFeaturizer(config['lm_word'])
        self.model_config=self.config['model_config']
        self.model_config.update({'input_vocab_size':self.vocab_featurizer.num_classes,'target_vocab_size':self.word_featurizer.num_classes})

    def load_model(self):
        self.model = Transformer(**self.model_config)
        self.model._build()
        try:
            self.load_checkpoint()
        except:
            logging.info('lm loading model failed.')
        self.model.start_id=self.word_featurizer.start
        self.model.end_id=self.word_featurizer.stop
    def load_checkpoint(self, ):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(self.config['running_config']["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
    def encode(self,word,token):
        x=[token.start]
        for i in word:
            x.append(token.token_to_index[i])
        x.append(token.stop)
        return np.array(x)[np.newaxis,:]
    def decode(self,out,token):
        de=[]
        for i in out[1:]:
            de.append(token.index_to_token[i])
        return de


    def predict(self,pins,return_string_list=True):
        x=self.encode(pins,self.vocab_featurizer)
        result=self.model.recognize(x)
        if return_string_list:
            result=self.decode(result,self.word_featurizer)
        return result



