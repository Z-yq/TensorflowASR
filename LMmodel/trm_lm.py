from LMmodel.tf2_trm import Transformer
import logging
import numpy as np
from utils.text_featurizers import TextFeaturizer
import os
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
class LM():
    def __init__(self,config):
        self.config=config
        self.am_featurizer = TextFeaturizer(config['am_token'])
        self.lm_featurizer = TextFeaturizer(config['lm_token'])
        self.model_config=self.config['model_config']
        self.model_config.update({'input_vocab_size':self.am_featurizer.num_classes,'target_vocab_size':self.lm_featurizer.num_classes})

    def load_model(self,training=True):
        self.model = Transformer(**self.model_config)


        if not training:
            self.model._build()
            self.load_checkpoint()

        self.model.start_id=self.lm_featurizer.start
        self.model.end_id=self.lm_featurizer.stop

    def convert_to_pb(self, export_path):
        import tensorflow as tf
        self.model.inference(np.ones([1,10],'int32'))

        concrete_func = self.model.inference.get_concrete_function()
        tf.saved_model.save(self.model, export_path, signatures=concrete_func)
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


    def predict(self,pins):
        x=self.encode(pins,self.am_featurizer)
        result=self.model.inference(x)
        return result



