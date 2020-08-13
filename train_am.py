from AMmodel.model import AM
from dataloaders.am_dataloader import AM_DataLoader
from utils.user_config import UserConfig
from trainer import ctc_runners,transducer_runners,las_runners
import tensorflow as tf
import numpy as np
import argparse
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
class AM_Trainer():
    def __init__(self,config):
        self.config=config['learning_config']

        self.am = AM(config)
        self.am.load_model(training=True)
        self.dg = AM_DataLoader(config)
        if self.am.config['decoder_config']['model_type']=='CTC':
            self.runner = ctc_runners.CTCTrainer(self.dg.speech_featurizer,self.dg.text_featurizer,self.config['running_config'])
        elif self.am.config['decoder_config']['model_type']=='LAS':
            self.runner=las_runners.LASTrainer(self.dg.speech_featurizer,self.dg.text_featurizer,self.config['running_config'])

        else:
            self.runner = transducer_runners.TransducerTrainer(self.config['running_config'],self.dg.text_featurizer )
        self.STT = self.am.model

        # self.opt = tf.keras.optimizers.Adamax(learning_rate=config['optimizer_config']['learning_rate'],
        #                                       beta_1=config['optimizer_config']['beta_1'],
        #                                       beta_2=config['optimizer_config']['beta_2'],
        #                                       epsilon=config['optimizer_config']['epsilon'])
        self.opt = tf.keras.optimizers.Adamax(**config['optimizer_config'])
        self.runner.set_total_train_steps(self.dg.get_per_epoch_steps() * self.config['running_config']['num_epochs'])
        self.runner.compile(self.STT,self.opt)

    def recevie_data(self,r):

        data = r.rpop(self.config['data_name'])
        data = eval(data)
        trains=[]
        for key in self.config['data_dict_key']:
            x = data[key]
            dtype = data['%s_dtype'%key]
            shape = data['%s_shape'%key]
            x = np.frombuffer(x, dtype)
            x = x.reshape(shape)
            trains.append(x)
        return trains


    def make_train_batch_data(self):
        batches=[]
        for _ in range(self.config['running_config']['train_steps_per_batches']):
            features, wavs, input_length, labels, label_length=self.dg.generator()
            batches.append(( features, wavs, input_length, labels, label_length))
            if self.dg.augment.available():
                features, wavs, input_length, labels, label_length=self.dg.augment_data(wavs, labels, label_length)
                batches.append((features, wavs, input_length, labels, label_length))
        return batches
    def make_eval_batch_data(self):
        batches = []
        for _ in range(self.config['running_config']['eval_steps_per_batches']):
            features, wavs, input_length, labels, label_length = self.dg.generator(train=False)
            batches.append((features, wavs, input_length, labels, label_length))

        return batches

    def train(self):
        while 1:
            train_batches=self.make_train_batch_data()
            eval_batches=self.make_eval_batch_data()
            self.runner.fit(train_batches,eval_batches,epoch=self.dg.epochs)
            if self.runner._finished():
                self.runner.save_checkpoint()
                logging.info('Finish training!')
if __name__ == '__main__':

    parse=argparse.ArgumentParser()
    parse.add_argument('--data_config',type=str,required=True,help='the am data config path')
    parse.add_argument('--model_config',type=str,required=True,help='the am model config path')
    args=parse.parse_args()

    config=UserConfig(args.data_config,args.model_config)
    train=AM_Trainer(config)
    train.train()
