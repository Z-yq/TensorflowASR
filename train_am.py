from AMmodel.model import AM
from dataloaders.am_dataloader import AM_DataLoader
from dataloaders.multi_task_dataloader import MultiTask_DataLoader
from utils.user_config import UserConfig
from trainer import ctc_runners,transducer_runners,las_runners,multi_runners
import tensorflow as tf
import numpy as np
import argparse
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
gpus = tf.config.experimental.list_physical_devices('GPU')
logging.info('valid gpus:%d' % len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
class AM_Trainer():
    def __init__(self,config):
        self.config=config['learning_config']

        self.am = AM(config)
        self.am.load_model(training=True)
        if self.am.model_type!='MultiTask':
            print('am_dataloder')
            self.dg = AM_DataLoader(config)
        else:
            self.dg=MultiTask_DataLoader(config)
        self.dg.speech_config['reduction_factor']=self.am.model.time_reduction_factor
        self.dg.load_state(self.config['running_config']['outdir'])
        if self.am.model_type=='CTC':
            self.runner = ctc_runners.CTCTrainer(self.dg.speech_featurizer,self.dg.text_featurizer,self.config['running_config'])
        elif self.am.model_type=='LAS':
            self.runner=las_runners.LASTrainer(self.dg.speech_featurizer,self.dg.text_featurizer,self.config['running_config'])
            self.dg.LAS=True
        elif self.am.model_type == 'MultiTask':
            self.runner = multi_runners.MultiTaskCTCTrainer(self.dg.speech_featurizer,
                                                 self.config['running_config'])


        else:

            self.runner = transducer_runners.TransducerTrainer(self.dg.speech_featurizer,self.dg.text_featurizer,self.config['running_config'] )
        self.STT = self.am.model

        if self.dg.augment.available():
            factor=2
        else:
            factor=1
        all_train_step=self.dg.get_per_epoch_steps() * self.config['running_config']['num_epochs']*factor
        lr=CustomSchedule(config['model_config']['dmodel'],warmup_steps=int(all_train_step*0.1))
        config['optimizer_config']['learning_rate']=lr
        self.opt = tf.keras.optimizers.Adamax(**config['optimizer_config'])
        self.runner.set_total_train_steps(all_train_step)
        self.runner.compile(self.STT,self.opt)
        self.dg.batch=self.runner.global_batch_size


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


    def train(self):

        train_datasets = tf.data.Dataset.from_generator(self.dg.generator,
                                                       self.dg.return_data_types(),
                                                        self.dg.return_data_shape(),
                                                        args=(True,))
        eval_datasets = tf.data.Dataset.from_generator(self.dg.generator,
                                                       self.dg.return_data_types(),
                                                       self.dg.return_data_shape(),
                                                       args=(False,))
        self.runner.set_datasets(train_datasets, eval_datasets)

        while 1:
            self.runner.fit(epoch=self.dg.epochs)
            if self.runner._finished():
                self.runner.save_checkpoint()
                logging.info('Finish training!')
                break
            if self.runner.steps%self.config['running_config']['save_interval_steps']==0:
                self.dg.save_state(self.config['running_config']['outdir'])
if __name__ == '__main__':

    parse=argparse.ArgumentParser()
    parse.add_argument('--data_config', type=str, default='./configs/am_data.yml', help='the am data config path')
    parse.add_argument('--model_config', type=str, default='./configs/deepspeech2.yml', help='the am model config path')
    args=parse.parse_args()

    config=UserConfig(args.data_config,args.model_config)
    train=AM_Trainer(config)
    train.train()
