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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
gpus = tf.config.experimental.list_physical_devices('GPU')
logging.info('valid gpus:%d' % len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
class AM_Trainer():
    def __init__(self,config):
        self.config=config['learning_config']

        self.am = AM(config)
        self.am.load_model(training=True)
        if self.am.model_type!='MultiTask':
            self.dg = AM_DataLoader(config)
        else:
            self.dg=MultiTask_DataLoader(config)
        self.dg.speech_config['reduction_factor']=self.am.model.time_reduction_factor
        if self.am.model_type=='CTC':
            self.runner = ctc_runners.CTCTrainer(self.dg.speech_featurizer,self.dg.text_featurizer,self.config['running_config'])
        elif self.am.model_type=='LAS':
            self.runner=las_runners.LASTrainer(self.dg.speech_featurizer,self.dg.text_featurizer,self.config['running_config'])
            self.dg.LAS=True
        elif self.am.model_type == 'MultiTask':
            self.runner = multi_runners.MultiTaskLASTrainer(self.dg.speech_featurizer, self.dg.token4_featurizer,
                                                 self.config['running_config'])


        else:

            self.runner = transducer_runners.TransducerTrainer(self.dg.speech_featurizer,self.dg.text_featurizer,self.config['running_config'] )
        self.STT = self.am.model

        if self.dg.augment.available():
            factor=2
        else:
            factor=1
        self.opt = tf.keras.optimizers.Adamax(**config['optimizer_config'])
        self.runner.set_total_train_steps(self.dg.get_per_epoch_steps() * self.config['running_config']['num_epochs']*factor)
        self.runner.compile(self.STT,self.opt)
        self.dg.batch=self.runner.global_batch_size

    def load_checkpoint(self,config,model):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(config['learning_config']['running_config']["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        self.init_steps= int(files[-1].split('_')[-1].replace('.h5', ''))



    def train(self):
        if self.am.model_type!='MultiTask':
            train_datasets = tf.data.Dataset.from_generator(self.dg.generator,
                                                           self.dg.return_data_types(),
                                                            self.dg.return_data_shape(),
                                                            args=(True,))
            eval_datasets = tf.data.Dataset.from_generator(self.dg.generator,
                                                           self.dg.return_data_types(),
                                                           self.dg.return_data_shape(),
                                                           args=(False,))
            self.runner.set_datasets(train_datasets, eval_datasets)
        else:
            self.runner.set_datasets(self.dg.generator(True), self.dg.generator(False))
        while 1:
            self.runner.fit(epoch=self.dg.epochs)
            if self.runner._finished():
                self.runner.save_checkpoint()
                logging.info('Finish training!')
                break
if __name__ == '__main__':

    parse=argparse.ArgumentParser()
    parse.add_argument('--data_config', type=str, default='./configs/am_data.yml', help='the am data config path')
    parse.add_argument('--model_config', type=str, default='./configs/MultiConformer.yml', help='the am model config path')
    args=parse.parse_args()

    config=UserConfig(args.data_config,args.model_config)
    train=AM_Trainer(config)
    train.train()
