from E2EModel.e2e_model import E2EModel
from dataloaders.e2e_dataloader import E2E_DataLoader
from utils.user_config import UserConfig
from trainer import e2e_runners
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
class E2E_Trainer():
    def __init__(self,config):

        self.config=config['learning_config']

        self.e2e = E2EModel(config,config)
        self.e2e.compile_model(True)
        self.dg=E2E_DataLoader(config,training=True)
        self.dg.speech_config['reduction_factor']=self.e2e.am.model.time_reduction_factor
        self.dg.load_state(self.config['running_config']['outdir'])

        self.runner = e2e_runners.E2ETrainer(self.config['running_config'])

        if self.dg.augment.available():
            factor=2
        else:
            factor=1
        self.am_opt = tf.keras.optimizers.Adamax(**config['optimizer_config'])
        self.lm_opt = tf.keras.optimizers.Adamax(**config['optimizer_config'])
        self.runner.set_total_train_steps(self.dg.get_per_epoch_steps() * self.config['running_config']['num_epochs']*factor)
        self.runner.compile(self.e2e.am.model,self.e2e.lm.model,self.am_opt,self.lm_opt)
        self.dg.batch=self.runner.global_batch_size

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
    parse.add_argument('--data_config', type=str, default='./configs/e2e_data.yml', help='the am data config path')
    parse.add_argument('--model_config', type=str, default='./configs/e2e_model.yml', help='the am model config path')
    args=parse.parse_args()

    config=UserConfig(args.data_config,args.model_config)
    train=E2E_Trainer(config)
    train.train()
