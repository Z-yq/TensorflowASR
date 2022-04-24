

import argparse
import logging
import os

import tensorflow as tf

from vad.dataloaders.vad_dataloader import VADDataLoader
from vad.trainer import vad_trainer
from utils.user_config import UserConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpus = tf.config.experimental.list_physical_devices('GPU')
logging.info('valid gpus:%d' % len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class VAD_Trainer():
    def __init__(self, config):
        self.config = config
        self.dg = VADDataLoader(config)

        self.runner = vad_trainer.VADTrainer(config)
        all_train_step = self.dg.get_per_epoch_steps() * self.config['running_config']['num_epochs']
        self.runner.set_total_train_steps(all_train_step)
        self.runner.compile()
        self.dg.batch = self.runner.global_batch_size

    def train(self):
        option = tf.data.Options()
        option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_datasets = tf.data.Dataset.from_generator(self.dg.generator,
                                                        self.dg.return_data_types(),
                                                        self.dg.return_data_shape(),
                                                        args=(True,)).with_options(option)
        eval_datasets = tf.data.Dataset.from_generator(self.dg.generator,
                                                       self.dg.return_data_types(),
                                                       self.dg.return_data_shape(),
                                                       args=(False,)).with_options(option)
        self.runner.set_datasets(train_datasets, eval_datasets)
        logging.warning('Training Start, first 5 steps will be slow........')
        while 1:
            self.runner.fit(epoch=self.dg.epochs)
            if self.runner._finished():
                self.runner.save_checkpoint()
                logging.info('Finish training!')
                break


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_config', type=str, default='./vad/configs/data.yml', help='the am data config path')
    parse.add_argument('--model_config', type=str, default='./vad/configs/model.yml',
                       help='the am model config path')
    args = parse.parse_args()

    config = UserConfig(args.data_config, args.model_config)
    train = VAD_Trainer(config)
    train.train()



