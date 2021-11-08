import argparse
import logging
import os

import tensorflow as tf

from asr.dataloaders.am_dataloader import AM_DataLoader
from asr.trainer import ctc_runners
from utils.user_config import UserConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
    def __init__(self, config):
        self.config = config
        self.dg = AM_DataLoader(config)

        self.runner = ctc_runners.CTCTrainer(config)

        if self.dg.augment.available():
            factor = 2
        else:
            factor = 1
        all_train_step = self.dg.get_per_epoch_steps() * self.config['running_config']['num_epochs'] * factor

        self.runner.set_total_train_steps(all_train_step)
        self.runner.compile()
        self.dg.batch = self.runner.global_batch_size

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


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_config', type=str, default='./configs/data.yml', help='the am data config path')
    parse.add_argument('--model_config', type=str, default='./configs/model.yml',
                       help='the am model config path')
    args = parse.parse_args()

    config = UserConfig(args.data_config, args.model_config)
    train = AM_Trainer(config)
    train.train()
