import argparse
import logging
import os

import tensorflow as tf

from asr.dataloaders.am_dataloader import AM_DataLoader
from asr.dataloaders.chunk_dataloader import Chunk_DataLoader
from asr.trainer import ctc_runners
from asr.models.chunk_conformer_blocks import ChunkConformer
from utils.user_config import UserConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpus = tf.config.experimental.list_physical_devices('GPU')
logging.info('valid gpus:%d' % len(gpus))
gpu_nums=len(gpus)
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
        if self.config['model_config']['name']=='ChunkConformer':
            self.mode=0
        else:
            self.mode=1
        if self.mode:
            self.dg = AM_DataLoader(config)

            self.runner = ctc_runners.CTCTrainer(config)

            if self.dg.augment.available():
                factor = 2
            else:
                factor = 1

            all_train_step = self.dg.get_per_epoch_steps() * self.config['running_config']['num_epochs'] * factor
            if gpu_nums > 0:
                all_train_step//=len(gpus)
            self.runner.set_total_train_steps(all_train_step)
            self.runner.compile()
            self.dg.batch = self.runner.global_batch_size
        else:
            from utils.text_featurizers import TextFeaturizer
            self.speech_config = config["speech_config"]
            self.model_config = config["model_config"]
            self.opt_config = config["optimizer_config"]
            self.running_config = config['running_config']
            self.phone_featurizer = TextFeaturizer(config["inp_config"])

            self.text_featurizer = TextFeaturizer(config["tar_config"])
            self.train_dg = Chunk_DataLoader(config)
            self.test_dg = Chunk_DataLoader(config, False)
            self.strategy = tf.distribute.MirroredStrategy()

            with self.strategy.scope():
                self.runner = ChunkConformer(self.config, self.phone_featurizer.num_classes,
                                             self.text_featurizer.num_classes)
                self.runner.compile(optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.opt_config['lr'], beta_1=self.opt_config['beta1'],
                    beta_2=self.opt_config['beta2'], epsilon=self.opt_config['epsilon']
                ))
                try:
                    self.runner.load_weights(
                    tf.train.latest_checkpoint(os.path.join(self.running_config['outdir'], 'all-ckpt')))
                except:
                    logging.info('load ckpt failed....')
            if gpu_nums>0:
                self.train_dg.batch = self.train_dg.batch *gpu_nums
                self.test_dg.batch = self.test_dg.batch *gpu_nums


    def chunk_train(self):
        def decay(epoch):
            if epoch < 50:
                return 1e-4
            elif epoch >= 50 and epoch < 100:
                return 5e-5
            else:
                return 2e-5

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.config["running_config"]['outdir'], 'all-ckpt', 'ckpt-{epoch}'),
                save_weights_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.running_config['outdir'], 'tensorboard')),
            tf.keras.callbacks.LearningRateScheduler(decay),
        ]
        if len(gpus)>1:
            logging.warning('The training is currently being performed on a single machine using multiple cards.'
                            'Initialization will take approximately 10-20 minutes, please be patient. '
                            'Alternatively, you can set up single-card training to quickly debug issues.')
        self.runner.fit(x=self.train_dg, epochs=self.config["running_config"]["num_epochs"], callbacks=callbacks,
                        shuffle=False, workers=10, use_multiprocessing=True, validation_data=self.test_dg)
        logging.info('Finish training!')
    def ohters_train(self):
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
    def train(self):
        if self.mode:
            self.ohters_train()
        else:
            self.chunk_train()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_config', type=str, default='./asr/configs/chunk_data.yml', help='the am data config path')
    parse.add_argument('--model_config', type=str, default='./asr/configs/chunk_conformerS.yml',
                       help='the am model config path')
    args = parse.parse_args()

    config = UserConfig(args.data_config, args.model_config)
    train = AM_Trainer(config)
    train.train()
