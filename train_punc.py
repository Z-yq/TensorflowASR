from utils.user_config import UserConfig
from punc_recover.trainer import punc_trainer
from punc_recover.dataloaders.punc_dataloader import Punc_DataLoader
import tensorflow as tf
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
class Punc_Trainer():
    def __init__(self,config,):
        self.dg = Punc_DataLoader(config)
        self.runner = punc_trainer.PuncTrainer(config)
        self.runner.set_total_train_steps(self.dg.get_per_epoch_steps() * config['running_config']['num_epochs'])
        self.runner.compile()

    def train(self):
        while 1:
            train_datasets = tf.data.Dataset.from_generator(self.dg.generator,
                                                            self.dg.return_data_types(),
                                                            self.dg.return_data_shape(),
                                                            args=(True,))
            eval_datasets = tf.data.Dataset.from_generator(self.dg.generator,
                                                           self.dg.return_data_types(),
                                                           self.dg.return_data_shape(),
                                                           args=(False,))
            self.runner.set_datasets(train_datasets, eval_datasets)

            self.runner.fit(epoch=self.dg.epochs)
            if self.runner._finished():
                self.runner.save_checkpoint()
                logging.info('Finish training!')
                break
if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_config', type=str, default='./punc_recover/configs/data.yml',
                       help='the lm data config path')
    parse.add_argument('--model_config', type=str, default='./punc_recover/configs/punc_settings.yml',
                       help='the lm model config path')
    args = parse.parse_args()
    punc_config = UserConfig(args.punc_config,args.punc_config)
    train=Punc_Trainer(punc_config)
    train.train()
