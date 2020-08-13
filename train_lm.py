from utils.user_config import UserConfig
from LMmodel.trm_lm import LM
from trainer import lm_runners
from dataloaders.lm_dataloader import LM_DataLoader
import tensorflow as tf
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class LM_Trainer():
    def __init__(self,config):
        self.config = config
        self.dg = LM_DataLoader(config)
        lm=LM(config)
        lm.load_model()
        self.model = lm.model
        self.optimizer = tf.keras.optimizers.Adamax(**config['optimizer_config'])
        self.runner = lm_runners.LMTrainer(self.config['running_config'],one2one=self.model.one2one)
        self.runner.set_total_train_steps(self.dg.get_per_epoch_steps() * self.config['running_config']['num_epochs'])
        self.runner.compile(self.model,self.optimizer)
    def make_train_batch_data(self):
        batches=[]
        for _ in range(self.config['running_config']['train_steps_per_batches']):
            x,y,feature=self.dg.generate()
            batches.append(( x,y,feature))

        return batches
    def make_eval_batch_data(self):
        batches = []
        for _ in range(self.config['running_config']['eval_steps_per_batches']):
            x, y, feature= self.dg.generate(train=False)
            batches.append((x,y,feature))

        return batches
    def train(self):
        while 1:
            train_batches=self.make_train_batch_data()
            eval_batches=self.make_eval_batch_data()
        # self.runner.set_train_data_loader(batches)
            self.runner.fit(train_batches,eval_batches,epoch=self.dg.epochs)

if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_config', str, required=True, help='the am data config path')
    parse.add_argument('--model_config', str, required=True, help='the am model config path')
    args = parse.parse_args()

    config = UserConfig(args.data_config,args.model_config)
    train=LM_Trainer(config)
    train.train()
