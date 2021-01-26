from utils.user_config import UserConfig
from LMmodel.trm_lm import LM
from trainer import lm_runners
from dataloaders.lm_dataloader import LM_DataLoader
import tensorflow as tf
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
class LM_Trainer():
    def __init__(self,config):
        self.config = config
        self.dg = LM_DataLoader(config)
        lm=LM(config)
        lm.load_model()
        self.model = lm.model

        all_train_step = self.dg.get_per_epoch_steps() * self.config['running_config']['num_epochs']
        lr = CustomSchedule(config['model_config']['d_model'], warmup_steps=int(all_train_step * 0.1))
        config['optimizer_config']['learning_rate'] = lr

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
            if self.runner.steps % self.config['running_config']['save_interval_steps'] == 0:
                self.dg.save_state(self.config['running_config']['outdir'])
if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_config', type=str, default='./configs/lm_data.yml',
                       help='the lm data config path')
    parse.add_argument('--model_config', type=str, default='./configs/transformerO2OE.yml',
                       help='the lm model config path')
    args = parse.parse_args()

    config = UserConfig(args.data_config,args.model_config)
    train=LM_Trainer(config)
    train.train()
