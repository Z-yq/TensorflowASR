from utils.user_config import UserConfig
from LMmodel.trm_lm import LM
from tester import lm_tester
from dataloaders.lm_dataloader import LM_DataLoader
import argparse
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
class LM_Tester():
    def __init__(self,config):
        self.config = config
        self.dg = LM_DataLoader(config,training=False)
        lm = LM(config)
        lm.load_model()
        self.model = lm.model

        self.runner = lm_tester.LMTester(self.config['running_config'],self.config['model_config']['one2one'])
        self.runner.set_progbar(self.dg.eval_per_epoch_steps())
        self.runner.compile(self.model)
    def make_eval_batch_data(self):
        batches = []
        for _ in range(self.config['running_config']['eval_steps_per_batches']):
            x,y= self.dg.eval_generate()

            batches.append((x,y))

        return batches

    def test(self):
        while 1:
            eval_batches=self.make_eval_batch_data()
            # print('now',self.dg.offset)
            self.runner.run(eval_batches)
            if self.dg.offset>len(self.dg.test_texts)-1:
                break
if __name__ == '__main__':

    parse=argparse.ArgumentParser()
    parse.add_argument('--data_config',type=str,required=True,help='the lm data config path')
    parse.add_argument('--model_config',type=str,required=True,help='the lm model config path')
    args=parse.parse_args()

    config=UserConfig(args.data_config,args.model_config)
    tester=LM_Tester(config)
    tester.test()
