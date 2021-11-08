
import abc
import os
from tqdm import tqdm
from colorama import Fore

import numpy as np
import tensorflow as tf

from utils.text_featurizers import TextFeaturizer
from utils.tools import preprocess_paths, get_num_batches, bytes_to_string
from utils.metrics import ErrorRate, wer, cer






class BaseTester():
    """ Customized tester module for all models
    This tester model will write results to test.tsv file in outdir
    After writing finished, it will calculate testing metrics
    """

    def __init__(self,
                 config: dict,

                 ):
        """
        Args:
            config: the 'learning_config' part in YAML config file
            saved_path: path to exported weights or model
            from_weights: choose to load from weights or from whole model
        """
        super().__init__()
        self.running_config=config

        self.output_file_path = os.path.join(self.running_config["outdir"], "test.tsv")
        self.eval_metrics = {
            "greed_ser": tf.keras.metrics.Mean(),
            "greed_cer": tf.keras.metrics.Mean()
        }
        self.ctc_nums = [0, 0, 0, 0]  # n,s,i,d
        self.translator_nums = [0, 0, 0, 0]  # n,s,i,d
        self.steps=0
        self.all_steps=0
    def set_all_steps(self,all_steps):
        self.all_steps=all_steps
    def set_progbar(self,total_steps):
        self.eval_progbar = tqdm(
            initial=0, total=total_steps, unit="batch",
            position=0, leave=True,
            bar_format="{desc} |%s{bar:20}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
            desc=f"[Eval] [Step {self.steps}]"
        )
    # -------------------------------- RUNNING -------------------------------------

    def compile(self,):
        raise NotImplementedError
    def run(self,):
        raise NotImplementedError
    def set_datasets(self,evaldataset):
        self.eval_datasets=evaldataset
    def _eval_batches(self, ):
        for batch in self.eval_datasets:

            self._eval_step(batch)
            self.eval_progbar.update(1)
            self._print_eval_metrics(self.eval_progbar)
            self.steps += 1
            if self.finished():
                break
    def _eval_step(self, batch):
        """
        One testing step
        Args:
            batch: a step fed from test dataset

        Returns:
            (file_paths, groundtruth, greedy, beamsearch, beamsearch_lm) each has shape [B]
        """
        raise NotImplementedError
    def _print_eval_metrics(self, progbar):
        result_dict = {}
        for key, value in self.eval_metrics.items():
            result_dict[f"{key}"] = str(value.result().numpy())
        result_dict['phone_s_i_d']='{}_{}_{}'.format(self.phone_nums[1],self.phone_nums[2],self.phone_nums[3])
        result_dict['trans_s_i_d']='{}_{}_{}'.format(self.translator_nums[1],self.translator_nums[2],self.translator_nums[3])
        progbar.set_postfix(result_dict)


    def finished(self):
        if self.steps>=self.all_steps:
            return True
        return False
