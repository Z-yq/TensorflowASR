import logging
import os

import numpy as np
import tensorflow as tf

from asr.models.chunk_conformer_blocks import ChunkConformer
from asr.tester.base_tester import BaseTester
from utils.text_featurizers import TextFeaturizer
from utils.xer import wer

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class AMTester(BaseTester):
    """ Trainer for CTC Models """

    def __init__(self,
                 config,
                 ):
        super(AMTester, self).__init__(config=config['running_config'])
        self.config = config
        self.speech_config = config['speech_config']
        self.model_config = config['model_config']
        self.opt_config = config['optimizer_config']
        self.phone_featurizer = TextFeaturizer(config['inp_config'])
        self.text_featurizer = TextFeaturizer(config['tar_config'])
        self.eval_metrics = {
            "ser": tf.keras.metrics.Mean(),
            "cer": tf.keras.metrics.Mean(),
        }

    def _eval_step(self, batch):
        features, input_length, phone_labels, phone_label_length, tar_label, tar_label_length = batch

        ctc_output = self.runner.predict(features)
        ctc_output = tf.nn.softmax(ctc_output, -1)
        new_inp_length = tf.ones_like(input_length, tf.int32) * ctc_output.shape[1]
        ctc_decode = tf.keras.backend.ctc_decode(ctc_output, new_inp_length)[0][0]
        ctc_decode = tf.cast(tf.clip_by_value(ctc_decode, 0, self.text_featurizer.num_classes), tf.int32)

        ctc_decode = ctc_decode.numpy()
        phone_labels = tar_label.numpy()

        for i, j in zip(ctc_decode, phone_labels):
            i = np.array(i).flatten().tolist()
            j = j.flatten().tolist()

            while self.phone_featurizer.pad in i:
                i.remove(self.text_featurizer.pad)

            while self.phone_featurizer.pad in j:
                j.remove(self.text_featurizer.pad)

            score, ws, wd, wi = wer(j, i)
            self.ctc_nums[0] += len(j)
            self.ctc_nums[1] += ws
            self.ctc_nums[2] += wi
            self.ctc_nums[3] += wd
            self.eval_metrics["ser"].update_state(0 if i == j else 1)
            self.eval_metrics["cer"].reset_states()
            self.eval_metrics["cer"].update_state(sum(self.ctc_nums[1:]) / (self.ctc_nums[0] + 1e-6))

    def compile(self, ):
        self.runner = ChunkConformer(self.config, self.phone_featurizer.num_classes, self.text_featurizer.num_classes)
        self.runner.compile()
        self.runner.load_weights(
            tf.train.latest_checkpoint(os.path.join(self.running_config['outdir'],'all-ckpt')))

    def run(self, ):

        self._eval_batches()


