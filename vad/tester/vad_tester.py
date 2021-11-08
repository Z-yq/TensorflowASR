import logging
import os
import tensorflow as tf
import tensorflow_addons
from vad.models.vad_model import CNN_Online_VAD,CNN_Offline_VAD
from vad.tester.base_tester import BaseTester


class VadTester(BaseTester):
    """ Trainer for CTC Models """

    def __init__(self,
                 config,

                 ):
        super(VadTester, self).__init__(config['running_config'])

        self.model_config = config['model_config']
        self.opt_config = config['optimizer_config']
        self.eval_metrics = {
            "acc": tf.keras.metrics.Mean(),
            "f1_score": tf.keras.metrics.Mean(),

        }

    def _eval_step(self, batch):
        x, labels,_ = batch

        pred = self.model.inference(x)
        acc =  tf.metrics.binary_accuracy(labels, pred)
        labels=tf.reshape(labels,[-1,1])
        pred=tf.reshape(pred,[-1,1])
        f1=self.f1_score.update_state(labels,pred)
        self.eval_metrics["acc"].update_state(acc)
        self.eval_metrics["f1_score"].update_state(f1.result)

    def compile(self, ):
        if self.model_config['streaming']:
            self.model = CNN_Online_VAD(self.model_config['dmodel'],name=self.model_config['name'])
        else:
            self.model = CNN_Offline_VAD(self.model_config['dmodel'],name=self.model_config['name'])

        self.model._build()
        self.f1_score=tensorflow_addons.metrics.f_scores.F1Score(2,threshold=self.running_config['voice_thread'])
        self.load_checkpoint()

        logging.info('trainer resume failed')
        self.model.summary(line_length=100)

    def run(self, ):
        self._eval_batches()

    def load_checkpoint(self, ):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        self.steps = int(files[-1].split('_')[-1].replace('.h5', ''))

