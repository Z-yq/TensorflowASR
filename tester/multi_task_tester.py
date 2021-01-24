import tensorflow as tf
from tester.base_tester import BaseTester
from utils.xer import wer

class MultiTaskTester(BaseTester):
    """ Trainer for CTC Models """

    def __init__(self,
                 config,

                 text3_feature,
                 ):
        super(MultiTaskTester, self).__init__(config)
        self.text3_featurizer = text3_feature
        self.p_cer_s = 0
        self.p_cer_d = 0
        self.p_cer_i = 0
        self.eval_metrics = {
            "greed_ser": tf.keras.metrics.Mean(),
            "greed_cer": tf.keras.metrics.Mean(),
        }
    def _eval_step(self, batch):

        x, input_length, py_label = batch

        pred_decode = self.model.recognize_pb(x ,input_length)


        pred_decode = tf.clip_by_value(pred_decode, 0, self.text3_featurizer.num_classes)

        for i, j in zip(pred_decode, py_label):
            i = i.numpy().flatten().tolist()
            j = j.flatten().tolist()

            while 0 in i:
                i.remove(0)

            while self.text3_featurizer.pad in j:
                j.remove(self.text3_featurizer.pad)

            score, ws, wd, wi = wer(j, i)
            self.p_cer_s += ws
            self.p_cer_d += wd
            self.p_cer_i += wi
            self.eval_metrics["greed_ser"].update_state(0 if i == j else 1)
            self.eval_metrics["greed_cer"].update_state(score)



    def compile(self, model: tf.keras.Model):
        self.model = model
        self.model.summary(line_length=100)
        self.model_type = self.model.name


    def run(self, eval_dataset):
        self._eval_batches(eval_dataset)
    def _print_eval_metrics(self, progbar):
        result_dict = {}
        for key, value in self.eval_metrics.items():
            result_dict[f"{key}"] = str(value.result().numpy())
        result_dict['del']=self.p_cer_d
        result_dict['ins']=self.p_cer_i
        result_dict['sub']=self.p_cer_s

        progbar.set_postfix(result_dict)