import tensorflow as tf
from tester.base_tester import BaseTester
from utils.xer import wer

class MultiTaskTester(BaseTester):
    """ Trainer for CTC Models """

    def __init__(self,
                 config,

                 text3_feature,
                 text4_feature,
                 ):
        super(MultiTaskTester, self).__init__(config)
        self.text3_featurizer = text3_feature
        self.text4_featurizer = text4_feature

        self.p_cer_s = 0
        self.p_cer_d = 0
        self.p_cer_i = 0
        self.eval_metrics = {
            "greed_ser": tf.keras.metrics.Mean(),
            "greed_cer": tf.keras.metrics.Mean(),
            "phone_greed_ser": tf.keras.metrics.Mean(),
            "phone_greed_cer": tf.keras.metrics.Mean()
        }
    def _eval_step(self, batch):

        x, wavs, input_length, py_label, txt_label = batch
        if self.model.mel_layer is not None:
            final_decode, pred_decode = self.model.recognize_pb(wavs, input_length)
        else:
            final_decode, pred_decode = self.model.recognize_pb(x, input_length)

        pred_decode = tf.clip_by_value(pred_decode, 0, self.text3_featurizer.num_classes)

        for i, j in zip(pred_decode, py_label):
            i = i.numpy().flatten().tolist()
            j = j.flatten().tolist()

            while 0 in i:
                i.remove(self.text3_featurizer.pad)

            while self.text3_featurizer.pad in j:
                j.remove(self.text3_featurizer.pad)

            score, ws, wd, wi = wer(i, j)
            self.p_cer_s += ws
            self.p_cer_d += wd
            self.p_cer_i += wi
            self.eval_metrics["phone_greed_ser"].update_state(0 if i == j else 1)
            self.eval_metrics["phone_greed_cer"].update_state(score)

        for i, j in zip(final_decode, txt_label):
            i = i.numpy().flatten().tolist()
            j = j.flatten().tolist()

            if self.text4_featurizer.stop in i:
                index = i.index(self.text4_featurizer.stop)
                i = i[:index]
            index = j.index(self.text4_featurizer.stop)
            j = j[:index]

            score, ws, wd, wi = wer(i, j)
            self.cer_s += ws
            self.cer_d += wd
            self.cer_i += wi
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
        result_dict['phone_del']=self.p_cer_d
        result_dict['phone_ins']=self.p_cer_i
        result_dict['phone_sub']=self.p_cer_s

        result_dict['del'] = self.cer_d
        result_dict['ins'] = self.cer_i
        result_dict['sub'] = self.cer_s
        progbar.set_postfix(result_dict)