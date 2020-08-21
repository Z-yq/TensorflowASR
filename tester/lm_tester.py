import tensorflow as tf
from tester.base_tester import BaseTester
from utils.xer import wer


class LMTester(BaseTester):
    """ Trainer for CTC Models """

    def __init__(self,
                 config,
              one2one,
                 ):
        super(LMTester, self).__init__(config)
        self.one2one=one2one


    def _eval_step(self, batch):
        x,labels = batch

        pred_decode = self.model.inference(x)


        for i,j in zip(pred_decode,labels):
            i=i.numpy().flatten().tolist()
            j = j.flatten().tolist()

            if self.one2one:

                index=j.index(self.model.end_id)
                i=i[1:index]
                j=j[1:index]
            else:
                if self.model.end_id in i:
                    index=i.index(self.model.end_id)
                    i=i[1:index]
                else:
                    i=i[1:]
                index = j.index(self.model.end_id)
                j = j[1:index]

            score, ws, wd, wi = wer(i, j)
            self.cer_s+=ws
            self.cer_d+=wd
            self.cer_i+=wi
            self.eval_metrics["greed_ser"].update_state(1 if i == j else 0)
            self.eval_metrics["greed_cer"].update_state(score)

    def compile(self, model: tf.keras.Model):

        self.model = model
        self.model.summary(line_length=100)
        self.model_type=self.model.name

    def run(self, eval_dataset):

        self._eval_batches(eval_dataset)


