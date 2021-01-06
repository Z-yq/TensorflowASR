import tensorflow as tf
from tester.base_tester import BaseTester
from utils.xer import wer
import numpy as np

class AMTester(BaseTester):
    """ Trainer for CTC Models """

    def __init__(self,
                 config,
                text_feature,
                 ):
        super(AMTester, self).__init__(config)
        self.text_featurizer=text_feature


    def _eval_step(self, batch):
        features,  input_length, labels, label_length = batch


        pred_decode = self.model.recognize_pb(features, input_length)[0]


        if 'CTC' in self.model_type:
            pred_decode=tf.clip_by_value(pred_decode,0,self.text_featurizer.num_classes)
        elif 'Transducer' in self.model_type:
            pred_decode_=[]
            decode_=[]
            for i in pred_decode:
                if i==1:
                    pred_decode_.append(decode_)
                    decode_=[]
                elif i==0:
                    continue
                else:
                    decode_.append(i)
            pred_decode=pred_decode_

        for i,j in zip(pred_decode,labels):
            i=np.array(i).flatten().tolist()
            j = j.flatten().tolist()

            if 'CTC' in self.model_type:
                while self.text_featurizer.pad in i:
                    i.remove(self.text_featurizer.pad)

                while self.text_featurizer.pad in j:
                    j.remove(self.text_featurizer.pad)
            elif 'Transducer' in self.model_type:
                if self.text_featurizer.pad in j:
                    index=j.index(self.text_featurizer.pad)

                    j=j[1:index]
                else:
                    j=j[1:]
            else:
                if self.text_featurizer.stop in i:
                    index=i.index(self.text_featurizer.stop)
                    i=i[:index]
                index=j.index(self.text_featurizer.stop)
                j=j[:index]
            score, ws, wd, wi = wer(j, i)
            self.cer_s+=ws
            self.cer_d+=wd
            self.cer_i+=wi
            self.eval_metrics["greed_ser"].update_state(0 if i == j else 1)
            self.eval_metrics["greed_cer"].update_state(score)

    def compile(self, model: tf.keras.Model):

        self.model = model
        self.model.summary(line_length=100)
        self.model_type=self.model.name

    def run(self, eval_dataset):

        self._eval_batches(eval_dataset)


