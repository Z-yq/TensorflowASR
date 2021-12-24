import logging
import numpy as np
import onnxruntime

from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ASR():
    def __init__(self, config):
        self.running_config = config['running_config']
        self.speech_config = config['speech_config']
        self.model_config = config['model_config']
        self.opt_config = config['optimizer_config']
        self.phone_featurizer = TextFeaturizer(config['inp_config'])
        self.text_featurizer = TextFeaturizer(config['tar_config'])
        self.speech_featurizer = SpeechFeaturizer(self.speech_config)
        self.chunk = self.speech_config['sample_rate'] * self.speech_config['streaming_bucket']

    def compile(self,path):
        self.encoder=onnxruntime.InferenceSession(os.path.join(path,'encoder.onnx'))
        self.ctc_model =onnxruntime.InferenceSession(os.path.join(path,'ctc_model.onnx'))
        self.translator = onnxruntime.InferenceSession(os.path.join(path,'translator.onnx'))

    def softmax(self,logits):

        max_value = np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits - max_value)
        exp_sum = np.sum(exp, axis=1, keepdims=True)
        dist = exp / exp_sum
        return dist
    def extract_feature(self,wav):
        wav=wav.reshape([1,-1,1])
        data = {self.encoder.get_inputs()[0].name:wav.astype('float32')}
        out = self.encoder.run([self.encoder.get_outputs()[0].name], input_feed=data)

        return out[0]

    def remove_blank(self,labels, blank=0):
        new_labels = []
        # 合并相同的标签
        previous = None
        for l in labels:
            if l != previous:
                new_labels.append(l)
                previous = l
        # 删除blank
        new_labels = [l for l in new_labels if l != blank]

        return new_labels



    def greedy_decode(self,y, blank=1331):
        # 按列取最大值，即每个时刻t上最大值对应的下标
        raw_rs = np.argmax(y, axis=1)
        # 移除blank,值为0的位置表示这个位置是blank
        rs = self.remove_blank(raw_rs, blank)
        return rs

    def decode(self,enc_features):

        if len(enc_features)>1:
            enc_outputs=np.hstack(enc_features)

        else:
            enc_outputs=enc_features[0]

        ctc_data = {self.ctc_model.get_inputs()[0].name: enc_outputs.astype('float32')}
        ctc_output=self.ctc_model.run([self.ctc_model.get_outputs()[0].name], input_feed=ctc_data)[0]
        ctc_output = self.softmax(ctc_output[0])

        ctc_result = self.greedy_decode(ctc_output,self.phone_featurizer.num_classes-1)

        ctc_result += [0] * 10
        translator_data={
            self.translator.get_inputs()[0].name:np.array([ctc_result], 'int32'),
            self.translator.get_inputs()[1].name:enc_outputs.astype('float32')
                         }
        translator_out = self.translator.run([self.translator.get_outputs()[0].name], input_feed=translator_data)[0]
        translator_out = np.argmax(translator_out, -1)

        txt_result = []
        for n in translator_out[0]:
            if n != 0 and n!=self.text_featurizer.endid():
                txt_result.append(n)
            if n==self.text_featurizer.endid():
                break

        txt = self.text_featurizer.iextract(txt_result)

        return ''.join(txt)

