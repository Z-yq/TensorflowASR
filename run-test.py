from utils.user_config import UserConfig
from AMmodel.model import AM
from LMmodel.trm_lm import LM
import pypinyin
import time
import os
class ASR():
    def __init__(self,am_config,lm_config):

        self.am=AM(am_config)
        self.am.load_model(False)

        self.lm=LM(lm_config)
        self.lm.load_model(False)

    def decode_am_result(self,result):
        return self.am.decode_result(result)
    def stt(self,wav_path):

        am_result=self.am.predict(wav_path)
        if self.am.model_type=='Transducer':
            am_result =self.decode_am_result(am_result[1:-1])
            lm_result = self.lm.predict(am_result)
            lm_result = self.lm.decode(lm_result[0].numpy(), self.lm.word_featurizer)
        else:
            am_result=self.decode_am_result(am_result[0])
            lm_result=self.lm.predict(am_result)
            lm_result = self.lm.decode(lm_result[0].numpy(), self.lm.word_featurizer)
        return am_result,lm_result
    def am_test(self,wav_path):
        #am_result is token id
        am_result = self.am.predict(wav_path)
        #token to vocab
        if self.am.model_type == 'Transducer':
            am_result = self.decode_am_result(am_result[1:-1])
        else:
            am_result = self.decode_am_result(am_result[0])
        return am_result
    def lm_test(self,txt):
        py=pypinyin.pinyin(txt)
        input_py=[i[0] for i in py]
        #now lm_result is token id
        lm_result=self.lm.predict(input_py)
        #token to vocab
        lm_result=self.lm.decode(lm_result[0].numpy(),self.lm.word_featurizer)
        return lm_result

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES']='2'
    am_config=UserConfig(r'./conformerCTC(M)/am_data.yml',r'./conformerCTC(M)/conformerM.yml')
    lm_config = UserConfig(r'./transformer-logs/lm_data.yml', r'./transformer-logs/transformerO2OE.yml')
    asr=ASR(am_config,lm_config)

    a,b=asr.stt(r'BAC009S0764W0121.wav')
    print(a)
    print(b)
    print(asr.am_test(r'BAC009S0764W0121.wav'))
    print(asr.lm_test('中介协会'))
