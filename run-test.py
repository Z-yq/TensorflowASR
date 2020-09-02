
from utils.user_config import UserConfig
from AMmodel.model import AM
from LMmodel.trm_lm import LM

class ASR():
    def __init__(self,am_config,lm_config):

        self.am=AM(am_config)
        self.am.load_model(False)

        self.lm=LM(lm_config)
        self.lm.load_model()

    def decode_am_result(self,result):
        return self.am.decode_result(result[0])
    def stt(self,wav_path):

        am_result=self.am.predict(wav_path)

        lm_result=self.lm.predict(self.decode_am_result(am_result))

        return am_result,lm_result

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    am_config=UserConfig(r'D:\TF2-ASR\configs\am_data.yml',r'D:\TF2-ASR\configs\conformer.yml')
    lm_config = UserConfig(r'D:\TF2-ASR\configs\lm_data.yml', r'D:\TF2-ASR\configs\transformer.yml')
    asr=ASR(am_config,lm_config)


