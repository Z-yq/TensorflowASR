
from utils.user_config import UserConfig
from AMmodel.model import AM
from LMmodel.trm_lm import LM

class ASR():
    def __init__(self,am_config,lm_config):

        self.am=AM(am_config)
        self.am.load_model(False)

        self.lm=LM(lm_config)
        self.lm.load_model()


    def stt(self,wav_path):

        am_result=self.am.predict(wav_path)

        lm_result=self.lm.predict(am_result)

        return am_result,lm_result

if __name__ == '__main__':
    am_config=UserConfig(r'D:\TF2-ASR\configs\am_data.yml',r'D:\TF2-ASR\configs\conformer.yml')
    lm_config = UserConfig(r'D:\TF2-ASR\configs\lm_data.yml', r'D:\TF2-ASR\configs\transformer.yml')
    asr=ASR(am_config,lm_config)
    am,lm=asr.stt(r'./BAC009S0724W0121.wav')
    print('here')
    print(' '.join(am))
    print(''.join(lm))
