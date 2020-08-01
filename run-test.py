import hparams
from AMmodel.model import AM
from LMmodel.trm_lm import LM

class ASR():
    def __init__(self,hparams):

        self.am=AM(hparams)

        self.lm=LM(hparams)


    def stt(self,wav_path):

        am_result=self.am.predict(wav_path)

        lm_result=self.lm.get(am_result)

        return am_result,lm_result

if __name__ == '__main__':
    asr=ASR(hparams)
    am,lm=asr.stt(r'D:\data\xiaobei\wavs\000001.wav')
    print(' '.join(am))
    print(''.join(lm))
