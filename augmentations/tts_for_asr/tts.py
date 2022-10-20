import onnxruntime
from utils.text_featurizers import TextFeaturizer
import soundfile as sf
import numpy as np
import os
PATH=os.path.split(os.path.realpath(__file__))[0]
class TTS():

    def __init__(self,):
        config={
            "vocabulary":os.path.join(PATH,'configs/tokens.txt'),
            "spker":os.path.join(PATH,'configs/spk.txt'),
            "maplist":os.path.join(PATH,'configs/pinyin_2_phone.map'),
            "blank_at_zero":True

        }
        self.text_feature=TextFeaturizer(config)
        self.tts_model=onnxruntime.InferenceSession(os.path.join(PATH,'models/tts_model.onnx'),providers=[onnxruntime.get_available_providers()[-1]])

    def synthesize(self,text,spk_ids,speed):
        """

        :param text:sil+待合成文本+sil，如果有标点符号，请手动替换成sil，egs:sil这是一个例子sil
        :param spk_ids:0~514，
        :param speed:速度控制，值大于1为变慢，小于1为加速。
        :return:合成的语音数据
        """

        inp_ids=self.text_feature.extract('sil'+text+'sil')
        inp_ids=np.array(inp_ids,'int32').reshape([1,-1])
        spk_ids=np.array([spk_ids],'int32').reshape([1,1])
        speed=np.array([speed],'float32').reshape([1,-1])
        wav=self.tts_model.run([self.tts_model.get_outputs()[0].name],input_feed={
            self.tts_model.get_inputs()[0].name:inp_ids,
            self.tts_model.get_inputs()[1].name:spk_ids,
            self.tts_model.get_inputs()[2].name:speed,
        })[0]

        wav = wav.flatten()
        return wav

if __name__ == '__main__':
    tts=TTS()
    wav=tts.synthesize('你好神眸sil隐私遮挡',100,1.0)
    sf.write('test.wav',wav,16000)