import onnxruntime
import librosa
import numpy as np
import soundfile as sf
import os
PATH=os.path.split(os.path.realpath(__file__))[0]
class VC_Aug():
    def __init__(self):
        self.vc_model=onnxruntime.InferenceSession(os.path.join(PATH,'models/vc_aug_model_L.onnx'),providers=onnxruntime.get_available_providers())
    def convert(self,wav,spk):
        """

        :param wav:语音数据，[-1,1]
        :param spk: 0~1882
        :return: 音色转换后的语音数据
        """
        spk=np.clip(0,1882,spk)
        T = len(wav) // 640 * 640
        wav = wav[:T]
        out = self.vc_model.run([self.vc_model.get_outputs()[0].name], input_feed={"x": wav.reshape([1,-1,1]).astype('float32'), "c": np.array(spk, 'int64').reshape(1,)})[0]
        return out.flatten()

if __name__ == '__main__':
    wav_path="test.wav"
    wav=librosa.load(wav_path,sr=16000)[0]
    wav/=np.abs(wav).max()
    augment=VC_Aug()
    # spk=np.random.randint(0,1883,1).flatten()
    spk=[1881]
    out=augment.convert(wav,spk)
    sf.write(wav_path.replace('.wav','_vcto_{}.wav'.format(spk[0])),out.flatten(),16000)