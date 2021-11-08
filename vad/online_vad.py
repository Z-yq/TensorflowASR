import numpy as np
import tensorflow as tf
import wave
class SilenceDetector(object):
    def __init__(self, threshold=20, bits_per_sample=16):
        self.cur_SPL = 0
        self.threshold = threshold
        self.bits_per_sample = bits_per_sample
        self.normal = 2.0 ** (bits_per_sample - 1)

    def is_silence(self, chunk):
        self.cur_SPL = self.soundPressureLevel(chunk)
        is_sil = self.cur_SPL < self.threshold
        return is_sil

    def soundPressureLevel(self, chunk):
        value = (self.localEnergy(chunk) ** 0.5)
        value = value / (len(chunk) + 1e-12)
        value = 20.0 * np.log(value)
        return value

    def localEnergy(self, chunk):
        chunk *= self.normal
        chunk = chunk ** 2
        power = np.sum(chunk)
        return power


class OnlineVAD():
    def __init__(self, model_path,max_sil_wait=3, sr=8000):

        self.init_params()
        self.sd = tf.saved_model.load(model_path)
        self.energe = SilenceDetector()
        self.energe.threshold = 20.
        self.max_sil_wait=max_sil_wait
        self.sr=sr

    def init_params(self):
        # data params
        self.chunk = np.array([], 'float32')  # streaming chunk
        self.wav_length = 0

        # result params
        self.live_result = {'start_time': 0., 'end_time': 0., 'live_text': '',
                            'decoded_result': []}  # 当前积累的语音流结果
        self.decoded_result = []
        self.now_start = 0.  # 该流任务已经积累的时间节点

        # VAD params
        self.threods = []
        self.vad_point = 0
        self.voice_data = np.zeros(2400)

        # event params
        self.inter_break = 0
        self.start_event = 0
        self.end_event = 0

        # state params
        self.send_flag = 0
        self.sil_record = []
        self.sil_times = 0
        self.sound_record = []
        self.chunk_point = 0
        self.sound_start = 0
        self.sound_end = 0


    def vad(self, wav):
        data = wav.copy()
        data = data[-800:]
        data = data.reshape([1, -1, 80])

        output = self.sd.inference(np.array(data, 'float32'))

        output = output.numpy().flatten()
        output = np.where(output >= 0., 1, 0)
        output = output.tolist()
        return output[-10:]

    def parse(self, new_data):
        new_data = np.frombuffer(new_data, 'int16')
        new_data = np.array(new_data, 'float32')
        new_data /= 32768
        self.wav_length += len(new_data) / 8000
        if self.sound_start:
            self.chunk = np.concatenate([self.chunk, new_data], 0)
        self.voice_data = np.hstack((self.voice_data, new_data))
        self.voice_data = self.voice_data[-2400:]

        if self.wav_length - self.vad_point >= 0.1:  # 每 100ms做一次VAD

            vad_pred = self.vad(self.voice_data)
            # print(vad_pred)
            # print(self.wav_length)
            if self.sound_start:
                self.sil_record += vad_pred
            else:
                self.sound_record += vad_pred
            self.vad_point = self.wav_length

        if self.sound_start:
            if len(self.sil_record) >= 20:

                if np.sum(self.sil_record[-10:]) <= 8 and self.sil_times == 0:
                    self.sil_times += 1


                    if self.sil_times == 1:
                        self.inter_break = 1

                        self.live_result['end_time'] = self.wav_length
                elif np.sum(self.sil_record[-10:]) <= 5 and self.sil_times == 1:
                    self.sil_times += 1



                elif np.sum(self.sil_record[-10:]) <= 2 and self.sil_times >= 2:
                    self.sil_times += 1



                else:

                    self.sil_times = 0
                self.sil_record = self.sil_record[-10:]
            if self.sil_times == self.max_sil_wait:
                self.sound_end = 1
                self.end_event = 1

                self.live_result['end_time'] = self.wav_length - self.max_sil_wait+0.1

                self.sil_record = []

                self.sound_start = 0
                self.sil_times = 0
                self.inter_break = 0


                return 1



        else:
            if len(self.sound_record) == 20:
                if np.sum(self.sound_record[-10:]) >= 5.:
                    self.sound_start = 1
                    self.start_event = 1

                    self.sound_record = []
                    self.chunk = self.voice_data[-1600:]
                    self.live_result['start_time'] = self.wav_length - 0.2
                    return 0

                else:
                    self.sound_record = self.sound_record[-10:]



    def end_asr(self):
        pass

    def final_parse(self):
        if len(self.chunk) < 800:
            return 0
        elif len(self.chunk) > 800 and self.sound_start:
            self.send_flag = 1
            self.sound_end = 1
            self.live_result['end_time'] = self.wav_length
            return 1
def audio_request_generator(wav_file):
    pkg_duration = 20
    pkg_frames = int(8000 * (pkg_duration / 1000))

    with wave.open(wav_file, mode="rb") as wav_reader:
        total_frames = wav_reader.getnframes()
        processed_frames = 0

        while processed_frames < total_frames:

            processed_frames += pkg_frames

            audio = wav_reader.readframes(pkg_frames)

            yield audio

if __name__ == '__main__':
    online_vad=OnlineVAD('./vad/online_vad_model')
    audio_generator=audio_request_generator('./vad/test.wav')
    for audio in audio_generator:
        result=online_vad.parse(audio)
        if result ==1:
            print('sound end',online_vad.live_result['end_time'])
            print('======================')
        elif result==0:
            print('sound start',online_vad.live_result['start_time'])
    result=online_vad.final_parse()
    if result==1:
        print('sound end', online_vad.live_result['end_time'])
        print('======================')