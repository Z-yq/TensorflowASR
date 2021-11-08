import numpy as np
import tensorflow as tf

class SilenceDetector(object):
    def __init__(self, threshold=20, bits_per_sample=16):
        self.cur_SPL = 0
        self.threshold = threshold
        self.bits_per_sample = bits_per_sample
        self.normal = 2.0 ** (bits_per_sample - 1)
        self.max_energ = 0

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
        chunk*=self.normal
        chunk=chunk**2
        power=np.sum(chunk)
        return power


class OfflineVAD():
    def __init__(self,model_path,min_duration=0.5,chuck_min_energe=20.,sr=8000,recover_thread=0.1,recover_max_duration=15.):
        self.init_params()
        self.sd = tf.saved_model.load(model_path)
        self.energe = SilenceDetector()
        self.energe.threshold = chuck_min_energe
        self.min_duration=min_duration
        self.sample_rate=sr
        self.recover_thread=recover_thread
        self.recover_max_duration=recover_max_duration
    def init_params(self):

        self.data = None  # 当前积累的语音流数据
        self.live_result = {'start_time': 0., 'end_time': 0.}  # 当前积累的语音流结果
        self.vad_result = []
        self.now_start = 0.  # 该流任务已经积累的时间节点

        self.send_flag = 0
        self.sil_record = []
        self.sound_record = []
        self.sound_pick = 0  # 0 找开始点，1 找结束点
        self.sound_start = 0
        #self.pm=np.zeros([2,1,320],'float32')

    def vad(self, wav):
        self.init_params()
        self.wav = wav
        data = wav.copy()
        data = data.reshape([1,-1,  1])

        output = self.sd.inference(np.array(data, 'float32'))

        output = tf.nn.sigmoid(output)
        output = output.numpy().flatten()

        output = np.where(output >= 0.5, 1, 0)
        output = output.tolist()
        self.parse(output)
        vad_result = [[round(i['start_time'], 3), round(i['end_time'], 3)] for i in self.vad_result]
        if len(vad_result)>=2:
            vad_result=self.recover(vad_result)
        return vad_result

    def parse(self, vad_preds):

        vad_length = len(vad_preds)
        # print(vad_length)
        for i in range(vad_length // 10 + 1):
            s = i * 10
            e = s + 10
            new_data = self.wav[int(s * 80):int(e * 80)]
            if self.data is None:
                self.data = new_data
            else:
                self.data = np.concatenate([self.data, new_data], 0)
            vad_pred = vad_preds[s:e]
            # print(s,e,vad_pred)
            if self.sound_pick:
                self.sil_record += vad_pred
            else:
                self.sound_record += vad_pred

            if self.sound_pick:
                if len(self.sil_record) == 20:
                    if np.sum(self.sil_record[-10:]) < 5.:
                        if not self.energe.is_silence(self.data) and len(self.data)>=self.min_duration*self.sample_rate:
                            self.send_flag = 1
                            self.live_result['start_time'] = self.now_start
                            self.live_result['end_time'] = round(self.now_start + len(self.data) / 8000, 3)

                            self.vad_result.append(self.live_result)
                        self.reset_live_state()

                        self.now_start += round(len(self.data) / 8000, 3)

                        self.sil_record = []
                        self.sound_pick = 0
                        self.sound_record = []
                        self.sound_start=0
                        self.data = None

                    else:
                        self.sil_record = self.sil_record[-10:]

            else:
                if len(self.sound_record) == 20:
                    if np.sum(self.sound_record[-10:]) > 6.:
                        self.sound_pick = 1
                        self.sound_start = 1
                    else:
                        self.sound_record = self.sound_record[-10:]
                        self.now_start += (round(len(self.data) / 8000, 3) / 2)
                        self.data = self.data[-len(self.data) // 2:]
        # print(e)
        self.final_parse()

    def final_parse(self):
        if self.data is None:
            return
        if len(self.data) > int(8000 * 0.2) and self.sound_start:
            self.send_flag = 1
            self.live_result['start_time'] = self.now_start
            self.live_result['end_time'] = round(self.now_start + len(self.data) / 8000, 3)
            self.vad_result.append(self.live_result)
        self.data = None

    def reset_live_state(self):
        self.live_result = {'start_time': 0., 'end_time': 0.}  # 当前积累的语音流结果
    def recover(self,results):
        new_results=[]
        s,e = results[0]
        for i in range(1,len(results)):
            now=results[i]
            # print(now)

            if now[0]-e<self.recover_thread and now[1]-s<self.recover_max_duration:
                e=now[1]
                if i==len(results)-1:
                    new_results.append([s,e])
            else:
                new_results.append([s,e])
                s=now[0]
                e=now[1]
                if i==len(results)-1:
                    new_results.append([s,e])
        # print(new_results)
        results_=[]
        for s,e in new_results:
            time_diff=e-s
            if time_diff >self.recover_max_duration:
                factor=time_diff//self.recover_max_duration
                if time_diff%self.recover_max_duration!=0:
                    num=time_diff/(factor+1)
                    factor+=1
                else:
                    num=time_diff/factor
                num=int(num)

                s_=s
                for i in range(int(factor)):

                    e_=s_+num if i!=factor-1 else e

                    results_.append([s_,e_])
                    s_=e_
            else:
                results_.append([s,e])
        return results_

if __name__ == '__main__':
    import librosa
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='-1'

    offline_vad = OfflineVAD('./vad/offline_vad_model')
    print('init over')
    wav=librosa.load('./vad/test.wav',8000)[0]
    results=offline_vad.vad(wav)
    print(results)