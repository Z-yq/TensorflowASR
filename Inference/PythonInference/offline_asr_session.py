from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import librosa
import numpy as np

from asr.src.asr import ASR
from vad.src.vad import VAD
from punc_recover.src.punc_recover import Punc
from utils.user_config import UserConfig

class ASRSession(object):
    """
    ASRSession 封装了端到端ASR。
    """

    def __init__(self,session='asr_1',sample_rate=16000,
                 ) -> None:

        self.session = session
        self.sample_rate=sample_rate
        logging.info('transcriber created. session [%s]', self.session)
        asr_config=UserConfig('./asr/src/configs/am_data.yml','./asr/src/configs/am_data.yml')
        punc_config=UserConfig('./punc_recover/src/configs/data.yml','./punc_recover/src/configs/punc_settings.yml')
        vad_config=UserConfig('./vad/src/configs/am_data.yml','./vad/src/configs/model.yml')
        self.start_time = time.time()
        self.sentence_id = 0
        self.total = 0
        self.asr=ASR(asr_config)
        self.asr.compile('./asr/models/offline')
        self.punc=Punc(punc_config)
        self.offline_vad=OfflineVAD(sr=sample_rate)
        self.offline_vad.compile(VAD(vad_config))

    def send(self,wav_path):
        wav=librosa.load(wav_path,self.sample_rate)[0]
        wav=wav[:len(wav)//80*80]
        vad_result=self.offline_vad.vad(wav)
        respones=[]
        for idx,(s,e) in enumerate(vad_result):
            data=wav[int(s*self.sample_rate):int(e*self.sample_rate)]
            feauter=self.asr.extract_feature(data)
            result=self.asr.decode([feauter])
            if len(result)>5:
                result=self.punc.punc_recover(result)
            respones.append({'session':'asr_1','sentence_index':idx,'sentence_begin_time':int(s*1000),'best_text':result,'sentence_end_time':int(e*1000)})
        return respones


class OfflineVAD():
    def __init__(self,min_duration=0.5,sr=8000,recover_thread=0.1,recover_max_duration=15.):
        self.init_params()

        self.min_duration=min_duration
        self.sample_rate=sr
        self.recover_thread=recover_thread
        self.recover_max_duration=recover_max_duration
    def compile(self,sd):
        self.sd=sd
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
        self.sil_times=0

    def vad(self, wav):
        self.init_params()
        self.wav = wav
        data = wav.copy()
        data=data[::2]

        data = data.reshape([1, -1, 80])

        output = self.sd.inference(np.array(data, 'float32'))

        output = output.flatten()

        output = np.where(output >= 0.0, 1, 0)
        output = output.tolist()
        self.parse(output)

        vad_result = [[round(i['start_time'], 3), round(i['end_time'], 3)] for i in self.vad_result]
        print(vad_result)
        if len(vad_result)>=2:
            vad_result=self.recover(vad_result)
        return vad_result

    def parse(self, vad_preds):

        vad_length = len(vad_preds)
        # print(vad_length)
        self.wav_length=0
        for i in range(vad_length // 10 + 1):
            s = i * 10
            e = s + 10
            # print(s,self.wav_length)
            new_data = self.wav[int(s * 160):int(e * 160)]
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

            if self.sound_start:

                if len(self.sil_record) >= 20:

                    if np.sum(self.sil_record[-10:]) <= 8 and self.sil_times == 0:
                        self.sil_times += 1

                    elif np.sum(self.sil_record[-10:]) <= 5 and self.sil_times == 1:
                        self.sil_times += 1

                    elif np.sum(self.sil_record[-10:]) <= 5 and self.sil_times >= 2:
                        self.sil_times += 1
                    else:

                        self.sil_times = 0
                    self.sil_record = self.sil_record[-10:]
                if self.sil_times == 3:
                    self.sound_end = 1
                    self.end_event = 1

                    self.live_result['end_time'] = self.wav_length -3 * 0.1 + 0.1

                    self.sil_record = []

                    self.sound_start = 0
                    self.sil_times = 0
                    self.inter_break = 0

                    self.vad_result.append(self.live_result)



            else:
                if len(self.sound_record) == 20:
                    if np.sum(self.sound_record[-10:]) >= 5.:
                        self.sound_start = 1
                        self.start_event = 1

                        self.sound_record = []

                        self.live_result['start_time'] = self.wav_length - 0.2


                    else:
                        self.sound_record = self.sound_record[-10:]
            self.wav_length+=0.1
        # print(e)
        self.final_parse()

    def final_parse(self):
        if self.data is None:
            return
        if len(self.data) > int(8000 * 0.2) and self.sound_start:


            self.live_result['end_time'] = self.wav_length-0.1
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
    session = ASRSession()
    respones=session.send('./BAC009S0764W0121.wav')
    for res in respones:
        print(res)

