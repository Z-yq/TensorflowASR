from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import uuid
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
        self.task_content = TaskContent(self.session, 0.5,sample_rate,5)
        self.sentence_id = 0
        self.total = 0
        self.asr=ASR(asr_config)
        self.asr.compile('./asr/models/streaming')
        self.punc=Punc(punc_config)
        self.task_content.compile(VAD(vad_config))

    def on_started(self, message):
        # 识别流程启动
        logging.info('on_start_event. session [%s] task [%s]' %
                     (self.session, message['task_id']))

    def on_sentence_begin(self, message):
        # 句首位置确定
        logging.info('on_sentence_begin. session [%s] ' %
                     (self.session))
        resp = dict(session=self.session,
                                event_type='sentence begin',
                                sentence_index=int(message['index']),
                                sentence_begin_time=int(message['start_time']))

        return resp

    def on_result_changed(self, message):
        # 句中结果变更
        logging.info('on_changed. session [%s] task [%s]' %
                     (self.session, message['task_id']))
        resp = dict(
            session=self.session,
            event_type='result change',
            sentence_begin_time=int(message['begin_time']),
            best_text=str(message['text'])
        )
        return resp

    def on_inter_break(self, message):
        # 句中结果变更
        logging.info('inter break. session [%s] task [%s]' %
                     (self.session, message['task_id']))
        resp = dict(
            session=self.session,
            event_type='inter break',
            sentence_begin_time=int(message['begin_time']),
            sentence_end_time=int(message['end_time']),
            best_text=str(message['text'])
        )
        return resp

    def on_sentence_end(self, message):
        # 句末位置确定
        logging.info('on_sentence_end. session [%s] task [%s] text[%s]' %
                     (self.session, message['task_id'], str(message['text'])))
        resp =dict(
            session=self.session,
            event_type='sentence end',
            sentence_index=int(message['index']),
            sentence_begin_time=int(message['begin_time']),
            best_text=str(message['text']),
            sentence_end_time=int(message['end_time']))
        return resp

    def on_completed(self, message):
        # 识别流程结束
        logging.info('on_complete_event. session [%s] task [%s]' %
                     (self.session, message['task_id']))

        return None

    def on_task_failed(self, message):
        logging.info('on_failed_event\t%s, status_text:%s. session[%s]' %
                     (message['header']['task_id'],
                      message['header']['status_text'], self.session))

    def on_channel_closed(self):
        logging.info('on_channel_close_event')

    def send(self, audio_data: bytes):
        new_data = audio_data

        self.task_content.parse(new_data)

        if self.task_content.start_event:
            return_value = self.on_sentence_begin(
                {'index': self.sentence_id, 'start_time': self.task_content.wav_length * 1000-200})
            self.task_content.start_event = 0
            return return_value
        return_value = None
        if not self.task_content.send_flag:
            return_value = None
        elif self.task_content.sound_end and self.task_content.send_flag:  # END EVENT
            logging.debug('end event')
            task_id = uuid.uuid4().__str__()
            audio = self.task_content.chunk
            enc_outputs = self.task_content.enc_outputs
            audio = np.array(audio, 'float32')
            live_result = self.task_content.streaming_live_out()
            live_result.update({'task_id': task_id})
            live_result.update({'index': self.sentence_id})
            s = time.time()

            if len(audio) > 800:

                enc_output = self.asr.extract_feature(audio)

                task_result = self.asr.decode(enc_outputs + [enc_output])
            else:
                task_result = self.asr.decode(enc_outputs)
            # print(task_result)
            if len(task_result)>=5:
                task_result=self.punc.punc_recover(task_result)
            # print(task_result)
            logging.debug('end event audio length {}'.format(len(audio)))
            e = time.time()
            logging.debug('session {} wav length {} asr cost time {},time at {}'.format(self.session, len(audio) / 8000,
                                                                                       round(e - s, 5), time.time()))
            live_result['live_text'] = ''.join(task_result)
            return_value = self.on_sentence_end({
                'index': live_result['index'],
                'begin_time': live_result['start_time'] * 1000,
                'end_time': live_result['end_time'] * 1000,
                'text': live_result[
                    'live_text'],
                'task_id': live_result['task_id'],
            })

            self.sentence_id += 1
            self.task_content.end_event = 0
            self.task_content.sound_end = 0
            self.task_content.sound_start = 0
            self.task_content.send_flag = 0
            self.task_content.reset_chunk_end()
            self.task_content.reset_live_result()
            logging.debug('enc outputs length:{}'.format(len(self.task_content.enc_outputs)))
        elif self.task_content.send_flag:

            if self.task_content.inter_break and self.task_content.sil_times == 1:
                logging.debug('inter_break event')
                self.task_content.inter_break = 0
                task_id = uuid.uuid4().__str__()
                audio = self.task_content.chunk
                enc_outputs = self.task_content.enc_outputs
                audio = np.array(audio, 'float32')
                live_result = self.task_content.streaming_live_out()
                live_result.update({'task_id': task_id})

                s = time.time()
                if len(audio) > 800:

                    enc_output = self.asr.extract_feature(audio)

                    task_result = self.asr.decode(enc_outputs + [enc_output])
                    if len(audio) >= self.task_content.chunk_max_duration:
                        enc_outputs.append(enc_output)
                        self.task_content.enc_outputs = enc_outputs
                else:
                    task_result = self.asr.decode(enc_outputs)

                if len(task_result) >= 5:
                    task_result = self.punc.punc_recover(task_result)

                e = time.time()
                logging.debug('session {} wav length {} asr cost time {}'.format(self.session, len(audio) / 8000,
                                                                                 round(e - s, 5)))
                live_result['live_text'] = ''.join(task_result)

                return_value = self.on_inter_break({
                    'begin_time': live_result['start_time'] * 1000,
                    'end_time': live_result['end_time'] * 1000,
                    'text': live_result[
                        'live_text'],
                    'task_id': live_result['task_id'],
                })

                self.task_content.send_flag = 0
                logging.debug('enc outputs length:{}'.format(len(self.task_content.enc_outputs)))
            else:
                # on_change
                logging.debug('change event')
                task_id = uuid.uuid4().__str__()
                audio = self.task_content.chunk

                audio = np.array(audio, 'float32')
                live_result = self.task_content.streaming_live_out()
                live_result.update({'task_id': task_id})

                s = time.time()
                enc_output = self.asr.extract_feature(audio)
                self.task_content.enc_outputs += [enc_output]
                e = time.time()
                logging.debug('session {} wav length {} asr cost time {}'.format(self.session, len(audio) / 8000,
                                                                                 round(e - s, 5)))
                self.task_content.send_flag = 0
                logging.debug('enc outputs length:{}'.format(len(self.task_content.enc_outputs)))
        self.task_content.chunk_length_check()

        return return_value

    def final_send(self):
        self.task_content.final_parse()
        if not self.task_content.send_asr():
            return_value = None
        else:

            audio = self.task_content.chunk
            enc_outputs = self.task_content.enc_outputs
            audio = np.array(audio, 'float32')
            live_result = self.task_content.streaming_live_out()
            live_result.update({'index': self.sentence_id})
            s = time.time()
            if len(audio) > 800:
                enc_output = self.asr.extract_feature(audio)
                task_result = self.asr.decode(enc_outputs + [enc_output])
            else:
                task_result = self.asr.decode(enc_outputs)
            if len(task_result) >= 5:
                task_result = self.punc.punc_recover(task_result)
            e = time.time()
            logging.debug('session {} wav length {} asr cost time {}'.format(self.session, len(audio) / 8000,
                                                                             round(e - s, 5)))
            live_result['live_text'] = ''.join(task_result)
            return_value = self.on_sentence_end({
                'index': live_result['index'],
                'begin_time': live_result['start_time'] * 1000,
                'end_time': live_result['end_time'] * 1000,
                'text': live_result['live_text'],
                'task_id': live_result['task_id'],
            })
            self.task_content.reset_live_result()
            self.sentence_id += 1
            self.task_content.end_event = 0

        self.task_content.init_params()
        return return_value

    def expired(self):
        """ 判断context是否过期

        :return: 已过期, 返回 True; 否则返回 False
        :rtype: bool
        """
        return time.time() - self.start_time >300



class TaskContent():
    def __init__(self, session, chunk_max_duration, sr=8000,
                 wait_sil=5, #等待300ms
                 vad_time=1,#100ms做一次VAD
                 start_thread=5,
                 end_thread=2,
                 ):
        self.session = session
        self.chunk_max_duration = chunk_max_duration * sr
        self.init_params()
        self.task_start_time = time.time()
        self.wait_sil=wait_sil
        self.sr=sr
        self.vad_time=vad_time
        self.start_thread=start_thread
        self.end_thread=end_thread
    def compile(self, sd):
        self.sd = sd


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

        self.enc_outputs = []

    def vad(self, wav):
        data = wav.copy()
        data=data[::2]
        data = data.reshape([1, -1, 80])

        output = self.sd.inference(np.array(data, 'float32'))
        output = output.flatten()
        output = np.where(output >= 0., 1, 0)
        output = output.tolist()
        return output[-int(10*self.vad_time):]

    def parse(self, new_data):
        new_data = np.frombuffer(new_data, 'int16')
        new_data = np.array(new_data, 'float32')
        new_data /= 32768
        self.wav_length += len(new_data) / self.sr
        if self.sound_start:
            self.chunk = np.concatenate([self.chunk, new_data], 0)
        self.voice_data = np.hstack((self.voice_data, new_data))
        self.voice_data = self.voice_data[-int((self.vad_time+2)*self.sr):]

        if self.wav_length - self.vad_point >= 0.1*self.vad_time:  # 每 100ms做一次VAD
            s = time.time()
            vad_pred = self.vad(self.voice_data)
            e = time.time()
            if self.sound_start:
                self.sil_record += vad_pred
            else:
                self.sound_record += vad_pred
            self.vad_point = self.wav_length
            logging.debug('{} vad cost {},wav length {},'.format(self.session, e - s, self.wav_length))
        if self.sound_start:
            if len(self.sil_record) >= 20:

                if np.sum(self.sil_record[-10:]) <= 8 and self.sil_times == 0:
                    self.sil_times += 1

                    logging.debug('inter_break {}'.format(self.sil_times))
                    if self.sil_times == 1:
                        self.inter_break = 1
                        # end_time = self.now_start + round(len(self.data) / 8000, 3)
                        self.live_result['end_time'] = self.wav_length
                elif np.sum(self.sil_record[-10:]) <= 5 and self.sil_times == 1:
                    self.sil_times += 1

                    logging.debug('inter_break {}'.format(self.sil_times))

                elif np.sum(self.sil_record[-10:]) <= self.end_thread and self.sil_times >= 2:
                    self.sil_times += 1

                    logging.debug('inter_break {}'.format(self.sil_times))

                else:

                    self.sil_times = 0
                self.sil_record = self.sil_record[-10:]
            if self.sil_times == self.wait_sil:
                self.sound_end = 1
                self.end_event = 1

                self.live_result['end_time'] = self.wav_length - self.wait_sil*0.1+0.1

                self.sil_record = []

                self.sound_start = 0
                self.sil_times = 0
                self.inter_break = 0

                self.send_flag = 1

            elif len(self.chunk) - self.chunk_point >= self.chunk_max_duration:
                self.send_flag = 1
                self.chunk_point = len(self.chunk)
            elif len(self.chunk) - self.chunk_point == 0:
                self.send_flag = 0

        else:
            if len(self.sound_record) == 20:
                if np.sum(self.sound_record[-10:]) >= self.start_thread:
                    self.sound_start = 1
                    self.start_event = 1

                    self.sound_record = []
                    self.chunk = self.voice_data[-int(self.sr*0.2):]
                    self.live_result['start_time'] = self.wav_length - 0.2


                else:
                    self.sound_record = self.sound_record[-10:]

    def reset_chunk_end(self):

        self.chunk = np.array([], 'float32')
        self.chunk_point = 0
        self.enc_outputs = []

    def reset_chunk(self):

        self.chunk = np.array([], 'float32')
        self.chunk_point = 0

    def chunk_length_check(self):
        if len(self.chunk) >= self.chunk_max_duration:
            self.reset_chunk()

    def end_asr(self):
        pass

    def final_parse(self):
        if len(self.chunk) < 800:
            return
        elif len(self.chunk) > 800 and self.sound_start:
            self.send_flag = 1
            self.sound_end = 1
            self.live_result['end_time'] = self.wav_length

    def streaming_live_out(self):

        return self.live_result

    def reset_live_result(self):

        self.chunk = np.array([], 'float32')
        self.chunk_point = 0
        self.live_result = {'start_time': 0., 'end_time': 0., 'live_text': '', 'decoded_result': []}

        self.end_event = 0
        self.sound_end = 0
        self.sound_start = 0
        self.send_flag = 0

        self.reset_chunk_end()

    def send_asr(self):
        return self.send_flag
def audio_request_generator(wav_file):
    pkg_duration = 20 #20ms
    pkg_frames = int(8000 * (pkg_duration / 1000))

    with wave.open(wav_file, mode="rb") as wav_reader:
        total_frames = wav_reader.getnframes()
        processed_frames = 0

        while processed_frames < total_frames:

            processed_frames += pkg_frames

            audio = wav_reader.readframes(pkg_frames)

            yield audio

if __name__ == '__main__':
    import wave
    session = ASRSession()
    audio_generator=audio_request_generator('./BAC009S0764W0121.wav')
    for audio in audio_generator:
        result=session.send(audio)
        if result is not None:
            print(result)

