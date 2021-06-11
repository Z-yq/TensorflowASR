from utils.user_config import UserConfig
from AMmodel.model import AM
from LMmodel.trm_lm import LM
import pypinyin

class ASR():
    def __init__(self, am_config, lm_config,punc_config=None):

        self.am = AM(am_config)
        self.am.load_model(False)

        # self.lm = LM(lm_config,punc_config)
        # self.lm.load_model(False)
        # if punc_config is not None:
        #     self.punc_recover=True
        # else:
        #     self.punc_recover=False
    def decode_am_result(self, result):
        return self.am.decode_result(result)

    def stt(self, wav_path):

        am_result = self.am.predict(wav_path)
        if self.am.model_type == 'Transducer':
            am_result = self.decode_am_result(am_result[1:-1])
            lm_result = self.lm.predict(am_result)
            lm_result = self.lm.decode(lm_result[0].numpy(), self.lm.lm_featurizer)
        else:
            am_result = self.decode_am_result(am_result[0])
            lm_result = self.lm.predict(am_result)
            lm_result = self.lm.decode(lm_result[0].numpy(), self.lm.lm_featurizer)
        if self.punc_recover:
            punc_result=self.lm.punc_predict(lm_result)
            lm_result=punc_result
        return am_result, lm_result

    def am_test(self, wav_path):
        # am_result is token id
        am_result = self.am.predict(wav_path)
        # token to vocab
        if self.am.model_type == 'Transducer':
            am_result = self.decode_am_result(am_result[1:-1])
        else:
            am_result = self.decode_am_result(am_result[0])
        return am_result


    def lm_test(self, txt):
        if self.lm.config['am_token']['for_multi_task']:
            pys = pypinyin.pinyin(txt, 8, neutral_tone_with_five=True)
            input_py = [i[0] for i in pys]

        else:
            pys = pypinyin.pinyin(txt)
            input_py = [i[0] for i in pys]

        # now lm_result is token id
        lm_result = self.lm.predict(input_py)
        # token to vocab
        lm_result = self.lm.decode(lm_result[0].numpy(), self.lm.lm_featurizer)
        if self.punc_recover:
            lm_result=self.lm.punc_predict(lm_result)
        return lm_result

    def punc_test(self,txt):
        return self.lm.punc_predict(list(txt))

if __name__ == '__main__':
    import time
    # USE CPU:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # USE one GPU:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # limit cpu to 1 core:
    # import tensorflow as tf
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    am_config = UserConfig(r'./conformerCTC(M)/am_data.yml', r'./conformerCTC(M)/conformerM.yml')
    lm_config = UserConfig(r'./configs/lm_data.yml', r'./configs/transformerO2OE.yml')
    asr=ASR(am_config,lm_config)
    asr.am_test(r'BAC009S0764W0121.wav')
    # am_config = UserConfig(r'./conformerCTC(M)/am_data.yml', r'./conformerCTC(M)/conformerM.yml')
    # lm_config = UserConfig(r'./transformer-logs/lm_data.yml', r'./transformer-logs/transformerO2OE.yml')
    # punc_config = UserConfig(r'./punc_model/punc_settings.yml', r'./punc_model/punc_settings.yml')
    # asr = ASR(am_config, lm_config,punc_config)
    #
    # # first inference will be slow,it is normal
    # s=time.time()
    # a, b = asr.stt(r'BAC009S0764W0121.wav')
    # e=time.time()
    # print(a)
    # print(b)
    # print('asr.stt first infenrence cost time:',e-s)
    #
    # # now it's OK
    # s = time.time()
    # a, b = asr.stt(r'BAC009S0764W0121.wav')
    # e = time.time()
    # print(a)
    # print(b)
    # print('asr.stt infenrence cost time:', e - s)
    # s=time.time()
    # print(asr.am_test(r'BAC009S0764W0121.wav'))
    # e=time.time()
    # print('asr.am_test cost time:',e-s)
    # s=time.time()
    # print(asr.lm_test('中介协会'))
    # e=time.time()
    # print('asr.lm_test cost time:',e-s)
    # s = time.time()
    # print(asr.punc_test('今日数学使用在不同的领域中包括科学工程医学经济学和金融学等'))
    # e = time.time()
    # print('asr.punc_test cost time:', e - s)


