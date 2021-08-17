from utils.user_config import UserConfig
from AMmodel.model import AM

class ASR():
    def __init__(self, am_config):

        self.am = AM(am_config)
        self.am.load_model(False)

    def decode_am_result(self, result):
        return self.am.decode_result(result)


    def am_test(self, wav_path):
        # am_result is token id
        am_result = self.am.predict(wav_path)
        # token to vocab
        if self.am.model_type == 'Transducer':
            am_result = self.decode_am_result(am_result[1:-1])
        else:
            am_result = self.decode_am_result(am_result[0])
        return am_result


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

    am_config = UserConfig(r'./streaming-logs/am_data.yml', r'./streaming-logs/Streaming_ConformerS.yml')

    asr = ASR(am_config)

    # first inference will be slow,it is normal

    print(asr.am_test(r'BAC009S0764W0121.wav'))

