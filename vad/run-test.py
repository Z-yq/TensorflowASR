import logging
import os
from vad.models.vad_model import CNN_Online_VAD,CNN_Offline_VAD,tf

from utils.user_config import UserConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class VAD():
    def __init__(self, config,):
        self.running_config = config['running_config']
        self.model_config = config['model_config']

        self.compile()

    def creat_mask(self, seq):
        seq_pad = tf.cast(tf.equal(seq, 0), tf.float32)
        return seq_pad[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    def compile(self):
        if self.model_config['streaming']:
            self.model = CNN_Online_VAD(self.model_config['dmodel'], name=self.model_config['name'])
        else:
            self.model = CNN_Offline_VAD(self.model_config['dmodel'], name=self.model_config['name'])

        self.model._build()
        self.load_checkpoint()
        self.model.summary(line_length=100)

    def load_checkpoint(self, ):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        self.steps = int(files[-1].split('_')[-1].replace('.h5', ''))
    def convert_to_pb(self,export_path):
        concrete_func = self.model.inference.get_concrete_function()
        tf.saved_model.save(self.model, export_path, signatures=concrete_func)







if __name__ == '__main__':

    # USE CPU:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # USE one GPU:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # limit cpu to 1 core:
    # import tensorflow as tf
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)

    config = UserConfig(r'./punc_recover/configs/data.yml', r'./punc_recover/configs/punc_settings.yml')
    # punc_config = UserConfig(r'./punc_model/punc_settings.yml', r'./punc_model/punc_settings.yml')
    punc = Punc(config)

    # first inference will be slow,it is normal
    print(punc.punc_recover('谢谢你的爱'))
