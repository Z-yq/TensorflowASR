import logging
import os

import numpy as np

from asr.src.models.conformer_blocks import ConformerEncoder, StreamingConformerEncoder, CTCDecoder, Translator, tf
from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer
from utils.user_config import UserConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ASR():
    def __init__(self, config):
        self.running_config = config['running_config']
        self.speech_config = config['speech_config']
        self.model_config = config['model_config']
        self.opt_config = config['optimizer_config']
        self.phone_featurizer = TextFeaturizer(config['inp_config'])
        self.text_featurizer = TextFeaturizer(config['tar_config'])
        self.speech_featurizer = SpeechFeaturizer(self.speech_config)
        self.chunk = self.speech_config['sample_rate'] * self.speech_config['streaming_bucket']
        self.compile()

    def compile(self):
        if not self.speech_config['streaming']:
            self.encoder = ConformerEncoder(dmodel=self.model_config['dmodel'],
                                            reduction_factor=self.model_config['reduction_factor'],
                                            num_blocks=self.model_config['num_blocks'],
                                            head_size=self.model_config['head_size'],
                                            num_heads=self.model_config['num_heads'],
                                            kernel_size=self.model_config['kernel_size'],
                                            fc_factor=self.model_config['fc_factor'],
                                            dropout=self.model_config['dropout'],
                                            add_wav_info=self.speech_config['add_wav_info'],
                                            sample_rate=self.speech_config['sample_rate'],
                                            n_mels=self.speech_config['num_feature_bins'],
                                            mel_layer_type=self.speech_config['mel_layer_type'],
                                            mel_layer_trainable=self.speech_config['mel_layer_trainable'],
                                            stride_ms=self.speech_config['stride_ms'],
                                            name="conformer_encoder", )
        else:
            self.encoder = StreamingConformerEncoder(dmodel=self.model_config['dmodel'],
                                                     reduction_factor=self.model_config['reduction_factor'],
                                                     num_blocks=self.model_config['num_blocks'],
                                                     head_size=self.model_config['head_size'],
                                                     num_heads=self.model_config['num_heads'],
                                                     kernel_size=self.model_config['kernel_size'],
                                                     fc_factor=self.model_config['fc_factor'],
                                                     dropout=self.model_config['dropout'],
                                                     add_wav_info=self.speech_config['add_wav_info'],
                                                     sample_rate=self.speech_config['sample_rate'],
                                                     n_mels=self.speech_config['num_feature_bins'],
                                                     mel_layer_type=self.speech_config['mel_layer_type'],
                                                     mel_layer_trainable=self.speech_config['mel_layer_trainable'],
                                                     stride_ms=self.speech_config['stride_ms'],
                                                     name="stream_conformer_encoder")

            self.encoder.add_chunk_size(
                chunk_size=int(self.speech_config['streaming_bucket'] * self.speech_config['sample_rate']),
                mel_size=self.speech_config['num_feature_bins'],
                hop_size=int(self.speech_config['stride_ms'] * self.speech_config['sample_rate'] // 1000) *
                         self.model_config['reduction_factor'])
        self.ctc_model = CTCDecoder(num_classes=self.phone_featurizer.num_classes,
                                    dmodel=self.model_config['dmodel'],
                                    num_blocks=self.model_config['ctcdecoder_num_blocks'],
                                    head_size=self.model_config['head_size'],
                                    num_heads=self.model_config['num_heads'],
                                    kernel_size=self.model_config['ctcdecoder_kernel_size'],
                                    dropout=self.model_config['ctcdecoder_dropout'],
                                    fc_factor=self.model_config['ctcdecoder_fc_factor'],
                                    )
        self.translator = Translator(inp_classes=self.phone_featurizer.num_classes,
                                     tar_classes=self.text_featurizer.num_classes,
                                     dmodel=self.model_config['dmodel'],
                                     num_blocks=self.model_config['translator_num_blocks'],
                                     head_size=self.model_config['head_size'],
                                     num_heads=self.model_config['num_heads'],
                                     kernel_size=self.model_config['translator_kernel_size'],
                                     dropout=self.model_config['translator_dropout'],
                                     fc_factor=self.model_config['translator_fc_factor'], )
        self.encoder._build()
        self.ctc_model._build()
        self.translator._build()

        self.load_checkpoint()




    def load_checkpoint(self, ):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "encoder-ckpt")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.encoder.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        logging.info('encoder load at {}'.format(os.path.join(self.checkpoint_dir, files[-1])))

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "ctc_decoder-ckpt")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.ctc_model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        logging.info('ctc_model load at {}'.format(os.path.join(self.checkpoint_dir, files[-1])))

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "translator-ckpt")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.translator.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        logging.info('translator load at {}'.format(os.path.join(self.checkpoint_dir, files[-1])))

    def extract_feature(self,wav):
        wav=wav.reshape([1,-1,1])
        out=self.encoder.inference(wav)
        return out
    def decode(self,enc_features):
        enc_outputs=tf.concat(enc_features,axis=1)
        ctc_output=self.ctc_model.inference(enc_outputs)
        ctc_output = tf.nn.softmax(ctc_output, -1)
        input_length = np.array([enc_outputs.shape[1]], 'int32')
        ctc_decode = tf.keras.backend.ctc_decode(ctc_output, input_length)[0][0]
        ctc_decode = tf.cast(tf.clip_by_value(ctc_decode, 0, self.phone_featurizer.num_classes), tf.int32)
        ctc_result = []
        for n in ctc_decode[0].numpy():
            if n != 0:
                ctc_result.append(n)
        ctc_result += [0] * 10
        translator_out = self.translator.inference(np.array([ctc_result], 'int32'), enc_outputs)
        translator_out = tf.argmax(translator_out, -1)
        txt_result = []
        for n in translator_out[0].numpy():
            if n != 0 and n!=self.text_featurizer.endid():
                txt_result.append(n)
            if n==self.text_featurizer.endid():
                break

        txt = self.text_featurizer.iextract(txt_result)
        return ''.join(txt)

