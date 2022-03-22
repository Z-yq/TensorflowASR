import logging
import os

import numpy as np
import tensorflow as tf

from asr.models.conformer_blocks import ConformerEncoder, StreamingConformerEncoder, CTCDecoder, Translator
from asr.tester.base_tester import BaseTester
from utils.text_featurizers import TextFeaturizer
from utils.xer import wer


class AMTester(BaseTester):
    """ Trainer for CTC Models """

    def __init__(self,
                 config,
                 ):
        super(AMTester, self).__init__(config=config['running_config'])
        self.config = config
        self.speech_config = config['speech_config']
        self.model_config = config['model_config']
        self.opt_config = config['optimizer_config']
        self.phone_featurizer = TextFeaturizer(config['inp_config'])
        self.text_featurizer = TextFeaturizer(config['tar_config'])
        self.eval_metrics = {
            "phone_ser": tf.keras.metrics.Mean(),
            "phone_cer": tf.keras.metrics.Mean(),
            "txt_ser": tf.keras.metrics.Mean(),
            "txt_cer": tf.keras.metrics.Mean()
        }


    def _eval_step(self, batch):
        features, input_length, phone_labels, phone_label_length, tar_label = batch
        enc_output = self.encoder(features, training=False)
        ctc_output = self.ctc_model(enc_output, training=False)
        ctc_output = tf.nn.softmax(ctc_output, -1)
        ctc_decode = tf.keras.backend.ctc_decode(ctc_output, input_length)[0][0]
        ctc_decode = tf.cast(tf.clip_by_value(ctc_decode, 0, self.phone_featurizer.num_classes), tf.int32)
        translator_out = self.translator([ctc_decode, enc_output], training=False)
        translator_out=tf.argmax(translator_out,-1)
        translator_out = translator_out.numpy()
        ctc_decode=ctc_decode.numpy()
        phone_labels=phone_labels.numpy()
        tar_label=tar_label.numpy()
        for i, j in zip(ctc_decode, phone_labels):
            i = np.array(i).flatten().tolist()
            j = j.flatten().tolist()

            while self.phone_featurizer.pad in i:
                i.remove(self.phone_featurizer.pad)

            while self.phone_featurizer.pad in j:
                j.remove(self.phone_featurizer.pad)

            score, ws, wd, wi = wer(j, i)
            self.ctc_nums[0] += len(j)
            self.ctc_nums[1] += ws
            self.ctc_nums[2] += wi
            self.ctc_nums[3] += wd
            self.eval_metrics["phone_ser"].update_state(0 if i == j else 1)
            self.eval_metrics["phone_cer"].reset_states()
            self.eval_metrics["phone_cer"].update_state(sum(self.ctc_nums[1:]) / (self.ctc_nums[0] + 1e-6))
        for i, j in zip(translator_out, tar_label):
            i = np.array(i).flatten().tolist()
            if 1 in i :
                cut_line=i.index(1)
                i=i[:cut_line]

            j = j.flatten().tolist()

            while self.text_featurizer.pad in i:
                i.remove(self.text_featurizer.pad)
            while self.text_featurizer.endid() in i:
                i.remove(self.text_featurizer.endid())
            while self.text_featurizer.pad in j:
                j.remove(self.text_featurizer.pad)
            while self.text_featurizer.endid() in j:
                j.remove(self.text_featurizer.endid())

            score, ws, wd, wi = wer(j, i)
            self.translator_nums[0] += len(j)
            self.translator_nums[1] += ws
            self.translator_nums[2] += wi
            self.translator_nums[3] += wd
            self.eval_metrics["txt_ser"].update_state(0 if i == j else 1)
            self.eval_metrics["txt_cer"].reset_states()
            self.eval_metrics["txt_cer"].update_state(sum(self.translator_nums[1:]) / (self.translator_nums[0] + 1e-6))

    def compile(self,):

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

        self.encoder.summary(line_length=100)
        self.ctc_model.summary(line_length=100)
        self.translator.summary(line_length=100)

    def run(self,):

        self._eval_batches()

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


