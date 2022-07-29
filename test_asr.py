import logging
import os

import numpy as np

from asr.models.conformer_blocks import ConformerEncoder, StreamingConformerEncoder, CTCDecoder, Translator, tf
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
            assert 'Streaming' in self.model_config['name'],'am_data.yml set streaming=True,But model.yml is OfflineCTC'
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
            self.encoder.set_inference_func()
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

    def load_checkpoint(self, ):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(self.running_config["outdir"], "encoder-ckpt")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.encoder.load_weights(os.path.join(self.checkpoint_dir, files[-1]),by_name=True)
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

    def stream_stt(self, wav_path):
        data = self.speech_featurizer.load_wav(wav_path)

        enc_outputs = None
        for i in range(9999):
            s = i * self.chunk
            e = s + self.chunk
            if s >= len(data):
                break
            input_wav = data[int(s):int(e)]
            input_wav = input_wav.reshape([1, -1, 1])
            es = time.time()
            enc_output = self.encoder.inference(input_wav)
            ee = time.time()
            enc_output = enc_output.numpy()
            if enc_outputs is not None:
                enc_outputs = np.hstack((enc_outputs, enc_output))
            else:
                enc_outputs = enc_output
            # 这里为每chunk预测一次，也可以最后预测一次
            ds = time.time()
            ctc_output = self.ctc_model(enc_outputs, training=False)
            de = time.time()
            ctc_output = tf.nn.softmax(ctc_output, -1)
            input_length = np.array([enc_outputs.shape[1]], 'int32')
            ctc_decode = tf.keras.backend.ctc_decode(ctc_output, input_length)[0][0]
            ctc_decode = tf.cast(tf.clip_by_value(ctc_decode, 0, self.phone_featurizer.num_classes), tf.int32)
            ts = time.time()
            ctc_result = []
            for n in ctc_decode[0].numpy():
                if n != 0:
                    ctc_result.append(n)
            ctc_result += [0] * 10
            translator_out = self.translator.inference(np.array([ctc_result], 'int32'), enc_outputs)
            translator_out = tf.argmax(translator_out, -1)
            te = time.time()
            print('extract cost time:', ee - es, 'ctc decode time:', de - ds, 'translator cost time:', te - ts)
        ctc_result = []
        for n in ctc_decode[0].numpy():
            if n != 0:
                ctc_result.append(n)
        txt_result = []
        for n in translator_out[0].numpy():
            if n != 0:
                txt_result.append(n)
            if n==self.text_featurizer.endid():
                break
        phone = self.phone_featurizer.iextract(ctc_result)
        txt = self.text_featurizer.iextract(txt_result)
        return ' '.join(phone), ''.join(txt)

    def remove_blank(self, labels, blank=0):
        new_labels = []
        # 合并相同的标签
        previous = None
        for l in labels:
            if l != previous:
                new_labels.append(l)
                previous = l
        # 删除blank
        new_labels = [l for l in new_labels if l != blank]

        return new_labels

    def greedy_decode(self, y, blank=1331):
        # 按列取最大值，即每个时刻t上最大值对应的下标
        raw_rs = np.argmax(y, axis=1)
        # 移除blank,值为0的位置表示这个位置是blank
        rs = self.remove_blank(raw_rs, blank)
        return rs
    def offline_stt(self, wav_path):
        # am_result is token id
        data = self.speech_featurizer.load_wav(wav_path)
        input_wav = data.reshape([1, -1, 1])
        es = time.time()
        enc_outputs = self.encoder(input_wav, training=False)
        ee = time.time()
        ds = time.time()
        ctc_output = self.ctc_model(enc_outputs, training=False)
        de = time.time()
        ctc_output = tf.nn.softmax(ctc_output, -1)
        input_length = np.array([enc_outputs.shape[1]], 'int32')
        ctc_decode = tf.keras.backend.ctc_decode(ctc_output, input_length)[0][0]

        ctc_decode = tf.cast(tf.clip_by_value(ctc_decode, 0, self.phone_featurizer.num_classes), tf.int32)
        # print(ctc_decode)
        ts = time.time()
        hot_words=["极大",'轻狂','补充','市场','不足']
        hot_words=[self.text_featurizer.extract(i) for i in hot_words]
        hot_words=np.array(hot_words,'int32')

        translator_out = self.translator([ctc_decode[:,:15], None,hot_words], training=False)
        translator_out = tf.argmax(translator_out, -1)
        te = time.time()
        print('extract feature cost:', ee - es, 'ctc cost time:', de - ds, 'translator cost time:', te - ts)
        ctc_result = []
        for n in ctc_decode[0].numpy():
            if n != 0:
                ctc_result.append(n)
        txt_result = []
        for n in translator_out[0].numpy():
            if n != 0:
                txt_result.append(n)
            if n==self.text_featurizer.endid():
                break
        phone = self.phone_featurizer.iextract(ctc_result)
        txt = self.text_featurizer.iextract(txt_result)

        return ' '.join(phone), ''.join(txt)

    def stt(self, wav_path):
        if self.speech_config['streaming']:
            return self.stream_stt(wav_path)
        else:
            return self.offline_stt(wav_path)
    def convert_to_onnx(self):
        import tf2onnx
        self.encoder.set_inference_func()
        self.ctc_model.set_inference_func()
        self.translator.set_inference_func()

        tf2onnx.convert.from_function(self.encoder.inference,
                                      input_signature=[ tf.TensorSpec([None, None,1], dtype=tf.float32)],opset=13,output_path='./encoder.onnx')

        tf2onnx.convert.from_function(self.ctc_model.inference,
                                      input_signature=[ tf.TensorSpec([None, None,self.ctc_model.dmodel], dtype=tf.float32),], opset=13,
                                      output_path='./ctc_model.onnx')

        tf2onnx.convert.from_function(self.translator.inference,
                                      input_signature=[ tf.TensorSpec([None, None], dtype=tf.int32),
                     tf.TensorSpec([None, None, self.translator.dmodel], dtype=tf.float32),], opset=13,
                                      output_path='./translator.onnx')


    def convert_to_pb(self,export_path):
        self.encoder.set_inference_func()
        self.ctc_model.set_inference_func()
        self.translator.set_inference_func()
        encoder=os.path.join(export_path,'encoder')
        ctc=os.path.join(export_path,'ctc_decoder')
        translator=os.path.join(export_path,'translator')
        concrete_func = self.encoder.inference.get_concrete_function()
        tf.saved_model.save(self.encoder, encoder, signatures=concrete_func)

        concrete_func = self.ctc_model.inference.get_concrete_function()
        tf.saved_model.save(self.ctc_model, ctc, signatures=concrete_func)

        concrete_func = self.translator.inference.get_concrete_function()
        tf.saved_model.save(self.translator, translator, signatures=concrete_func)
if __name__ == '__main__':
    import time

    # USE CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # USE one GPU:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # limit cpu to 1 core:
    # import tensorflow as tf
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)

    am_config = UserConfig(r'./asr/configs/am_data.yml', r'./asr/configs/conformerS.yml')
    asr = ASR(am_config)
    # print(asr.stt('./asr/BAC009S0764W0121.wav'))
    print(asr.stt(r'D:\data\data_aishell\wav\train\S0050\BAC009S0050W0251.wav'))
    # asr.convert_to_onnx()
