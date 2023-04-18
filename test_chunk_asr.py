import logging
import os

import numpy as np

from asr.models.chunk_conformer_blocks import (
    ChunkConformer,
    tf,
)

from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer
from utils.user_config import UserConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ASR:
    def __init__(self, config):
        self.running_config = config["running_config"]
        self.speech_config = config["speech_config"]
        self.model_config = config["model_config"]
        self.opt_config = config["optimizer_config"]
        self.phone_featurizer = TextFeaturizer(config["inp_config"])
        self.text_featurizer = TextFeaturizer(config["tar_config"])
        self.speech_featurizer = SpeechFeaturizer(self.speech_config)
        self.strategy = tf.distribute.MirroredStrategy()
        self.config = config
        self.compile()

    def compile(self):
        with self.strategy.scope():

            self.runner = ChunkConformer(self.config, self.phone_featurizer.num_classes,
                                         self.text_featurizer.num_classes)
            self.runner.compile()
            self.runner.load_weights(tf.train.latest_checkpoint(os.path.join(self.running_config['outdir'],'all-ckpt')))

            self.wav_buf_length=self.runner.front.wav_buf_length



    def stream_call(self, wav_path):
        data = self.speech_featurizer.load_wav(wav_path)
        data = data / np.abs(data.max())

        caches = self.runner.init_picker_caches(1)
        caches2 = self.runner.init_decoder_caches(1)

        valid_txt_outs = tf.zeros([1, 0, self.text_featurizer.num_classes])
        valid_phone_outs = tf.zeros([1, 0, self.phone_featurizer.num_classes])
        unvalid_txt_outs = tf.zeros([1, 0, self.text_featurizer.num_classes])
        txt_outs_offline = self.runner.predict(data.reshape([1, -1, 1]))
        # print(txt_outs_offline[0][0])
        # exit()
        for i in range(99999):
            s = i * self.wav_buf_length
            e = s + self.wav_buf_length
            if s >= len(data):
                break

            input_wav = data[int(s): int(e)]
            input_wav = input_wav.reshape([1, -1, 1])

            valid_phone_out, _, valid_hidden_out, caches = self.runner.picker_stream_predict(input_wav, caches)
            for cache in caches:
                print(cache.shape)

            if valid_phone_out.shape[1] == 0:
                continue
            else:
                feature_outputs, picked_phone_out = self.runner.feature_pick(valid_hidden_out, valid_phone_out)
                if feature_outputs.shape[1] != 0:
                    valid_ctc_out, unvalid_txt_outs, caches2 = self.runner.decoder_stream_predict(feature_outputs,
                                                                                                 caches2)
                    # print(valid_ctc_out.shape, unvalid_ctc_out.shape)
                    valid_txt_outs = tf.concat([valid_txt_outs, valid_ctc_out], axis=1)
                    valid_phone_outs = tf.concat([valid_phone_outs, picked_phone_out], axis=1)
            txt_output = tf.concat([valid_txt_outs, unvalid_txt_outs], axis=1)

            if txt_output.shape[1] == 0:
                continue
            txt_output = tf.nn.softmax(txt_output, -1)
            input_length = np.array([txt_output.shape[1]], "int32")
            ctc_decode = tf.keras.backend.ctc_decode(txt_output, input_length)[0][0]
            ctc_decode = tf.cast(
                tf.clip_by_value(ctc_decode, 0, self.text_featurizer.num_classes),
                tf.int32,
            )

            ctc_result = []
            for n in ctc_decode[0].numpy():
                if n != 0:
                    ctc_result.append(n)

            text = self.text_featurizer.iextract(ctc_result)

            ctc_output = valid_phone_outs

            if ctc_output.shape[1] == 0:
                continue
            ctc_output = tf.nn.softmax(ctc_output, -1)
            input_length = np.array([ctc_output.shape[1]], "int32")
            ctc_decode = tf.keras.backend.ctc_decode(ctc_output, input_length)[0][0]
            ctc_decode = tf.cast(
                tf.clip_by_value(ctc_decode, 0, self.phone_featurizer.num_classes),
                tf.int32,
            )

            ctc_result = []
            for n in ctc_decode[0].numpy():
                if n != 0:
                    ctc_result.append(n)

            phone = self.phone_featurizer.iextract(ctc_result)

            print('time:',e / 16000,)
            print('streaming phone out:', phone)
            print("streaming texts out:", text)


        ctc_output = tf.nn.softmax(txt_outs_offline, -1)
        input_length = np.array([ctc_output.shape[1]], "int32")
        ctc_decode = tf.keras.backend.ctc_decode(ctc_output, input_length)[0][0]
        ctc_decode = tf.cast(
            tf.clip_by_value(ctc_decode, 0, self.text_featurizer.num_classes),
            tf.int32,
        )
        ctc_result = []
        for n in ctc_decode[0].numpy():
            if n != 0:
                ctc_result.append(n)
        texts = self.text_featurizer.iextract(ctc_result)

        print("offline texts out:", texts)

    def convert_to_onnx(self, outdir=''):
        import tf2onnx
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        tf2onnx.convert.from_function(
            self.runner.picker_onnx_convert(),
            input_signature=self.runner.picker_inp_sig,
            opset=15,
            output_path=os.path.join(outdir, "picker.onnx"),
        )

        tf2onnx.convert.from_function(
            self.runner.decoder_onnx_convert(),
            input_signature=self.runner.decoder_inp_sig,
            opset=15,
            output_path=os.path.join(outdir, "decoder.onnx"),
        )

    def get_shape(self, shape):
        shape_ = []
        for n in shape:
            if type(n) != str:
                shape_.append(n)
            else:
                shape_.append(0)
        return shape_

    def pick_feature(self, enc, logits):
        new_enc = []
        enc_ = enc[0]
        logits_ = logits[0]
        for i in range(enc_.shape[0]):
            if np.argmax(logits_[i]) != logits_.shape[-1] - 1:
                new_enc.append(enc_[i:i + 1])
        if len(new_enc) > 0:
            new_enc = np.vstack(new_enc)
            new_enc = new_enc[np.newaxis]
        return new_enc

    def onnx_stream_call(self, wav_path, model_dir=''):
        import onnxruntime
        picker = onnxruntime.InferenceSession(os.path.join(model_dir, 'picker.onnx'), providers=[
            'CPUExecutionProvider'])  # ['CUDAExecutionProvider', 'CPUExecutionProvider']
        decoder = onnxruntime.InferenceSession(os.path.join(model_dir, 'decoder.onnx'),
                                               providers=['CPUExecutionProvider'])
        pick_inps = picker.get_inputs()
        pick_outs = picker.get_outputs()
        pick_outs = [i.name for i in pick_outs]
        picker_inp_states = {}
        for inp in pick_inps:
            picker_inp_states[inp.name] = np.zeros(self.get_shape(inp.shape), 'float32')

        dec_inps = decoder.get_inputs()
        dec_outs = decoder.get_outputs()
        dec_outs = [i.name for i in dec_outs]
        decoder_inp_states = {}
        for inp in dec_inps:
            decoder_inp_states[inp.name] = np.zeros(self.get_shape(inp.shape), 'float32')

        data = self.speech_featurizer.load_wav(wav_path)
        data = data / np.abs(data.max())

        valid_txt_outs = tf.zeros([1, 0, self.text_featurizer.num_classes])
        unvalid_txt_outs = tf.zeros([1, 0, self.text_featurizer.num_classes])
        txt_outs_offline = self.runner.predict(data.reshape([1, -1, 1]))

        for i in range(99999):
            s = i * self.wav_buf_length
            e = s + self.wav_buf_length
            if s >= len(data):
                break
            ss = time.time()
            input_wav = data[int(s): int(e)]
            input_wav = input_wav.reshape([1, -1, 1])
            picker_inp_states['input_wav'] = input_wav

            # valid_ctc_out, unvalid_ctc_out, valid_hidden_out, front_wav_cache, front_sub_cache, encoder_mha_cache, encoder_cnn_cache, picker_mha_cache, picker_cnn_cache, dec_inp
            picker_outputs = picker.run(pick_outs, input_feed=picker_inp_states)
            valid_phone_out, unvalid_phone_out, valid_hidden_out = picker_outputs[:3]
            pick_caches = picker_outputs[3:]
            for state, inp in zip(pick_caches, pick_inps[1:]):
                picker_inp_states[inp.name] = state

            if valid_hidden_out.shape[1] == 0:
                continue
            else:
                feature_outputs = self.pick_feature(valid_hidden_out, valid_phone_out)
                if len(feature_outputs) > 0:
                    # valid_ctc_out, unvalid_ctc_out, helper_mha_cache, helper_cnn_cache, decoder_mha_cache, decoder_cnn_cache, dec_inp
                    decoder_inp_states['valid_enc_out'] = feature_outputs
                    decoder_outputs = decoder.run(dec_outs, decoder_inp_states)
                    valid_ctc_out, unvalid_txt_outs = decoder_outputs[:2]
                    dec_caches = decoder_outputs[2:]
                    valid_txt_outs = tf.concat([valid_txt_outs, valid_ctc_out], axis=1)
                    for state, inp in zip(dec_caches, dec_inps[1:]):
                        decoder_inp_states[inp.name] = state

            txt_output = tf.concat([valid_txt_outs, unvalid_txt_outs], axis=1)

            if txt_output.shape[1] == 0:
                continue
            txt_output = tf.nn.softmax(txt_output, -1)
            input_length = np.array([txt_output.shape[1]], "int32")
            ctc_decode = tf.keras.backend.ctc_decode(txt_output, input_length)[0][0]
            ctc_decode = tf.cast(
                tf.clip_by_value(ctc_decode, 0, self.text_featurizer.num_classes),
                tf.int32,
            )

            ctc_result = []
            for n in ctc_decode[0].numpy():
                if n != 0:
                    ctc_result.append(n)

            text = self.text_featurizer.iextract(ctc_result)

            ee = time.time()
            print('time:', e / 16000, 'inference cost:', ee - ss)
            print("onnx stream texts out:", text)

        ctc_output = tf.nn.softmax(txt_outs_offline, -1)
        input_length = np.array([ctc_output.shape[1]], "int32")
        ctc_decode = tf.keras.backend.ctc_decode(ctc_output, input_length)[0][0]
        ctc_decode = tf.cast(
            tf.clip_by_value(ctc_decode, 0, self.text_featurizer.num_classes),
            tf.int32,
        )
        ctc_result = []
        for n in ctc_decode[0].numpy():
            if n != 0:
                ctc_result.append(n)
        texts = self.text_featurizer.iextract(ctc_result)

        print("tensorflow texts out:", texts)



if __name__ == "__main__":
    import time

    # USE CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # USE one GPU:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # limit cpu to 1 core:
    # import tensorflow as tf
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)

    am_config = UserConfig(
        r"./chunk_conformer-logs/am_data.yml", r"./chunk_conformer-logs/chunk_conformerS.yml"
    )
    asr = ASR(am_config)

    asr.stream_call('./asr/BAC009S0764W0121.wav')
    print('convert to onnx...')
    asr.convert_to_onnx('./onnx_models')
    print('convert success....')
    print('do onnx stream test....')
    asr.onnx_stream_call('./asr/BAC009S0764W0121.wav','./onnx_models')
