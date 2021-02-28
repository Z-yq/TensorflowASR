
"""
Read https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
to use cuDNN-LSTM
"""
import numpy as np
import tensorflow as tf

from utils.tools import append_default_keys_dict, get_rnn
from AMmodel.layers.row_conv_1d import RowConv1D
from AMmodel.layers.sequence_wise_batch_norm import SequenceBatchNorm
from AMmodel.layers.transpose_time_major import TransposeTimeMajor
from AMmodel.layers.merge_two_last_dims import Merge2LastDims
from AMmodel.ctc_wrap import CtcModel
from AMmodel.las_wrap import LAS,LASConfig
from AMmodel.transducer_wrap import Transducer
from AMmodel.wav_model import WavePickModel
DEFAULT_CONV = {
    "conv_type": 2,
    "conv_kernels": ((11, 41), (11, 21), (11, 21)),
    "conv_strides": ((2, 2), (1, 2), (1, 2)),
    "conv_filters": (32, 32, 96),
    "conv_dropout": 0.2
}

DEFAULT_RNN = {
    "rnn_layers": 3,
    "rnn_type": "gru",
    "rnn_units": 350,
    "rnn_activation": "tanh",
    "rnn_bidirectional": True,
    "rnn_rowconv": False,
    "rnn_rowconv_context": 2,
    "rnn_dropout": 0.2
}

DEFAULT_FC = {
    "fc_units": (1024,),
    "fc_dropout": 0.2
}
class DeepSpeech2(tf.keras.Model):
    def __init__(self,arch_config,**kwargs):
        super(DeepSpeech2, self).__init__()
        conv_conf = append_default_keys_dict(DEFAULT_CONV, arch_config.get("conv_conf", {}))
        rnn_conf = append_default_keys_dict(DEFAULT_RNN, arch_config.get("rnn_conf", {}))
        fc_conf = append_default_keys_dict(DEFAULT_FC, arch_config.get("fc_conf", {}))
        assert len(conv_conf["conv_strides"]) == \
               len(conv_conf["conv_filters"]) == len(conv_conf["conv_kernels"])
        assert conv_conf["conv_type"] in [1, 2]
        assert rnn_conf["rnn_type"] in ["lstm", "gru", "rnn"]
        assert conv_conf["conv_dropout"] >= 0.0 and rnn_conf["rnn_dropout"] >= 0.0
        layer=[]
        if conv_conf["conv_type"] == 2:
            conv = tf.keras.layers.Conv2D
        else:
            layer += [Merge2LastDims("conv1d_features")]
            conv = tf.keras.layers.Conv1D
            ker_shape = np.shape(conv_conf["conv_kernels"])
            stride_shape = np.shape(conv_conf["conv_strides"])
            filter_shape = np.shape(conv_conf["conv_filters"])
            assert len(ker_shape) == 1 and len(stride_shape) == 1 and len(filter_shape) == 1
        for i, fil in enumerate(conv_conf["conv_filters"]):
            layer += [conv(filters=fil, kernel_size=conv_conf["conv_kernels"][i],
                           strides=conv_conf["conv_strides"][i], padding="same",
                           activation=None, dtype=tf.float32, name=f"cnn_{i}")]
            layer += [tf.keras.layers.BatchNormalization(name=f"cnn_bn_{i}")]
            layer += [tf.keras.layers.ReLU(name=f"cnn_relu_{i}")]
            layer += [tf.keras.layers.Dropout(conv_conf["conv_dropout"],
                                              name=f"cnn_dropout_{i}")]
        last_dim=fil
        if conv_conf["conv_type"] == 2:
            layer += [Merge2LastDims("reshape_conv2d_to_rnn")]
        layer+=[tf.keras.layers.Dense(last_dim,name='feature_projector')]
        self.Cnn_feature_extractor=tf.keras.Sequential(layer)

        self.add_wav_info=kwargs['add_wav_info']
        if kwargs['add_wav_info']:
            hop_size=kwargs['hop_size']
            for i, fil in enumerate(conv_conf["conv_filters"]):
                hop_size*=conv_conf["conv_strides"][i][0]
            self.wav_layer=WavePickModel(last_dim,hop_size)
        layer=[]
        rnn = get_rnn(rnn_conf["rnn_type"])

        # To time major
        if rnn_conf["rnn_bidirectional"]:
            layer += [TransposeTimeMajor("transpose_to_time_major")]

        # RNN layers
        for i in range(rnn_conf["rnn_layers"]):
            if rnn_conf["rnn_bidirectional"]:
                layer += [tf.keras.layers.Bidirectional(
                    rnn(rnn_conf["rnn_units"], activation=rnn_conf["rnn_activation"],
                        time_major=True, dropout=rnn_conf["rnn_dropout"],
                        return_sequences=True, use_bias=True),
                    name=f"b{rnn_conf['rnn_type']}_{i}")]
                layer += [SequenceBatchNorm(time_major=True, name=f"sequence_wise_bn_{i}")]
            else:
                layer += [rnn(rnn_conf["rnn_units"], activation=rnn_conf["rnn_activation"],
                              dropout=rnn_conf["rnn_dropout"], return_sequences=True, use_bias=True,
                              name=f"{rnn_conf['rnn_type']}_{i}")]
                layer += [SequenceBatchNorm(time_major=False, name=f"sequence_wise_bn_{i}")]
                if rnn_conf["rnn_rowconv"]:
                    layer += [RowConv1D(filters=rnn_conf["rnn_units"],
                                        future_context=rnn_conf["rnn_rowconv_context"],
                                        name=f"row_conv_{i}")]

        # To batch major
        if rnn_conf["rnn_bidirectional"]:
            layer += [TransposeTimeMajor("transpose_to_batch_major")]

        # FC Layers
        if fc_conf["fc_units"]:
            assert fc_conf["fc_dropout"] >= 0.0

            for idx, units in enumerate(fc_conf["fc_units"]):
                layer += [tf.keras.layers.Dense(units=units, activation=None,
                                                use_bias=True, name=f"hidden_fc_{idx}")]
                layer += [tf.keras.layers.BatchNormalization(name=f"hidden_fc_bn_{idx}")]
                layer += [tf.keras.layers.ReLU(name=f"hidden_fc_relu_{idx}")]
                layer += [tf.keras.layers.Dropout(fc_conf["fc_dropout"],
                                                  name=f"hidden_fc_dropout_{idx}")]
        self.Rnn_feature_extractor=tf.keras.Sequential(layer)
        self.dmodel=fc_conf["fc_units"][-1]
    def call(self,inputs,training=True):
        if self.add_wav_info:
            mel,wav=inputs
            wav_outputs=self.wav_layer(wav,training=training)
            mel_outputs=self.Cnn_feature_extractor(mel,training=training)
            outputs=mel_outputs+wav_outputs
        else:
            outputs=self.Cnn_feature_extractor(inputs,training=training)
        outputs=self.Rnn_feature_extractor(outputs,training=training)
        return outputs
    def get_config(self):
        conf = super(DeepSpeech2, self).get_config()
        if self.add_wav_info:
            conf.update(self.wav_layer.get_config())
        conf.update(self.Cnn_feature_extractor.get_config())
        conf.update(self.Rnn_feature_extractor.get_config())
        return conf


class DeepSpeech2CTC(CtcModel):
    def __init__(self,
                 input_shape: list,
                 arch_config: dict,
                 num_classes: int,
                 name: str = "deepspeech2",
                 speech_config=dict):
        super(DeepSpeech2CTC, self).__init__(
            encoder=DeepSpeech2( arch_config=arch_config,
                                  name=name,
                                 add_wav_info=speech_config['add_wav_info'],
                                 hop_size=int(speech_config['stride_ms'] * speech_config['sample_rate'] // 1000)

                                 ),
            num_classes=num_classes,
            name=f"{name}_ctc",
            speech_config=speech_config
        )
        self.time_reduction_factor = 1
        for s in arch_config["conv_conf"]["conv_strides"]:
            self.time_reduction_factor *= s[0]
class DeepSpeech2LAS(LAS):
    def __init__(self,
                 config,
                 input_shape: list,
                 training,
                 name: str = "LAS",
                 enable_tflite_convertible=False,
                 speech_config=dict):
        config['LAS_decoder'].update({'encoder_dim': config['fc_conf']['fc_units'][-1]})
        decoder_config = LASConfig(**config['LAS_decoder'])

        super(DeepSpeech2LAS, self).__init__(
            encoder=DeepSpeech2(arch_config=config,
                                name=name,
                                add_wav_info=speech_config['add_wav_info'],
                                hop_size=int(speech_config['stride_ms'] * speech_config['sample_rate'] // 1000)

                                ),
            config=decoder_config, training=training,enable_tflite_convertible=enable_tflite_convertible,
            speech_config=speech_config
        )
        self.time_reduction_factor = 1
        for s in config["conv_conf"]["conv_strides"]:
            self.time_reduction_factor *= s[0]
class DeepSpeech2Transducer(Transducer):
    def __init__(self,
                 input_shape: list,
                 config,
                 name: str = "deepspeech2",
                 speech_config=dict):

        super(DeepSpeech2Transducer, self).__init__(
            encoder=DeepSpeech2(arch_config=config,
                                name=name,
                                add_wav_info=speech_config['add_wav_info'],
                                hop_size=int(speech_config['stride_ms'] * speech_config['sample_rate'] // 1000)

                                ),
            vocabulary_size=config['Transducer_decoder']['vocabulary_size'],
            embed_dim=config['Transducer_decoder']['embed_dim'],
            embed_dropout=config['Transducer_decoder']['embed_dropout'],
            num_lstms=config['Transducer_decoder']['num_lstms'],
            lstm_units=config['Transducer_decoder']['lstm_units'],
            joint_dim=config['Transducer_decoder']['joint_dim'],
            name=name+'_transducer',
            speech_config=speech_config
        )
        self.time_reduction_factor = 1
        for s in  config["conv_conf"]["conv_strides"]:
            self.time_reduction_factor *= s[0]
