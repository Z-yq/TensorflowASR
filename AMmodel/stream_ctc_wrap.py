
import tensorflow as tf
from AMmodel.layers.time_frequency import Spectrogram,Melspectrogram


import numpy as np
try:
    from ctc_decoders import ctc_greedy_decoder, ctc_beam_search_decoder
    USE_TF=0
except:
    USE_TF=1

from utils.text_featurizers import TextFeaturizer



class CtcModel(tf.keras.Model):
    def __init__(self,
                 encoder: tf.keras.Model,
                 num_classes: int,
                 speech_config,
                 name="ctc_model",

                 **kwargs):
        super(CtcModel, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        # Fully connected layer
        self.dmodel=encoder.dmodel
        self.cell_nums=encoder.cell_nums
        self.speech_config=speech_config
        self.mel_layer=None
        if speech_config['use_mel_layer']:
            if speech_config['mel_layer_type']=='Melspectrogram':
                self.mel_layer=Melspectrogram(sr=speech_config['sample_rate'],n_mels=speech_config['num_feature_bins'],
                                              n_hop=int(speech_config['stride_ms']*speech_config['sample_rate']//1000),
                                              n_dft=1024,
                                              trainable_fb=speech_config['trainable_kernel']
                                              )
            else:
                self.mel_layer = Spectrogram(
                                                n_hop=int(speech_config['stride_ms'] * speech_config['sample_rate']//1000),
                                                n_dft=1024,
                                                trainable_kernel=speech_config['trainable_kernel']
                                                )
            self.mel_layer.trainable=speech_config['trainable_kernel']
        self.wav_info=speech_config['add_wav_info']
        self.chuck_size=int(self.speech_config['sample_rate']*self.speech_config['streaming_bucket'])
        if self.wav_info:
            assert speech_config['use_mel_layer']==True,'shold set use_mel_layer is True'

        self.fc = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=num_classes, activation="linear",
                                  use_bias=True), name="fully_connected")

        self.recognize_pb=None
    def _build(self, shape):


        inputs = np.random.normal(size=shape).astype(np.float32)

        self(inputs)




    def summary(self, line_length=None, **kwargs):
        self.encoder.summary(line_length=line_length, **kwargs)
        super(CtcModel, self).summary(line_length, **kwargs)

    def add_featurizers(self,
                        text_featurizer: TextFeaturizer,
                        ):

        self.text_featurizer = text_featurizer


    def call(self,x, training=False, **kwargs):
        features=x
        B = tf.shape(features)[0]
        features = tf.reshape(features, [-1, self.chuck_size, 1])

        if self.mel_layer is not None:
            if self.wav_info:
                wav = features
                wav = tf.reshape(wav, [B, -1, self.chuck_size, 1])
                features = self.mel_layer(features)
                D = tf.shape(features)[1]
                V = tf.shape(features)[2]
                features = tf.reshape(features, [B, -1, D, V, 1])
            else:
                features = self.mel_layer(features)
                D = tf.shape(features)[1]
                V = tf.shape(features)[2]
                features = tf.reshape(features, [B, -1, D, V, 1])
        if self.wav_info:
            enc, _ = self.encoder([features, wav], training=training)
        else:
            enc, _= self.encoder(features, training=training)

        V =self.dmodel

        enc = tf.reshape(enc, [B, -1, V])
        outputs = self.fc(enc, training=training)
        return outputs
    def inference(self,inputs,states):

        features=inputs
        if self.mel_layer is not None:
            if self.wav_info:
                wav = features
                features = self.mel_layer(features)

            else:
                features = self.mel_layer(features)

        if self.wav_info:
            enc, states = self.encoder.inference([features, wav],states)
        else:
            enc, states = self.encoder.inference(features,states)

        outputs = self.fc(enc, training=False)

        return outputs,states
    def initial_states(self,inputs):
        states=self.encoder.get_init_states(inputs)
        return states
    def return_pb_function(self,shape):
        @tf.function(
            experimental_relax_shapes=True,
            # input_signature=[
            #     tf.TensorSpec(shape, dtype=tf.float32),
            #     tf.TensorSpec([None,1], dtype=tf.int32),
            # ]
        )
        def recognize_function(features,length,states):

            logits,states = self.inference(features,states)

            probs = tf.nn.softmax(logits)
            decoded = tf.keras.backend.ctc_decode(
                y_pred=probs, input_length=tf.squeeze(length,-1), greedy=True
            )[0][0]
            return decoded,states
        self.recognize_pb =recognize_function

    def get_config(self):
        if self.mel_layer is not None:
            config=self.mel_layer.get_config()
            config.update(self.encoder.get_config())
        else:
            config = self.encoder.get_config()
        config.update(self.fc.get_config())
        return config
