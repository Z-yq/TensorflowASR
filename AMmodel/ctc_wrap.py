
import numpy as np
import tensorflow as tf
from AMmodel.layers.time_frequency import Spectrogram,Melspectrogram
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
        self.fc = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=num_classes, activation="linear",
                                  use_bias=True), name="fully_connected")
        self.recognize_pb=None
    def _build(self, sample_shape):
        features = tf.random.normal(shape=sample_shape)
        self(features, training=False)

    def summary(self, line_length=None, **kwargs):
        self.encoder.summary(line_length=line_length, **kwargs)
        super(CtcModel, self).summary(line_length, **kwargs)

    def add_featurizers(self,
                        text_featurizer: TextFeaturizer,
                        ):

        self.text_featurizer = text_featurizer


    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        if self.mel_layer is not None:
            inputs=self.mel_layer(inputs)
            # print(inputs.shape)
        outputs = self.encoder(inputs, training=training)
        outputs = self.fc(outputs, training=training)
        return outputs

    def return_pb_function(self,shape,beam=False):
        @tf.function(
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape, dtype=tf.float32),
                tf.TensorSpec([None,1], dtype=tf.int32),

            ]
        )
        def recognize_tflite(features,length):

            logits = self.call(features, training=False)

            probs = tf.nn.softmax(logits)
            decoded = tf.keras.backend.ctc_decode(
                y_pred=probs, input_length=tf.squeeze(length,-1), greedy=True
            )[0][0]
            return [decoded]

        @tf.function(
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape, dtype=tf.float32),
                tf.TensorSpec([None, 1], dtype=tf.int32),
            ]
        )
        def recognize_beam_tflite( features,length):

            logits = self.call(features, training=False)

            probs = tf.nn.softmax(logits)
            decoded = tf.keras.backend.ctc_decode(
                y_pred=probs, input_length=tf.squeeze(length,-1), greedy=False,
                beam_width=self.text_featurizer.decoder_config["beam_width"]
            )[0][0]
            return [decoded]

        self.recognize_pb =recognize_tflite if not beam else recognize_beam_tflite



    def get_config(self):
        if self.mel_layer is not None:
            config=self.mel_layer.get_config()
            config.update(self.encoder.get_config())
        else:
            config = self.encoder.get_config()
        config.update(self.fc.get_config())
        return config
class CtcE2eModel(tf.keras.Model):
    def __init__(self,encoder: tf.keras.Model,
                 num_classes: int,
                 lm_embedding: int,
                 speech_config,
                 name="ctc_model",
                 **kwargs):
        super(CtcE2eModel, self).__init__()
        self.encoder = encoder
        self.speech_config = speech_config
        self.mel_layer = None
        if speech_config['use_mel_layer']:
            if speech_config['mel_layer_type'] == 'Melspectrogram':
                self.mel_layer = Melspectrogram(sr=speech_config['sample_rate'],
                                                n_mels=speech_config['num_feature_bins'],
                                                n_hop=int(
                                                    speech_config['stride_ms'] * speech_config['sample_rate'] // 1000),
                                                n_dft=1024,
                                                trainable_fb=speech_config['trainable_kernel']
                                                )
            else:
                self.mel_layer = Spectrogram(
                    n_hop=int(speech_config['stride_ms'] * speech_config['sample_rate'] // 1000),
                    n_dft=1024,
                    trainable_kernel=speech_config['trainable_kernel']
                )
        self.lm_projecter=tf.keras.layers.Dense(lm_embedding)
        self.fc = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=num_classes, activation="linear",
                                  use_bias=True), name="fully_connected")
        self.recognize_pb = None

    def _build(self, sample_shape):
        features = tf.random.normal(shape=sample_shape)
        self(features, training=False)

    def summary(self, line_length=None, **kwargs):
        self.encoder.summary(line_length=line_length, **kwargs)
        super(CtcE2eModel, self).summary(line_length, **kwargs)

    def add_featurizers(self,
                        text_featurizer: TextFeaturizer,
                        ):

        self.text_featurizer = text_featurizer


    def call(self, inputs, training=False, **kwargs):
        if self.mel_layer is not None:
            inputs = self.mel_layer(inputs)
            # print(inputs.shape)
        encoder_outputs = self.encoder(inputs, training=training)
        lm_embedding=self.lm_projecter(encoder_outputs)
        outputs = self.fc(lm_embedding, training=training)
        return outputs,lm_embedding

    def return_pb_function(self, shape, beam=False):
        @tf.function(
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape, dtype=tf.float32),
                tf.TensorSpec([None, 1], dtype=tf.int32),

            ]
        )
        def recognize_tflite(features, length):
            logits,_ = self.call(features, training=False)

            probs = tf.nn.softmax(logits)
            decoded = tf.keras.backend.ctc_decode(
                y_pred=probs, input_length=tf.squeeze(length, -1), greedy=True
            )[0][0]
            return [decoded]

        @tf.function(
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape, dtype=tf.float32),
                tf.TensorSpec([None, 1], dtype=tf.int32),
            ]
        )
        def recognize_beam_tflite(features, length):
            logits, _ = self.call(features, training=False)

            probs = tf.nn.softmax(logits)
            decoded = tf.keras.backend.ctc_decode(
                y_pred=probs, input_length=tf.squeeze(length, -1), greedy=False,
                beam_width=self.text_featurizer.decoder_config["beam_width"]
            )[0][0]
            return [decoded]

        self.recognize_pb = recognize_tflite if not beam else recognize_beam_tflite

    def get_config(self):
        if self.mel_layer is not None:
            config = self.mel_layer.get_config()
            config.update(self.encoder.get_config())
        else:
            config = self.encoder.get_config()
        config.update(self.fc.get_config())
        return config