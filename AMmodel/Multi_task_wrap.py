import collections
import numpy as np
import tensorflow as tf
from AMmodel.layers.time_frequency import Spectrogram,Melspectrogram




class MultiTaskCTC(tf.keras.Model):

    def __init__(self, encoder1,encoder2,encoder3,classes1,classes2,classes3,dmodel,speech_config=dict, **kwargs):
        super().__init__(self, **kwargs)
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3
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
            self.mel_layer.trainable = speech_config['trainable_kernel']
        self.wav_info = speech_config['add_wav_info']
        if self.wav_info:
            assert speech_config['use_mel_layer'] == True, 'shold set use_mel_layer is True'
        self.fc1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=classes1, activation="linear",
                                  use_bias=True), name="fully_connected1")

        self.fc2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=classes2, activation="linear",
                                  use_bias=True), name="fully_connected2")

        self.fc3 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=classes3, activation="linear",
                                  use_bias=True), name="fully_connected3")

        self.fc_to_project_1=tf.keras.layers.Dense(dmodel,name='word_prob_projector')
        self.fc_to_project_2=tf.keras.layers.Dense(dmodel,name='phone_prob_projector')
        self.fc_to_project_3=tf.keras.layers.Dense(dmodel,name='py_prob_projector')
        self.fc_final_class = tf.keras.layers.Conv1D(
            classes3,32, padding='same',name="cnn_final_class")

    def setup_window(self, win_front, win_back):
        """Call only for inference."""
        self.use_window_mask = True
        self.win_front = win_front
        self.win_back = win_back

    def setup_maximum_iterations(self, maximum_iterations):
        """Call only for inference."""
        self.maximum_iterations = maximum_iterations

    def _build(self, sample_shape):
        features = tf.random.normal(shape=sample_shape)
        self(features, training=False)

    def summary(self, line_length=None, **kwargs):
        # self.encoder.summary(line_length=line_length, **kwargs)
        super(MultiTaskCTC, self).summary(line_length, **kwargs)

    def add_featurizers(self,
                        text_featurizer,
                        ):

        self.text_featurizer = text_featurizer

        # @tf.function(experimental_relax_shapes=True)
    def encoder1_enc(self,inputs,training=False):
        if self.mel_layer is not None:
            if self.wav_info:
                wav = inputs
                inputs = self.mel_layer(inputs)
            else:
                inputs = self.mel_layer(inputs)
            # print(inputs.shape)
        if self.wav_info:
            enc_outputs = self.encoder1([inputs, wav], training=training)
        else:
            enc_outputs = self.encoder1(inputs, training=training)
        outputs = self.fc1(enc_outputs[-1], training=training)
        for i in range(12, 15):
            outputs += self.fc1(enc_outputs[i], training=training)
        return enc_outputs[-1],outputs

    def encoder2_enc(self, inputs,enc1, training=False):
        if self.mel_layer is not None:
            if self.wav_info:
                wav = inputs
                inputs = self.mel_layer(inputs)
            else:
                inputs = self.mel_layer(inputs)
            # print(inputs.shape)
        if self.wav_info:
            enc_outputs = self.encoder2([inputs, wav,enc1], training=training)
        else:
            enc_outputs = self.encoder2([inputs,enc1], training=training)
        outputs = self.fc2(enc_outputs[-1], training=training)
        for i in range(12, 15):
            outputs += self.fc2(enc_outputs[i], training=training)
        return enc_outputs[-1], outputs
    def encoder3_enc(self, inputs, enc2,training=False):
        if self.mel_layer is not None:
            if self.wav_info:
                wav = inputs
                inputs = self.mel_layer(inputs)
            else:
                inputs = self.mel_layer(inputs)
            # print(inputs.shape)
        if self.wav_info:
            enc_outputs = self.encoder3([inputs, wav,enc2], training=training)
        else:
            enc_outputs = self.encoder3([inputs,enc2], training=training)
        outputs = self.fc3(enc_outputs[-1], training=training)
        for i in range(12, 15):
            outputs += self.fc3(enc_outputs[i], training=training)
        return enc_outputs[-1], outputs
    def call(self, inputs, training=False, **kwargs):
        enc1,outputs1=self.encoder1_enc(inputs,training)
        enc2,outputs2=self.encoder2_enc(inputs,enc1,training)
        enc3,outputs3=self.encoder3_enc(inputs,enc2,training)
        outputs1_=self.fc_to_project_1(outputs1)
        outputs2_=self.fc_to_project_2(outputs2)
        outputs3_=self.fc_to_project_3(outputs3)
        output=outputs1_+outputs2_+outputs3_+enc1+enc2+enc3
        outputs=self.fc_final_class(output)
        outputs+=outputs3
        return outputs1,outputs2,outputs3,outputs

    def return_pb_function(self, shape, beam=False):
        @tf.function(
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape, dtype=tf.float32),
                tf.TensorSpec([None, 1], dtype=tf.int32),

            ]
        )
        def recognize_tflite(features, length):
            logits = self.call(features, training=False)[-1]

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
            logits = self.call(features, training=False)[-1]

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



