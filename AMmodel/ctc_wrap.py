
import numpy as np
import tensorflow as tf
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
                 name="ctc_model",
                 **kwargs):
        super(CtcModel, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        # Fully connected layer
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
        outputs = self.encoder(inputs, training=training)
        outputs = self.fc(outputs, training=training)
        return outputs

    def return_pb_function(self,f,c,beam=False):
        @tf.function(
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec([None,None,f,c], dtype=tf.float32),
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
                tf.TensorSpec([None, None, f, c], dtype=tf.float32),
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
        config = self.encoder.get_config()
        config.update(self.fc.get_config())
        return config
