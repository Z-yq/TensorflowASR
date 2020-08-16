
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

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([None, None, 80,4], dtype=tf.float32)
        ]
    )
    def recognize(self, features):
        logits = self.call(features, training=False)
        probs = tf.nn.softmax(logits)
        if USE_TF:
            decoded = tf.keras.backend.ctc_decode(
                y_pred=probs, input_length=tf.expand_dims(tf.shape(probs)[1],0), greedy=True
            )[0]
            return decoded
        else:
            def map_fn(prob):
                return tf.numpy_function(self.perform_greedy,
                                         inp=[prob], Tout=tf.string)

            return [tf.map_fn(map_fn, probs, dtype=tf.string)]

    def perform_greedy(self, probs: np.ndarray):
        decoded = ctc_greedy_decoder(probs, vocabulary=self.text_featurizer.vocab_array)
        return tf.convert_to_tensor(decoded, dtype=tf.string)
    def return_pb_function(self,f,c,beam=False):
        @tf.function(
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec([None,None,f,c], dtype=tf.float32),

            ]
        )
        def recognize_tflite(features):
            """
            Function to convert to tflite using greedy decoding
            Args:
                features: tf.Tensor with shape [None,None,None,None] indicating a input

            Return:
                decoded: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
            """

            # features=self.speech_featurizer.extract(audio)

            logits = self.call(features, training=False)
            probs = tf.nn.softmax(logits)
            decoded = tf.keras.backend.ctc_decode(
                y_pred=probs, input_length=tf.expand_dims(tf.shape(probs)[1],0), greedy=True
            )[0][0]
            # transcript = self.text_featurizer.index2upoints(tf.cast(decoded[0][0], dtype=tf.int32))
            return decoded

        @tf.function(
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec([None, None, f, c], dtype=tf.float32),
            ]
        )
        def recognize_beam_tflite( features):

            logits = self.call(features, training=False)
            probs = tf.nn.softmax(logits)
            decoded = tf.keras.backend.ctc_decode(
                y_pred=probs, input_length=tf.expand_dims(tf.shape(probs)[1],0), greedy=False,
                beam_width=self.text_featurizer.decoder_config["beam_width"]
            )[0][0]
            # transcript = self.text_featurizer.index2upoints(tf.cast(decoded[0][0], dtype=tf.int32))
            return decoded

        self.recognize_pb =recognize_tflite if not beam else recognize_beam_tflite
        # transcript = self.text_featurizer.index2upoints(tf.cast(decoded[0][0], dtype=tf.int32))
        # return tf.squeeze(transcript, axis=0)

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([None, None, None, None], dtype=tf.float32),
            tf.TensorSpec([], dtype=tf.bool)
        ]
    )
    def recognize_beam(self, features, lm=False):
        logits = self.call(features, training=False)
        probs = tf.nn.softmax(logits)

        def map_fn(prob):
            return tf.numpy_function(self.perform_beam_search,
                                     inp=[prob, lm], Tout=tf.string)

        return tf.map_fn(map_fn, probs, dtype=tf.string)

    def perform_beam_search(self,
                            probs: np.ndarray,
                            lm: bool = False):
        decoded = ctc_beam_search_decoder(
            probs_seq=probs,
            vocabulary=self.text_featurizer.vocab_array,
            beam_size=self.text_featurizer.decoder_config["beam_width"],
            ext_scoring_func=self.text_featurizer.scorer if lm else None
        )
        decoded = decoded[0][-1]

        return tf.convert_to_tensor(decoded, dtype=tf.string)



    def get_config(self):
        config = self.encoder.get_config()
        config.update(self.fc.get_config())
        return config
