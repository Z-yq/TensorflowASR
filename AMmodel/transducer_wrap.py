

import os
import collections
import tensorflow as tf

from utils.tools import shape_list, get_shape_invariants
from utils.text_featurizers import TextFeaturizer



class TransducerPrediction(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 embed_dim: int,
                 embed_dropout: float = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 512,
                 name="transducer_prediction",
                 **kwargs):
        super(TransducerPrediction, self).__init__(name=name, **kwargs)
        self.embed = tf.keras.layers.Embedding(
            input_dim=vocabulary_size, output_dim=embed_dim, mask_zero=False)
        self.do = tf.keras.layers.Dropout(embed_dropout)
        self.lstm_cells = []
        # lstms units must equal (for using beam search)
        for i in range(num_lstms):
            lstm = tf.keras.layers.LSTMCell(units=lstm_units,
                                        )
            self.lstm_cells.append(lstm)
        self.decoder_lstms = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            self.lstm_cells, name="decoder_lstms"
        ),return_sequences=True,return_state=True)

    def get_initial_state(self, input_sample):

        return self.decoder_lstms.get_initial_state(input_sample)

    # @tf.function(experimental_relax_shapes=True)
    def call(self,
             inputs,
             training=False,
             p_memory_states=None,
             **kwargs):
        # inputs has shape [B, U]
        outputs = self.embed(inputs, training=training)
        outputs = self.do(outputs, training=training)
        if p_memory_states is None:  # Zeros mean no initial_state
            p_memory_states = self.get_initial_state(outputs)
        # n_memory_states = []
        # for i, lstm in enumerate(self.lstms):
        outputs = self.decoder_lstms(outputs, training=training, initial_state=p_memory_states)
        new_memory_states = outputs[1:]
        outputs = outputs[0]
        # n_memory_states.append(new_memory_states)

        # return shapes [B, T, P], ([num_lstms, B, P], [num_lstms, B, P]) if using lstm
        return outputs, new_memory_states

    def get_config(self):
        conf = super(TransducerPrediction, self).get_config()
        conf.update(self.embed.get_config())
        conf.update(self.do.get_config())
        for lstm in self.lstms:
            conf.update(lstm.get_config())
        return conf


class TransducerJoint(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 joint_dim: int = 1024,
                 name="tranducer_joint",
                 **kwargs):
        super(TransducerJoint, self).__init__(name=name, **kwargs)
        self.ffn_enc = tf.keras.layers.Dense(joint_dim)
        self.ffn_pred = tf.keras.layers.Dense(joint_dim)
        self.ffn_out = tf.keras.layers.Dense(vocabulary_size)

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        # enc has shape [B, T, E]
        # pred has shape [B, U, P]
        enc, pred = inputs
        enc_out = self.ffn_enc(enc, training=training)  # [B, T ,E] => [B, T, V]
        pred_out = self.ffn_pred(pred, training=training)  # [B, U, P] => [B, U, V]
        # => [B, T, U, V]
        outputs = tf.nn.tanh(tf.expand_dims(enc_out, axis=2) + tf.expand_dims(pred_out, axis=1))
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(TransducerJoint, self).get_config()
        conf.update(self.ffn_enc.get_config())
        conf.update(self.ffn_pred.get_config())
        conf.update(self.ffn_out.get_config())
        return conf


class Transducer(tf.keras.Model):
    """ Transducer Model Warper """

    def __init__(self,
                 encoder: tf.keras.Model,
                 vocabulary_size: int,
                 embed_dim: int = 512,
                 embed_dropout: float = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 320,
                 joint_dim: int = 1024,
                 name="transducer",
                 **kwargs):
        super(Transducer, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.predict_net = TransducerPrediction(
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_lstms=num_lstms,
            lstm_units=lstm_units,
            name=f"{name}_prediction"
        )
        self.joint_net = TransducerJoint(
            vocabulary_size=vocabulary_size,
            joint_dim=joint_dim,
            name=f"{name}_joint"
        )
        self.kept_hyps = None
        self.startid=0
        self.endid=1
        self.max_iter=10
    def _build(self, sample_shape):  # Call on real data for building model
        features = tf.random.normal(shape=sample_shape)
        predicted = tf.constant([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        return  self([features, predicted], training=True)

    def save_seperate(self, path_to_dir: str):
        self.encoder.save(os.path.join(path_to_dir, "encoder"))
        self.predict_net.save(os.path.join(path_to_dir, "prediction"))
        self.joint_net.save(os.path.join(path_to_dir, "joint"))

    def summary(self, line_length=None, **kwargs):
        self.encoder.summary(line_length=line_length, **kwargs)
        self.predict_net.summary(line_length=line_length, **kwargs)
        self.joint_net.summary(line_length=line_length, **kwargs)
        super(Transducer, self).summary(line_length=line_length, **kwargs)

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        """
        Transducer Model call function
        Args:
            features: audio features in shape [B, T, F, C]
            predicted: predicted sequence of character ids, in shape [B, U]
            training: python boolean
            **kwargs: sth else

        Returns:
            `logits` with shape [B, T, U, vocab]
        """
        features, predicted = inputs
        enc = self.encoder(features, training=training)
        pred, _ = self.predict_net(predicted, training=training)
        outputs = self.joint_net([enc, pred], training=training)

        return outputs
    #




    def add_featurizers(self,
                        text_featurizer: TextFeaturizer):
        """
        Function to add featurizer to model to convert to end2end tflite
        Args:
            speech_featurizer: SpeechFeaturizer instance
            text_featurizer: TextFeaturizer instance
            scorer: external language model scorer
        """

        self.text_featurizer = text_featurizer

    def return_pb_function(self,f,c):
        @tf.function(experimental_relax_shapes=True, input_signature=[
            tf.TensorSpec([None, None, f, c], dtype=tf.float32),  # features
            tf.TensorSpec([None, 1], dtype=tf.int32),  # features

        ])
        def recognize_pb(features, length, training=False):
            enc = self.encoder(features, training=training)
            batch = tf.shape(enc)[0]

            decoded = tf.zeros([batch, 1], tf.int32)
            stop_flag = tf.zeros([batch, ], tf.float32)
            b_i = 0
            B = self.max_iter

            def _cond(b_i, B, stop_flag, decoded):
                return tf.less(b_i, B)

            def _body(b_i, B, stop_flag, decoded):
                pred, _ = self.predict_net(decoded, training=training)
                outputs = self.joint_net([enc, pred], training=training)
                outputs = tf.reduce_sum(outputs, 2)
                outputs = tf.nn.softmax(outputs, -1)
                length_ = tf.squeeze(length, -1)
                ctc_out = tf.keras.backend.ctc_decode(outputs, length_)[0][0]
                step_flag = tf.cast(tf.reduce_any(tf.equal(ctc_out, self.endid), -1), 1)
                stop_flag += step_flag
                ctc_out = tf.cast(tf.clip_by_value(ctc_out, 0, self.text_featurizer.blank), tf.int32)
                ctc_out_value = tf.where(ctc_out == 0, 1, 0) * self.text_featurizer.blank
                ctc_out += tf.cast(ctc_out_value, tf.int32)

                hyps = tf.cond(
                    tf.reduce_all(tf.cast(stop_flag, tf.bool)),
                    true_fn=lambda: B,
                    false_fn=lambda: b_i + 1,
                )
                decoded = tf.pad(ctc_out, [[0, 0], [1, 0]])
                return hyps, B, stop_flag, decoded

            _, _, stop_flag, decoded = tf.while_loop(
                _cond,
                _body,
                loop_vars=(b_i, B, stop_flag, decoded),
                shape_invariants=(
                    tf.TensorShape([1]),
                    tf.TensorShape([1]),
                    tf.TensorShape([None, ]),
                    tf.TensorShape([None, None]),

                )
            )
            return [decoded]

        self.recognize_pb= recognize_pb



    def get_config(self):
        conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf
