

import os
import collections
import tensorflow as tf
import collections
from utils.tools import shape_list, get_shape_invariants,merge_repeated
from utils.text_featurizers import TextFeaturizer
from AMmodel.layers.time_frequency import Melspectrogram,Spectrogram

Hypotheses = collections.namedtuple(
    "Hypotheses",
    ("scores", "yseqs", "p_memory_states")
)


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
                 name="transducer", speech_config=dict,
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
        self.speech_config = speech_config
        self.mel_layer = None
        if speech_config['use_mel_layer']:
            if speech_config['mel_layer_type'] == 'Melspectrogram':
                self.mel_layer = Melspectrogram(sr=speech_config['sample_rate'],
                                                n_mels=speech_config['num_feature_bins'],
                                                n_hop=int(speech_config['stride_ms'] * speech_config['sample_rate']//1000),
                                                n_dft=1024,
                                                trainable_fb=speech_config['trainable_kernel']
                                                )
            else:
                self.mel_layer = Spectrogram(
                                             n_hop=int(speech_config['stride_ms'] * speech_config['sample_rate']//1000),
                                             n_dft=1024,
                                             trainable_kernel=speech_config['trainable_kernel']
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

        features, predicted = inputs
        if self.mel_layer is not None:
            features=self.mel_layer(features)
        enc = self.encoder(features, training=training)
        pred, _ = self.predict_net(predicted, training=training)
        outputs = self.joint_net([enc, pred], training=training)

        return outputs

    def add_featurizers(self,
                        text_featurizer: TextFeaturizer):

        self.text_featurizer = text_featurizer

    def return_pb_function(self,shape):
        @tf.function(experimental_relax_shapes=True, input_signature=[
            tf.TensorSpec(shape, dtype=tf.float32),  # features
            tf.TensorSpec([None, 1], dtype=tf.int32),  # features

        ])
        def recognize_pb(features, length, training=False):
            decoded=self.perform_greedy(features)
            return [decoded]

        self.recognize_pb= recognize_pb

    @tf.function(experimental_relax_shapes=True)
    def perform_greedy(self,
                       features):
        batch = tf.shape(features)[0]
        new_hyps = Hypotheses(
            tf.zeros([batch],tf.float32),
            self.text_featurizer.start * tf.ones([batch, 1], dtype=tf.int32),
            self.predict_net.get_initial_state(features)
        )
        if self.mel_layer is not None:
            features=self.mel_layer(features)
        enc = self.encoder(features, training=False)  # [B, T, E]
        # enc = tf.squeeze(enc, axis=0)  # [T, E]
        stop_flag = tf.zeros([batch,1 ], tf.float32)
        T = tf.cast(shape_list(enc)[1], dtype=tf.int32)

        i = tf.constant(0, dtype=tf.int32)

        def _cond(enc, i, new_hyps, T, stop_flag):
            return tf.less(i, T)

        def _body(enc, i, new_hyps, T, stop_flag):
            hi = enc[:, i:i + 1]  # [B, 1, E]
            y, n_memory_states = self.predict_net(
                inputs=new_hyps[1][:,-1:],  # [1, 1]
                p_memory_states=new_hyps[2],
                training=False
            )  # [1, 1, P], [1, P], [1, P]
            # [1, 1, E] + [1, 1, P] => [1, 1, 1, V]
            ytu = tf.nn.log_softmax(self.joint_net([hi, y], training=False))
            ytu = tf.squeeze(ytu, axis=None)  # [B, 1, 1, V] => [B,V]
            n_predict = tf.expand_dims(tf.argmax(ytu, axis=-1, output_type=tf.int32),-1)  # => argmax []

            # print(stop_flag.shape,n_predict.shape)
            new_hyps =Hypotheses(new_hyps[0]+1,
            tf.concat([new_hyps[1], tf.reshape(n_predict,[-1,1])], -1),
             n_memory_states)

            stop_flag += tf.cast(tf.equal(tf.reshape(n_predict, [-1,1]), self.text_featurizer.stop), tf.float32)
            n_i = tf.cond(
                tf.reduce_all(tf.cast(stop_flag, tf.bool)),
                true_fn=lambda: T,
                false_fn=lambda: i + 1,
            )

            return enc, n_i, new_hyps, T,stop_flag

        _, _, new_hyps, _, stop_flag = tf.while_loop(
            _cond,
            _body,
            loop_vars=(enc, i, new_hyps, T, stop_flag),
            shape_invariants=(
                tf.TensorShape([None, None,None]),
                tf.TensorShape([]),
                Hypotheses(
                    tf.TensorShape([None]),
                    tf.TensorShape([None, None]),
                    tf.nest.map_structure(get_shape_invariants, new_hyps[-1])
                ),
                tf.TensorShape([]),
                tf.TensorShape([None,1 ]),
            )
        )

        return new_hyps[1]
    def recognize(self, features):
        decoded=self.perform_greedy(features)

        return decoded

    def get_config(self):
        if self.mel_layer is not None:
            conf=self.mel_layer.get_config()
            conf.update(self.encoder.get_config())
        else:
            conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf
