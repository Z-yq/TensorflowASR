import os
import tensorflow as tf
import numpy as np
from utils.tools import shape_list, get_shape_invariants, merge_repeated

from utils.text_featurizers import TextFeaturizer
from AMmodel.layers.time_frequency import Melspectrogram, Spectrogram
from AMmodel.layers.LayerNormLstmCell import LayerNormLSTMCell
from AMmodel.ctc_attention import CTCAttention
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
            lstm = LayerNormLSTMCell(units=lstm_units,dropout=embed_dropout,recurrent_dropout=embed_dropout)
            self.lstm_cells.append(lstm)
        self.decoder_lstms = tf.keras.layers.RNN(
            self.lstm_cells, return_sequences=True, return_state=True)

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
        # new_memory_states = outputs[1:]
        outputs = outputs[0]
        # n_memory_states.append(new_memory_states)

        # return shapes [B, T, P], ([num_lstms, B, P], [num_lstms, B, P]) if using lstm
        return outputs  # , new_memory_states

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

        self.ctc_classes=tf.keras.layers.Dense(vocabulary_size,name='ctc_classes')
        self.ctc_attention=CTCAttention(encoder.dmodel,vocabulary_size,self.ctc_classes,embed_layer=self.predict_net.embed)
        self.wav_info = speech_config['add_wav_info']
        if self.wav_info:
            assert speech_config['use_mel_layer'] == True, 'shold set use_mel_layer is True'
        self.kept_decode = None
        self.startid = 0
        self.endid = 1
        self.max_iter = 10

    def _build(self, shape):  # Call on real data for building model


        batch = shape[0]
        inputs = np.random.normal(size=shape).astype(np.float32)
        if self.mel_layer is not None:
            input_lengths = np.array([shape[1] // 4 // self.mel_layer.n_hop] * batch, 'int32')
        else:
            input_lengths = np.array([shape[1] // 4] * batch, 'int32')

        targets = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]] * batch)
        targets = targets[:, :, np.newaxis]
        targets_lengths = np.array([9] * batch)
        self([inputs, input_lengths,
             targets, targets_lengths],training=True)




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
    def call(self,inputs, training=False):
        features, input_lengths,targets, targets_lengths=inputs

        if self.mel_layer is not None:
            if self.wav_info :
                wav=features
                features = self.mel_layer(features)
            else:
                features = self.mel_layer(features)
            # print(inputs.shape)
        if self.wav_info :
            enc = self.encoder([features,wav], training=training)
        else:
            enc = self.encoder(features, training=training)

        pred = self.predict_net(targets, training=training)
        outputs = self.joint_net([enc, pred], training=training)
        ctc_outputs=self.ctc_classes(enc,training=training)
        ctc_att_out,ctc_alig_out=self.ctc_attention(enc,input_lengths,targets,targets_lengths)
        # outputs = self.joint_net([enc, pred], training=training)

        return outputs,ctc_outputs,ctc_att_out,ctc_alig_out



    def add_featurizers(self,
                        text_featurizer: TextFeaturizer):

        self.text_featurizer = text_featurizer

    def return_pb_function(self, shape):
        @tf.function(input_signature=[
            tf.TensorSpec(shape, dtype=tf.float32),  # features
            tf.TensorSpec([None, 1], dtype=tf.int32),  # features

        ])
        def recognize_pb(features, lengths):
            b_i = tf.constant(0, dtype=tf.int32)

            B = tf.shape(features)[0]

            decoded = tf.constant([], dtype=tf.int32)

            def _cond(b_i, B, features, decoded): return tf.less(b_i, B)

            def _body(b_i, B, features, decoded):
                yseq = self.perform_greedy(tf.expand_dims(features[b_i], axis=0),
                                           streaming=False)

                yseq = tf.concat([yseq, tf.constant([[self.text_featurizer.stop]], tf.int32)], axis=-1)
                decoded = tf.concat([decoded, yseq[0]], axis=0)
                return b_i + 1, B, features, decoded

            _, _, _, decoded = tf.while_loop(
                _cond,
                _body,
                loop_vars=(b_i, B, features, decoded),
                shape_invariants=(
                    tf.TensorShape([]),
                    tf.TensorShape([]),
                    get_shape_invariants(features),
                    tf.TensorShape([None])
                )
            )

            return [decoded]

        self.recognize_pb = recognize_pb

    @tf.function(experimental_relax_shapes=True)
    def perform_greedy(self,
                       features,
                       streaming=False):
        if self.wav_info:
            wav=features
        if self.mel_layer is not None:
            features = self.mel_layer(features)

        decoded = tf.constant([self.text_featurizer.start])
        if self.kept_decode is not None:
            decoded = self.kept_decode

        if self.wav_info:
            enc = self.encoder([features,wav], training=False)[-1]  # [1, T, E]
        else:
            enc = self.encoder(features, training=False)[-1]  # [1, T, E]
        enc = tf.squeeze(enc, axis=0)  # [T, E]

        T = tf.cast(tf.shape(enc)[0], dtype=tf.int32)

        i = tf.constant(0, dtype=tf.int32)

        def _cond(enc, i, decoded, T):
            return tf.less(i, T)

        def _body(enc, i, decoded, T):
            hi = tf.reshape(enc[i], [1, 1, -1])  # [1, 1, E]
            y = self.predict_net(
                inputs=tf.reshape(decoded, [1, -1]),  # [1, 1]
                p_memory_states=None,
                training=False
            )
            y = y[:, -1:]
            # [1, 1, P], [1, P], [1, P]
            # [1, 1, E] + [1, 1, P] => [1, 1, 1, V]
            ytu = tf.nn.log_softmax(self.joint_net([hi, y], training=False))
            ytu = tf.squeeze(ytu, axis=None)  # [1, 1, 1, V] => [V]
            n_predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []
            n_predict = tf.reshape(n_predict, [1])

            def return_no_blank():
                return tf.concat([decoded, n_predict], axis=0)

            decoded = tf.cond(
                n_predict != self.text_featurizer.blank and n_predict != 0,
                true_fn=return_no_blank,
                false_fn=lambda: decoded
            )

            return enc, i + 1, decoded, T

        _, _, decoded, _ = tf.while_loop(
            _cond,
            _body,
            loop_vars=(enc, i, decoded, T),
            shape_invariants=(
                tf.TensorShape([None, None]),
                tf.TensorShape([]),

                tf.TensorShape([None]),

                tf.TensorShape([])
            )
        )

        return tf.expand_dims(decoded, axis=0)

    def recognize(self, features):
        decoded = self.perform_greedy(features)

        return decoded

    def get_config(self):
        if self.mel_layer is not None:
            conf = self.mel_layer.get_config()
            conf.update(self.encoder.get_config())
        else:
            conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        conf.update(self.ctc_classes.get_config())
        conf.update(self.ctc_attention.get_config())
        return conf
