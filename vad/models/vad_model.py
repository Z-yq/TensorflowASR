# encoding=utf-8

import tensorflow as tf


class CNN_Online_VAD(tf.keras.Model):
    def __init__(self,
                 dmodel: int,
                 name="cnn_online_vad",
                 **kwargs):
        super(CNN_Online_VAD, self).__init__(name=name, **kwargs)

        self.embed = tf.keras.layers.Dense(dmodel)

        self.cnn1 = tf.keras.layers.Conv1D(dmodel * 2, 3, padding='causal', activation='relu')
        self.dense1 = tf.keras.layers.Conv1D(dmodel, 1, padding='causal', activation='relu')
        self.cnn2 = tf.keras.layers.Conv1D(dmodel * 2, 3, padding='causal', activation='relu')
        self.dense2 = tf.keras.layers.Dense(dmodel, activation='relu')
        self.dense3 = tf.keras.layers.Dense(dmodel, activation='relu')

        self.fc = tf.keras.layers.Dense(1)

        self.fc3 = tf.keras.layers.Dense(80, name='audio_voice_mask')

    def _build(self):
        fake = tf.random.uniform([1, 80, 80])
        self(fake)

    def call(self,
             inputs,
             training=True,
             **kwargs):
        outputs = self.embed(inputs, training=training)
        outputs = self.dense1(outputs, training=training)
        outputs = self.cnn1(outputs, training=training)
        outputs = self.dense2(outputs, training=training)
        outputs = self.cnn2(outputs, training=training)
        outputs = self.dense3(outputs, training=training)
        output1 = self.fc(outputs)
        output3 = self.fc3(outputs)
        output4 = inputs * output3
        return output1, output4

    @tf.function(input_signature=[
        tf.TensorSpec([None, None, 80], dtype=tf.float32),  # TODO:根据自己的frame_input修改

    ])
    def inference(self, inputs, training=False):
        outputs = self.embed(inputs, training=training)
        outputs = self.dense1(outputs, training=training)
        outputs = self.cnn1(outputs, training=training)
        outputs = self.dense2(outputs, training=training)
        outputs = self.cnn2(outputs, training=training)
        outputs = self.dense3(outputs, training=training)
        output1 = self.fc(outputs)
        return output1


class CNN_Offline_VAD(tf.keras.Model):
    def __init__(self,
                 dmodel: int,
                 name="cnn_offline_vad",
                 **kwargs):
        super(CNN_Offline_VAD, self).__init__(name=name, **kwargs)
        self.embed = tf.keras.layers.Dense(dmodel)
        self.cnn1 = tf.keras.layers.Conv1D(dmodel, 5, padding='same', activation='relu')
        self.dense1 = tf.keras.layers.Dense(dmodel, activation='relu')
        self.cnn2 = tf.keras.layers.Conv1D(dmodel, 5, padding='same', activation='relu', dilation_rate=2)
        self.cnn3 = tf.keras.layers.Conv1D(dmodel, 5, padding='same', activation='relu', dilation_rate=4)
        self.cnn4 = tf.keras.layers.Conv1D(dmodel, 5, padding='same', activation='relu', dilation_rate=8)

        self.dense2 = tf.keras.layers.Dense(dmodel, activation='relu')
        self.fc = tf.keras.layers.Dense(1)

    def _build(self):
        fake = tf.random.uniform([1, 80, 80])
        self(fake)

    # @tf.function(experimental_relax_shapes=True)
    def call(self,
             inputs,
             training=False,
             **kwargs):
        # inputs has shape [B, U]
        outputs = self.embed(inputs, training=training)
        outputs = tf.squeeze(outputs, -1)

        outputs = self.dense1(outputs)
        outputs = self.cnn1(outputs, training=training)
        outputs = self.cnn2(outputs, training=training)
        outputs = self.cnn3(outputs, training=training)
        outputs = self.cnn4(outputs, training=training)
        outputs = self.dense2(outputs, training=training)
        outputs = self.fc(outputs)
        # return shapes [B, T, P], ([num_lstms, B, P], [num_lstms, B, P]) if using lstm
        return outputs

    @tf.function(input_signature=[
        tf.TensorSpec([None, None, 80], dtype=tf.float32),  # features

    ])
    def infrence(self, inputs, training=False):
        outputs = self.embed(inputs, training=training)
        outputs = tf.squeeze(outputs, -1)

        outputs = self.dense1(outputs)
        outputs = self.cnn1(outputs, training=training)
        outputs = self.cnn2(outputs, training=training)
        outputs = self.cnn3(outputs, training=training)
        outputs = self.cnn4(outputs, training=training)
        outputs = self.dense2(outputs, training=training)
        outputs = self.fc(outputs)
        # return shapes [B, T, P], ([num_lstms, B, P], [num_lstms, B, P]) if using lstm
        return outputs