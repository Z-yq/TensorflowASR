# encoding=utf-8

import tensorflow as tf


class CNN_Online_VAD(tf.keras.Model):
    def __init__(self,
                 dmodel: int,
                 frame: int,
                 name="cnn_online_vad",
                 **kwargs):
        super(CNN_Online_VAD, self).__init__(name=name, **kwargs)
        self.frame=frame
        self.embed = tf.keras.layers.Dense(dmodel)

        self.cnn1 = tf.keras.layers.Conv1D(dmodel * 2, 3, padding='causal', activation='relu')
        self.dense1 = tf.keras.layers.Conv1D(dmodel, 1, padding='causal', activation='relu')
        self.cnn2 = tf.keras.layers.Conv1D(dmodel * 2, 3, padding='causal', activation='relu')
        self.dense2 = tf.keras.layers.Dense(dmodel, activation='relu')
        self.dense3 = tf.keras.layers.Dense(dmodel, activation='relu')

        self.fc = tf.keras.layers.Dense(1)

        self.fc3 = tf.keras.layers.Dense(frame, name='audio_voice_mask')

    def _build(self):
        fake = tf.random.uniform([1, 80, self.frame])
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
    def set_inferenc_func(self):
        @tf.function(input_signature=[
            tf.TensorSpec([None, None, self.frame], dtype=tf.float32),

        ])
        def inference( inputs):
            training=False
            outputs = self.embed(inputs, training=training)
            outputs = self.dense1(outputs, training=training)
            outputs = self.cnn1(outputs, training=training)
            outputs = self.dense2(outputs, training=training)
            outputs = self.cnn2(outputs, training=training)
            outputs = self.dense3(outputs, training=training)
            output1 = self.fc(outputs)
            return output1
        self.inference=inference

class CNN_Offline_VAD(tf.keras.Model):
    def __init__(self,
                 dmodel: int,
                 frame: int,
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
        self.frame=frame
        self.fc3 = tf.keras.layers.Dense(frame, name='audio_voice_mask')
    def _build(self):
        fake = tf.random.uniform([1, 80, self.frame])
        self(fake)

    def call(self,
             inputs,
             training=False,
             **kwargs):

        outputs = self.embed(inputs, training=training)
        outputs = self.dense1(outputs)
        outputs = self.cnn1(outputs, training=training)
        outputs = self.cnn2(outputs, training=training)
        outputs = self.cnn3(outputs, training=training)
        outputs = self.cnn4(outputs, training=training)
        outputs = self.dense2(outputs, training=training)
        output1 = self.fc(outputs)
        output3 = self.fc3(outputs)
        output4 = inputs * output3
        return output1, output4

    def set_inference_func(self):
        @tf.function(input_signature=[
            tf.TensorSpec([None, None, self.frame], dtype=tf.float32),  # features

        ])
        def infrence(inputs,):
            training = False
            outputs = self.embed(inputs, training=training)
            outputs = self.dense1(outputs)
            outputs = self.cnn1(outputs, training=training)
            outputs = self.cnn2(outputs, training=training)
            outputs = self.cnn3(outputs, training=training)
            outputs = self.cnn4(outputs, training=training)
            outputs = self.dense2(outputs, training=training)
            outputs = self.fc(outputs)
            return outputs
        self.inference=infrence