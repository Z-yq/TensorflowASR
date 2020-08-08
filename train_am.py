from AMmodel.model import AM,ctc_lambda_func
from dataloader import AM_DataLoader
import hparams
import tensorflow as tf
import os
import logging
import numpy as np
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
class AM_Trainer():
    def __init__(self,hparams):
        self.hp=hparams
        self.am = AM(self.hp)
        self.dg = AM_DataLoader(self.am.itokens1, self.am.itokens2, self.am.itokens3, self.am.itokens4, SampleRate=self.hp.sample_rate, n_mels=self.hp.num_mels,
                        batch=self.hp.batch_size,train_list=self.hp.am_train_list)

        self.STT = self.am.STT

        self.opt = tf.keras.optimizers.Adamax(1e-4)

    def recevie_data(self,r):

        data = r.rpop(self.hp.data_name)
        data = eval(data)
        x = data['x']
        dtype = data['x_dtype']
        shape = data['x_shape']
        x = np.frombuffer(x, dtype)
        x = x.reshape(shape)

        wavs = data['wavs']
        dtype = data['wavs_dtype']
        shape = data['wavs_shape']
        wavs = np.frombuffer(wavs, dtype)
        wavs = wavs.reshape(shape)

        input_length = data['input_length']
        dtype = data['input_length_dtype']
        shape = data['input_length_shape']
        input_length = np.frombuffer(input_length, dtype)
        input_length = input_length.reshape(shape)

        y1 = data['y1']
        dtype = data['y1_dtype']
        shape = data['y1_shape']
        y1 = np.frombuffer(y1, dtype)
        y1 = y1.reshape(shape)

        label_length1 = data['label_length1']
        dtype = data['label_length1_dtype']
        shape = data['label_length1_shape']
        label_length1 = np.frombuffer(label_length1, dtype)
        label_length1 = label_length1.reshape(shape)

        y2 = data['y2']
        dtype = data['y2_dtype']
        shape = data['y2_shape']
        y2 = np.frombuffer(y2, dtype)
        y2 = y2.reshape(shape)

        label_length2 = data['label_length2']
        dtype = data['label_length2_dtype']
        shape = data['label_length2_shape']
        label_length2 = np.frombuffer(label_length2, dtype)
        label_length2 = label_length2.reshape(shape)

        y3 = data['y3']
        dtype = data['y3_dtype']
        shape = data['y3_shape']
        y3 = np.frombuffer(y3, dtype)
        y3 = y3.reshape(shape)

        label_length3 = data['label_length3']
        dtype = data['label_length3_dtype']
        shape = data['label_length3_shape']
        label_length3 = np.frombuffer(label_length3, dtype)
        label_length3 = label_length3.reshape(shape)
        wavs = wavs[:, :x.shape[1] * self.hp.hop_size]
        return [x, wavs, np.expand_dims(input_length, -1), y1, np.expand_dims(label_length1, -1), y2,
                np.expand_dims(label_length2, -1), y3, np.expand_dims(label_length3, -1)]


    # @tf.function(experimental_relax_shapes=True)
    def test_op(self,x, y):
        ctc1, ctc2, ctc3=self.STT.call(x[:2],training=False)
        loss = tf.reduce_mean(ctc_lambda_func([ctc1,x[3],x[2],x[4]])) + tf.reduce_mean(ctc_lambda_func([ctc2,x[5],x[2],x[6]])) + tf.reduce_mean(ctc_lambda_func([ctc3,x[7],x[2],x[8]]))
        self.t1_loss(loss)
        return tf.reduce_mean(ctc_lambda_func([ctc1,x[3],x[2],x[4]])), tf.reduce_mean(ctc_lambda_func([ctc2,x[5],x[2],x[6]])),tf.reduce_mean(ctc_lambda_func([ctc3,x[7],x[2],x[8]]))


    @tf.function(experimental_relax_shapes=True)
    def train_op(self,x, y):
        with tf.GradientTape() as tape:
            outputs = self.STT(x[:2])
            ctc1, ctc2, ctc3= outputs
            loss = tf.math.log(tf.reduce_mean(ctc_lambda_func([ctc1,x[3],x[2],x[4]])) + 1.) + tf.math.log(tf.reduce_mean(ctc_lambda_func([ctc2,x[5],x[2],x[6]])) + 1.) + tf.math.log(
                tf.reduce_mean(ctc_lambda_func([ctc3,x[7],x[2],x[8]])) + 1.)
        gradients = tape.gradient(loss, self.STT.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.STT.trainable_variables))
        self.ctc1_loss(tf.math.log(tf.reduce_mean(ctc_lambda_func([ctc1,x[3],x[2],x[4]])) + 1.) )
        self.ctc2_loss(tf.math.log(tf.reduce_mean(ctc_lambda_func([ctc2,x[5],x[2],x[6]])) + 1.) )
        self.ctc3_loss( tf.math.log(
                tf.reduce_mean(ctc_lambda_func([ctc3,x[7],x[2],x[8]])) + 1.))

    def train(self):
        step = 0
        os.makedirs(os.path.join(self.hp.am_save_path, 'model'), exist_ok=True)
        self.ctc1_loss = tf.keras.metrics.Mean(name='ctc1_loss')
        self.ctc2_loss = tf.keras.metrics.Mean(name='ctc2_loss')
        self.ctc3_loss = tf.keras.metrics.Mean(name='ctc3_loss')
        self.t1_loss = tf.keras.metrics.Mean(name='t1_loss')
        if self.hp.use_redis:
            r = self.redis.Redis(self.hp.ip, self.hp.port)
        while 1:
            out = self.dg.generator()
            if out is None:
                continue
            else:
                x, wavs, input_length, y1, label_length1, y2, label_length2, y3, label_length3, y4, label_length4 = out
            input_length//=2
            self.train_op([tf.constant(x), tf.constant(wavs),
                                tf.constant(np.expand_dims(input_length, -1)), tf.constant(y1),
                                tf.constant(np.expand_dims(label_length1, -1)), tf.constant(y2),
                                tf.constant(np.expand_dims(label_length2, -1)), tf.constant(y3),
                                tf.constant(np.expand_dims(label_length3, -1))], np.zeros_like(x))
            if self.hp.am_add_noise:
                x, wavs = self.dg.add_noise(wavs)

                self.train_op([tf.constant(x), tf.constant(wavs),
                                    tf.constant(np.expand_dims(input_length, -1)), tf.constant(y1),
                                    tf.constant(np.expand_dims(label_length1, -1)), tf.constant(y2),
                                    tf.constant(np.expand_dims(label_length2, -1)), tf.constant(y3),
                                    tf.constant(np.expand_dims(label_length3, -1))], np.zeros_like(x))

            if self.hp.use_redis:
                if r.llen(self.hp.data_name) > 0 :
                    train_x1 = self.recevie_data(r)
                    self.train_op(train_x1, np.zeros([train_x1[0].shape[0], 1]))
                    train_x = self.recevie_data(r)
                    self.train_op(train_x, np.zeros([train_x[0].shape[0], 1]))
            if step % 10 == 0:
                x, wavs, input_length, y1, label_length1, y2, label_length2, y3, label_length3, y4, label_length4 = self.dg.generator(
                    train=False)
                input_length//=2
                tctc1, tctc2, tctc3 = self.test_op([tf.constant(x), tf.constant(wavs),
                                                        tf.constant(np.expand_dims(input_length, -1)), tf.constant(y1),
                                                        tf.constant(np.expand_dims(label_length1, -1)), tf.constant(y2),
                                                        tf.constant(np.expand_dims(label_length2, -1)), tf.constant(y3),
                                                        tf.constant(np.expand_dims(label_length3, -1))], np.zeros_like(x))


                info = 'run-model epoch:{} step:{} out1:{}  out2:{}  out3:{}|||test:{} ctc1:{} ctc2:{} ctc3:{}'.format(
                    self.dg.epochs, step,
                    self.ctc1_loss.result().numpy().round(2),
                    self.ctc2_loss.result().numpy().round(2),
                    self.ctc3_loss.result().numpy().round(2),

                    self.t1_loss.result().numpy().round(2),
                    tctc1.numpy().round(2),
                    tctc2.numpy().round(2),
                    tctc3.numpy().round(2))
                logging.info(info)
                # print(info)

            if step %self.hp.save_step == 0 and step > 0:

                self.STT.save_weights(os.path.join(self.hp.am_save_path,'model/am'))

            step += 1
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    train=AM_Trainer(hparams)
    train.train()
