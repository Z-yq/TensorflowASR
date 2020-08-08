from LMmodel.tf2_trm import create_masks
from LMmodel.trm_lm import LM
from dataloader import LM_DataLoader
import tensorflow as tf
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class LM_Trainer():
    def __init__(self,hparams):
        self.hp=hparams
        self.dg = LM_DataLoader(self.hp.lm_train_list)
        lm=LM(hparams)
        self.model = lm.model
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        self.optimizer = tf.keras.optimizers.Adamax(1e-4)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
    def mask_mse(self,real, pred):
        mask = tf.cast(tf.not_equal(real, -10.), 1.)
        loss = tf.square(real - pred)
        loss *= mask
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-6)

    def loss_function(self,real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)
    @tf.function(experimental_relax_shapes=True)
    def train_step(self,inp, tar, feature):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, out_feature = self.model(inp, tar_inp,
                                             True,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)
            loss = self.loss_function(tar_real, predictions) + self.mask_mse(feature, out_feature)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)
    def train(self,):
        step=0
        while 1:
            try:
                x,y,feature=self.dg.generate(8)
            except:
                logging.info('data genrator failed,you can check here')
                continue
            self.train_step(x,y,feature)
            if step%10==0:
                logging.info('Step {}  Loss {:.4f} Accuracy {:.4f}'.format(
                step, self.train_loss.result(), self.train_accuracy.result()))

            if step%1000==0:
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()
            step+=1
            if step%self.hp.save_step==0 and step>1:
                self.model.save_weights(os.path.join(self.hp.lm_save_path,'model/lm'))

if __name__ == '__main__':
    import hparams
    train=LM_Trainer(hparams)
    train.train()
