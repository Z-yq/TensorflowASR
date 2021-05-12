import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from trainer.base_runners import BaseTrainer
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PuncTrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,

                 config: dict,
                 is_mixed_precision: bool = False,
                 one2one:bool =False,
                 strategy=None
                 ):
        super(PuncTrainer, self).__init__(config=config,)

        self.is_mixed_precision = is_mixed_precision
        self.one2one=one2one
        self.set_strategy(strategy)

    def creat_mask(self, seq):
        seq_pad = tf.cast(tf.equal(seq, 0), tf.float32)
        return seq_pad[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def set_train_metrics(self):
        lists=['bd_loss','feature_map_loss','bd_acc']

        self.train_metrics={}
        for item in lists:
            self.train_metrics.update({
            item: tf.keras.metrics.Mean("train_{}".format(item), dtype=tf.float32)
        })

    def set_eval_metrics(self):
        lists = ['bd_loss','feature_map_loss','bd_acc']

        self.eval_metrics={}
        for item in lists:
            self.eval_metrics.update({
            item: tf.keras.metrics.Mean("eval_{}".format(item), dtype=tf.float32),
        })

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        inp, tar_bd,feature= batch


        mask=self.creat_mask(inp)
        with tf.GradientTape() as tape:
            bd_pred, out_feature = self.model([inp,mask],
                                                  training=True)

            bd_loss=self.classes_loss(tar_bd,bd_pred)
            feature_map_loss=self.bert_feature_loss(feature, out_feature)
            train_loss =feature_map_loss*10.+bd_loss

            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)
            train_loss = tf.nn.compute_average_loss(train_loss,
                                                    global_batch_size=self.global_batch_size)
        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics['feature_map_loss'].update_state(feature_map_loss)
        self.train_metrics['bd_loss'].update_state(bd_loss)
        self.train_metrics['bd_acc'].update_state(self.classes_acc(tar_bd,bd_pred))

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        inp, tar_bd, feature = batch


        mask=self.creat_mask(inp)
        pred_bd,out_feature = self.model([inp,mask],training=False)

        feature_map_loss = self.bert_feature_loss(feature, out_feature)
        bd_loss=self.classes_loss(tar_bd,pred_bd)

        self.eval_metrics['feature_map_loss'].update_state(feature_map_loss)
        self.eval_metrics['bd_loss'].update_state(bd_loss)
        self.eval_metrics['bd_acc'].update_state(self.classes_acc(tar_bd,pred_bd))


    def bert_feature_loss(self, real, pred):
        T1=tf.shape(real)[1]
        T2=tf.shape(pred)[1]
        T=tf.reduce_min([T1,T2])
        real=real[:,:T]
        pred=pred[:,:T]
        mask = tf.cast(tf.not_equal(real, -10.), 1.)
        loss = tf.square(real - pred)
        loss *= mask
        return tf.reduce_mean(tf.reduce_sum(loss,-1) / (tf.reduce_sum(mask,-1) + 1e-6),-1,True)

    def classes_loss(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = tf.keras.losses.sparse_categorical_crossentropy(real, pred,True)
        mask_one= tf.math.logical_not(tf.math.equal(real, 1))
        mask = tf.cast(mask, dtype=loss.dtype)
        mask_one = tf.cast(mask_one, dtype=loss.dtype)
        mask_one*=mask
        loss_all =loss* mask
        final=tf.reduce_sum(loss_all,-1,True)/(tf.reduce_sum(mask,-1,True)+1e-6)
        loss_other = loss_all*mask_one
        final2 = tf.reduce_sum(loss_other, -1, True) / (tf.reduce_sum(mask_one, -1, True)+1e-6)
        return final+final2


    def classes_acc(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accs = tf.keras.metrics.sparse_categorical_accuracy(real,pred)

        mask = tf.cast(mask, dtype=accs.dtype)
        accs *= mask
        final=tf.reduce_sum(accs,-1)/tf.reduce_sum(mask,-1)

        return tf.reduce_mean(final)

    def _eval_batches(self):
        """One epoch evaluation."""

        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()

        for idx, batch in enumerate(self.eval_datasets):

            self.strategy.run(self._eval_step, args=(batch,))

            self.eval_progbar.update(1)
            self._print_eval_metrics(self.eval_progbar)
            if idx >= self.eval_steps_per_epoch:
                break
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")
    def _train_batches(self):
        """Train model one epoch."""

        for idx,batch in enumerate(self.train_datasets):

            self.strategy.run(self._train_step,args=(batch,))
            self.steps+=1
            self.train_progbar.update(1)
            self._print_train_metrics(self.train_progbar)
            self._check_log_interval()
            self._check_save_interval()


            if idx>self.train_steps_per_epoch:
                break
    def compile(self, model: tf.keras.Model,
                optimizer: any,
                max_to_keep: int = 10):
        with self.strategy.scope():
            self.model = model

            self.optimizer = tf.keras.optimizers.get(optimizer)
            self.model._build()
            try:

                self.load_checkpoint()
            except:
                logging.info('lm trainer resume failed')
            if self.is_mixed_precision:
                self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, "dynamic")

        self.set_progbar()
        # self.load_checkpoint()

    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs=epoch
            self.train_progbar.set_description_str(
                f"[Train] [Epoch {epoch}/{self.config['num_epochs']}]")


        self._train_batches()

        self._check_eval_interval()
            # self._eval_batches(eval_dataset)


