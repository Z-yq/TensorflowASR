import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision
from LMmodel.tf2_trm import create_masks
from trainer.base_runners import BaseTrainer



class LMTrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,

                 config: dict,
                 is_mixed_precision: bool = False,
                 one2one:bool =False,
                 ):
        super(LMTrainer, self).__init__(config=config,)

        self.is_mixed_precision = is_mixed_precision
        self.global_batch_size=config['batch_size']
        self.one2one=one2one
    def set_train_metrics(self):
        lists=['lm_loss','feature_map_loss','lm_acc']

        self.train_metrics={}
        for item in lists:
            self.train_metrics.update({
            item: tf.keras.metrics.Mean("train_{}".format(item), dtype=tf.float32)
        })

    def set_eval_metrics(self):
        lists = ['lm_loss','feature_map_loss', 'lm_acc']

        self.eval_metrics={}
        for item in lists:
            self.eval_metrics.update({
            item: tf.keras.metrics.Mean("eval_{}".format(item), dtype=tf.float32),
        })

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        inp, tar, feature= batch
        if self.one2one:
            tar_inp = tar
            tar_real = tar
        else:
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            feature=feature[:,1:]

        with tf.GradientTape() as tape:
            predictions, out_feature = self.model(inp, tar_inp,
                                                  training=True)
            classes_loss=self.classes_loss(tar_real, predictions)
            feature_map_loss=self.bert_feature_loss(feature, out_feature)
            train_loss =classes_loss+feature_map_loss

            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)

        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics['lm_loss'].update_state(classes_loss)
        self.train_metrics['feature_map_loss'].update_state(feature_map_loss)
        self.train_metrics['lm_acc'].update_state(self.classes_acc(tar_real,predictions))





    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        inp, tar, feature = batch
        if self.one2one:
            tar_inp = tar
            tar_real = tar
        else:
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            feature = feature[:, 1:]

        predictions, out_feature = self.model(inp, tar_inp,
                                              training=False
                                             )
        classes_loss = self.classes_loss(tar_real, predictions)
        feature_map_loss = self.bert_feature_loss(feature, out_feature)

        self.eval_metrics['lm_loss'].update_state(classes_loss)
        self.eval_metrics['feature_map_loss'].update_state(feature_map_loss)
        self.eval_metrics['lm_acc'].update_state(self.classes_acc(tar_real, predictions))


    def bert_feature_loss(self, real, pred):
        mask = tf.cast(tf.not_equal(real, -10.), 1.)
        loss = tf.square(real - pred)
        loss *= mask
        return tf.reduce_sum(tf.reduce_sum(loss,-1) / (tf.reduce_sum(mask,-1) + 1e-6))

    def classes_loss(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = tf.keras.losses.sparse_categorical_crossentropy(real, pred,True)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        final=tf.reduce_sum(loss,-1)/tf.reduce_sum(mask,-1)

        return tf.reduce_sum(final)
    def classes_acc(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accs = tf.keras.metrics.sparse_categorical_accuracy(real,pred)

        mask = tf.cast(mask, dtype=accs.dtype)
        accs *= mask
        final=tf.reduce_sum(accs,-1)/tf.reduce_sum(mask,-1)

        return tf.reduce_mean(final)
    def compile(self, model: tf.keras.Model,
                optimizer: any,
                max_to_keep: int = 10):

        self.model = model

        self.optimizer = tf.keras.optimizers.get(optimizer)

        if self.is_mixed_precision:
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, "dynamic")

        self.set_progbar()
        # self.load_checkpoint()

    def fit(self, train_dataset, eval_dataset=None,epoch=None):
        if epoch is not None:
            self.epochs=epoch
            self.train_progbar.set_description_str(
                f"[Train] [Epoch {epoch}/{self.config['num_epochs']}]")


        self._train_batches(train_dataset)
        if eval_dataset is not None:

            self._eval_batches(eval_dataset)


