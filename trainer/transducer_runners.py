

import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from trainer.base_runners import BaseTrainer
from losses.rnnt_losses import USE_TF,tf_rnnt_loss,rnnt_loss
from AMmodel.transducer_wrap import Transducer
from utils.text_featurizers import TextFeaturizer
import logging
import random
class TransducerTrainer(BaseTrainer):
    def __init__(self,
                 speech_featurizer,
                 text_featurizer: TextFeaturizer,
                 config: dict,
                 is_mixed_precision: bool = False,
                 strategy=None
                ):
        """
        Args:
            config: the 'running_config' part in YAML config file'
            text_featurizer: the TextFeaturizer instance
            is_mixed_precision: a boolean for using mixed precision or not
        """
        super(TransducerTrainer, self).__init__(config)
        self.speech_featurizer=speech_featurizer
        self.text_featurizer = text_featurizer
        self.is_mixed_precision = is_mixed_precision
        self.set_strategy(strategy)
        if USE_TF:
            self.rnnt_loss=tf_rnnt_loss
        else:
            self.rnnt_loss=rnnt_loss
    def set_train_metrics(self):
        self.train_metrics = {
            "transducer_loss": tf.keras.metrics.Mean("train_transducer_loss", dtype=tf.float32),
            "ctc_loss": tf.keras.metrics.Mean("ctc_loss", dtype=tf.float32),
            "ctc_att_loss": tf.keras.metrics.Mean("ctc_att_loss", dtype=tf.float32),
            "ctc_alig_loss": tf.keras.metrics.Mean("ctc_alig_loss", dtype=tf.float32),
            "ctc_acc": tf.keras.metrics.Mean("ctc_acc", dtype=tf.float32),
            "att_acc": tf.keras.metrics.Mean("att_acc", dtype=tf.float32),

        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "transducer_loss": tf.keras.metrics.Mean("eval_transducer_loss", dtype=tf.float32),
            "ctc_loss": tf.keras.metrics.Mean("ctc_loss", dtype=tf.float32),
            "ctc_att_loss": tf.keras.metrics.Mean("ctc_att_loss", dtype=tf.float32),
            "ctc_alig_loss": tf.keras.metrics.Mean("ctc_alig_loss", dtype=tf.float32),
            "ctc_acc": tf.keras.metrics.Mean("ctc_acc", dtype=tf.float32),
            "att_acc": tf.keras.metrics.Mean("att_acc", dtype=tf.float32),

        }

    def ctc_acc(self, labels, y_pred):
        T1 = y_pred.shape[1]
        T2 = labels.shape[1]
        T = min([T1, T2])
        y_pred = y_pred[:, :T]
        labels = labels[:, :T]

        mask = tf.cast(tf.not_equal(labels, 0), 1.)
        y_pred = tf.cast(y_pred, tf.float32)
        labels = tf.cast(labels, tf.float32)

        value = tf.cast(labels == y_pred, tf.float32)

        accs = tf.reduce_sum(value, -1) / (tf.reduce_sum(mask, -1) + 1e-6)
        return accs

    def ctc_att_acc(self, labels, y_pred):
        mask = tf.cast(tf.not_equal(labels, 0), 1.)
        y_pred = tf.argmax(y_pred, -1)
        y_pred = tf.cast(y_pred, tf.float32)
        labels = tf.cast(labels, tf.float32)
        value = tf.cast(labels == y_pred, tf.float32)

        accs = tf.reduce_sum(value, -1) / (tf.reduce_sum(mask, -1) + 1e-6)
        return accs

    def mask_loss(self,y_true,y_pred):
        mask=tf.cast(tf.not_equal(y_true,0),1.)
        loss=tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred,True)
        mask_loss=loss*mask
        total_loss=tf.reduce_sum(mask_loss,-1)/(tf.reduce_sum(mask,-1)+1e-6)
        return total_loss
    def alig_loss(self,guide_matrix,y_pred):
        attention_masks = tf.cast(tf.math.not_equal(guide_matrix, -1.0), tf.float32)
        loss_att = tf.reduce_sum(
            tf.abs(y_pred * guide_matrix) * attention_masks,-1
        )
        loss_att /= tf.reduce_sum(attention_masks,-1)
        return tf.reduce_sum(loss_att,-1)*20.
    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        features,  input_length, labels, label_length,guide_matrix = batch

        pred_inp=labels
        target=labels[:,1:]
        label_length-=1
        ctc_label = tf.where(target==self.text_featurizer.blank,0,target)

        with tf.GradientTape() as tape:


            logits,ctc_logits,ctc_att_logits,ctc_alig_output = self.model([features,  input_length, pred_inp,ctc_label, label_length], training=True)
            # print(logits.shape,target.shape)
            if USE_TF:
                per_train_loss=self.rnnt_loss(logits=logits, labels=target
                                              , label_length=label_length, logit_length=input_length)
                per_train_loss = tf.clip_by_value(per_train_loss, 0., 500.)
            else:
                per_train_loss = self.rnnt_loss(
                logits=logits, labels=labels, label_length=label_length,
                logit_length=(input_length // self.model.time_reduction_factor),
                blank=self.text_featurizer.blank)
            ctc_loss = tf.nn.ctc_loss(ctc_label, ctc_logits, label_length, input_length, False, blank_index=-1)
            ctc_loss = tf.clip_by_value(ctc_loss, 0., 1000.)
            att_loss = self.mask_loss(ctc_label, ctc_att_logits)
            real_length = tf.shape(ctc_att_logits)[1]
            alig_loss = self.alig_loss(guide_matrix[:, :real_length,:-1], ctc_alig_output[:, :real_length])

            train_loss = tf.nn.compute_average_loss(per_train_loss+ctc_loss+att_loss+alig_loss,
                                                    global_batch_size=self.global_batch_size)

            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)
        ctc_pred=tf.keras.backend.ctc_decode(tf.nn.softmax(ctc_logits,-1),input_length)[0][0]
        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics["transducer_loss"].update_state(per_train_loss)
        self.train_metrics["ctc_loss"].update_state(ctc_loss)
        self.train_metrics["ctc_att_loss"].update_state(att_loss)
        self.train_metrics["ctc_alig_loss"].update_state(alig_loss)
        return ctc_label,ctc_pred,ctc_att_logits

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        features,input_length, labels, label_length,guide_matrix = batch
        pred_inp = labels
        target = labels[:, 1:]
        label_length -= 1
        ctc_label = tf.where(target == self.text_featurizer.blank, 0, target)

        logits, ctc_logits, ctc_att_logits, ctc_alig_output = self.model(
            [features, input_length, pred_inp, ctc_label, label_length], training=False)
        # print(logits.shape,target.shape)
        if USE_TF:
            eval_loss = self.rnnt_loss(logits=logits, labels=target
                                            , label_length=label_length,
                                            logit_length=input_length,
                                           )
        else:
            eval_loss = self.rnnt_loss(
                logits=logits, labels=target, label_length=label_length,
                logit_length=(input_length // self.model.time_reduction_factor),
                blank=self.text_featurizer.blank)
        ctc_pred = tf.keras.backend.ctc_decode(tf.nn.softmax(ctc_logits,-1), input_length)[0][0]
        ctc_loss = tf.nn.ctc_loss(ctc_label, ctc_logits, label_length, input_length, False, blank_index=-1)
        ctc_loss = tf.clip_by_value(ctc_loss, 0., 500.)
        att_loss = self.mask_loss(ctc_label, ctc_att_logits)
        real_length = tf.shape(ctc_att_logits)[1]
        alig_loss = self.alig_loss(guide_matrix[:, :real_length,:-1], ctc_alig_output[:, :real_length])
        self.eval_metrics["transducer_loss"].update_state(eval_loss)
        self.eval_metrics["ctc_loss"].update_state(ctc_loss)
        self.eval_metrics["ctc_att_loss"].update_state(att_loss)
        self.eval_metrics["ctc_alig_loss"].update_state(alig_loss)
        return ctc_label,ctc_pred,ctc_att_logits
    def compile(self,
                model: Transducer,
                optimizer: any,
                max_to_keep: int = 10):
        f, c = self.speech_featurizer.compute_feature_dim()
        with self.strategy.scope():
            self.model = model
            if self.model.mel_layer is not None:
                self.model._build([1, 16000, 1])
            else:
                self.model._build([1, 80, f, c])
            self.model.summary(line_length=100)
            try:
                self.load_checkpoint()
            except:
                logging.info('trainer resume failed,use init state')
            self.optimizer = tf.keras.optimizers.get(optimizer)
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

    def _train_batches(self):
        """Train model one epoch."""

        for batch in self.train_datasets:
            try:

                ctc_label,ctc_logits,ctc_att_logits=self.strategy.run(self._train_step,args=(batch,))

                ctc_acc=self.ctc_acc(ctc_label,ctc_logits)
                att_acc=self.ctc_att_acc(ctc_label,ctc_att_logits)
                self.train_metrics['ctc_acc'].update_state(ctc_acc)
                self.train_metrics['att_acc'].update_state(att_acc)
                self.steps+=1
                self.train_progbar.update(1)
                self._print_train_metrics(self.train_progbar)
                self._check_log_interval()

                if self._check_save_interval():
                    break

            except tf.errors.OutOfRangeError:
                continue
    def _eval_batches(self):
        """One epoch evaluation."""

        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()
        n=0
        for batch in self.eval_datasets:
            try:

                ctc_label,ctc_logits,ctc_att_logits=self.strategy.run(self._eval_step,args=(batch,))
                ctc_acc = self.ctc_acc(ctc_label, ctc_logits)
                att_acc = self.ctc_att_acc(ctc_label, ctc_att_logits)
                self.eval_metrics['ctc_acc'].update_state(ctc_acc)
                self.eval_metrics['att_acc'].update_state(att_acc)

            except tf.errors.OutOfRangeError:

                pass
            n+=1

            # Update steps
            self.eval_progbar.update(1)


            # Print eval info to progress bar
            self._print_eval_metrics(self.eval_progbar)
            if n>=self.eval_steps_per_epoch:
                break
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")