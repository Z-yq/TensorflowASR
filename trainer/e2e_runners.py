import logging

import numpy as np
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from trainer.base_runners import BaseTrainer


class E2ETrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,
                 config: dict,
                 is_mixed_precision: bool = False,
                 strategy=None
                 ):
        super(E2ETrainer, self).__init__(config=config, )
        self.is_mixed_precision = is_mixed_precision
        self.set_strategy(strategy)

    def set_train_metrics(self):
        self.train_metrics = {
            "am_py_ctc_loss": tf.keras.metrics.Mean("py_ctc_loss", dtype=tf.float32),
            "am_ch_ctc_loss": tf.keras.metrics.Mean("ch_ctc_loss", dtype=tf.float32),
            "lm_class_loss": tf.keras.metrics.Mean("lm_loss", dtype=tf.float32),
            "lm_bert_loss": tf.keras.metrics.Mean("lm_bert_loss", dtype=tf.float32),
            "lm_acc": tf.keras.metrics.Mean("lm_acc", dtype=tf.float32),
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "am_py_ctc_loss": tf.keras.metrics.Mean("py_ctc_loss", dtype=tf.float32),
            "am_ch_ctc_loss": tf.keras.metrics.Mean("ch_ctc_loss", dtype=tf.float32),
            "lm_class_loss": tf.keras.metrics.Mean("lm_loss", dtype=tf.float32),
            "lm_bert_loss": tf.keras.metrics.Mean("lm_bert_loss", dtype=tf.float32),
            "lm_acc": tf.keras.metrics.Mean("lm_acc", dtype=tf.float32),
        }

    @tf.function(experimental_relax_shapes=True)
    def am_train_step(self, batch):
        speech_features, input_length, py_label, py_label_length, txt_label, txt_label_length, bert_features, guide_matrix = batch

        with tf.GradientTape() as tape, tf.GradientTape() as tape2:

            predictions, bert_out_feature, py_out_feature = self.lm_model(py_label,self.lm_model.create_padding_mask(py_label),
                                                                          training=True)
            classes_loss = self.classes_loss(txt_label, predictions)
            feature_map_loss = self.bert_feature_loss(bert_features, bert_out_feature)
            lm_loss = classes_loss + feature_map_loss
            if self.am_model.mel_layer is not None:
                encoder_embedding, y_pred = self.am_model(speech_features, training=True)
            else:
                encoder_embedding, y_pred = self.am_model(speech_features, training=True)
            y_pred = tf.nn.softmax(y_pred, -1)

            py_ctc_loss = tf.keras.backend.ctc_batch_cost(tf.cast(py_label, tf.int32),
                                                          tf.cast(y_pred, tf.float32),
                                                          tf.cast(input_length[:, tf.newaxis], tf.int32),
                                                          tf.cast(py_label_length[:, tf.newaxis], tf.int32),
                                                          )
            B = tf.shape(encoder_embedding)[0]
            T = tf.shape(encoder_embedding)[1]
            mask = tf.zeros([B, 1, 1, T])
            lm_result, _, _ = self.lm_model.decoder_part(encoder_embedding, mask)
            lm_result = tf.nn.softmax(lm_result, -1)
            ch_ctc_loss = tf.keras.backend.ctc_batch_cost(tf.cast(txt_label - 1, tf.int32),
                                                          tf.cast(lm_result, tf.float32),
                                                          tf.cast(input_length[:, tf.newaxis], tf.int32),
                                                          tf.cast(txt_label_length[:, tf.newaxis], tf.int32),
                                                          )
            real_length = tf.reduce_min([tf.shape(guide_matrix)[1], tf.shape(encoder_embedding)[1]])
            alig_loss, sim_matrix = self.alig_loss(guide_matrix[:, :real_length], encoder_embedding[:, :real_length],
                                                   py_out_feature)
            am_train_loss = tf.nn.compute_average_loss(alig_loss + ch_ctc_loss + py_ctc_loss,
                                                       global_batch_size=self.global_batch_size)
            lm_train_loss = tf.nn.compute_average_loss(lm_loss,
                                                       global_batch_size=self.global_batch_size)

        am_gradients = tape.gradient(am_train_loss, self.am_model.trainable_variables)
        lm_gradients = tape2.gradient(lm_train_loss, self.lm_model.trainable_variables)
        self.am_optimizer.apply_gradients(zip(am_gradients, self.am_model.trainable_variables))
        self.lm_optimizer.apply_gradients(zip(lm_gradients, self.lm_model.trainable_variables))

        self.train_metrics["am_py_ctc_loss"].update_state(py_ctc_loss)
        self.train_metrics["am_ch_ctc_loss"].update_state(ch_ctc_loss)
        self.train_metrics["lm_class_loss"].update_state(classes_loss)
        self.train_metrics["lm_bert_loss"].update_state(feature_map_loss)

    def alig_loss(self, guide_matrix, encoder_outputs, lm_features):
        lm_features = tf.nn.l2_normalize(lm_features, -1)
        encoder_outputs = tf.nn.l2_normalize(encoder_outputs, -1)
        y_pred = tf.keras.backend.batch_dot(encoder_outputs, lm_features, [2, 2])
        y_pred = y_pred * 10. - 5.
        attention_masks = tf.cast(tf.math.not_equal(guide_matrix, -1.0), tf.float32)
        mask_value = tf.where(attention_masks == 1., 0., -np.inf)
        y_pred += mask_value
        y_pred = tf.nn.softmax(y_pred, -1)
        loss_att = tf.reduce_sum(
            tf.abs(y_pred * guide_matrix) * attention_masks, -1
        )
        loss_att /= tf.reduce_sum(attention_masks, -1)
        return (tf.reduce_mean(loss_att)), y_pred

    @tf.function(experimental_relax_shapes=True)
    def lm_train_step(self, batch):
        inp, tar, feature = batch

        with tf.GradientTape() as tape3:
            predictions, out_feature,_ = self.lm_model(inp,self.lm_model.create_padding_mask(inp),
                                                  training=True)
            classes_loss = self.classes_loss(tar, predictions)
            feature_map_loss = self.bert_feature_loss(feature, out_feature)
            train_loss = classes_loss + feature_map_loss


            train_loss = tf.nn.compute_average_loss(train_loss,
                                                    global_batch_size=self.global_batch_size)

        gradients = tape3.gradient(train_loss, self.lm_model.trainable_variables)
        self.lm_optimizer.apply_gradients(zip(gradients, self.lm_model.trainable_variables))

        self.train_metrics['lm_loss'].update_state(classes_loss)
        self.train_metrics['lm_bert_loss'].update_state(feature_map_loss)
        self.train_metrics['lm_acc'].update_state(self.classes_acc(tar, predictions))

    @tf.function(experimental_relax_shapes=True)
    def am_eval_step(self, batch):
        speech_features, input_length, py_label, py_label_length, txt_label, txt_label_length, bert_features, guide_matrix = batch

        predictions, bert_out_feature, py_out_feature = self.lm_model(py_label,
                                                                      self.lm_model.create_padding_mask(py_label),
                                                                      training=False)
        classes_loss = self.classes_loss(txt_label, predictions)
        feature_map_loss = self.bert_feature_loss(bert_features, bert_out_feature)

        if self.am_model.mel_layer is not None:
            encoder_embedding, y_pred = self.am_model(speech_features, training=False)
        else:
            encoder_embedding, y_pred = self.am_model(speech_features, training=False)
        y_pred = tf.nn.softmax(y_pred, -1)

        py_ctc_loss = tf.keras.backend.ctc_batch_cost(tf.cast(py_label, tf.int32),
                                                      tf.cast(y_pred, tf.float32),
                                                      tf.cast(input_length[:, tf.newaxis], tf.int32),
                                                      tf.cast(py_label_length[:, tf.newaxis], tf.int32),
                                                      )
        B = tf.shape(encoder_embedding)[0]
        T = tf.shape(encoder_embedding)[1]
        mask = tf.zeros([B, 1, 1, T])
        lm_result, _, _ = self.lm_model.decoder_part(encoder_embedding, mask)
        lm_result = tf.nn.softmax(lm_result, -1)
        ch_ctc_loss = tf.keras.backend.ctc_batch_cost(tf.cast(txt_label - 1, tf.int32),
                                                      tf.cast(lm_result, tf.float32),
                                                      tf.cast(input_length[:, tf.newaxis], tf.int32),
                                                      tf.cast(txt_label_length[:, tf.newaxis], tf.int32),
                                                      )

        self.eval_metrics["am_py_ctc_loss"].update_state(py_ctc_loss)
        self.eval_metrics["am_ch_ctc_loss"].update_state(ch_ctc_loss)
        self.eval_metrics["lm_class_loss"].update_state(classes_loss)
        self.eval_metrics["lm_bert_loss"].update_state(feature_map_loss)

    @tf.function(experimental_relax_shapes=True)
    def lm_eval_step(self, batch):
        inp, tar, feature = batch

        predictions, out_feature, _ = self.lm_model(inp, self.lm_model.create_padding_mask(inp),
                                                    training=False)
        classes_loss = self.classes_loss(tar, predictions)
        feature_map_loss = self.bert_feature_loss(feature, out_feature)

        self.eval_metrics['lm_class_loss'].update_state(classes_loss)
        self.eval_metrics['lm_bert_loss'].update_state(feature_map_loss)
        self.eval_metrics['lm_acc'].update_state(self.classes_acc(tar, predictions))
    def bert_feature_loss(self, real, pred):
        mask = tf.cast(tf.not_equal(real, -10.), 1.)
        loss = tf.square(real - pred)
        loss *= mask
        return tf.reduce_mean(tf.reduce_sum(loss, -1) / (tf.reduce_sum(mask, -1) + 1e-6), -1, True)

    def classes_loss(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = tf.keras.losses.sparse_categorical_crossentropy(real, pred, True)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        final = tf.reduce_sum(loss, -1, True) / tf.reduce_sum(mask, -1, True)

        return final

    def classes_acc(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accs = tf.keras.metrics.sparse_categorical_accuracy(real, pred)

        mask = tf.cast(mask, dtype=accs.dtype)
        accs *= mask
        final = tf.reduce_sum(accs, -1) / tf.reduce_sum(mask, -1)

        return tf.reduce_mean(final)

    def compile(self,
                am_model: tf.keras.Model,
                lm_model: tf.keras.Model,
                am_optimizer: any,
                lm_optimizer: any,
                max_to_keep: int = 10):

        with self.strategy.scope():
            self.am_model = am_model
            self.lm_model = lm_model
            if self.am_model.mel_layer is not None:
                self.am_model._build([1, 16000, 1])
            else:

                self.am_model._build([1, 80, 80, 1])
            self.lm_model._build()
            try:
                self.load_checkpoint()
            except:
                logging.info('trainer resume failed')
            self.am_model.summary(line_length=100)
            self.lm_model.summary(line_length=100)
            self.am_optimizer = tf.keras.optimizers.get(am_optimizer)
            self.lm_optimizer = tf.keras.optimizers.get(lm_optimizer)
            if self.is_mixed_precision:
                self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, "dynamic")
        self.set_progbar()
        # self.load_checkpoint()

    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs = epoch
            self.train_progbar.set_description_str(
                f"[Train] [Epoch {epoch}/{self.config['num_epochs']}]")
        self._train_batches()

        self._check_eval_interval()

    def _train_batches(self):
        """Train model one epoch."""

        for batch in self.train_datasets:
            try:
                speech_features, input_length, py_label, py_label_length, txt_label, txt_label_length, bert_features, guide_matrix, lm_pys, lm_chs, lm_ch_features = batch

                self.strategy.run(self.am_train_step, args=(
                [speech_features, input_length, py_label, py_label_length, txt_label, txt_label_length, bert_features,
                 guide_matrix],))

                self.strategy.run(self.lm_train_step, args=(
                    [lm_pys, lm_chs, lm_ch_features],))
                self.steps += 1
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
                speech_features, input_length, py_label, py_label_length, txt_label, txt_label_length, bert_features, guide_matrix, lm_pys, lm_chs, lm_ch_features = batch
                self.strategy.run(self.am_eval_step,args=(
                [speech_features, input_length, py_label, py_label_length, txt_label, txt_label_length, bert_features,
                 guide_matrix],))
                self.strategy.run(self.lm_eval_step, args=(
                [lm_pys, lm_chs, lm_ch_features],))
            except tf.errors.OutOfRangeError:

                pass
            n+=1
            self.eval_progbar.update(1)
            self._print_eval_metrics(self.eval_progbar)
            if n>self.eval_steps_per_epoch:
                break
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")
    def load_checkpoint(self,):
        """Load checkpoint."""
        import os
        self.am_checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints","am")
        self.lm_checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints",'lm')

        files = os.listdir(self.am_checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.am_model.load_weights(os.path.join(self.am_checkpoint_dir, files[-1]))
        self.steps= int(files[-1].split('_')[-1].replace('.h5', ''))

        files = os.listdir(self.lm_checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.lm_model.load_weights(os.path.join(self.lm_checkpoint_dir, files[-1]))

    def save_checkpoint(self, max_save=10):
        """Save checkpoint."""
        import os
        self.am_checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints","am")
        self.lm_checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints","lm")

        if not os.path.exists(self.am_checkpoint_dir):
            os.makedirs(self.am_checkpoint_dir)
        self.am_model.save_weights(os.path.join(self.am_checkpoint_dir, 'model_{}.h5'.format(self.steps)))
        if len(os.listdir(self.am_checkpoint_dir)) > max_save:
            files = os.listdir(self.am_checkpoint_dir)
            files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
            os.remove(os.path.join(self.am_checkpoint_dir, files[0]))

        if not os.path.exists(self.lm_checkpoint_dir):
            os.makedirs(self.lm_checkpoint_dir)
        self.lm_model.save_weights(os.path.join(self.lm_checkpoint_dir, 'model_{}.h5'.format(self.steps)))
        if len(os.listdir(self.lm_checkpoint_dir)) > max_save:
            files = os.listdir(self.lm_checkpoint_dir)
            files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
            os.remove(os.path.join(self.lm_checkpoint_dir, files[0]))

        self.train_progbar.set_postfix_str("Successfully Saved Checkpoint")