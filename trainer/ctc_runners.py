

import logging
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer
from trainer.base_runners import BaseTrainer



class CTCTrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 config: dict,
                 is_mixed_precision: bool = False,
                 strategy=None
                 ):
        super(CTCTrainer, self).__init__(config=config,)
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.is_mixed_precision = is_mixed_precision
        self.set_strategy(strategy)

    def set_train_metrics(self):
        self.train_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("train_ctc_loss", dtype=tf.float32),
            "ctc_acc": tf.keras.metrics.Mean("train_ctc_acc", dtype=tf.float32),
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("eval_ctc_loss", dtype=tf.float32),
            "ctc_acc": tf.keras.metrics.Mean("eval_ctc_acc", dtype=tf.float32),
        }

    def ctc_acc(self, labels, y_pred):
        T1 = tf.shape(y_pred)[1]
        T2 = tf.shape(labels)[1]
        T = tf.reduce_min([T1, T2])
        y_pred = y_pred[:, :T]
        labels = labels[:, :T]

        mask = tf.cast(tf.not_equal(labels, self.text_featurizer.pad), 1.)
        y_pred = tf.cast(y_pred, tf.float32)
        labels = tf.cast(labels, tf.float32)

        value = tf.cast(labels == y_pred, tf.float32)

        accs = tf.reduce_sum(value, -1) / (tf.reduce_sum(mask, -1) + 1e-6)
        return accs
    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        features, input_length, labels, label_length= batch

        with tf.GradientTape() as tape:

            y_pred = self.model(features, training=True)
            y_pred=tf.nn.softmax(y_pred,-1)
            tape.watch(y_pred)

            train_loss=tf.keras.backend.ctc_batch_cost(tf.cast(labels, tf.int32),
                                            tf.cast(y_pred, tf.float32),
                                            tf.cast(input_length[:,tf.newaxis], tf.int32),
                                            tf.cast(label_length[:,tf.newaxis], tf.int32),
                                            )
            train_loss = tf.nn.compute_average_loss(train_loss,
                                                    global_batch_size=self.global_batch_size)

            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)

        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        ctc_pred = tf.keras.backend.ctc_decode(y_pred, input_length)[0][0]
        ctc_acc = self.ctc_acc(labels, ctc_pred)
        self.train_metrics['ctc_acc'].update_state(ctc_acc)
        self.train_metrics["ctc_loss"].update_state(train_loss)

    @tf.function(experimental_relax_shapes=True)
    def streaming_train_step(self, batch):
        features, input_length, labels, label_length = batch

        with tf.GradientTape() as tape:

            y_pred,y_pred2 = self.model(features, training=True)
            y_pred = tf.nn.softmax(y_pred, -1)
            y_pred2 = tf.nn.softmax(y_pred2, -1)


            train_loss = tf.keras.backend.ctc_batch_cost(tf.cast(labels, tf.int32),
                                                         tf.cast(y_pred, tf.float32),
                                                         tf.cast(input_length[:, tf.newaxis], tf.int32),
                                                         tf.cast(label_length[:, tf.newaxis], tf.int32),
                                                         )+tf.keras.backend.ctc_batch_cost(tf.cast(labels, tf.int32),
                                                         tf.cast(y_pred2, tf.float32),
                                                         tf.cast(input_length[:, tf.newaxis], tf.int32),
                                                         tf.cast(label_length[:, tf.newaxis], tf.int32),
                                                         )
            train_loss = tf.nn.compute_average_loss(train_loss,
                                                    global_batch_size=self.global_batch_size)

            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)

        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        ctc_pred = tf.keras.backend.ctc_decode(y_pred2, input_length)[0][0]
        ctc_acc = self.ctc_acc(labels, ctc_pred)
        self.train_metrics['ctc_acc'].update_state(ctc_acc)
        self.train_metrics["ctc_loss"].update_state(train_loss)


    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        features,  input_length, labels, label_length = batch

        logits = self.model(features, training=False)
        logits=tf.nn.softmax(logits,-1)
        per_eval_loss = tf.keras.backend.ctc_batch_cost(tf.cast(labels, tf.int32),
                                            tf.cast(logits, tf.float32),
                                            tf.cast(input_length[:,tf.newaxis], tf.int32),
                                            tf.cast(label_length[:,tf.newaxis], tf.int32),
                                            )

        # Update metrics
        ctc_pred = tf.keras.backend.ctc_decode(logits, input_length)[0][0]
        ctc_acc = self.ctc_acc(labels, ctc_pred)
        self.eval_metrics["ctc_loss"].update_state(per_eval_loss)
        self.eval_metrics['ctc_acc'].update_state(ctc_acc)

    @tf.function(experimental_relax_shapes=True)
    def streaming_eval_step(self, batch):
        features, input_length, labels, label_length = batch

        _,logits = self.model(features, training=False)
        logits = tf.nn.softmax(logits, -1)
        per_eval_loss = tf.keras.backend.ctc_batch_cost(tf.cast(labels, tf.int32),
                                                        tf.cast(logits, tf.float32),
                                                        tf.cast(input_length[:, tf.newaxis], tf.int32),
                                                        tf.cast(label_length[:, tf.newaxis], tf.int32),
                                                        )

        # Update metrics
        ctc_pred = tf.keras.backend.ctc_decode(logits, input_length)[0][0]
        ctc_acc = self.ctc_acc(labels, ctc_pred)
        self.eval_metrics["ctc_loss"].update_state(per_eval_loss)
        self.eval_metrics['ctc_acc'].update_state(ctc_acc)

    def compile(self, model: tf.keras.Model,
                optimizer: any,
                max_to_keep: int = 10):
        f,c=self.speech_featurizer.compute_feature_dim()
        with self.strategy.scope():
            self.model = model
            if self.model.mel_layer is not None:
                self.model._build([1, 16000 if self.config['streaming'] is False else self.model.chunk_size *3 , 1])
            else:
                self.model._build([1, 80, f, c])
            try:
                self.load_checkpoint()
            except:
                logging.info('trainer resume failed')
            self.model.summary(line_length=100)
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
                if self.config['streaming']:
                    self.strategy.run(self.streaming_train_step, args=(batch,))
                else:
                    self.strategy.run(self._train_step, args=(batch,))

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
        n = 0
        for batch in self.eval_datasets:
            try:
                if self.config['streaming']:
                    self.strategy.run(self.streaming_eval_step, args=(batch,))
                else:
                    self.strategy.run(self._eval_step, args=(batch,))

            except tf.errors.OutOfRangeError:

                pass
            n += 1

            self.eval_progbar.update(1)

            self._print_eval_metrics(self.eval_progbar)
            if n > self.eval_steps_per_epoch:
                break
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")