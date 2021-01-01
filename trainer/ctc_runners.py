

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
            "ctc_loss": tf.keras.metrics.Mean("train_ctc_loss", dtype=tf.float32)
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("eval_ctc_loss", dtype=tf.float32),
        }

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
        self.eval_metrics["ctc_loss"].update_state(per_eval_loss)

    def compile(self, model: tf.keras.Model,
                optimizer: any,
                max_to_keep: int = 10):
        f,c=self.speech_featurizer.compute_feature_dim()
        with self.strategy.scope():
            self.model = model
            if self.model.mel_layer is not None:
                self.model._build([1, 16000, 1])
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



