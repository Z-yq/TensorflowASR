
import logging
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer

from trainer.base_runners import BaseTrainer



class MultiTaskCTCTrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,
                 speech_featurizer: SpeechFeaturizer,

                 config: dict,
                 is_mixed_precision: bool = False,
                 strategy=None
                 ):
        super(MultiTaskCTCTrainer, self).__init__(config=config,)
        self.speech_featurizer = speech_featurizer

        self.is_mixed_precision = is_mixed_precision
        self.set_strategy(strategy)
    def set_train_metrics(self):
        lists=['classes_loss','ctc1_loss','ctc2_loss','ctc3_loss','acc']

        self.train_metrics={}
        for item in lists:
            self.train_metrics.update({
            item: tf.keras.metrics.Mean("train_{}".format(item), dtype=tf.float32)
        })

    def set_eval_metrics(self):
        lists = ['classes_loss','ctc1_loss','ctc2_loss','ctc3_loss','acc']

        self.eval_metrics={}
        for item in lists:
            self.eval_metrics.update({
            item: tf.keras.metrics.Mean("eval_{}".format(item), dtype=tf.float32),
        })

    def classes_acc(self, real, pred):
        real=tf.cast(real,tf.int32)
        pred=tf.clip_by_value(pred,0,9999)
        pred=tf.cast(pred,tf.int32)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accs = tf.cast(real==pred,tf.float32)

        mask = tf.cast(mask, dtype=accs.dtype)
        accs *= mask
        final=tf.reduce_sum(accs,-1)/tf.reduce_sum(mask,-1)

        return tf.reduce_mean(final)
    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        speech_features, input_length, words_label, words_label_length, phone_label, phone_label_length, py_label, py_label_length= batch

        with tf.GradientTape() as tape:

            ctc1_output, ctc2_output, ctc3_output, final_decoded = self.model(
                    speech_features,
                    training=True)

            ctc1_output = tf.nn.softmax(ctc1_output, -1)
            ctc2_output = tf.nn.softmax(ctc2_output, -1)
            ctc3_output = tf.nn.softmax(ctc3_output, -1)
            final_decoded = tf.nn.softmax(final_decoded, -1)


            ctc1_loss=tf.keras.backend.ctc_batch_cost(words_label,ctc1_output,input_length[:,tf.newaxis],words_label_length[:,tf.newaxis])

            ctc2_loss=tf.keras.backend.ctc_batch_cost(phone_label,ctc2_output,input_length[:,tf.newaxis],phone_label_length[:,tf.newaxis])

            ctc3_loss=tf.keras.backend.ctc_batch_cost(py_label,ctc3_output,input_length[:,tf.newaxis],py_label_length[:,tf.newaxis])
            classes_loss=tf.keras.backend.ctc_batch_cost(py_label,final_decoded,input_length[:,tf.newaxis],py_label_length[:,tf.newaxis])

            train_loss = classes_loss +ctc1_loss+ctc2_loss+ctc3_loss
            train_loss = tf.nn.compute_average_loss(train_loss,
                                                    global_batch_size=self.global_batch_size)
            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)
        decoded_result=tf.keras.backend.ctc_decode(final_decoded,input_length)[0][0]
        acc=self.classes_acc(py_label,decoded_result)
        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics['classes_loss'].update_state(classes_loss)

        self.train_metrics["ctc1_loss"].update_state(ctc1_loss)
        self.train_metrics["ctc2_loss"].update_state(ctc2_loss)
        self.train_metrics["ctc3_loss"].update_state(ctc3_loss)
        self.train_metrics["acc"].update_state(acc)


    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        speech_features, input_length, words_label, words_label_length, phone_label, phone_label_length, py_label, py_label_length = batch
        ctc1_output, ctc2_output, ctc3_output, final_decoded = self.model(
            speech_features,
            training=False)

        ctc1_output = tf.nn.softmax(ctc1_output, -1)
        ctc2_output = tf.nn.softmax(ctc2_output, -1)
        ctc3_output = tf.nn.softmax(ctc3_output, -1)
        final_decoded = tf.nn.softmax(final_decoded, -1)

        ctc1_loss = tf.keras.backend.ctc_batch_cost(words_label, ctc1_output, input_length[:, tf.newaxis],
                                                    words_label_length[:, tf.newaxis])

        ctc2_loss = tf.keras.backend.ctc_batch_cost(phone_label, ctc2_output, input_length[:, tf.newaxis],
                                                    phone_label_length[:, tf.newaxis])

        ctc3_loss = tf.keras.backend.ctc_batch_cost(py_label, ctc3_output, input_length[:, tf.newaxis],
                                                    py_label_length[:, tf.newaxis])
        classes_loss = tf.keras.backend.ctc_batch_cost(py_label, final_decoded, input_length[:, tf.newaxis],
                                                       py_label_length[:, tf.newaxis])

        self.eval_metrics['classes_loss'].update_state(classes_loss)
        self.eval_metrics["ctc1_loss"].update_state(ctc1_loss)
        self.eval_metrics["ctc2_loss"].update_state(ctc2_loss)
        self.eval_metrics["ctc3_loss"].update_state(ctc3_loss)



    def compile(self, model: tf.keras.Model,
                optimizer: any,
                max_to_keep: int = 10):
        f,c=self.speech_featurizer.compute_feature_dim()
        with self.strategy.scope():
            self.model = model
            if self.model.mel_layer is not None:
                self.model._build([1, 16000,1])
            else:
                self.model._build([1, 80, f, c])
            try:
                self.load_checkpoint()
            except:
                logging.info('trainer resume failed')
            self.optimizer = tf.keras.optimizers.get(optimizer)

            if self.is_mixed_precision:
                self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, "dynamic")
        self.model.summary()
        self.set_progbar()
    def _train_batches(self):
        """Train model one epoch."""

        for idx,batch in enumerate(self.train_datasets):
            try:
                self.strategy.run(self._train_step,args=(batch,))
                self.steps+=1
                self.train_progbar.update(1)
                self._print_train_metrics(self.train_progbar)
                self._check_log_interval()
                if self._check_save_interval():
                    break

            except tf.errors.OutOfRangeError:
                continue
            if idx>self.train_steps_per_epoch:
                break
    def _eval_batches(self):
        """One epoch evaluation."""

        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()

        for idx, batch in enumerate(self.eval_datasets):
            try:
                self.strategy.run(self._eval_step,args=(batch,))

            except tf.errors.OutOfRangeError:

                pass

            # Update steps
            self.eval_progbar.update(1)


            # Print eval info to progress bar
            self._print_eval_metrics(self.eval_progbar)
            if idx >=self.eval_steps_per_epoch:
                break
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")
    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs=epoch
            self.train_progbar.set_description_str(
                f"[Train] [Epoch {epoch}/{self.config['num_epochs']}]")


        self._train_batches()

        self._check_eval_interval()

